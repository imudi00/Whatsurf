# labeling/groq/groq_client.py
"""
Groq API 클라이언트
키 + 모델 로테이션으로 RPM/RPD 한도 초과 시 자동 전환

.env 설정:
    GROQ_API_KEYS=key1,key2,key3             # 콤마 구분, 따옴표 없이
    GROQ_API_KEY=key1                        # 단일 키 (하위 호환)

    GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant   # 모델 로테이션
    GROQ_MODEL=llama-3.3-70b-versatile                         # 단일 모델 (하위 호환)

    GROQ_CALL_INTERVAL_SEC=2               (30RPM → 2초 간격)
    GROQ_MAX_RETRIES=4
    GROQ_RPD_LIMIT=14400                   (키당 일일 한도)
    GROQ_COUNTER_FILE=.groq_usage.json

로테이션 규칙:
    - RPM 한도(rate_limit) → MAX_RETRIES 재시도 후 다음 키 (같은 모델)
    - RPD 한도(0 remaining) → 즉시 다음 키
    - 모델 에러(deprecated/not_found) → 즉시 다음 모델, 키 초기화
    - 기타 서버 에러 → 다음 키 (계속 시도)
    - 모든 키 소진 → 다음 모델, 키 초기화
    - 세션 내에서 이전 키/모델로 절대 돌아가지 않음 (0→1→0 방지)
"""
import os, re, time, json, threading
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

for _p in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env", override=True); break

try:
    from groq import Groq
    # SDK exception types (더 정확한 에러 감지용)
    try:
        from groq import RateLimitError      as _GroqRateLimit
        from groq import NotFoundError       as _GroqNotFound
        from groq import BadRequestError     as _GroqBadRequest
        from groq import InternalServerError as _GroqServerError
        from groq import APIConnectionError  as _GroqConnError
    except ImportError:
        _GroqRateLimit = _GroqNotFound = _GroqBadRequest = None
        _GroqServerError = _GroqConnError = None
except ImportError:
    raise ImportError("pip install groq 로 설치하세요.")

# ── 설정 ────────────────────────────────────────────────────

CALL_INTERVAL_SEC = float(os.getenv("GROQ_CALL_INTERVAL_SEC", "2"))
MAX_RETRIES       = int(os.getenv("GROQ_MAX_RETRIES", "4"))
RPD_LIMIT         = int(os.getenv("GROQ_RPD_LIMIT", "14400"))
COUNTER_FILE      = Path(os.getenv("GROQ_COUNTER_FILE", ".groq_usage.json"))


# ── API 키 + 모델 목록 로드 ──────────────────────────────────

def _get_api_keys() -> list[str]:
    """GROQ_API_KEYS (콤마 구분) 또는 GROQ_API_KEY 로드. 따옴표 자동 제거."""
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi:
        keys = [k.strip().strip("'\"") for k in multi.split(",") if k.strip().strip("'\"")]
        if keys:
            return keys
    single = os.getenv("GROQ_API_KEY", "").strip().strip("'\"")
    if single:
        return [single]
    raise EnvironmentError(
        "Groq API 키 미설정. .env에 GROQ_API_KEYS=key1,key2 (따옴표 없이) 또는 GROQ_API_KEY=key 를 추가하세요."
    )


def _get_models() -> list[str]:
    """GROQ_MODELS (콤마 구분) 또는 GROQ_MODEL 로드."""
    multi = os.getenv("GROQ_MODELS", "")
    if multi:
        models = [m.strip().strip("'\"") for m in multi.split(",") if m.strip().strip("'\"")]
        if models:
            return models
    return [os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()]


# ── 에러 분류 ────────────────────────────────────────────────

def _is_rate_limit(e: Exception) -> bool:
    """RPM/RPD 한도 초과 에러 여부 (재시도 + 키 로테이션 대상)"""
    if _GroqRateLimit and isinstance(e, _GroqRateLimit):
        return True
    msg = str(e).lower()
    return "rate_limit" in msg or "rate limit" in msg or "429" in str(e)


def _is_model_error(e: Exception) -> bool:
    """모델 사용 불가 에러 (deprecated / not found → 모델 로테이션 대상)"""
    if _GroqNotFound and isinstance(e, _GroqNotFound):
        return True
    msg = str(e).lower()
    return (
        "model_not_found" in msg
        or "model not found" in msg
        or "deprecated" in msg
        or ("404" in str(e) and "model" in msg)
        or "decommissioned" in msg
    )


# ── RPD 카운터 ───────────────────────────────────────────────

def _load_counter() -> dict:
    if COUNTER_FILE.exists():
        try:
            return json.loads(COUNTER_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_counter(data: dict):
    COUNTER_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _rpd_remaining_for(key_idx: int) -> int:
    data  = _load_counter()
    entry = data.get(f"key_{key_idx}", {})
    if entry.get("date") != str(date.today()):
        return RPD_LIMIT
    return max(0, RPD_LIMIT - entry.get("count", 0))


def _increment_counter(key_idx: int):
    data  = _load_counter()
    today = str(date.today())
    k     = f"key_{key_idx}"
    entry = data.get(k, {})
    if entry.get("date") != today:
        entry = {"date": today, "count": 0}
    entry["count"] += 1
    data[k] = entry
    _save_counter(data)


def groq_rpd_remaining() -> int:
    keys = _get_api_keys()
    return max((_rpd_remaining_for(i) for i in range(len(keys))), default=0)


def get_rpd_status() -> dict:
    keys   = _get_api_keys()
    models = _get_models()
    return {
        "keys": {
            f"key_{i}": {
                "remaining": _rpd_remaining_for(i),
                "used":      RPD_LIMIT - _rpd_remaining_for(i),
                "limit":     RPD_LIMIT,
            }
            for i in range(len(keys))
        },
        "models": models,
        "current_model": models[_cur_model] if _cur_model < len(models) else None,
        "current_key":   _cur_key,
    }


# ── 클라이언트 캐시 ──────────────────────────────────────────

_clients: dict[int, Groq] = {}
_last_call_time = 0.0


def _get_client(key_idx: int) -> Groq:
    if key_idx not in _clients:
        keys = _get_api_keys()
        _clients[key_idx] = Groq(api_key=keys[key_idx])
    return _clients[key_idx]


# ── 세션 로테이션 상태 (절대 후퇴 없음) ──────────────────────

_cur_model: int = 0
_cur_key:   int = 0
_state_lock = threading.Lock()   # 멀티스레드 병렬 호출 시 상태 보호

last_call_info: dict = {}
rotation_log:   list[dict] = []


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        time.sleep(CALL_INTERVAL_SEC - elapsed)


def _parse_retry_delay(msg: str) -> float:
    m = re.search(r"retry[_ ]?(?:in|delay|after)[:\s'\"]*([0-9.]+)", msg, re.IGNORECASE)
    return float(m.group(1)) + 1 if m else 30.0


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


# ── 핵심 호출 함수 ──────────────────────────────────────────

def call_groq(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
) -> str:
    """
    Groq 호출 (thread-safe — _state_lock으로 상태 보호).
    rate_limit  → MAX_RETRIES 재시도 후 다음 키 (같은 모델)
    model error → 즉시 다음 모델 (키 초기화)
    기타 에러   → 다음 키 (계속 시도)
    모든 키 소진 → 다음 모델
    모든 조합 소진 → RuntimeError
    세션 내에서 이전 키/모델로 절대 돌아가지 않음.
    """
    global _cur_model, _cur_key, _last_call_time

    models = _get_models()
    keys   = _get_api_keys()

    while _cur_model < len(models):
        model = models[_cur_model]
        _rotate_model_now = False   # 모델 에러로 즉시 전환 플래그

        while _cur_key < len(keys) and not _rotate_model_now:
            key_idx = _cur_key

            # RPD 소진 키는 건너뜀
            remaining = _rpd_remaining_for(key_idx)
            if remaining <= 0:
                print(f"  [Groq] key_{key_idx} RPD 소진 → 다음 키")
                rotation_log.append({
                    "reason": "rpd_exhausted", "model": model,
                    "from_key": key_idx, "to_key": key_idx + 1, "ts": _now_iso(),
                })
                _cur_key += 1
                continue

            client = _get_client(key_idx)
            print(f"  [Groq] {model} / key_{key_idx} 사용 (잔여 RPD: {remaining})")

            for attempt in range(1, MAX_RETRIES + 1):
                _wait_rpm()
                try:
                    _last_call_time = time.time()
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.1,
                    )
                    _increment_counter(key_idx)
                    last_call_info.update({"model": model, "key_idx": key_idx})
                    return resp.choices[0].message.content.strip()

                except Exception as e:
                    if _is_rate_limit(e):
                        delay = _parse_retry_delay(str(e))
                        if attempt < MAX_RETRIES:
                            print(f"  [Groq] {model}/key_{key_idx} RPM 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                            time.sleep(delay)
                            continue
                        # MAX_RETRIES 소진 → 다음 키
                        print(f"  [Groq] {model}/key_{key_idx} 재시도 {MAX_RETRIES}회 소진 → 다음 키")
                        rotation_log.append({
                            "reason": "rpm_retry_exhausted", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break  # while _cur_key 루프로

                    elif _is_model_error(e):
                        # 모델 deprecated/not found → 즉시 다음 모델
                        next_m = models[_cur_model + 1] if _cur_model + 1 < len(models) else None
                        print(f"  [Groq] 모델 {model} 사용 불가 → {'다음 모델: ' + next_m if next_m else '사용 가능한 모델 없음'}")
                        rotation_log.append({
                            "reason": "model_unavailable", "from_model": model,
                            "to_model": next_m, "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _rotate_model_now = True
                        break  # for loop 탈출 → while _cur_key도 탈출 예정

                    else:
                        # 기타 에러 (서버 에러 등) → 현재 키 포기, 다음 키 시도
                        print(f"  [Groq] {model}/key_{key_idx} 에러 → 다음 키 시도: {str(e)[:80]}")
                        rotation_log.append({
                            "reason": "other_error", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break  # while _cur_key 루프로

        # 이 모델의 모든 키 소진 (또는 모델 에러)
        next_model = models[_cur_model + 1] if _cur_model + 1 < len(models) else None
        reason_str = "사용 불가" if _rotate_model_now else "모든 키 소진"
        if next_model:
            print(f"  [Groq] 모델 {model} {reason_str} → 다음 모델: {next_model}")
        else:
            print(f"  [Groq] 모델 {model} {reason_str}, 사용 가능한 모델 없음")

        if not _rotate_model_now:
            # 정상 소진 시에만 model_rotated 로그 (model_unavailable은 이미 기록됨)
            rotation_log.append({
                "reason": "model_rotated", "from_model": model,
                "to_model": next_model, "ts": _now_iso(),
            })
        _cur_model += 1
        _cur_key = 0

    raise RuntimeError(
        f"Groq: 모든 모델×키 조합({len(models)}모델×{len(keys)}키) 한도 소진.\n"
        f"내일 다시 실행하거나 .env에 키/모델을 추가하세요.\n"
        f"현황: {get_rpd_status()}"
    )


def call_groq_serial(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
) -> str:
    """Thread-safe wrapper: 한 번에 하나의 Groq 호출만 허용 (병렬 feature 처리 시 사용)."""
    with _state_lock:
        return call_groq(prompt, system)


def _fix_json_string(text: str) -> str:
    """LLM JSON 출력의 일반적인 문법 오류를 수정."""
    # <think>...</think> 블록 제거 (Qwen3, llama-4 등 reasoning 모델)
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    # 마크다운 코드블록 제거
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    # 전각 따옴표 → 반각
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # trailing comma 제거: ,} 또는 ,]
    text = re.sub(r',\s*([\]\}])', r'\1', text)
    # 객체 사이 쉼표 누락: } { → },{  (배열 내부)
    text = re.sub(r'\}\s*\n\s*\{', '},\n{', text)
    return text


def _robust_json_parse(raw: str):
    """
    LLM 출력에서 JSON 배열/객체를 최대한 강건하게 파싱.

    처리하는 LLM 실수:
      1. 마크다운 코드블록
      2. trailing comma  → {"a":1,}
      3. 객체 사이 쉼표 누락  → } {  →  },{
      4. 전각 따옴표  → "key"
      5. 앞뒤 자연어 텍스트 무시
      6. 줄 단위 JSON 객체 (NDJSON-like)
    """
    text = _fix_json_string(raw)

    # 전략 1: 전체 텍스트 직접 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 전략 2: JSON 배열 [ ... ] 추출
    m_arr = re.search(r'\[[\s\S]*\]', text)
    if m_arr:
        candidate = _fix_json_string(m_arr.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 전략 3: JSON 객체 { ... } 추출 → 배열로 감싸기
    m_obj = re.search(r'\{[\s\S]*\}', text)
    if m_obj:
        candidate = _fix_json_string(m_obj.group(0))
        try:
            return [json.loads(candidate)]
        except json.JSONDecodeError:
            pass

    # 전략 4: 줄 단위 { } 객체 수집 (NDJSON-like 응답)
    objects = []
    for line in text.splitlines():
        line = line.strip().rstrip(',')
        if line.startswith('{') and line.endswith('}'):
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if objects:
        return objects

    raise json.JSONDecodeError(
        f"JSON 파싱 전략 모두 실패. 원본 (앞 200자): {raw[:200]}", raw, 0
    )


def call_groq_json(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
    _retry_json: int = 2,   # JSON 파싱 실패 시 API 재호출 횟수
):
    """
    Groq 호출 후 JSON 파싱.
    파싱 실패 시 _robust_json_parse로 재시도, 그래도 실패하면 API를 재호출.
    """
    last_raw = ""
    for attempt in range(1, _retry_json + 2):   # 첫 시도 + _retry_json 회
        if attempt == 1:
            raw = call_groq(prompt, system)
        else:
            # JSON이 불완전할 때 더 짧게 출력하도록 유도하는 재시도 프롬프트
            retry_prompt = (
                prompt
                + "\n\n⚠ 이전 응답이 유효한 JSON이 아니었습니다. "
                "반드시 유효한 JSON 배열만 출력하세요. 설명이나 마크다운 없이."
            )
            print(f"  [Groq JSON] 파싱 실패 → API 재시도 ({attempt-1}/{_retry_json})")
            raw = call_groq(retry_prompt, system)

        last_raw = raw
        try:
            return _robust_json_parse(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt <= _retry_json:
                continue
            break

    raise json.JSONDecodeError(
        f"Groq JSON 파싱 {_retry_json+1}회 시도 모두 실패.\n원본: {last_raw[:300]}",
        last_raw, 0
    )
