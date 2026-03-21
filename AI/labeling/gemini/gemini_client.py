# labeling/gemini/gemini_client.py
"""
Gemini Pro 클라이언트 — omission_risk 전용
키 + 모델 로테이션으로 RPD/RPM 한도 초과 시 자동 전환

.env 설정:
    GEMINI_API_KEYS=key1,key2,key3           # 콤마 구분, 따옴표 없이
    GEMINI_API_KEY=key1                      # 단일 키 (하위 호환)

    GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash   # 모델 로테이션 (콤마 구분)
    GEMINI_PRO_MODEL=gemini-2.5-pro                     # 단일 모델 (하위 호환)

    GEMINI_PRO_RPD_LIMIT=100                (키당 일일 한도)
    GEMINI_PRO_CALL_INTERVAL_SEC=13         (5RPM → 12초 최소, 13초 여유)
    GEMINI_PRO_COUNTER_FILE=.gemini_pro_usage.json

로테이션 규칙:
    - 키 소진(RPD/RPM) → 다음 키 (같은 모델)
    - 모든 키 소진     → 다음 모델, 키 인덱스 초기화
    - 모든 모델+키 소진 → RuntimeError
    - 세션 내에서는 절대 이전 키로 돌아가지 않음 (0→1→0 방지)
"""
import os, re, time, json
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# ── 에러 분류 ────────────────────────────────────────────────

def _is_resource_exhausted(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()

def _is_model_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "model_not_found" in msg or "model not found" in msg
        or "deprecated" in msg or "not found" in msg
        or "invalid_argument" in msg
    )

for _p in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env", override=True); break

# ── 설정 ────────────────────────────────────────────────────

RPD_LIMIT         = int(os.getenv("GEMINI_PRO_RPD_LIMIT", "100"))
CALL_INTERVAL_SEC = float(os.getenv("GEMINI_PRO_CALL_INTERVAL_SEC", "13"))
MAX_RETRIES       = 3
COUNTER_FILE      = Path(os.getenv("GEMINI_PRO_COUNTER_FILE", ".gemini_pro_usage.json"))


# ── API 키 + 모델 목록 로드 ──────────────────────────────────

def _get_api_keys() -> list[str]:
    """GEMINI_API_KEYS (콤마 구분) 또는 GEMINI_API_KEY 로드. 따옴표 자동 제거."""
    multi = os.getenv("GEMINI_API_KEYS", "")
    if multi:
        keys = [k.strip().strip("'\"") for k in multi.split(",") if k.strip().strip("'\"")]
        if keys:
            return keys
    single = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
    if single:
        return [single]
    raise EnvironmentError(
        "Gemini API 키 미설정. .env에 GEMINI_API_KEYS=key1,key2 (따옴표 없이) 또는 GEMINI_API_KEY=key 를 추가하세요."
    )


def _get_models() -> list[str]:
    """GEMINI_PRO_MODELS (콤마 구분) 또는 GEMINI_PRO_MODEL 로드."""
    multi = os.getenv("GEMINI_PRO_MODELS", "")
    if multi:
        models = [m.strip().strip("'\"") for m in multi.split(",") if m.strip().strip("'\"")]
        if models:
            return models
    return [os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro").strip()]


# ── RPD 카운터 (키별 파일 영속) ─────────────────────────────

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


def rpd_remaining() -> int:
    """현재 가용 키 중 가장 많은 잔여 RPD"""
    keys = _get_api_keys()
    return max((_rpd_remaining_for(i) for i in range(len(keys))), default=0)


def get_rpd_status() -> dict:
    """모든 키의 RPD 현황 + 사용 모델 목록 반환 (리포트용)"""
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


# ── 클라이언트 캐시 (키별) ───────────────────────────────────

_clients: dict[int, object] = {}
_last_call_time = 0.0


def _get_client(key_idx: int):
    if key_idx not in _clients:
        keys = _get_api_keys()
        _clients[key_idx] = genai.Client(api_key=keys[key_idx])
    return _clients[key_idx]


# ── 세션 로테이션 상태 (절대 후퇴 없음) ──────────────────────

_cur_model: int = 0   # 현재 모델 인덱스 (세션 내 단방향 증가)
_cur_key:   int = 0   # 현재 키 인덱스 (모델 전환 시 초기화)

# 마지막 성공 호출 정보
last_call_info: dict = {}     # {"model": "...", "key_idx": 0}

# 로테이션 이벤트 로그 (외부에서 report에 기록 가능)
rotation_log: list[dict] = []


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        time.sleep(CALL_INTERVAL_SEC - elapsed)


def _parse_retry_delay(msg: str) -> float:
    m = re.search(r"retry[_ ]?(?:in|delay)[:\s'\"]*(\d+)", msg, re.IGNORECASE)
    return float(m.group(1)) + 2 if m else 60.0


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


# ── 핵심 호출 함수 ──────────────────────────────────────────

def call_gemini_pro(prompt: str) -> str:
    """
    Gemini Pro 호출.
    키 소진 → 다음 키 (같은 모델).
    모든 키 소진 → 다음 모델 + 키 초기화.
    모든 조합 소진 → RuntimeError.
    세션 내에서 이전 키/모델로 절대 돌아가지 않음.
    """
    global _cur_model, _cur_key, _last_call_time

    models = _get_models()
    keys   = _get_api_keys()

    while _cur_model < len(models):
        model = models[_cur_model]
        _rotate_model_now = False

        while _cur_key < len(keys) and not _rotate_model_now:
            key_idx = _cur_key

            remaining = _rpd_remaining_for(key_idx)
            if remaining <= 0:
                print(f"  [Gemini] key_{key_idx} RPD 소진 → 다음 키")
                rotation_log.append({
                    "reason": "rpd_exhausted", "model": model,
                    "from_key": key_idx, "to_key": key_idx + 1, "ts": _now_iso(),
                })
                _cur_key += 1
                continue

            client = _get_client(key_idx)
            print(f"  [Gemini] {model} / key_{key_idx} 사용 (잔여 RPD: {remaining})")

            for attempt in range(1, MAX_RETRIES + 1):
                _wait_rpm()
                try:
                    _last_call_time = time.time()
                    resp = client.models.generate_content(model=model, contents=prompt)
                    _increment_counter(key_idx)
                    last_call_info.update({"model": model, "key_idx": key_idx})
                    return resp.text.strip()

                except Exception as e:
                    if _is_resource_exhausted(e):
                        delay = _parse_retry_delay(str(e))
                        if attempt < MAX_RETRIES:
                            print(f"  [Gemini] {model}/key_{key_idx} 한도 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                            time.sleep(delay)
                            continue
                        # 재시도 소진 → 다음 키
                        print(f"  [Gemini] {model}/key_{key_idx} 재시도 {MAX_RETRIES}회 소진 → 다음 키")
                        rotation_log.append({
                            "reason": "rpm_retry_exhausted", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

                    elif _is_model_error(e):
                        next_m = models[_cur_model + 1] if _cur_model + 1 < len(models) else None
                        print(f"  [Gemini] 모델 {model} 사용 불가 → {'다음 모델: ' + next_m if next_m else '없음'}")
                        rotation_log.append({
                            "reason": "model_unavailable", "from_model": model,
                            "to_model": next_m, "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _rotate_model_now = True
                        break

                    else:
                        # 기타 에러 → 다음 키 시도
                        print(f"  [Gemini] {model}/key_{key_idx} 에러 → 다음 키: {str(e)[:80]}")
                        rotation_log.append({
                            "reason": "other_error", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": _now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

        # 이 모델의 모든 키 소진 (또는 모델 에러) → 다음 모델
        next_model = models[_cur_model + 1] if _cur_model + 1 < len(models) else None
        reason_str = "사용 불가" if _rotate_model_now else "모든 키 소진"
        print(f"  [Gemini] 모델 {model} {reason_str} → {'다음 모델: ' + next_model if next_model else '사용 가능한 모델 없음'}")
        if not _rotate_model_now:
            rotation_log.append({
                "reason": "model_rotated", "from_model": model,
                "to_model": next_model, "ts": _now_iso(),
            })
        _cur_model += 1
        _cur_key = 0

    raise RuntimeError(
        f"Gemini Pro: 모든 모델×키 조합({len(models)}모델×{len(keys)}키) 한도 소진.\n"
        f"내일 다시 실행하거나 .env에 키/모델을 추가하세요.\n"
        f"현황: {get_rpd_status()}"
    )


def _fix_json_string(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text).strip()
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = re.sub(r',\s*([\]\}])', r'\1', text)
    text = re.sub(r'\}\s*\n\s*\{', '},\n{', text)
    return text


def _robust_json_parse(raw: str):
    """groq_client._robust_json_parse 와 동일한 강건 JSON 파싱 로직."""
    text = _fix_json_string(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m_arr = re.search(r'\[[\s\S]*\]', text)
    if m_arr:
        candidate = _fix_json_string(m_arr.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    m_obj = re.search(r'\{[\s\S]*\}', text)
    if m_obj:
        candidate = _fix_json_string(m_obj.group(0))
        try:
            result = json.loads(candidate)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

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
        f"Gemini JSON 파싱 실패. 원본: {raw[:200]}", raw, 0
    )


def call_gemini_pro_json(prompt: str, _retry_json: int = 2):
    """Gemini 호출 후 강건 JSON 파싱. 실패 시 최대 _retry_json회 재시도."""
    last_raw = ""
    for attempt in range(1, _retry_json + 2):
        if attempt == 1:
            raw = call_gemini_pro(prompt)
        else:
            retry_prompt = (
                prompt
                + "\n\n⚠ 이전 응답이 유효한 JSON이 아니었습니다. "
                "반드시 유효한 JSON 배열만 출력하세요. 설명 없이."
            )
            print(f"  [Gemini JSON] 파싱 실패 → 재시도 ({attempt-1}/{_retry_json})")
            raw = call_gemini_pro(retry_prompt)

        last_raw = raw
        try:
            return _robust_json_parse(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt <= _retry_json:
                continue
            break

    raise json.JSONDecodeError(
        f"Gemini JSON {_retry_json+1}회 시도 실패. 원본: {last_raw[:300]}",
        last_raw, 0
    )
