# labeling/groq/groq_client.py
"""
Groq API 클라이언트 — 키 + 모델 로테이션으로 RPM/RPD 한도 초과 시 자동 전환

.env 설정:
    GROQ_API_KEYS=key1,key2,key3
    GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant
    GROQ_CALL_INTERVAL_SEC=2   GROQ_MAX_RETRIES=4   GROQ_RPD_LIMIT=14400

로테이션 규칙:
    RPM 한도 → MAX_RETRIES 재시도 후 다음 키 (같은 모델)
    RPD 소진 → 즉시 다음 키
    모델 에러 → 즉시 다음 모델, 키 초기화
    기타 에러 → 다음 키
    모든 조합 소진 → RuntimeError
    세션 내에서 이전 키/모델로 절대 돌아가지 않음
"""
import json
import os
import time
import threading
from collections import deque
from pathlib import Path
from dotenv import load_dotenv

from ..api_client_base import (
    load_api_keys, load_models, RpdCounter,
    parse_retry_delay, now_iso,
    robust_json_parse, retry_json_call,
)

for _p in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env", override=True); break

try:
    from groq import Groq
    try:
        from groq import RateLimitError as _GroqRateLimit
        from groq import NotFoundError  as _GroqNotFound
    except ImportError:
        _GroqRateLimit = _GroqNotFound = None
except ImportError:
    raise ImportError("pip install groq 로 설치하세요.")

# ── 설정 ────────────────────────────────────────────────────

CALL_INTERVAL_SEC = float(os.getenv("GROQ_CALL_INTERVAL_SEC", "2"))
MAX_RETRIES       = int(os.getenv("GROQ_MAX_RETRIES", "4"))
RPD_LIMIT         = int(os.getenv("GROQ_RPD_LIMIT", "14400"))

_counter  = RpdCounter(Path(os.getenv("GROQ_COUNTER_FILE", ".groq_usage.json")), RPD_LIMIT)
_API_KEYS = load_api_keys("GROQ_API_KEYS", "GROQ_API_KEY")
_MODELS   = load_models("GROQ_MODELS", "GROQ_MODEL", "llama-3.3-70b-versatile")


# ── 에러 분류 ────────────────────────────────────────────────

def _is_rate_limit(e: Exception) -> bool:
    if _GroqRateLimit and isinstance(e, _GroqRateLimit):
        return True
    msg = str(e).lower()
    return "rate_limit" in msg or "rate limit" in msg or "429" in str(e)


def _is_model_error(e: Exception) -> bool:
    if _GroqNotFound and isinstance(e, _GroqNotFound):
        return True
    msg = str(e).lower()
    return (
        "model_not_found" in msg or "model not found" in msg
        or "deprecated" in msg or "decommissioned" in msg
        or ("404" in str(e) and "model" in msg)
    )


# ── RPD 공개 API ─────────────────────────────────────────────

def groq_rpd_remaining() -> int:
    return _counter.max_remaining(len(_API_KEYS))


def get_rpd_status() -> dict:
    return {
        "keys":          _counter.status(len(_API_KEYS)),
        "models":        _MODELS,
        "current_model": _MODELS[_cur_model] if _cur_model < len(_MODELS) else None,
        "current_key":   _cur_key,
    }


# ── 클라이언트 캐시 ──────────────────────────────────────────

_clients: dict[int, Groq] = {}
_last_call_time = 0.0


def _get_client(key_idx: int) -> Groq:
    if key_idx not in _clients:
        _clients[key_idx] = Groq(api_key=_API_KEYS[key_idx])
    return _clients[key_idx]


# ── 세션 로테이션 상태 (절대 후퇴 없음) ──────────────────────

_cur_model: int = 0
_cur_key:   int = 0
_state_lock = threading.Lock()

last_call_info: dict         = {}
rotation_log:   deque[dict]  = deque(maxlen=500)


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        time.sleep(CALL_INTERVAL_SEC - elapsed)


# ── 핵심 호출 함수 ──────────────────────────────────────────

def call_groq(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
) -> str:
    """
    Groq 호출. 직접 호출은 thread-safe 하지 않음 — 병렬 환경에서는 call_groq_serial 사용.
    rate_limit  → MAX_RETRIES 재시도 후 다음 키
    model error → 즉시 다음 모델 (키 초기화)
    기타 에러   → 다음 키
    모든 조합 소진 → RuntimeError
    """
    global _cur_model, _cur_key, _last_call_time

    while _cur_model < len(_MODELS):
        model = _MODELS[_cur_model]
        _rotate_model_now = False

        while _cur_key < len(_API_KEYS) and not _rotate_model_now:
            key_idx   = _cur_key
            remaining = _counter.remaining(key_idx)

            if remaining <= 0:
                print(f"  [Groq] key_{key_idx} RPD 소진 → 다음 키")
                rotation_log.append({
                    "reason": "rpd_exhausted", "model": model,
                    "from_key": key_idx, "to_key": key_idx + 1, "ts": now_iso(),
                })
                _cur_key += 1
                continue

            client = _get_client(key_idx)
            print(f"  [Groq] {model} / key_{key_idx} (잔여 RPD: {remaining})")

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
                    _counter.increment(key_idx)
                    last_call_info.update({"model": model, "key_idx": key_idx})
                    return resp.choices[0].message.content.strip()

                except Exception as e:
                    if _is_rate_limit(e):
                        delay = parse_retry_delay(str(e))
                        if attempt < MAX_RETRIES:
                            print(f"  [Groq] {model}/key_{key_idx} RPM 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                            time.sleep(delay)
                            continue
                        print(f"  [Groq] {model}/key_{key_idx} 재시도 {MAX_RETRIES}회 소진 → 다음 키")
                        rotation_log.append({
                            "reason": "rpm_retry_exhausted", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

                    elif _is_model_error(e):
                        next_m = _MODELS[_cur_model + 1] if _cur_model + 1 < len(_MODELS) else None
                        print(f"  [Groq] 모델 {model} 사용 불가 → {'다음 모델: ' + next_m if next_m else '없음'}")
                        rotation_log.append({
                            "reason": "model_unavailable", "from_model": model,
                            "to_model": next_m, "ts": now_iso(), "error": str(e)[:120],
                        })
                        _rotate_model_now = True
                        break

                    else:
                        print(f"  [Groq] {model}/key_{key_idx} 에러 → 다음 키: {str(e)[:80]}")
                        rotation_log.append({
                            "reason": "other_error", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

        next_model = _MODELS[_cur_model + 1] if _cur_model + 1 < len(_MODELS) else None
        reason_str = "사용 불가" if _rotate_model_now else "모든 키 소진"
        print(f"  [Groq] 모델 {model} {reason_str} → {'다음 모델: ' + next_model if next_model else '없음'}")
        if not _rotate_model_now:
            rotation_log.append({
                "reason": "model_rotated", "from_model": model,
                "to_model": next_model, "ts": now_iso(),
            })
        _cur_model += 1
        _cur_key = 0

    raise RuntimeError(
        f"Groq: 모든 모델×키 조합({len(_MODELS)}×{len(_API_KEYS)}) 한도 소진. "
        f"현황: {get_rpd_status()}"
    )


def call_groq_serial(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
) -> str:
    """Thread-safe wrapper: 한 번에 하나의 Groq 호출만 허용."""
    with _state_lock:
        return call_groq(prompt, system)


def call_groq_json(
    prompt: str,
    system: str = "You are a helpful assistant. Respond in JSON only.",
    _retry_json: int = 2,
):
    """Groq 호출 후 JSON 파싱. 실패 시 최대 _retry_json회 재시도."""
    return retry_json_call(
        lambda p: call_groq(p, system),
        prompt, _retry_json=_retry_json, label="Groq",
    )
