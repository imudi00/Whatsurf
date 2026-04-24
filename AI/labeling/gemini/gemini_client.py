# labeling/gemini/gemini_client.py
"""
Gemini Pro 클라이언트 — omission_risk 전용
키 + 모델 로테이션으로 RPD/RPM 한도 초과 시 자동 전환

.env 설정:
    GEMINI_API_KEYS=key1,key2,key3
    GEMINI_PRO_MODELS=gemini-2.5-pro,gemini-2.0-flash
    GEMINI_PRO_RPD_LIMIT=100   GEMINI_PRO_CALL_INTERVAL_SEC=13

로테이션 규칙:
    키 소진(RPD/RPM) → 다음 키 (같은 모델)
    모든 키 소진     → 다음 모델, 키 인덱스 초기화
    모든 조합 소진   → RuntimeError
    세션 내에서 이전 키/모델로 절대 돌아가지 않음
"""
import os
import time
import threading
from collections import deque
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from ..api_client_base import (
    load_api_keys, load_models, RpdCounter,
    parse_retry_delay, now_iso,
    robust_json_parse, retry_json_call,
)

for _p in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env", override=True); break

# ── 설정 ────────────────────────────────────────────────────

RPD_LIMIT         = int(os.getenv("GEMINI_PRO_RPD_LIMIT", "100"))
CALL_INTERVAL_SEC = float(os.getenv("GEMINI_PRO_CALL_INTERVAL_SEC", "13"))
MAX_RETRIES       = 3

_counter  = RpdCounter(Path(os.getenv("GEMINI_PRO_COUNTER_FILE", ".gemini_pro_usage.json")), RPD_LIMIT)
_API_KEYS = load_api_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
_MODELS   = load_models("GEMINI_PRO_MODELS", "GEMINI_PRO_MODEL", "gemini-2.5-pro")


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


# ── RPD 공개 API ─────────────────────────────────────────────

def rpd_remaining() -> int:
    return _counter.max_remaining(len(_API_KEYS))


def get_rpd_status() -> dict:
    return {
        "keys":          _counter.status(len(_API_KEYS)),
        "models":        _MODELS,
        "current_model": _MODELS[_cur_model] if _cur_model < len(_MODELS) else None,
        "current_key":   _cur_key,
    }


# ── 클라이언트 캐시 ──────────────────────────────────────────

_clients: dict[int, object] = {}
_last_call_time = 0.0


def _get_client(key_idx: int):
    if key_idx not in _clients:
        _clients[key_idx] = genai.Client(api_key=_API_KEYS[key_idx])
    return _clients[key_idx]


# ── 세션 로테이션 상태 (절대 후퇴 없음) ──────────────────────

_cur_model: int = 0
_cur_key:   int = 0
_state_lock = threading.Lock()

last_call_info: dict        = {}
rotation_log:   deque[dict] = deque(maxlen=500)


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        time.sleep(CALL_INTERVAL_SEC - elapsed)


# ── 핵심 호출 함수 ──────────────────────────────────────────

def call_gemini_pro(prompt: str) -> str:
    """
    Gemini Pro 호출. 직접 호출은 thread-safe 하지 않음 — 병렬 환경에서는 call_gemini_pro_serial 사용.
    키 소진 → 다음 키. 모든 키 소진 → 다음 모델. 모든 조합 소진 → RuntimeError.
    """
    global _cur_model, _cur_key, _last_call_time

    while _cur_model < len(_MODELS):
        model = _MODELS[_cur_model]
        _rotate_model_now = False

        while _cur_key < len(_API_KEYS) and not _rotate_model_now:
            key_idx   = _cur_key
            remaining = _counter.remaining(key_idx)

            if remaining <= 0:
                print(f"  [Gemini] key_{key_idx} RPD 소진 → 다음 키")
                rotation_log.append({
                    "reason": "rpd_exhausted", "model": model,
                    "from_key": key_idx, "to_key": key_idx + 1, "ts": now_iso(),
                })
                _cur_key += 1
                continue

            client = _get_client(key_idx)
            print(f"  [Gemini] {model} / key_{key_idx} (잔여 RPD: {remaining})")

            for attempt in range(1, MAX_RETRIES + 1):
                _wait_rpm()
                try:
                    _last_call_time = time.time()
                    resp = client.models.generate_content(model=model, contents=prompt)
                    _counter.increment(key_idx)
                    last_call_info.update({"model": model, "key_idx": key_idx})
                    return resp.text.strip()

                except Exception as e:
                    if _is_resource_exhausted(e):
                        delay = parse_retry_delay(str(e), default=60.0)
                        if attempt < MAX_RETRIES:
                            print(f"  [Gemini] {model}/key_{key_idx} 한도 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                            time.sleep(delay)
                            continue
                        print(f"  [Gemini] {model}/key_{key_idx} 재시도 {MAX_RETRIES}회 소진 → 다음 키")
                        rotation_log.append({
                            "reason": "rpm_retry_exhausted", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

                    elif _is_model_error(e):
                        next_m = _MODELS[_cur_model + 1] if _cur_model + 1 < len(_MODELS) else None
                        print(f"  [Gemini] 모델 {model} 사용 불가 → {'다음 모델: ' + next_m if next_m else '없음'}")
                        rotation_log.append({
                            "reason": "model_unavailable", "from_model": model,
                            "to_model": next_m, "ts": now_iso(), "error": str(e)[:120],
                        })
                        _rotate_model_now = True
                        break

                    else:
                        print(f"  [Gemini] {model}/key_{key_idx} 에러 → 다음 키: {str(e)[:80]}")
                        rotation_log.append({
                            "reason": "other_error", "model": model,
                            "from_key": key_idx, "to_key": key_idx + 1,
                            "ts": now_iso(), "error": str(e)[:120],
                        })
                        _cur_key += 1
                        break

        next_model = _MODELS[_cur_model + 1] if _cur_model + 1 < len(_MODELS) else None
        reason_str = "사용 불가" if _rotate_model_now else "모든 키 소진"
        print(f"  [Gemini] 모델 {model} {reason_str} → {'다음 모델: ' + next_model if next_model else '없음'}")
        if not _rotate_model_now:
            rotation_log.append({
                "reason": "model_rotated", "from_model": model,
                "to_model": next_model, "ts": now_iso(),
            })
        _cur_model += 1
        _cur_key = 0

    raise RuntimeError(
        f"Gemini Pro: 모든 모델×키 조합({len(_MODELS)}×{len(_API_KEYS)}) 한도 소진. "
        f"현황: {get_rpd_status()}"
    )


def call_gemini_pro_serial(prompt: str) -> str:
    """Thread-safe wrapper: 한 번에 하나의 Gemini 호출만 허용."""
    with _state_lock:
        return call_gemini_pro(prompt)


def call_gemini_pro_json(prompt: str, _retry_json: int = 2):
    """Gemini 호출 후 JSON 파싱. 실패 시 최대 _retry_json회 재시도."""
    return retry_json_call(
        call_gemini_pro, prompt, _retry_json=_retry_json, label="Gemini",
    )
