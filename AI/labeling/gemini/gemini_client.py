# labeling/gemini/gemini_client.py
"""
Gemini Pro 클라이언트 — omission_risk 전용
키 로테이션으로 RPD/RPM 한도 초과 시 자동 전환 (모델 고정)

.env 설정:
    GEMINI_API_KEYS=key1,key2,key3
    GEMINI_PRO_MODELS=gemini-2.5-pro   (첫 번째 항목만 사용)
    GEMINI_PRO_RPD_LIMIT=100   GEMINI_PRO_CALL_INTERVAL_SEC=13

로테이션 규칙:
    RPM/한도 초과 → MAX_RETRIES 재시도 후 다음 키
    RPD 소진      → 즉시 다음 키
    기타 에러     → 다음 키
    모든 키 소진  → RuntimeError
    세션 내에서 이전 키로 절대 돌아가지 않음
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
_MODEL    = load_models("GEMINI_PRO_MODELS", "GEMINI_PRO_MODEL", "gemini-2.5-pro")[0]


# ── RPD 공개 API ─────────────────────────────────────────────

def rpd_remaining() -> int:
    return _counter.max_remaining(len(_API_KEYS))


def get_rpd_status() -> dict:
    return {
        "keys":        _counter.status(len(_API_KEYS)),
        "model":       _MODEL,
        "current_key": _cur_key,
    }


# ── 클라이언트 캐시 ──────────────────────────────────────────

_clients: dict[int, object] = {}
_last_call_time = 0.0


def _get_client(key_idx: int):
    if key_idx not in _clients:
        _clients[key_idx] = genai.Client(api_key=_API_KEYS[key_idx])
    return _clients[key_idx]


# ── 세션 키 로테이션 상태 (절대 후퇴 없음) ──────────────────

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
    Gemini Pro 호출 (모델 고정, 키만 로테이션).
    직접 호출은 thread-safe 하지 않음 — 병렬 환경에서는 call_gemini_pro_serial 사용.
    RPD 소진 → 즉시 다음 키. RPM 초과 → MAX_RETRIES 재시도 후 다음 키.
    모든 키 소진 → RuntimeError.
    """
    global _cur_key, _last_call_time

    while _cur_key < len(_API_KEYS):
        key_idx   = _cur_key
        remaining = _counter.remaining(key_idx)

        if remaining <= 0:
            print(f"  [Gemini] key_{key_idx} RPD 소진 → 다음 키")
            rotation_log.append({
                "reason": "rpd_exhausted", "model": _MODEL,
                "from_key": key_idx, "to_key": key_idx + 1, "ts": now_iso(),
            })
            _cur_key += 1
            continue

        client = _get_client(key_idx)
        print(f"  [Gemini] {_MODEL} / key_{key_idx} (잔여 RPD: {remaining})")

        for attempt in range(1, MAX_RETRIES + 1):
            _wait_rpm()
            try:
                _last_call_time = time.time()
                resp = client.models.generate_content(model=_MODEL, contents=prompt)
                _counter.increment(key_idx)
                last_call_info.update({"model": _MODEL, "key_idx": key_idx})
                return resp.text.strip()

            except Exception as e:
                msg = str(e)
                is_exhausted = "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

                if is_exhausted and attempt < MAX_RETRIES:
                    delay = parse_retry_delay(msg, default=60.0)
                    print(f"  [Gemini] {_MODEL}/key_{key_idx} 한도 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue

                reason = "rpm_retry_exhausted" if is_exhausted else "other_error"
                print(f"  [Gemini] {_MODEL}/key_{key_idx} → 다음 키: {msg[:80]}")
                rotation_log.append({
                    "reason": reason, "model": _MODEL,
                    "from_key": key_idx, "to_key": key_idx + 1,
                    "ts": now_iso(), "error": msg[:120],
                })
                _cur_key += 1
                break

    raise RuntimeError(
        f"Gemini Pro: 모든 키({len(_API_KEYS)}개) 한도 소진. "
        f"모델: {_MODEL}, 현황: {get_rpd_status()}"
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
