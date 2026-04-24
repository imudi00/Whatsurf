# llm/llm_client.py
"""
Gemini LLM 공통 클라이언트 (auto_labeler.py 전용)
RPM / TPM / RPD 예외처리 및 모델 자동 폴백

예외 처리 흐름:
    RPM 초과  → retryDelay 만큼 대기 후 재시도
    TPM 초과  → 지수 백오프 (15→30→60→120s)
    RPD 소진  → MODEL_FALLBACK_LIST 다음 모델로 전환
    전체 소진 → RuntimeError

.env 키:
    GEMINI_API_KEY / GEMINI_API_KEYS
    LLM_MODEL_1=gemini-2.5-flash-lite
    LLM_MODEL_2=gemini-2.0-flash-lite
    LLM_MODEL_3=gemini-1.5-flash
    LLM_CALL_INTERVAL_SEC=7   LLM_MAX_RETRIES=4
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

for _p in [Path(__file__).resolve().parents[i] for i in range(1, 5)]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env"); break
else:
    load_dotenv()

# shared JSON parser from labeling package
_ai_dir = str(Path(__file__).resolve().parents[1])
if _ai_dir not in sys.path:
    sys.path.insert(0, _ai_dir)
from labeling.api_client_base import robust_json_parse, parse_retry_delay

MODEL_FALLBACK_LIST: list[str] = [
    os.getenv("LLM_MODEL_1", "gemini-2.5-flash-lite"),
    os.getenv("LLM_MODEL_2", "gemini-2.0-flash-lite"),
    os.getenv("LLM_MODEL_3", "gemini-1.5-flash"),
]
CALL_INTERVAL_SEC = float(os.getenv("LLM_CALL_INTERVAL_SEC", "7"))
MAX_RETRIES       = int(os.getenv("LLM_MAX_RETRIES", "4"))

_api_keys:  list[str]       = []
_key_index: int             = 0
_client:    genai.Client | None = None
_model_index = 0
_last_call_time = 0.0


def get_client() -> genai.Client:
    global _client, _api_keys, _key_index
    if _client is None:
        raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if not raw:
            raise EnvironmentError("GEMINI_API_KEYS가 .env에 설정되지 않았습니다.")
        _api_keys = [k.strip() for k in raw.replace(',', ' ').split() if k.strip()]
        if not _api_keys:
            raise ValueError("사용 가능한 Gemini API 키가 없습니다.")
        _client = genai.Client(api_key=_api_keys[_key_index])
        print(f"  [LLM] 키 로드 완료: {len(_api_keys)}개")
    return _client


def rotate_key() -> bool:
    global _client, _key_index
    if len(_api_keys) > 1:
        _key_index = (_key_index + 1) % len(_api_keys)
        _client = genai.Client(api_key=_api_keys[_key_index])
        print(f"  [LLM] 다음 API 키로 전환: {_key_index + 1}번 키")
        return True
    return False


def current_model() -> str:
    return MODEL_FALLBACK_LIST[_model_index]


def _next_model() -> bool:
    global _model_index
    if _model_index + 1 < len(MODEL_FALLBACK_LIST):
        _model_index += 1
        print(f"  [LLM] 모델 전환 → {current_model()}")
        return True
    return False


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        w = CALL_INTERVAL_SEC - elapsed
        print(f"  [LLM] RPM 간격 대기 {w:.1f}s...")
        time.sleep(w)


def call_llm(prompt: str) -> str:
    """단일 프롬프트 전송. RPM/TPM/RPD 예외 자동 처리."""
    global _last_call_time
    client, attempt = get_client(), 0

    while True:
        attempt += 1
        _wait_rpm()
        try:
            _last_call_time = time.time()
            resp = client.models.generate_content(model=current_model(), contents=prompt)
            text = resp.text
            if not text:
                try:
                    reason = resp.candidates[0].finish_reason if resp.candidates else "UNKNOWN"
                except Exception:
                    reason = "UNKNOWN"
                print(f"  [LLM] 빈 응답 (finish_reason={reason}) — 재시도 {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Gemini 빈 응답 {MAX_RETRIES}회 반복 (finish_reason={reason})")
            return text.strip()

        except genai_errors.ClientError as e:
            msg = str(e)

            if "RESOURCE_EXHAUSTED" in msg and ("per day" in msg.lower() or "daily" in msg.lower()):
                print(f"  [LLM] RPD 소진 ({current_model()})")
                if not _next_model(): raise RuntimeError("모든 Gemini 모델 RPD 소진") from e
                attempt = 0; continue

            if "404" in msg or "NOT_FOUND" in msg:
                print(f"  [LLM] 모델 없음/지원 중단 ({current_model()}) → 다음 모델")
                if not _next_model(): raise RuntimeError("모든 Gemini 모델이 지원되지 않음") from e
                attempt = 0; continue

            if "RESOURCE_EXHAUSTED" in msg and attempt < MAX_RETRIES:
                rotated = rotate_key()
                client = get_client()
                delay = parse_retry_delay(msg) if not rotated else 2
                print(f"  [LLM] RPM 초과 — {'키 전환' if rotated else f'{delay:.0f}s 대기'} ({attempt}/{MAX_RETRIES})")
                if not rotated:
                    time.sleep(delay)
                continue

            if ("429" in msg or "token" in msg.lower()) and attempt < MAX_RETRIES:
                wait = 15 * (2 ** (attempt - 1))
                print(f"  [LLM] TPM 초과 — {wait}s 백오프 ({attempt}/{MAX_RETRIES})")
                time.sleep(wait); continue

            if attempt >= MAX_RETRIES:
                if not _next_model(): raise
                attempt = 0; continue

            raise


def call_llm_json(prompt: str, _retry: int = 2):
    """call_llm 후 JSON 파싱. 실패 시 재시도."""
    last_raw = ""
    suffix = "\n\n⚠ 이전 응답이 유효한 JSON이 아니었습니다. 반드시 유효한 JSON만 출력하세요. 설명이나 마크다운 없이."
    for attempt in range(1, _retry + 2):
        raw = call_llm(prompt if attempt == 1 else prompt + suffix)
        if attempt > 1:
            print(f"  [LLM JSON] 파싱 실패 → 재시도 ({attempt - 1}/{_retry})")
        last_raw = raw
        try:
            return robust_json_parse(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt <= _retry:
                continue
    raise json.JSONDecodeError(
        f"JSON 파싱 {_retry + 1}회 모두 실패. 원본: {last_raw[:300]}", last_raw, 0
    )
