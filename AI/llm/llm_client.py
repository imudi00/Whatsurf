# llm/llm_client.py
"""
Gemini LLM 공통 클라이언트
RPM / TPM / RPD 예외처리 및 모델 자동 폴백

예외 처리 흐름:
    RPM 초과  → 에러의 retryDelay 만큼 대기 후 재시도
    TPM 초과  → 지수 백오프 (15→30→60→120s)
    RPD 소진  → MODEL_FALLBACK_LIST 다음 모델로 전환
    전체 소진 → RuntimeError

.env 키:
    GEMINI_API_KEY
    LLM_MODEL_1=gemini-2.5-flash-lite   (기본)
    LLM_MODEL_2=gemini-2.0-flash-lite
    LLM_MODEL_3=gemini-1.5-flash-8b
    LLM_CALL_INTERVAL_SEC=7
    LLM_MAX_RETRIES=4
"""
import os, re, time, json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# .env 탐색: llm/ → AI/ → 프로젝트루트
for _p in [Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]]:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env"); break

MODEL_FALLBACK_LIST: list = [
    os.getenv("LLM_MODEL_1", "gemini-2.5-flash-lite"),
    os.getenv("LLM_MODEL_2", "gemini-2.0-flash-lite"),
    os.getenv("LLM_MODEL_3", "gemini-1.5-flash-8b"),
]
CALL_INTERVAL_SEC: float = float(os.getenv("LLM_CALL_INTERVAL_SEC", "7"))
MAX_RETRIES:       int   = int(os.getenv("LLM_MAX_RETRIES", "4"))

_client         = None
_model_index    = 0
_last_call_time = 0.0


def get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY 미설정")
        _client = genai.Client(api_key=key)
    return _client


def current_model() -> str:
    return MODEL_FALLBACK_LIST[_model_index]


def _next_model() -> bool:
    global _model_index
    if _model_index + 1 < len(MODEL_FALLBACK_LIST):
        _model_index += 1
        print(f"  [LLM] ▶ 모델 전환 → {current_model()}")
        return True
    return False


def _wait_rpm():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        w = CALL_INTERVAL_SEC - elapsed
        print(f"  [LLM] RPM 간격 대기 {w:.1f}s...")
        time.sleep(w)


def _parse_retry_delay(msg: str) -> float:
    m = re.search(r"retry[_ ]?(?:in|delay)[:\s'\"]*(\d+)", msg, re.IGNORECASE)
    return float(m.group(1)) + 2 if m else 60.0


def call_llm(prompt: str) -> str:
    """단일 프롬프트 전송. RPM/TPM/RPD 예외 자동 처리."""
    global _last_call_time
    client, attempt = get_client(), 0

    while True:
        attempt += 1
        _wait_rpm()
        try:
            _last_call_time = time.time()
            return client.models.generate_content(model=current_model(), contents=prompt).text.strip()

        except genai_errors.ClientError as e:
            msg = str(e)

            # RPD 소진 → 모델 전환
            if "RESOURCE_EXHAUSTED" in msg and ("per day" in msg.lower() or "daily" in msg.lower()):
                print(f"  [LLM] ✕ RPD 소진 ({current_model()})")
                if not _next_model(): raise RuntimeError("모든 Gemini 모델 RPD 소진") from e
                attempt = 0; continue

            # RPM 초과 → retryDelay 대기
            if "RESOURCE_EXHAUSTED" in msg and attempt < MAX_RETRIES:
                delay = _parse_retry_delay(msg)
                print(f"  [LLM] ✕ RPM 초과 — {delay:.0f}s 대기 ({attempt}/{MAX_RETRIES})")
                time.sleep(delay); continue

            # TPM 초과 → 지수 백오프
            if ("429" in msg or "token" in msg.lower()) and attempt < MAX_RETRIES:
                wait = 15 * (2 ** (attempt - 1))
                print(f"  [LLM] ✕ TPM 초과 — {wait}s 백오프 ({attempt}/{MAX_RETRIES})")
                time.sleep(wait); continue

            # 재시도 소진 → 모델 전환 시도
            if attempt >= MAX_RETRIES:
                if not _next_model(): raise
                attempt = 0; continue

            raise


def call_llm_json(prompt: str):
    """call_llm 후 JSON 파싱"""
    raw = call_llm(prompt)
    return json.loads(raw.replace("```json", "").replace("```", "").strip())
