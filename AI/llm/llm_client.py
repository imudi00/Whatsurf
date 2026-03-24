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
# [여기서부터 교체 시작]
# .env 탐색: 현재 파일 위치에서 위로 4단계까지 올라가며 찾습니다.
found_env = False
for i in range(1, 5):
    _p = Path(__file__).resolve().parents[i]
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        found_env = True
        break

# 만약 위에서 못 찾았다면 현재 실행 중인 폴더(루트)에서 마지막으로 시도합니다.
if not found_env:
    load_dotenv()
# [여기까지 교체 끝]

MODEL_FALLBACK_LIST: list = [
    os.getenv("LLM_MODEL_1", "gemini-2.5-flash-lite"),
    os.getenv("LLM_MODEL_2", "gemini-2.0-flash-lite"),
    os.getenv("LLM_MODEL_3", "gemini-1.5-flash"),      # gemini-1.5-flash-8b deprecated → gemini-1.5-flash
]
CALL_INTERVAL_SEC: float = float(os.getenv("LLM_CALL_INTERVAL_SEC", "7"))
MAX_RETRIES:       int   = int(os.getenv("LLM_MAX_RETRIES", "4"))

# --- [수정 시작] ---
_api_keys = []      # 쪼개진 키들을 담을 리스트
_key_index = 0      # 현재 사용 중인 키의 번호
_client = None      # 실제 구동될 클라이언트
_model_index = 0
_last_call_time = 0.0

def get_client() -> genai.Client:
    global _client, _api_keys, _key_index
    
    if _client is None:
        load_dotenv()
        # 1. .env에서 'S'가 붙은 전체 키 뭉치를 가져옵니다.
        raw_keys = os.getenv("GEMINI_API_KEYS")
        
        if not raw_keys:
            # 혹시 모르니 단수형도 한번 더 체크합니다.
            raw_keys = os.getenv("GEMINI_API_KEY")
            
        if not raw_keys:
            raise EnvironmentError("GEMINI_API_KEYS가 .env에 설정되지 않았습니다.")
            
        # 2. 쉼표(,)나 공백으로 구분된 키들을 리스트로 쪼개고 공백을 제거합니다.
        _api_keys = [k.strip() for k in raw_keys.replace(',', ' ').split() if k.strip()]
        
        if not _api_keys:
            raise ValueError("사용 가능한 Gemini API 키가 리스트에 없습니다.")
            
        # 3. 첫 번째 키로 클라이언트를 만듭니다.
        _client = genai.Client(api_key=_api_keys[_key_index])
        print(f"  [LLM] 키 로드 완료: 총 {len(_api_keys)}개의 키를 찾았습니다.")
        
    return _client

# 추가: 키가 한도 초과(429)일 때 다음 키로 바꿔주는 함수
def rotate_key():
    global _client, _key_index
    if len(_api_keys) > 1:
        _key_index = (_key_index + 1) % len(_api_keys)
        _client = genai.Client(api_key=_api_keys[_key_index])
        print(f"  [LLM] 🔑 다음 API 키로 전환: {_key_index + 1}번 키 사용")
        return True
    return False
# --- [수정 끝] ---


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
            resp = client.models.generate_content(model=current_model(), contents=prompt)
            text = resp.text  # safety filter / quota 시 None 가능
            if not text:
                # finish_reason 정보 출력 후 재시도
                try:
                    reason = resp.candidates[0].finish_reason if resp.candidates else "UNKNOWN"
                except Exception:
                    reason = "UNKNOWN"
                print(f"  [LLM] ⚠️  빈 응답 (finish_reason={reason}) — 재시도 {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"Gemini 빈 응답 {MAX_RETRIES}회 반복 (finish_reason={reason})")
            return text.strip()

        except genai_errors.ClientError as e:
            msg = str(e)

            # RPD 소진 → 모델 전환
            if "RESOURCE_EXHAUSTED" in msg and ("per day" in msg.lower() or "daily" in msg.lower()):
                print(f"  [LLM] ✕ RPD 소진 ({current_model()})")
                if not _next_model(): raise RuntimeError("모든 Gemini 모델 RPD 소진") from e
                attempt = 0; continue

            # 404 모델 없음 → 즉시 다음 모델로 전환
            if "404" in msg or "NOT_FOUND" in msg:
                print(f"  [LLM] ✕ 모델 없음/지원 중단 ({current_model()}) → 다음 모델로 전환")
                if not _next_model(): raise RuntimeError("모든 Gemini 모델이 지원되지 않음") from e
                attempt = 0; continue

            # RPM 초과 → 키 로테이션 후 대기
            if "RESOURCE_EXHAUSTED" in msg and attempt < MAX_RETRIES:
                rotated = rotate_key()  # 다음 키로 전환 (여러 키가 있을 때)
                client = get_client()   # 새 클라이언트 반영
                delay = _parse_retry_delay(msg) if not rotated else 2
                print(f"  [LLM] ✕ RPM 초과 — {'키 전환' if rotated else f'{delay:.0f}s 대기'} ({attempt}/{MAX_RETRIES})")
                if not rotated:
                    time.sleep(delay)
                continue

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


def _robust_json_parse(raw: str):
    """
    LLM 출력에서 JSON 객체/배열을 강건하게 파싱.
    모델이 backtick, 설명글, trailing comma 등을 붙여도 처리.
    """
    import re as _re

    def _clean(text: str) -> str:
        # 마크다운 코드블록 제거
        text = _re.sub(r"```(?:json)?\s*", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r"```", "", text).strip()
        # trailing comma 제거: ,} 또는 ,]
        text = _re.sub(r",\s*([\]\}])", r"\1", text)
        return text

    text = _clean(raw)

    # 전략 1: 전체 텍스트 직접 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 전략 2: JSON 객체 { ... } 추출
    m = _re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(_clean(m.group(0)))
        except json.JSONDecodeError:
            pass

    # 전략 3: JSON 배열 [ ... ] 추출
    m = _re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(_clean(m.group(0)))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"JSON 파싱 실패. 원본 앞 200자: {raw[:200]}", raw, 0
    )


def call_llm_json(prompt: str, _retry: int = 2):
    """call_llm 후 JSON 파싱 (강건한 파서 사용, 실패 시 재시도)."""
    last_raw = ""
    for attempt in range(1, _retry + 2):
        if attempt == 1:
            raw = call_llm(prompt)
        else:
            retry_prompt = (
                prompt
                + "\n\n⚠ 이전 응답이 유효한 JSON이 아니었습니다. "
                "반드시 유효한 JSON만 출력하세요. 설명이나 마크다운 없이."
            )
            print(f"  [LLM JSON] 파싱 실패 → 재시도 ({attempt - 1}/{_retry})")
            raw = call_llm(retry_prompt)
        last_raw = raw
        try:
            return _robust_json_parse(raw)
        except (json.JSONDecodeError, ValueError):
            if attempt <= _retry:
                continue
            break
    raise json.JSONDecodeError(
        f"JSON 파싱 {_retry + 1}회 모두 실패. 원본: {last_raw[:300]}", last_raw, 0
    )
