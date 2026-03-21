# llm_client.py
"""
Gemini LLM 클라이언트 공통 모듈
논조 피처(frame, logic, stance) 전체에서 공유

Free Tier 제한: gemini-2.5-flash-lite 기준 분당 10회 (RPM 10)
기사 1개당 LLM 3회 호출(frame/logic/stance)이므로
CALL_INTERVAL_SEC=7 로 설정 시 분당 ~8회로 안전하게 유지됨
"""
import os
import json
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

_client = None

# Free Tier RPM 한도 대응: 호출 간 최소 간격 (초)
# 분당 10회 제한 → 6초 간격이 최소, 7초로 여유 확보
CALL_INTERVAL_SEC: float = float(os.getenv("LLM_CALL_INTERVAL_SEC", "10"))

# RESOURCE_EXHAUSTED 시 최대 재시도 횟수
MAX_RETRIES: int = 3

_last_call_time: float = 0.0


def get_client() -> genai.Client:
    """싱글턴 클라이언트 반환"""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다.")
        _client = genai.Client(api_key=api_key)
    return _client


def _wait_for_rate_limit() -> None:
    """직전 호출로부터 CALL_INTERVAL_SEC 이상 경과했는지 확인하고 대기"""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < CALL_INTERVAL_SEC:
        wait = CALL_INTERVAL_SEC - elapsed
        print(f"  [LLM] rate limit 대기 {wait:.1f}s...")
        time.sleep(wait)


def _parse_retry_delay(error_message: str) -> float:
    """에러 메시지에서 retryDelay 값(초)을 파싱. 없으면 60 반환"""
    match = re.search(r"retry[_ ]?(?:in|delay)[:\s'\"]*(\d+)", str(error_message), re.IGNORECASE)
    return float(match.group(1)) + 2 if match else 60.0


def call_llm(prompt: str, model: str = "gemini-2.5-flash-lite") -> str:
    """
    프롬프트를 LLM에 전달하고 텍스트 응답을 반환

    - 호출 간격을 CALL_INTERVAL_SEC 로 자동 조절
    - RESOURCE_EXHAUSTED(429) 발생 시 에러 메시지의 retryDelay 만큼 대기 후 재시도
    """
    global _last_call_time
    client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        _wait_for_rate_limit()
        try:
            _last_call_time = time.time()
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()

        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < MAX_RETRIES:
                delay = _parse_retry_delay(str(e))
                print(f"  [LLM] RESOURCE_EXHAUSTED — {delay:.0f}s 후 재시도 ({attempt}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError("LLM 호출 최대 재시도 횟수 초과")


def parse_json_response(raw: str) -> dict:
    """LLM 응답에서 JSON 파싱 (코드블록 제거 포함)"""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)
