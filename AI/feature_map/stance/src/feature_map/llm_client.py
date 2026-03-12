# llm_client.py
"""
Gemini LLM 클라이언트 공통 모듈
논조 피처(frame, logic, stance) 전체에서 공유
"""
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def get_client() -> genai.Client:
    """싱글턴 클라이언트 반환"""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다.")
        _client = genai.Client(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str = "gemini-2.5-flash-lite") -> str:
    """프롬프트를 LLM에 전달하고 텍스트 응답을 반환"""
    client = get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def parse_json_response(raw: str) -> dict:
    """LLM 응답에서 JSON 파싱 (코드블록 제거 포함)"""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)
