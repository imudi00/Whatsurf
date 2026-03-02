# llm_features.py
'''
 few-shot 예시(기사 2~3개 + 정답 라벨)를 프롬프트에 추가해서 정확도 향상 예정.
 사용 모델은 Gemini 2.0 Flash. 분당 15회ㅡ 하루 1500회 제한 무료 모델
'''
from google import genai
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]

def extract_tone_features(text: str, keywords: dict) -> dict:
    entity_summary = ", ".join([e['word'] for e in keywords["entities"][:10]])

    prompt = f"""다음은 한국어 뉴스 기사 본문과 주요 키워드입니다.

[주요 키워드]
{entity_summary}

[기사 본문 (앞 600자)]
{text[:1000]}

아래 세 가지를 분석하여 JSON으로만 출력하세요. 다른 텍스트 없이 JSON만 출력.

1. frame: 기사의 주제 접근 방식. 반드시 다음 중 하나만 선택:
   {FRAME_LABELS}

2. logic: 기사의 핵심 논거 유형. 반드시 다음 중 하나만 선택:
   {LOGIC_LABELS}

3. stance_score: 기사가 주요 사안을 옹호(+1) 또는 비판(-1)하는 정도.
   -1.0 ~ 1.0 사이 소수점 한 자리 숫자.

출력 형식:
{{"frame": "경제적가치", "logic": "파급효과", "stance_score": -0.4}}"""

    response = client.models.generate_content(model = 'gemini-2.5-flash-lite',contents=prompt)
    raw = response.text.strip()

    # ```json ... ``` 마크다운 펜스 제거
    raw = raw.replace("```json", "").replace("```", "").strip()

    result = json.loads(raw)

    # 유효성 검증
    if result.get("frame") not in FRAME_LABELS:
        result["frame"] = "인과"
    if result.get("logic") not in LOGIC_LABELS:
        result["logic"] = "사실/정보 전달"
    result["stance_score"] = max(-1.0, min(1.0, float(result["stance_score"])))

    return result