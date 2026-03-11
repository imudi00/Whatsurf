import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def extract_stance(struct: dict, frame: str, logic: str) -> dict:
    prompt = f"""뉴스 기사의 주요 사안에 대한 논조(stance)를 분석하세요.

## 점수 기준
- +1.0: 사안을 전적으로 옹호/긍정
- +0.5: 주로 우호적, 일부 비판 포함
-  0.0: 중립 또는 양쪽 균형
- -0.5: 주로 비판적, 일부 긍정 포함
- -1.0: 사안을 강하게 비난/반대

## 이미 분석된 맥락
- 프레임: {frame}
- 논거 유형: {logic}

## 분석 근거 (우선순위)
1. 판단/감정 동사의 방향과 강도: {struct['judgment_words']}
2. 인용 주체의 포지션: {struct['quotes']}
3. 리드문의 방향성: {struct['lead'][:150]}
4. 수치가 긍정/부정 맥락인가: {struct['numbers']}

JSON만 출력:
{{"stance_score": 0.0, "dominant_tone": "비판적/우호적/중립", "key_evidence": "근거 한 줄"}}"""

    response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["stance_score"] = max(-1.0, min(1.0, float(result["stance_score"])))
    return result