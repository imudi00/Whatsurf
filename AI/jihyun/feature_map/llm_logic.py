import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]

FEW_SHOT = """
예시1)
출처: 야당 원내대표, 시민단체
판단어: 규탄, 비판, 철회 촉구
인용: "명백한 실정이다", "즉각 사퇴하라"
→ logic: 정책적 비난

예시2)
출처: 경제연구원, 교수, 분석가
수치: GDP 0.3% 하락, 실업률 0.5%p 상승
인용: "장기적으로 소비 위축이 불가피"
→ logic: 전문가 견해

예시3)
출처: 피해자 가족, 생존자
판단어: 호소, 눈물, 절박
인용: "아이를 잃은 뒤 삶이 무너졌다"
→ logic: 피해자 서사

예시4)
출처: 정부 부처, 연구소
수치: 물가 3.2%, 전기료 인상분 월 4000원
판단어: 없음
인용: 공식 발표 위주
→ logic: 사실/정보 전달
"""

def extract_logic(struct: dict) -> dict:
    prompt = f"""뉴스 기사의 핵심 논거 유형을 분류하세요.

## 논거 유형 정의
- 정책적 비난: 정부/기관 정책을 직접 공격
- 전문가 견해: 전문가/연구 데이터로 주장 뒷받침
- 피해자 서사: 피해 당사자의 경험/감정 중심
- 파급효과: 사건이 미치는 사회/경제적 영향 분석
- 해결책 제시: 대안/정책 제안 중심
- 사실/정보 전달: 판단 없이 팩트 나열

## 판단 순서
1. 인용 주체가 누구인가 (전문가 vs 피해자 vs 정치인)
2. 수치가 논거로 사용되는가
3. 판단/감정 동사의 강도

{FEW_SHOT}

## 분석 대상
출처: {struct['sources']}
수치: {struct['numbers']}
인용구: {struct['quotes']}
판단어: {struct['judgment_words']}
문장 샘플: {struct['sampled_sentences']}

반드시 다음 중 하나만 JSON으로 출력:
{{"logic": "...", "reason": "한 줄 근거"}}"""

    response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)

    if result.get("logic") not in LOGIC_LABELS:
        result["logic"] = "사실/정보 전달"
    return result