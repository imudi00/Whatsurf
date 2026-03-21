import json
from google import genai
from google.genai import types

client = genai.Client()

def generate_cluster_headline_summary(cluster_for_llm: dict) -> dict:
    prompt = f"""
너는 뉴스 타임라인 생성을 위한 뉴스 클러스터 요약기다.

반드시 JSON만 출력해라.

형식:
{{
  "headline": "대표 headline",
  "summary": "2~3문장 요약"
}}

규칙:
- headline은 클러스터 전체를 대표해야 한다
- summary는 핵심 내용만 간결하게 정리한다
- 과장 금지
- 추측 금지
- 한국어로 작성
- JSON 외 텍스트 금지

입력:
{json.dumps(cluster_for_llm, ensure_ascii=False, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=300,
        )
    )

    return json.loads(response.text)