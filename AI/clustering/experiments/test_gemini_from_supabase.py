import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PACKAGE_PARENT not in sys.path:
    sys.path.append(PACKAGE_PARENT)

from google import genai
from google.genai import types
from AI.source.data_loader import load_news_df

client = genai.Client()

df = load_news_df(limit=20)

print("[INFO] loaded_rows =", len(df))
print("[INFO] columns =", df.columns.tolist())

if "title" not in df.columns:
    raise ValueError("title 컬럼이 없습니다.")

df["title"] = df["title"].fillna("").astype(str)
titles = [t for t in df["title"].tolist() if t.strip()]

if len(titles) < 5:
    raise ValueError("title이 5개보다 적습니다.")

cluster_input = {
    "representative_title": titles[0],
    "titles": titles[:5],
    "meta": {
        "source": "supabase.news",
        "article_count": len(titles[:5])
    }
}

prompt = f"""
너는 뉴스 클러스터 요약기다.
입력은 같은 사건에 대한 기사 제목 묶음이다.

반드시 JSON만 출력해라.
형식:
{{
  "headline": "대표 headline",
  "summary": "2~3문장 요약"
}}

조건:
- headline은 클러스터 전체를 대표해야 한다
- summary는 간결하게 핵심만 정리한다
- 과장 금지
- 추측 금지
- 한국어로 작성
- JSON 외 텍스트 금지

입력:
{json.dumps(cluster_input, ensure_ascii=False, indent=2)}
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

print("\n=== INPUT ===")
print(json.dumps(cluster_input, ensure_ascii=False, indent=2))

print("\n=== OUTPUT ===")
print(response.text)