import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN이 .env에 없습니다.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

INPUT_PATH = Path("./AI/clustering/experiments/artifacts/cluster_kospi_1000/cluster_summary.json")
OUTPUT_PATH = Path("./AI/clustering/experiments/artifacts/llm_cluster_summary_hf_kospi.json")

MODEL_NAME = "openai/gpt-oss-20b"
MAX_CLUSTERS = 4
MAX_TITLES_PER_CLUSTER = 5
REQUEST_TIMEOUT = 60
SLEEP_SEC = 1.0


def call_hf_chat(cluster_input: dict) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 뉴스 편집자다. 반드시 한국어 JSON 객체만 출력하라. "
                    "설명문, 코드블록, 마크다운 없이 아래 형식만 출력한다. "
                    '{"headline":"한 줄 제목","summary":"2~3문장 요약"}'
                )
            },
            {
                "role": "user",
                "content": json.dumps(cluster_input, ensure_ascii=False, indent=2)
            }
        ],
        "temperature": 0.2
    }

    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def extract_message_content(response_json: dict) -> str:
    try:
        return response_json["choices"][0]["message"]["content"]
    except Exception:
        return ""


def parse_model_json(text: str) -> dict:
    text = text.strip()

    # 코드블록 제거
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    # JSON 시작~끝 부분만 추출 시도
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "headline": "",
            "summary": "",
            "parse_error": True,
            "raw_text": text
        }


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

clusters = data.get("clusters", [])[:MAX_CLUSTERS]

for cluster in clusters:
    article_titles = [a["title"] for a in cluster.get("articles", [])[:MAX_TITLES_PER_CLUSTER]]

    cluster_input = {
        "query": data.get("query", ""),
        "cluster_id": cluster.get("cluster_id"),
        "article_count": cluster.get("article_count"),
        "representative_title": cluster.get("representative_article", {}).get("title", ""),
        "top_titles": article_titles,
    }

    result_item = {
        "cluster_id": cluster.get("cluster_id"),
        "input": cluster_input,
        "model": MODEL_NAME,
        "success": False,
        "headline": "",
        "summary": "",
        "raw_output": "",
        "error": ""
    }

    try:
        response_json = call_hf_chat(cluster_input)
        content = extract_message_content(response_json)
        parsed = parse_model_json(content)

        result_item["raw_output"] = content
        result_item["headline"] = parsed.get("headline", "")
        result_item["summary"] = parsed.get("summary", "")
        result_item["success"] = not parsed.get("parse_error", False)

        if parsed.get("parse_error", False):
            result_item["error"] = "모델 응답 JSON 파싱 실패"

    except requests.exceptions.Timeout:
        result_item["error"] = "요청 시간 초과"
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        error_text = e.response.text if e.response is not None else str(e)
        result_item["error"] = f"HTTPError {status_code}: {error_text}"
    except Exception as e:
        result_item["error"] = str(e)

    results.append(result_item)
    time.sleep(SLEEP_SEC)

output_data = {
    "query": data.get("query", ""),
    "model": MODEL_NAME,
    "cluster_count": len(results),
    "results": results
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("saved:", OUTPUT_PATH)