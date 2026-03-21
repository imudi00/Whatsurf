import json
from pathlib import Path

INPUT_PATH = Path("./AI/clustering/experiments/artifacts/llm_cluster_summary_gemini_kospi.json")
OUTPUT_PATH = Path("./AI/clustering/experiments/artifacts/llm_cluster_summary_gemini_kospi_parsed.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

parsed = []

for item in data:
    raw = item["raw_output"].strip()

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = {
            "headline": "",
            "summary": "",
            "parse_error": True,
            "raw_output": raw,
        }

    parsed.append({
        "cluster_id": item["cluster_id"],
        "article_count": item["article_count"],
        "representative_title": item["representative_title"],
        "headline": obj.get("headline", ""),
        "summary": obj.get("summary", ""),
        "parse_error": obj.get("parse_error", False),
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(parsed, f, ensure_ascii=False, indent=2)

print("saved:", OUTPUT_PATH)