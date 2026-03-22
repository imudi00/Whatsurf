# LLM 자동 평가

import os
import sys
import json
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

for path in [PACKAGE_PARENT, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

from AI.clustering.src.clustering.evaluation import (
    evaluate_cluster_input_quality,
    jaccard_similarity,
    normalize_text,
)

INPUT_PATH = Path("./llm_results/llm_cluster_summary_gemini_samsung.json")
OUTPUT_PATH = Path("./llm_results/llm_cluster_summary_gemini_samsung_evaluated.json")


def headline_length_ok(headline: str) -> bool:
    headline = str(headline).strip()
    return 8 <= len(headline) <= 40


def summary_length_ok(summary: str) -> bool:
    summary = str(summary).strip()
    return 30 <= len(summary) <= 300


def text_redundancy_score(text: str) -> float:
    words = normalize_text(text).split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return round(1 - unique_ratio, 4)


def headline_relevance_score(headline: str, input_titles: list[str]) -> float:
    if not headline or not input_titles:
        return 0.0
    context = " ".join(input_titles)
    return round(jaccard_similarity(headline, context), 4)


def summary_relevance_score(summary: str, key_sentences: list[str], titles: list[str]) -> float:
    if not summary:
        return 0.0

    context_parts = []
    if key_sentences:
        context_parts.extend(key_sentences)
    if titles:
        context_parts.extend(titles)

    if not context_parts:
        return 0.0

    context = " ".join(context_parts)
    return round(jaccard_similarity(summary, context), 4)


def evaluate_llm_output(item: dict) -> dict:
    cluster_input = item.get("input", {})
    titles = cluster_input.get("titles", [])
    key_sentences = cluster_input.get("key_sentences", [])

    headline = item.get("generated_headline", "") or item.get("headline", "")
    summary = item.get("generated_summary", "") or item.get("summary", "")

    return {
        "format_ok": bool(headline and summary),
        "headline_length_ok": headline_length_ok(headline),
        "summary_length_ok": summary_length_ok(summary),
        "headline_relevance": headline_relevance_score(headline, titles),
        "summary_relevance": summary_relevance_score(summary, key_sentences, titles),
        "headline_redundancy": text_redundancy_score(headline),
        "summary_redundancy": text_redundancy_score(summary),
    }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    evaluated_results = []

    summary_stats = {
        "count": 0,
        "format_ok_count": 0,
        "headline_length_ok_count": 0,
        "summary_length_ok_count": 0,
        "headline_relevance_sum": 0.0,
        "summary_relevance_sum": 0.0,
        "headline_redundancy_sum": 0.0,
        "summary_redundancy_sum": 0.0,
        "title_cohesion_sum": 0.0,
        "representative_score_sum": 0.0,
        "key_sentence_relevance_sum": 0.0,
        "low_relevance_ratio_sum": 0.0,
    }

    for item in results:
        cluster_input = item.get("input", {})
        non_llm_eval = evaluate_cluster_input_quality(cluster_input)
        llm_eval = evaluate_llm_output(item)

        item["non_llm_evaluation"] = non_llm_eval
        item["llm_evaluation"] = llm_eval
        evaluated_results.append(item)

        summary_stats["count"] += 1
        summary_stats["format_ok_count"] += int(llm_eval["format_ok"])
        summary_stats["headline_length_ok_count"] += int(llm_eval["headline_length_ok"])
        summary_stats["summary_length_ok_count"] += int(llm_eval["summary_length_ok"])
        summary_stats["headline_relevance_sum"] += llm_eval["headline_relevance"]
        summary_stats["summary_relevance_sum"] += llm_eval["summary_relevance"]
        summary_stats["headline_redundancy_sum"] += llm_eval["headline_redundancy"]
        summary_stats["summary_redundancy_sum"] += llm_eval["summary_redundancy"]

        summary_stats["title_cohesion_sum"] += non_llm_eval["title_cohesion"]
        summary_stats["representative_score_sum"] += non_llm_eval["representative_score"]
        summary_stats["key_sentence_relevance_sum"] += non_llm_eval["key_sentence_relevance"]
        summary_stats["low_relevance_ratio_sum"] += non_llm_eval["low_relevance_ratio"]

    n = max(summary_stats["count"], 1)

    output = {
        "model": data.get("model", ""),
        "cluster_count": summary_stats["count"],
        "aggregate": {
            "format_ok_rate": round(summary_stats["format_ok_count"] / n, 4),
            "headline_length_ok_rate": round(summary_stats["headline_length_ok_count"] / n, 4),
            "summary_length_ok_rate": round(summary_stats["summary_length_ok_count"] / n, 4),
            "avg_headline_relevance": round(summary_stats["headline_relevance_sum"] / n, 4),
            "avg_summary_relevance": round(summary_stats["summary_relevance_sum"] / n, 4),
            "avg_headline_redundancy": round(summary_stats["headline_redundancy_sum"] / n, 4),
            "avg_summary_redundancy": round(summary_stats["summary_redundancy_sum"] / n, 4),
            "avg_title_cohesion": round(summary_stats["title_cohesion_sum"] / n, 4),
            "avg_representative_score": round(summary_stats["representative_score_sum"] / n, 4),
            "avg_key_sentence_relevance": round(summary_stats["key_sentence_relevance_sum"] / n, 4),
            "avg_low_relevance_ratio": round(summary_stats["low_relevance_ratio_sum"] / n, 4),
        },
        "results": evaluated_results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("saved:", OUTPUT_PATH)
    print("aggregate:", json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()