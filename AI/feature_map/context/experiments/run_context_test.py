# experiments/run_context_test.py
"""
맥락 피처(body_depth / omission_risk) 실험 실행 스크립트
사용법:
    python run_context_test.py --input ../data_samples/context_sample.csv
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_map.keyword_extractor import extract_features
from src.feature_map.body_depth import compute_body_depth, describe_body_depth
from src.feature_map.omission_risk import compute_omission_risk
from src.feature_map.io_utils import load_news_csv, save_json


def run_body_depth_pipeline(texts: list) -> list:
    """전체 기사 목록에 대해 body_depth 점수 계산"""
    results = []
    for text in texts:
        features = extract_features(text)
        score = compute_body_depth(text, features)
        results.append({
            "body_depth": score,
            "level": describe_body_depth(score),
            "paragraph_count": features["paragraph_count"],
            "sentence_count": features["sentence_count"],
            "info_sentence_ratio": round(features["source_count"] / max(features["sentence_count"], 1), 4),
        })
    return results


def run_omission_risk_pipeline(texts: list) -> list:
    """전체 기사 목록에 대해 omission_risk 계산 (각 기사 vs 나머지)"""
    results = []
    for i, text in enumerate(texts):
        cluster = [t for j, t in enumerate(texts) if j != i]
        risk = compute_omission_risk(text, cluster, extract_features)
        results.append({"omission_risk": risk})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="뉴스 CSV 경로 (body 컬럼 필요)")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_news_csv(args.input)
    texts = df["body"].tolist()[:30]  # 실험용 샘플 제한

    # body_depth
    depth_results = run_body_depth_pipeline(texts)
    save_json({"body_depth": depth_results}, os.path.join(args.out_dir, "body_depth_results.json"))

    # omission_risk
    omission_results = run_omission_risk_pipeline(texts)
    save_json({"omission_risk": omission_results}, os.path.join(args.out_dir, "omission_risk_results.json"))

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
