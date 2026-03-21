# experiments/run_context_test.py
"""
맥락 피처(body_depth / omission_risk) 실험 실행 스크립트 (Supabase 연동)
사용법:
    python run_context_test.py
    python run_context_test.py --limit 10
"""
import argparse
import os
import sys

# experiments/ 기준: ../..  → feature_map/ (source이 feature_map 안일 때)
# experiments/ 기준: ../../.. → 프로젝트 루트 (source이 feature_map과 같은 레벨일 때)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.feature_map.keyword_extractor import extract_features
from src.feature_map.body_depth import compute_body_depth, describe_body_depth
from src.feature_map.omission_risk import compute_omission_risk
from src.feature_map.io_utils import save_json
from source.data_loader import load_ai_test_df


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
    parser.add_argument("--limit", type=int, default=10, help="Supabase에서 가져올 기사 수 (기본: 10)")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Supabase에서 데이터 로드
    print(f"[Supabase] news 테이블에서 {args.limit}개 로드 중...")
    df = load_ai_test_df(limit=args.limit)
    print(f"[Supabase] 로드 완료: {len(df)}행, 컬럼: {list(df.columns)}")

    texts = df["body"].fillna("").astype(str).tolist()

    # body_depth
    depth_results = run_body_depth_pipeline(texts)
    save_json({"body_depth": depth_results}, os.path.join(args.out_dir, "body_depth_results.json"))

    # omission_risk
    omission_results = run_omission_risk_pipeline(texts)
    save_json({"omission_risk": omission_results}, os.path.join(args.out_dir, "omission_risk_results.json"))

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
