# experiments/run_stance_test.py
"""
논조 피처(frame / logic / stance) 실험 실행 스크립트 (Supabase 연동)
사용법:
    python run_stance_test.py
    python run_stance_test.py --limit 10
"""
import argparse
import os
import sys

# experiments/ 기준: ../..  → feature_map/ (source이 feature_map 안일 때)
# experiments/ 기준: ../../.. → 프로젝트 루트 (source이 feature_map과 같은 레벨일 때)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.feature_map.preprocessor import build_article_struct
from src.feature_map.stance_combined import extract_stance_all
from src.feature_map.io_utils import save_json
from source.data_loader import load_ai_test_df


def run_stance_pipeline(texts: list) -> list:
    """단일 기사 목록에 대해 frame/logic/stance를 LLM 1회로 분석"""
    results = []
    for i, text in enumerate(texts):
        print(f"  [{i+1}/{len(texts)}] 분석 중...")
        struct = build_article_struct(text)
        r = extract_stance_all(struct)

        results.append({
            "frame":         r["frame"],
            "frame_reason":  r.get("frame_reason"),
            "logic":         r["logic"],
            "logic_reason":  r.get("logic_reason"),
            "stance_score":  r["stance_score"],
            "dominant_tone": r.get("dominant_tone"),
            "key_evidence":  r.get("key_evidence"),
        })
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

    results = run_stance_pipeline(texts)
    save_json({"stance": results}, os.path.join(args.out_dir, "stance_results.json"))

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
