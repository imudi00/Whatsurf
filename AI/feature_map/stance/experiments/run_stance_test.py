# experiments/run_stance_test.py
"""
논조 피처(frame / logic / stance) 실험 실행 스크립트
사용법:
    python run_stance_test.py --input ../data_samples/stance_sample.csv
"""
import argparse
import os
import sys
import io
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_map.preprocessor import build_article_struct
from src.feature_map.frame import extract_frame
from src.feature_map.logic import extract_logic
from src.feature_map.stance import extract_stance
from src.feature_map.io_utils import save_json


def _load_csv(path: str) -> pd.DataFrame:
    raw = open(path, "rb").read()
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df = pd.read_csv(io.StringIO(raw.decode(enc)))
            df.columns = [c.strip().lower() for c in df.columns]
            return df
        except Exception:
            continue
    raise ValueError(f"CSV 로드 실패: {path}")


def run_stance_pipeline(texts: list) -> list:
    """단일 기사 목록에 대해 frame → logic → stance 순서로 분석"""
    results = []
    for text in texts:
        struct = build_article_struct(text)

        frame_result  = extract_frame(struct)
        logic_result  = extract_logic(struct)
        stance_result = extract_stance(
            struct,
            frame=frame_result["frame"],
            logic=logic_result["logic"],
        )

        results.append({
            "frame":         frame_result["frame"],
            "frame_reason":  frame_result.get("reason"),
            "logic":         logic_result["logic"],
            "logic_reason":  logic_result.get("reason"),
            "stance_score":  stance_result["stance_score"],
            "dominant_tone": stance_result.get("dominant_tone"),
            "key_evidence":  stance_result.get("key_evidence"),
        })
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

    df = _load_csv(args.input)
    texts = df["body"].fillna("").astype(str).tolist()[:20]  # 실험용 샘플 제한

    results = run_stance_pipeline(texts)
    save_json({"stance": results}, os.path.join(args.out_dir, "stance_results.json"))

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
