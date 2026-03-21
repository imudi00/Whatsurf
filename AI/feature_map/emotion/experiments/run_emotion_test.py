# experiments/run_emotion_test.py
"""
감정 피처 실험 실행 스크립트 (Supabase 연동)
사용법:
    python run_emotion_test.py
    python run_emotion_test.py --limit 10
"""
import argparse
import os
import sys

# experiments/ 기준: ../..  → feature_map/ (source이 feature_map 안일 때)
# experiments/ 기준: ../../.. → 프로젝트 루트 (source이 feature_map과 같은 레벨일 때)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.feature_map.io_utils import save_json
from src.feature_map.emotion_model import EmotionService
from src.feature_map.loaded_words import detect_loaded_words, loaded_word_density
from src.feature_map.bias_vector import compute_bias_vector, normalize_bias_vector
from source.data_loader import load_ai_test_df


def run_emotion_pipeline(texts: list, service: EmotionService) -> list:
    """텍스트 목록에 대해 감정 예측 수행"""
    results = []
    for text in texts:
        output = service.predict(text)
        results.append({
            "label": output.label,
            "intensity": output.intensity,
            "probs": output.probs,
        })
    return results


def run_loaded_words_pipeline(texts: list) -> list:
    """텍스트 목록에 대해 편향 단어 감지 수행"""
    return [
        {
            "found": detect_loaded_words(t),
            "density": round(loaded_word_density(t), 4),
        }
        for t in texts
    ]


def run_bias_vector_pipeline(texts: list) -> list:
    """텍스트 목록에 대해 편향 벡터 계산 수행"""
    results = []
    for t in texts:
        raw = compute_bias_vector(t)
        norm = normalize_bias_vector(raw)
        results.append({
            "tone_dem": norm.tone_dem,
            "tone_rep": norm.tone_rep,
            "lr_score": norm.lr_score,
            "rationale": norm.rationale,
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

    # 감정 분류
    service = EmotionService()
    emotion_results = run_emotion_pipeline(texts, service)
    save_json({"emotions": emotion_results}, os.path.join(args.out_dir, "emotions_results.json"))

    # 편향 단어 감지
    loaded_results = run_loaded_words_pipeline(texts)
    save_json({"loaded_words": loaded_results}, os.path.join(args.out_dir, "loaded_words_results.json"))

    # 편향 벡터
    bias_results = run_bias_vector_pipeline(texts)
    save_json({"bias_vector": bias_results}, os.path.join(args.out_dir, "bias_vector_results.json"))

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()
