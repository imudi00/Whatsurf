# experiments/run_emotion_test.py
"""
감정 피처 실험 실행 스크립트
사용법:
    python run_emotion_test.py --input data_samples/emotions_9class_sample.csv
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_map.io_utils import load_emotions_csv, save_json
from src.feature_map.emotion_model import EmotionService
from src.feature_map.loaded_words import detect_loaded_words, loaded_word_density
from src.feature_map.bias_vector import compute_bias_vector, normalize_bias_vector


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
    parser.add_argument("--input", required=True, help="감정 레이블 CSV 경로")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 데이터 로드
    dataset = load_emotions_csv(args.input)
    texts = dataset.texts[:50]  # 실험용 샘플 제한

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
