from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from src.emotion.data import load_emotions_csv, load_loaded_words_csv, load_bias_csv
from src.emotion.emotions import EmotionsService
from src.emotion.loaded_words import LoadedWordsPipeline
from src.emotion.bias_vector import BiasVectorService


ARTIFACT_DIR = Path("experiments/artifacts/emotion")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) emotions
    emo_ds = load_emotions_csv("data/emotion/sample/emotions_sample.csv")
    emotions = EmotionsService(base_model="klue/bert-base")
    # 샘플로 "학습이 실제로 도는지" 확인 (정확도는 목적 아님)
    emotions.train_simple(emo_ds.texts, emo_ds.labels, epochs=1, batch_size=2)

    emo_results = []
    for t in emo_ds.texts:
        out = emotions.predict(t)
        emo_results.append(
            {
                "text": t,
                "label": out.label,
                "intensity": out.intensity,
                "probs": out.probs,
            }
        )

    # 2) loaded_words
    lw_df = load_loaded_words_csv("data/emotion/sample/loaded_words_sample.csv")
    pipeline = LoadedWordsPipeline()

    lw_results = []
    for t in lw_df["text"].fillna("").astype(str).tolist():
        r = pipeline.run(t)
        lw_results.append(
            {
                "text": t,
                "is_biased": r.is_biased,
                "bias_terms": r.bias_terms,
                "masked_text": r.masked_text,
            }
        )

    # 3) bias_vector (dual-axis)
    bdf = load_bias_csv("data/emotion/sample/bias_sample.csv")
    bias = BiasVectorService(base_model="klue/bert-base")
    # 샘플 회귀도 "도는지"만 확인
    bias.train_simple(
        bdf["text"].astype(str).tolist(),
        bdf["tone_dem"].astype(float).tolist(),
        bdf["tone_rep"].astype(float).tolist(),
        epochs=1,
        batch_size=2,
    )

    bias_results = []
    for _, row in bdf.iterrows():
        out = bias.predict(str(row["text"]), rationale=row.get("rationale", None))
        bias_results.append(
            {
                "text": row["text"],
                "tone_dem": out.tone_dem,
                "tone_rep": out.tone_rep,
                "lr_score": out.lr_score,
                "rationale": out.rationale,
            }
        )

    # save artifacts
    (ARTIFACT_DIR / "emotions_results.json").write_text(json.dumps(emo_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "loaded_words_results.json").write_text(json.dumps(lw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "bias_vector_results.json").write_text(json.dumps(bias_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 간단 요약(기능명세 연결: "감정 강도 top-n")
    top = sorted(emo_results, key=lambda x: x["intensity"], reverse=True)[:3]
    summary_lines = ["[Top-3 intensity texts]"]
    for i, r in enumerate(top, 1):
        summary_lines.append(f"{i}. intensity={r['intensity']:.3f} label={r['label']} text={r['text']}")
    (ARTIFACT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("✅ emotion feature dev test finished.")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()