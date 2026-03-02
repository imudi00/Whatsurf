from __future__ import annotations
import json
from pathlib import Path

from AI_test1.emotion.src.emotion.data import load_emotions_csv, load_loaded_words_csv, load_bias_csv
from AI_test1.emotion.src.emotion.emotions import EmotionsService
from AI_test1.emotion.src.emotion.loaded_words import LoadedWordsPipeline
from AI_test1.emotion.src.emotion.bias_vector import BiasVectorService

ARTIFACT_DIR = Path("AI_test1/emotion/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    emo_ds = load_emotions_csv("AI_test1/emotion/data/sample/emotions_sample.csv")
    emotions = EmotionsService(base_model="klue/bert-base")
    emotions.train_simple(emo_ds.texts, emo_ds.labels, epochs=1, batch_size=2)

    emo_results = []
    for t in emo_ds.texts:
        out = emotions.predict(t)
        emo_results.append({"text": t, "label": out.label, "intensity": out.intensity, "probs": out.probs})

    lw_df = load_loaded_words_csv("AI_test1/emotion/data/sample/loaded_words_sample.csv")
    pipeline = LoadedWordsPipeline()
    lw_results = []
    for t in lw_df["text"].fillna("").astype(str).tolist():
        r = pipeline.run(t)
        lw_results.append({"text": t, "is_biased": r.is_biased, "bias_terms": r.bias_terms, "masked_text": r.masked_text})

    bdf = load_bias_csv("AI_test1/emotion/data/sample/bias_sample.csv")
    bias = BiasVectorService(base_model="klue/bert-base")
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
        bias_results.append({"text": row["text"], "tone_dem": out.tone_dem, "tone_rep": out.tone_rep, "lr_score": out.lr_score, "rationale": out.rationale})

    (ARTIFACT_DIR / "emotions_results.json").write_text(json.dumps(emo_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "loaded_words_results.json").write_text(json.dumps(lw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "bias_vector_results.json").write_text(json.dumps(bias_results, ensure_ascii=False, indent=2), encoding="utf-8")

    top = sorted(emo_results, key=lambda x: x["intensity"], reverse=True)[:3]
    summary = ["[Top-3 intensity texts]"]
    for i, r in enumerate(top, 1):
        summary.append(f"{i}. intensity={r['intensity']:.3f} label={r['label']} text={r['text']}")
    (ARTIFACT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("✅ emotion feature dev test finished.")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
