from __future__ import annotations

from pathlib import Path
import json
import sys

# ---------------------------------------------------------------------
# Path setup (stable regardless of cwd)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../ai_emotion_test
SRC = ROOT / "src"
DATA_DIR = ROOT / "data_samples"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------
# Imports from our src modules
# ---------------------------------------------------------------------
from emotion_datasets import load_emotions_csv, load_loaded_words_csv, load_bias_csvfrom emotion_model import EmotionsService
from loaded_words_detector import detect_loaded_words, mask_loaded_words
from bias_vector_dual_axis import compute_bias_vector


def main():
    # ---------------------------
    # 1) Load sample datasets
    # ---------------------------
    emotions_path = DATA_DIR / "emotions_9class_sample.csv"
    loaded_words_path = DATA_DIR / "loaded_words_sample.csv"
    bias_path = DATA_DIR / "biaslab_dual_axis_sample.csv"

    emo_ds = load_emotions_csv(str(emotions_path))  # TextLabelDataset(texts, labels)
    lw_df = load_loaded_words_csv(str(loaded_words_path))  # pd.DataFrame: text,is_biased,bias_terms
    b_df = load_bias_csv(str(bias_path))  # pd.DataFrame: text,tone_dem,tone_rep,rationale

    # ---------------------------
    # 2) Emotions: quick train + predict
    # ---------------------------
    emo_service = EmotionsService(label_list=None)

    if emo_ds.labels:
        emo_service.train_simple(emo_ds.texts, emo_ds.labels, epochs=1)

    emo_results = []
    for t in emo_ds.texts:
        out = emo_service.predict(t)
        emo_results.append(
            {
                "text": t,
                "label": out.label,
                "intensity": float(out.intensity),
                "probs": out.probs,
            }
        )

    # ---------------------------
    # 3) Loaded words: Dbias-style recognition + masking
    # ---------------------------
    lw_results = []
    for _, row in lw_df.iterrows():
        text = str(row.get("text", ""))

        terms = detect_loaded_words(text)
        masked = mask_loaded_words(text)

        lw_results.append(
            {
                "text": text,
                "is_biased": bool(len(terms) > 0),
                "bias_terms": terms,
                "masked_text": masked,
            }
        )

    # ---------------------------
    # 4) Bias vector: BiasLab-style dual-axis + rationale field
    # ---------------------------
    bias_results = []
    for _, row in b_df.iterrows():
        text = str(row.get("text", ""))
        rationale = str(row.get("rationale", ""))

        out = compute_bias_vector(text)

        bias_results.append(
            {
                "text": text,
                "tone_dem": float(out.tone_dem),
                "tone_rep": float(out.tone_rep),
                "lr_score": float(out.lr_score),
                "rationale": out.rationale if hasattr(out, "rationale") else rationale,
            }
        )

    # ---------------------------
    # 5) Save artifacts
    # ---------------------------
    (ARTIFACT_DIR / "emotions_results.json").write_text(
        json.dumps(emo_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "loaded_words_results.json").write_text(
        json.dumps(lw_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "bias_vector_results.json").write_text(
        json.dumps(bias_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # summary (Top-3 intensity for emotions)
    summary_lines = ["[Top-3 intensity texts]"]
    top3 = sorted(emo_results, key=lambda x: x["intensity"], reverse=True)[:3]
    for i, r in enumerate(top3, 1):
        summary_lines.append(
            f'{i}. intensity={r["intensity"]:.3f} label={r["label"]} text={r["text"]}'
        )

    (ARTIFACT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()