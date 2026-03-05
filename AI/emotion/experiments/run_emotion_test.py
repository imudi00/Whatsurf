from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from emotion.src.emotion.loaded_words import detect_loaded_words, mask_loaded_words
from emotion.src.emotion.bias_vector import compute_bias_vector

def main():

    texts = [
        "This radical policy is dangerous",
        "Government regulation should increase",
        "Neutral factual report about economy"
    ]

    for text in texts:

        loaded = detect_loaded_words(text)
        masked = mask_loaded_words(text)

        bias = compute_bias_vector(text)

        print({
            "text": text,
            "loaded_words": loaded,
            "masked_text": masked,
            "tone_dem": bias.tone_dem,
            "tone_rep": bias.tone_rep,
            "lr_score": bias.lr_score,
            "rationale": bias.rationale
        })


if __name__ == "__main__":
    main()