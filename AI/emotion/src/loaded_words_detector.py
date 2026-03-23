import re

# Dbias 논문 기반 bias word list (샘플)
BIAS_WORDS = [
    "radical",
    "racist",
    "extremist",
    "illegal",
    "dangerous",
    "propaganda",
    "fake",
    "biased"
]

def detect_loaded_words(text: str):
    """
    Dbias 기반 Loaded Words Detection
    """
    found = []
    for w in BIAS_WORDS:
        if re.search(rf"\b{w}\b", text.lower()):
            found.append(w)

    return found


def mask_loaded_words(text: str):
    """
    Dbias Bias Masking
    """
    masked = text
    for w in BIAS_WORDS:
        masked = re.sub(rf"\b{w}\b", "[MASK]", masked, flags=re.IGNORECASE)

    return masked