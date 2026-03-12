# loaded_words.py
"""
Dbias 논문 기반 편향 단어(Loaded Words) 감지 및 마스킹
"""
import re
from typing import List

# ──────────────────────────────────────────────
# 편향 단어 사전 (Dbias 논문 기반 샘플)
# ──────────────────────────────────────────────

BIAS_WORDS: List[str] = [
    "radical",
    "racist",
    "extremist",
    "illegal",
    "dangerous",
    "propaganda",
    "fake",
    "biased",
]


# ──────────────────────────────────────────────
# 감지 함수
# ──────────────────────────────────────────────

def detect_loaded_words(text: str) -> List[str]:
    """텍스트에서 편향 단어 목록을 반환"""
    return [w for w in BIAS_WORDS if re.search(rf"\b{w}\b", text.lower())]


def count_loaded_words(text: str) -> int:
    """편향 단어 등장 횟수 합산"""
    return sum(
        len(re.findall(rf"\b{w}\b", text.lower())) for w in BIAS_WORDS
    )


def loaded_word_density(text: str) -> float:
    """편향 단어 밀도 (단어 수 대비 비율)"""
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    return count_loaded_words(text) / total_words


# ──────────────────────────────────────────────
# 마스킹 함수
# ──────────────────────────────────────────────

def mask_loaded_words(text: str, mask_token: str = "[MASK]") -> str:
    """편향 단어를 mask_token으로 치환"""
    masked = text
    for w in BIAS_WORDS:
        masked = re.sub(rf"\b{w}\b", mask_token, masked, flags=re.IGNORECASE)
    return masked


def replace_loaded_words(text: str, replacement: str = "***") -> str:
    """편향 단어를 임의 문자열로 치환 (표시용)"""
    return mask_loaded_words(text, mask_token=replacement)
