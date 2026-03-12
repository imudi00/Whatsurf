# text_utils.py
"""
감정 피처에서 사용하는 텍스트 전처리 유틸리티
"""
import re

_SENT_SPLIT_RE = re.compile(r"(?<=[\.?!])\s+|(?<=다\.)\s+|(?<=다\?)\s+|(?<=다\!)\s+")


def normalize_text(s: str) -> str:
    s = s.replace("\u200b", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def split_sentences_kor(text: str):
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 5]


def split_sentences_basic(text: str, min_len: int = 5):
    """구두점 기준 기본 문장 분리"""
    return [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > min_len]
