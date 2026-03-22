# labeling/features/preprocessor.py
"""
뉴스 본문을 LLM 입력용 구조체로 분해
(구 feature_map/stance/src/feature_map/preprocessor.py)
"""
from .text_utils import (
    extract_headline,
    extract_lead,
    extract_quotes,
    extract_numbers,
    extract_judgment_words,
    extract_sources,
    sample_sentences,
)
import re


def build_article_struct(text: str) -> dict:
    """
    본문 텍스트를 헤드라인/리드/인용/수치/판단어/출처/샘플문장으로 분해

    Returns:
        dict: LLM 프롬프트에 주입 가능한 구조체
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]

    return {
        "headline":          extract_headline(text),
        "lead":              extract_lead(text),
        "quotes":            extract_quotes(text),
        "numbers":           extract_numbers(text),
        "judgment_words":    extract_judgment_words(text),
        "sources":           extract_sources(text),
        "sampled_sentences": sample_sentences(text),
        "total_sentences":   len(sentences),
    }
