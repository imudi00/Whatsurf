# labeling/features/keyword_extractor.py
"""
KPF-BERT NER 기반 키워드/통계 피처 추출
(구 feature_map/context/src/feature_map/keyword_extractor.py)
"""
import re
from typing import List, Dict
from transformers import pipeline

# ──────────────────────────────────────────────
# NER 모델 (한국어 뉴스 도메인 특화)
# ──────────────────────────────────────────────

_ner = None


def get_ner_pipeline():
    """싱글턴 NER 파이프라인 반환"""
    global _ner
    if _ner is None:
        _ner = pipeline("ner", model="KPF/KPF-bert-ner", aggregation_strategy="simple")
    return _ner


# ──────────────────────────────────────────────
# 패턴 추출 함수
# ──────────────────────────────────────────────

def extract_number_tokens(text: str) -> List[str]:
    """수치/통계 토큰 추출 (예: 30%, 3조원)"""
    return re.findall(r'\d+[%억만원개명]+', text)


def extract_quote_tokens(text: str) -> List[str]:
    """인용구 토큰 추출"""
    quotes = re.findall(r'"[^"]{5,}"', text)
    quotes += re.findall(r'라고|이라고', text)
    return quotes


def extract_source_tokens(text: str) -> List[str]:
    """출처 표현 토큰 추출"""
    return re.findall(r'에 따르면|관계자|전문가|밝혔다|설명했다', text)


# ──────────────────────────────────────────────
# 통합 추출 함수
# ──────────────────────────────────────────────

def extract_ner_entities(text: str, token_limit: int = 512) -> List[Dict]:
    """NER 엔티티 목록 반환"""
    ner = get_ner_pipeline()
    return ner(text[:token_limit])


def extract_features(text: str) -> dict:
    """
    본문에서 키워드/통계 피처를 통합 추출

    Returns:
        dict: sentences, entities, entity_count, entity_density,
              num_density, quote_count, source_count,
              paragraph_count, sentence_count
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 5]

    entities      = extract_ner_entities(text)
    num_tokens    = extract_number_tokens(text)
    quote_tokens  = extract_quote_tokens(text)
    source_tokens = extract_source_tokens(text)

    return {
        "sentences":       sentences,
        "entities":        entities,
        "entity_count":    len(entities),
        "entity_density":  len(entities) / max(len(sentences), 1),
        "num_density":     len(num_tokens) / max(len(sentences), 1),
        "quote_count":     len(quote_tokens),
        "source_count":    len(source_tokens),
        "paragraph_count": len(text.split('\n\n')),
        "sentence_count":  len(sentences),
    }
