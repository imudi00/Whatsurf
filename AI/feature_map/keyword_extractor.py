# keyword_extractor.py
from transformers import pipeline
import re

# KPF-BERT NER (한국어 뉴스 도메인 특화)
ner = pipeline("ner", model="KPF/KPF-bert-ner", aggregation_strategy="simple")

def extract_features(text: str) -> dict:
    """본문 → 키워드/통계 피처 추출"""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 5]
    
    # NER
    entities = ner(text[:512])  # 토큰 제한
    entity_types = [e['entity_group'] for e in entities]
    
    # 수치/통계 밀도
    num_pattern = re.findall(r'\d+[%억만원개명]+', text)
    
    # 인용구 패턴
    quote_pattern = re.findall(r'"[^"]{5,}"', text) + re.findall(r'라고|이라고', text)
    
    # 출처 표현
    source_pattern = re.findall(r'에 따르면|관계자|전문가|밝혔다|설명했다', text)
    
    return {
        "sentences": sentences,
        "entities": entities,
        "entity_count": len(entities),
        "entity_density": len(entities) / max(len(sentences), 1),
        "num_density": len(num_pattern) / max(len(sentences), 1),
        "quote_count": len(quote_pattern),
        "source_count": len(source_pattern),
        "paragraph_count": len(text.split('\n\n')),
        "sentence_count": len(sentences),
    }