# labeling/features/text_utils.py
"""
논조 피처에서 사용하는 텍스트 전처리 유틸리티
(구 feature_map/stance/src/feature_map/text_utils.py)
"""
import re
from typing import List


def normalize_text(s: str) -> str:
    s = s.replace("\u200b", " ").strip()
    return re.sub(r"\s+", " ", s)


def extract_headline(text: str) -> str:
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[0] if lines else ""


def extract_lead(text: str, max_len: int = 200) -> str:
    paragraphs = text.strip().split('\n\n')
    return (paragraphs[0].strip() if paragraphs else "")[:max_len]


def extract_quotes(text: str, max_count: int = 5) -> List[str]:
    """인용구 추출"""
    quotes = re.findall(r'["\'"](.*?)["\'""]', text)
    quotes += re.findall(r'(.{5,30}?)(?:라고|이라고|고)\s(?:밝혔|말했|전했|주장했|강조했)', text)
    return list(set(quotes))[:max_count]


def extract_numbers(text: str, max_count: int = 5) -> List[str]:
    """수치/통계 추출"""
    return re.findall(r'\d+[\.,]?\d*\s*[%억만원명개건회%]', text)[:max_count]


def extract_judgment_words(text: str, max_count: int = 5) -> List[str]:
    """판단/감정 술어 추출"""
    return re.findall(
        r'[\w]+(?:비판|우려|촉구|반발|환영|지적|경고|강조|주장|요구|반대|찬성|규탄|촉구)[\w]*',
        text,
    )[:max_count]


def extract_sources(text: str, max_count: int = 5) -> List[str]:
    """출처 표현 추출"""
    sources = re.findall(
        r'([가-힣\w]+(?:부|처|청|원|회|단|협|측|관계자|전문가|교수|연구원))', text
    )
    return list(set(sources))[:max_count]


def sample_sentences(text: str) -> List[str]:
    """앞·중·뒤 문장 샘플링 (토큰 절약)"""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    n = len(sentences)
    sampled = []
    if n > 0: sampled.append(sentences[0])
    if n > 2: sampled.append(sentences[n // 2])
    if n > 1: sampled.append(sentences[-1])
    return sampled
