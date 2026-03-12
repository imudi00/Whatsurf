# text_utils.py
"""
맥락 피처에서 사용하는 텍스트 전처리 유틸리티
"""
import re
from typing import List


def normalize_text(s: str) -> str:
    s = s.replace("\u200b", " ").strip()
    return re.sub(r"\s+", " ", s)


def split_sentences(text: str, min_len: int = 5) -> List[str]:
    """구두점 기준 문장 분리"""
    return [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > min_len]


def count_paragraphs(text: str) -> int:
    """문단 수 계산"""
    return len([p for p in text.split('\n\n') if p.strip()])


def find_info_sentences(sentences: List[str]) -> List[str]:
    """수치/인용/출처 포함 정보문장 필터링"""
    pattern = re.compile(r'\d+[%억만원개명]|"[^"]{3,}"|에 따르면|관계자|밝혔다|분석|연구|조사')
    return [s for s in sentences if pattern.search(s)]


def find_background_keywords(text: str) -> List[str]:
    """배경/맥락 관련 키워드 탐색"""
    keywords = ['배경', '원인', '역사', '맥락', '이유', '경위', '전후', '과정']
    return [kw for kw in keywords if kw in text]
