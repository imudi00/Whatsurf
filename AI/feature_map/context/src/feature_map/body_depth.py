# body_depth.py
"""
본문 맥락 깊이(body_depth) 점수 계산
규칙 기반 / 빠른 실행
"""
from .text_utils import (
    split_sentences,
    count_paragraphs,
    find_info_sentences,
    find_background_keywords,
)


# ──────────────────────────────────────────────
# 세부 점수 계산 함수
# ──────────────────────────────────────────────

def compute_paragraph_score(paragraph_count: int, target: int = 5) -> float:
    """문단 수 정규화 점수 (target 이상이면 1.0)"""
    return min(paragraph_count / target, 1.0)


def compute_info_ratio(text: str, sentences: list) -> float:
    """수치/인용/출처 포함 정보문장 비율"""
    info_sents = find_info_sentences(sentences)
    return len(info_sents) / max(len(sentences), 1)


def compute_background_score(text: str) -> float:
    """배경/맥락 키워드 존재 여부 (있으면 1.0)"""
    return 1.0 if find_background_keywords(text) else 0.0


# ──────────────────────────────────────────────
# 통합 계산 함수
# ──────────────────────────────────────────────

def compute_body_depth(text: str, features: dict) -> float:
    """
    body_depth = 0.4 * (문단수 정규화) + 0.3 * (정보문장 비율) + 0.3 * (배경키워드 존재)

    Args:
        text: 기사 전문
        features: extract_features() 반환값

    Returns:
        float: 0.0 ~ 1.0 범위 깊이 점수
    """
    para_score = compute_paragraph_score(features["paragraph_count"])
    info_ratio = compute_info_ratio(text, features["sentences"])
    bg_score   = compute_background_score(text)

    depth = 0.4 * para_score + 0.3 * info_ratio + 0.3 * bg_score
    return round(depth, 4)


def describe_body_depth(score: float) -> str:
    """점수를 사람이 읽을 수 있는 레이블로 변환"""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"
