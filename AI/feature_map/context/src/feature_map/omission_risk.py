# omission_risk.py
"""
동종 클러스터 대비 핵심 엔티티 누락 위험도 계산
"""
from collections import Counter
from typing import List, Callable


# ──────────────────────────────────────────────
# 클러스터 분석 함수
# ──────────────────────────────────────────────

def collect_cluster_entities(cluster_articles: List[str], features_fn: Callable) -> Counter:
    """클러스터 전체 기사에서 엔티티 빈도 집계"""
    counter = Counter()
    for article in cluster_articles:
        f = features_fn(article)
        for e in f["entities"]:
            counter[e['word']] += 1
    return counter


def find_core_entities(counter: Counter, total_articles: int, threshold: float = 0.3) -> set:
    """전체 기사 중 threshold 이상 등장한 핵심 엔티티 반환"""
    return {
        entity
        for entity, count in counter.items()
        if count / total_articles >= threshold
    }


def compute_omission_ratio(target_entities: set, core_entities: set) -> float:
    """타겟 기사의 핵심 엔티티 누락 비율 계산"""
    if not core_entities:
        return 0.0
    missing = core_entities - target_entities
    return len(missing) / len(core_entities)


# ──────────────────────────────────────────────
# 등급 변환 함수
# ──────────────────────────────────────────────

def ratio_to_risk_level(ratio: float) -> str:
    """누락 비율 → 위험도 등급 (low / medium / high)"""
    if ratio < 0.2:
        return "low"
    elif ratio < 0.5:
        return "medium"
    else:
        return "high"


# ──────────────────────────────────────────────
# 통합 계산 함수
# ──────────────────────────────────────────────

def compute_omission_risk(
    target_article: str,
    cluster_articles: List[str],
    features_fn: Callable,
    threshold: float = 0.3,
) -> str:
    """
    동종 클러스터 대비 핵심 엔티티 누락 위험도를 반환

    Args:
        target_article: 분석 대상 기사
        cluster_articles: 같은 이슈의 타 기사 목록
        features_fn: extract_features 함수
        threshold: 핵심 엔티티 기준 등장 비율 (기본 30%)

    Returns:
        str: "low" | "medium" | "high"
    """
    if not cluster_articles:
        return "low"

    counter = collect_cluster_entities(cluster_articles, features_fn)
    core_entities = find_core_entities(counter, len(cluster_articles), threshold)

    target_features = features_fn(target_article)
    target_entities = {e['word'] for e in target_features["entities"]}

    ratio = compute_omission_ratio(target_entities, core_entities)
    return ratio_to_risk_level(ratio)
