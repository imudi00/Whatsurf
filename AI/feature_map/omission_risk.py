# omission_risk.py
from collections import Counter

def compute_omission_risk(target_article: str, cluster_articles: list[str], features_fn) -> str:
    """
    동종 클러스터 대비 핵심 엔티티 누락 위험도 계산
    target_article: 분석 대상 기사
    cluster_articles: 같은 이슈의 타 기사들
    """
    # 클러스터 전체 엔티티 수집
    cluster_entity_counter = Counter()
    for article in cluster_articles:
        f = features_fn(article)
        for e in f["entities"]:
            cluster_entity_counter[e['word']] += 1
    
    # 클러스터에서 자주 언급된 핵심 엔티티 (상위 30%)
    total = len(cluster_articles)
    core_entities = {
        entity for entity, count in cluster_entity_counter.items()
        if count / total >= 0.3  # 30% 이상 기사에서 등장
    }
    
    # 타겟 기사의 엔티티
    target_features = features_fn(target_article)
    target_entities = {e['word'] for e in target_features["entities"]}
    
    # 누락 비율
    if not core_entities:
        return "low"
    
    missing = core_entities - target_entities
    omission_ratio = len(missing) / len(core_entities)
    
    if omission_ratio < 0.2:
        return "low"
    elif omission_ratio < 0.5:
        return "medium"
    else:
        return "high"