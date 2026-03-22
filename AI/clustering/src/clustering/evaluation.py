"""from sklearn.metrics import silhouette_score

def evaluate(X, labels):
    score = silhouette_score(X, labels)
    print("Silhouette Score:", round(score, 4))
    return score"""
    
from sklearn.metrics import silhouette_score


def evaluate_cluster(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    return round(score, 4)


# 비LLM 자동 성능 평가
import re
from itertools import combinations


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r'[^0-9a-zA-Z가-힣\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    a_set = token_set(a)
    b_set = token_set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def average_pairwise_title_similarity(titles: list[str]) -> float:
    if len(titles) < 2:
        return 0.0

    sims = []
    for a, b in combinations(titles, 2):
        sims.append(jaccard_similarity(a, b))

    return round(sum(sims) / len(sims), 4) if sims else 0.0


def representative_title_score(representative_title: str, titles: list[str]) -> float:
    if not representative_title or not titles:
        return 0.0

    sims = [jaccard_similarity(representative_title, t) for t in titles if t]
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def key_sentence_relevance_score(key_sentences: list[str], titles: list[str]) -> float:
    if not key_sentences or not titles:
        return 0.0

    cluster_context = " ".join(titles)
    scores = [jaccard_similarity(sent, cluster_context) for sent in key_sentences]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def low_relevance_title_ratio(representative_title: str, titles: list[str], threshold: float = 0.15) -> float:
    if not representative_title or not titles:
        return 0.0

    low_count = 0
    total = 0

    for t in titles:
        if not t:
            continue
        total += 1
        if jaccard_similarity(representative_title, t) < threshold:
            low_count += 1

    return round(low_count / total, 4) if total else 0.0

# 비LLM 자동 성능평가

import re
from itertools import combinations


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    a_set = token_set(a)
    b_set = token_set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def average_pairwise_title_similarity(titles: list[str]) -> float:
    if len(titles) < 2:
        return 0.0

    sims = []
    for a, b in combinations(titles, 2):
        sims.append(jaccard_similarity(a, b))

    return round(sum(sims) / len(sims), 4) if sims else 0.0


def representative_title_score(representative_title: str, titles: list[str]) -> float:
    if not representative_title or not titles:
        return 0.0

    sims = [jaccard_similarity(representative_title, t) for t in titles if t]
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def key_sentence_relevance_score(key_sentences: list[str], titles: list[str]) -> float:
    if not key_sentences or not titles:
        return 0.0

    cluster_context = " ".join(titles)
    scores = [jaccard_similarity(sent, cluster_context) for sent in key_sentences]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def low_relevance_title_ratio(representative_title: str, titles: list[str], threshold: float = 0.15) -> float:
    if not representative_title or not titles:
        return 0.0

    low_count = 0
    total = 0

    for t in titles:
        if not t:
            continue
        total += 1
        if jaccard_similarity(representative_title, t) < threshold:
            low_count += 1

    return round(low_count / total, 4) if total else 0.0


def evaluate_cluster_input_quality(cluster_input: dict) -> dict:
    representative_title = cluster_input.get("representative_title", "")
    titles = cluster_input.get("titles", [])
    key_sentences = cluster_input.get("key_sentences", [])

    return {
        "title_cohesion": average_pairwise_title_similarity(titles),
        "representative_score": representative_title_score(representative_title, titles),
        "key_sentence_relevance": key_sentence_relevance_score(key_sentences, titles),
        "low_relevance_ratio": low_relevance_title_ratio(representative_title, titles),
    }