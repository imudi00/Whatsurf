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


def average_pairwise_similarity(texts: list[str]) -> float:
    texts = [t for t in texts if str(t).strip()]
    if len(texts) < 2:
        return 0.0

    sims = []
    for a, b in combinations(texts, 2):
        sims.append(jaccard_similarity(a, b))

    return round(sum(sims) / len(sims), 4) if sims else 0.0


def text_redundancy_score(text: str) -> float:
    words = normalize_text(text).split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return round(1 - unique_ratio, 4)


def phase_headline_length_ok(headline: str) -> bool:
    headline = str(headline).strip()
    return 6 <= len(headline) <= 40


def phase_summary_length_ok(summary: str) -> bool:
    summary = str(summary).strip()
    return 20 <= len(summary) <= 250


def timeline_point_relevance_score(point: dict) -> float:
    """
    phase_headline / phase_summary가 해당 시점 기사 제목들과 얼마나 관련 있는지
    """
    titles = point.get("llm_input", {}).get("top_article_titles", [])
    headline = point.get("phase_headline", "")
    summary = point.get("phase_summary", "")

    context = " ".join(titles)
    if not context.strip():
        return 0.0

    headline_score = jaccard_similarity(headline, context) if headline else 0.0
    summary_score = jaccard_similarity(summary, context) if summary else 0.0

    return round((headline_score + summary_score) / 2, 4)


def update_focus_score(point: dict) -> float:
    """
    요약이 새 기사 / 업데이트 정보와 얼마나 관련 있는지
    """
    llm_input = point.get("llm_input", {})
    context_parts = []

    for t in llm_input.get("new_article_titles", []):
        if str(t).strip():
            context_parts.append(str(t).strip())

    for s in llm_input.get("key_sentences", []):
        if str(s).strip():
            context_parts.append(str(s).strip())

    context = " ".join(context_parts).strip()
    if not context:
        return 0.0

    summary = point.get("phase_summary", "") or point.get("update_summary", "")
    if not summary:
        return 0.0

    return round(jaccard_similarity(summary, context), 4)


def timeline_point_distinctiveness(points: list[dict]) -> float:
    """
    각 타임포인트 headline이 서로 얼마나 중복되지 않는지
    높을수록 다양한 시점으로 구성됨
    """
    headlines = [p.get("phase_headline", "") for p in points if p.get("phase_headline", "").strip()]
    if len(headlines) < 2:
        return 0.0

    avg_sim = average_pairwise_similarity(headlines)
    return round(1 - avg_sim, 4)


def burst_alignment_score(points: list[dict]) -> float:
    """
    importance가 높은 시점일수록 기사 수가 많은 경향을 가지는지 간단 점검
    """
    if not points:
        return 0.0

    importance_sum = 0.0
    count_sum = 0.0

    for p in points:
        importance_sum += float(p.get("importance", 0.0))
        count_sum += float(p.get("count", 0))

    if importance_sum == 0 or count_sum == 0:
        return 0.0

    ratios = []
    for p in points:
        imp = float(p.get("importance", 0.0)) / importance_sum if importance_sum else 0.0
        cnt = float(p.get("count", 0)) / count_sum if count_sum else 0.0
        ratios.append(1 - abs(imp - cnt))

    return round(sum(ratios) / len(ratios), 4) if ratios else 0.0


def timeline_coverage_score(points: list[dict]) -> float:
    """
    전체 타임라인 중 기사 수가 0이 아닌 시점 비율
    """
    if not points:
        return 0.0

    valid = sum(1 for p in points if int(p.get("count", 0)) > 0)
    return round(valid / len(points), 4)


def evaluate_timeline_point(point: dict) -> dict:
    titles = point.get("llm_input", {}).get("top_article_titles", [])
    new_titles = point.get("llm_input", {}).get("new_article_titles", [])

    return {
        "article_count": int(point.get("count", 0)),
        "new_article_count": int(point.get("new_count", 0)),
        "title_cohesion": average_pairwise_similarity(titles),
        "new_title_cohesion": average_pairwise_similarity(new_titles),
        "phase_headline_length_ok": phase_headline_length_ok(point.get("phase_headline", "")),
        "phase_summary_length_ok": phase_summary_length_ok(point.get("phase_summary", "")),
        "point_relevance": timeline_point_relevance_score(point),
        "update_focus_score": update_focus_score(point),
        "phase_headline_redundancy": text_redundancy_score(point.get("phase_headline", "")),
        "phase_summary_redundancy": text_redundancy_score(point.get("phase_summary", "")),
        "llm_success": bool(
            str(point.get("phase_headline", "")).strip() and
            str(point.get("phase_summary", "")).strip() and
            not str(point.get("llm_error", "")).strip()
        ),
    }


def evaluate_timeline(points: list[dict]) -> dict:
    if not points:
        return {
            "point_count": 0,
            "llm_success_rate": 0.0,
            "avg_title_cohesion": 0.0,
            "avg_new_title_cohesion": 0.0,
            "avg_point_relevance": 0.0,
            "avg_update_focus_score": 0.0,
            "avg_phase_headline_redundancy": 0.0,
            "avg_phase_summary_redundancy": 0.0,
            "phase_headline_length_ok_rate": 0.0,
            "phase_summary_length_ok_rate": 0.0,
            "timeline_distinctiveness": 0.0,
            "burst_alignment_score": 0.0,
            "coverage_score": 0.0,
        }

    evaluated = [evaluate_timeline_point(p) for p in points]
    n = len(evaluated)

    return {
        "point_count": n,
        "llm_success_rate": round(sum(int(e["llm_success"]) for e in evaluated) / n, 4),
        "avg_title_cohesion": round(sum(e["title_cohesion"] for e in evaluated) / n, 4),
        "avg_new_title_cohesion": round(sum(e["new_title_cohesion"] for e in evaluated) / n, 4),
        "avg_point_relevance": round(sum(e["point_relevance"] for e in evaluated) / n, 4),
        "avg_update_focus_score": round(sum(e["update_focus_score"] for e in evaluated) / n, 4),
        "avg_phase_headline_redundancy": round(sum(e["phase_headline_redundancy"] for e in evaluated) / n, 4),
        "avg_phase_summary_redundancy": round(sum(e["phase_summary_redundancy"] for e in evaluated) / n, 4),
        "phase_headline_length_ok_rate": round(sum(int(e["phase_headline_length_ok"]) for e in evaluated) / n, 4),
        "phase_summary_length_ok_rate": round(sum(int(e["phase_summary_length_ok"]) for e in evaluated) / n, 4),
        "timeline_distinctiveness": timeline_point_distinctiveness(points),
        "burst_alignment_score": burst_alignment_score(points),
        "coverage_score": timeline_coverage_score(points),
    }