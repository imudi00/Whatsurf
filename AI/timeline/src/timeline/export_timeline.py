from .summarize_extractive import summarize_timepoint


def choose_representative_article(day_articles):
    # 개발 테스트: 가장 긴 본문을 대표 기사로 선택(간단 휴리스틱)
    idx = day_articles["body"].astype(str).str.len().idxmax()
    row = day_articles.loc[idx]
    return {"title": str(row["title"]), "url": str(row.get("url", ""))}


def _build_article_key(row):
    """
    기사 중복/차집합 비교용 키 생성
    id 컬럼이 있으면 id 우선 사용
    없으면 title + url 조합 사용
    """
    if "id" in row and row["id"] is not None:
        return f"id::{row['id']}"
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    return f"title_url::{title}::{url}"


def _filter_new_articles(day_df, prev_article_keys):
    """
    이전 시점에 없던 새 기사만 추출
    """
    if len(day_df) == 0:
        return day_df.copy()

    keys = day_df.apply(_build_article_key, axis=1)
    mask = ~keys.isin(prev_article_keys)
    return day_df[mask].copy()


def _make_update_summary(new_articles, query, max_eojel=17, top_sentences=2, alpha=0.5):
    """
    이전 시점 대비 새로 들어온 기사들만 요약
    """
    if len(new_articles) == 0:
        return "이전 시점 대비 핵심 업데이트 없음"

    return summarize_timepoint(
        new_articles,
        query=query,
        max_eojel=max_eojel,
        top_sentences=top_sentences,
        alpha=alpha,
    )


def build_timeline(
    df,
    query,
    ranked_timepoints,
    top_timepoints=8,
    max_eojel=17,
    top_sentences=3,
    alpha=0.5,
):
    top = ranked_timepoints.head(top_timepoints)
    dates = top["date"].tolist()

    timeline = []
    prev_article_keys = set()

    for d in sorted(dates):
        day = df[df["date"].astype(str) == d].copy()
        if len(day) == 0:
            continue

        summary = summarize_timepoint(
            day,
            query=query,
            max_eojel=max_eojel,
            top_sentences=top_sentences,
            alpha=alpha,
        )

        rep = choose_representative_article(day)

        # 이전 시점 대비 새 기사만 추출
        new_articles = _filter_new_articles(day, prev_article_keys)

        # 업데이트 요약 생성
        update_summary = _make_update_summary(
            new_articles,
            query=query,
            max_eojel=max_eojel,
            top_sentences=max(1, top_sentences - 1),
            alpha=alpha,
        )

        timeline.append({
            "date": d,
            "importance": float(top[top["date"] == d]["importance"].iloc[0]),
            "summary": summary,
            "update_summary": update_summary,
            "representative": rep,
            "articles": [
                {
                    "id": str(row["id"]) if "id" in day.columns else "",
                    "title": str(row["title"]),
                    "url": str(row.get("url", "")),
                    "published": str(row.get("published", "")),
                }
                for _, row in day.iterrows()
            ],
            "new_articles": [
                {
                    "id": str(row["id"]) if "id" in new_articles.columns else "",
                    "title": str(row["title"]),
                    "url": str(row.get("url", "")),
                    "published": str(row.get("published", "")),
                }
                for _, row in new_articles.iterrows()
            ],
            "count": int(len(day)),
            "new_count": int(len(new_articles)),
        })

        # 현재 날짜 기사들을 다음 시점 비교 기준으로 누적
        current_keys = set(day.apply(_build_article_key, axis=1).tolist())
        prev_article_keys.update(current_keys)

    return timeline