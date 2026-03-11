from .summarize_extractive import summarize_timepoint

def choose_representative_article(day_articles):
    # 개발 테스트: 가장 긴 본문을 대표 기사로 선택(간단 휴리스틱)
    idx = day_articles["body"].astype(str).str.len().idxmax()
    row = day_articles.loc[idx]
    return {"title": str(row["title"]), "url": str(row.get("url", ""))}

def build_timeline(df, query, ranked_timepoints, top_timepoints=8, max_eojel=17, top_sentences=3, alpha=0.5):
    top = ranked_timepoints.head(top_timepoints)
    dates = top["date"].tolist()

    timeline = []
    for d in sorted(dates):
        day = df[df["date"].astype(str) == d].copy()
        if len(day) == 0:
            continue

        summary = summarize_timepoint(
            day, query=query, max_eojel=max_eojel, top_sentences=top_sentences, alpha=alpha
        )
        rep = choose_representative_article(day)

        timeline.append({
            "date": d,
            "importance": float(top[top["date"] == d]["importance"].iloc[0]),
            "summary": summary,
            "representative": rep,
            "articles": [{"title": t, "url": u} for t, u in zip(day["title"].tolist(), day["url"].tolist())],
            "count": int(len(day)),
        })

    return timeline
