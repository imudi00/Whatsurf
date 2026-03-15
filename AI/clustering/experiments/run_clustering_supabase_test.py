import argparse
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AI.source.data_loader import load_news_df, load_ai_test_df
from AI.clustering.src.clustering.service_pipeline import run_service_clustering


def choose_representative_article(cluster_df: pd.DataFrame):
    idx = cluster_df["body"].fillna("").astype(str).str.len().idxmax()
    row = cluster_df.loc[idx]
    return {
        "id": str(row["id"]) if "id" in cluster_df.columns else "",
        "title": str(row.get("title", "")),
        "url": str(row.get("url", "")),
        "published": str(row.get("published", "")),
    }


def summarize_cluster(cluster_df: pd.DataFrame, max_titles: int = 3):
    titles = (
        cluster_df["title"]
        .fillna("")
        .astype(str)
        .head(max_titles)
        .tolist()
    )
    titles = [t.strip() for t in titles if t.strip()]
    if not titles:
        return "군집 요약 생성 불가"

    return "관련 기사: " + " / ".join(titles[:max_titles])


def build_cluster_summary(result_df: pd.DataFrame, table: str, query: str | None):
    valid_df = result_df[result_df["cluster_label"] != -1].copy()
    cluster_ids = sorted(valid_df["cluster_label"].unique().tolist())

    clusters = []
    for cid in cluster_ids:
        cluster_df = valid_df[valid_df["cluster_label"] == cid].copy()
        cluster_df = cluster_df.sort_values("published") if "published" in cluster_df.columns else cluster_df

        rep = choose_representative_article(cluster_df)
        summary = summarize_cluster(cluster_df)
        title = rep["title"] if rep["title"] else f"Cluster {cid}"

        clusters.append({
            "cluster_id": int(cid),
            "article_count": int(len(cluster_df)),
            "title": title,
            "summary": summary,
            "representative_article": rep,
            "top_publishers": [],
            "articles": [
                {
                    "id": str(row["id"]) if "id" in cluster_df.columns else "",
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "published": str(row.get("published", "")),
                }
                for _, row in cluster_df.iterrows()
            ]
        })

    return {
        "table": table,
        "query": query,
        "article_count": int(len(result_df)),
        "cluster_count": int(len(clusters)),
        "noise_count": int((result_df["cluster_label"] == -1).sum()),
        "clusters": clusters
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["news", "ai_test"], default="ai_test")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--query", default=None)
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 데이터 로드
    if args.table == "news":
        df = load_news_df(limit=args.limit)
    else:
        df = load_ai_test_df(limit=args.limit)

    print(f"[INFO] table={args.table}, loaded_rows={len(df)}")

    # 2. 컬럼 기본 정리
    if "title" not in df.columns:
        df["title"] = ""
    if "body" not in df.columns:
        df["body"] = ""
    if "published" not in df.columns:
        df["published"] = ""
    if "url" not in df.columns:
        df["url"] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["cluster_text"] = (df["title"] + " " + df["body"]).str.strip()

    # 3. query 필터링 (선택)
    if args.query:
        query_tokens = [t.strip() for t in str(args.query).split() if t.strip()]

        def token_match(series):
            mask = pd.Series(True, index=series.index)
            for token in query_tokens:
                mask &= series.str.contains(token, case=False, na=False, regex=False)
            return mask

        title_mask = token_match(df["title"])
        body_mask = token_match(df["body"])

        if "keyword" in df.columns:
            df["keyword"] = df["keyword"].fillna("").astype(str)
            keyword_mask = token_match(df["keyword"])
            mask = title_mask | body_mask | keyword_mask
        else:
            mask = title_mask | body_mask

        df = df[mask].copy()
        print(f"[INFO] filtered_rows={len(df)}")

    # 4. 빈 텍스트 제거
    df = df[df["cluster_text"].str.len() > 0].copy()
    print(f"[INFO] clusterable_rows={len(df)}")

    if df.empty:
        print("[WARN] 클러스터링 가능한 기사가 없습니다.")
        return

    texts = df["cluster_text"].tolist()

    # 5. 클러스터링 실행
    labels = run_service_clustering(texts)

    # 6. 기사별 결과 저장
    result_df = df.copy()
    result_df["cluster_label"] = labels

    csv_path = os.path.join(args.out_dir, "clustered_articles.csv")
    result_df[["id", "title", "published", "url", "cluster_label"]].to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. 군집별 요약 저장
    summary_data = build_cluster_summary(result_df, table=args.table, query=args.query)

    json_path = os.path.join(args.out_dir, "cluster_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("[OK] Saved clustered articles to:", csv_path)
    print("[OK] Saved cluster summary to:", json_path)


if __name__ == "__main__":
    main()