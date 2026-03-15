import argparse
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AI.source.data_loader import load_news_df, load_ai_test_df
from src.timeline.io_utils import save_json
from src.timeline.burst import compute_daily_counts, detect_burst_points
from src.timeline.plot_counts import plot_daily_counts
from src.timeline.timepoint_rank import rank_timepoints
from src.timeline.export_timeline import build_timeline


def token_contains(series: pd.Series, query: str) -> pd.Series:
    """
    query를 공백 단위 토큰으로 나눠서
    모든 토큰이 포함된 행만 True로 반환
    예: '등록금 인상' -> '등록금'과 '인상'이 둘 다 있어야 매칭
    """
    tokens = [t.strip() for t in str(query).split() if t.strip()]

    if not tokens:
        return pd.Series(False, index=series.index)

    mask = pd.Series(True, index=series.index)
    for token in tokens:
        mask &= series.str.contains(token, case=False, na=False, regex=False)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--table", choices=["news", "ai_test"], default="news")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    )
    parser.add_argument("--top_timepoints", type=int, default=8)
    parser.add_argument("--max_eojel", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_sentences", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Supabase에서 데이터 불러오기
    if args.table == "news":
        df = load_news_df(limit=args.limit)
    else:
        df = load_ai_test_df(limit=args.limit)

    print(f"[INFO] table={args.table}, loaded_rows={len(df)}")

    # 2) 필수 컬럼 기본 정리
    if "title" not in df.columns:
        df["title"] = ""
    if "body" not in df.columns:
        df["body"] = ""
    if "published" not in df.columns:
        raise ValueError("'published' 컬럼이 없어 타임라인을 생성할 수 없습니다.")

    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # 3) query 기반 필터링
    # 연속 문자열 포함이 아니라, query 토큰들이 모두 포함되는지 검사
    title_mask = token_contains(df["title"], args.query)
    body_mask = token_contains(df["body"], args.query)

    if "keyword" in df.columns:
        df["keyword"] = df["keyword"].fillna("").astype(str)
        keyword_mask = token_contains(df["keyword"], args.query)
        mask = title_mask | body_mask | keyword_mask
    else:
        mask = title_mask | body_mask

    df = df[mask].copy()
    print(f"[INFO] filtered_rows={len(df)}")

    if df.empty:
        print(f"[WARN] query='{args.query}' 에 해당하는 기사가 없습니다.")
        save_json(
            {
                "query": args.query,
                "burst_dates": [],
                "timeline": [],
                "message": "No matching articles found."
            },
            os.path.join(args.out_dir, "timeline_new.json")
        )
        return

    print(df[["title", "published"]].head(10))

    # 4) 날짜 전처리
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"]).sort_values("published")

    if df.empty:
        print("[WARN] published 전처리 후 남은 데이터가 없습니다.")
        save_json(
            {
                "query": args.query,
                "burst_dates": [],
                "timeline": [],
                "message": "No valid published dates found."
            },
            os.path.join(args.out_dir, "timeline_new.json")
        )
        return

    # timeline 내부 로직이 기대하는 date 컬럼 생성
    df["date"] = df["published"].dt.date.astype(str)

    # 5) 일자별 기사 수 / burst 탐지 / 그래프
    counts = compute_daily_counts(df)
    burst_dates = detect_burst_points(counts, z_threshold=2.0, min_count=2)

    plot_daily_counts(
        counts,
        burst_dates,
        os.path.join(args.out_dir, "daily_counts.png")
    )

    # 6) 중요 시점 랭킹
    ranked = rank_timepoints(df, top_keywords_per_day=20)
    save_json(
        {"ranked_timepoints": ranked.to_dict(orient="records")},
        os.path.join(args.out_dir, "ranked_timepoints.json")
    )

    # 7) 타임라인 생성
    timeline = build_timeline(
        df=df,
        query=args.query,
        ranked_timepoints=ranked,
        top_timepoints=args.top_timepoints,
        max_eojel=args.max_eojel,
        top_sentences=args.top_sentences,
        alpha=args.alpha
    )

    # 8) 결과 저장
    save_json(
        {
            "query": args.query,
            "table": args.table,
            "filtered_count": int(len(df)),
            "burst_dates": burst_dates,
            "timeline": timeline
        },
        os.path.join(args.out_dir, "timeline_new.json")
    )

    print("[OK] Saved artifacts to:", args.out_dir)


if __name__ == "__main__":
    main()