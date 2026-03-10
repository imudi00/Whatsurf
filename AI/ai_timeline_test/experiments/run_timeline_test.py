import argparse
import os
import sys

sys.path.append(os.path.abspath("."))

from src.timeline.io_utils import load_news_csv, save_json
from src.timeline.burst import compute_daily_counts, detect_burst_points
from src.timeline.plot_counts import plot_daily_counts
from src.timeline.timepoint_rank import rank_timepoints
from src.timeline.export_timeline import build_timeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--out_dir", default="experiments/artifacts")
    parser.add_argument("--top_timepoints", type=int, default=8)
    parser.add_argument("--max_eojel", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_sentences", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_news_csv(args.input)

    counts = compute_daily_counts(df)
    burst_dates = detect_burst_points(counts, z_threshold=2.0, min_count=2)
    plot_daily_counts(counts, burst_dates, os.path.join(args.out_dir, "daily_counts.png"))

    ranked = rank_timepoints(df, top_keywords_per_day=20)
    save_json({"ranked_timepoints": ranked.to_dict(orient="records")}, os.path.join(args.out_dir, "ranked_timepoints.json"))

    timeline = build_timeline(
        df=df,
        query=args.query,
        ranked_timepoints=ranked,
        top_timepoints=args.top_timepoints,
        max_eojel=args.max_eojel,
        top_sentences=args.top_sentences,
        alpha=args.alpha
    )

    save_json({"query": args.query, "burst_dates": burst_dates, "timeline": timeline},
              os.path.join(args.out_dir, "timeline.json"))

    print("[OK] Saved artifacts to:", args.out_dir)

if __name__ == "__main__":
    main()
