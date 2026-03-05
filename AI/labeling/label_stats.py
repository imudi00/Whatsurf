"""
label_stats.py
라벨링 완료 후 결과 품질 검증 및 통계 출력

사용법:
    python label_stats.py --input labeled_output.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

FRAME_LABELS = ["인과", "대립", "개인", "경제적가치", "도덕", "안보", "권리"]
LOGIC_LABELS = ["정책적 비난", "전문가 견해", "피해자 서사", "파급효과", "해결책 제시", "사실/정보 전달"]


def print_stats(df: pd.DataFrame):
    print(f"\n{'='*50}")
    print(f"총 라벨링 건수: {len(df)}")
    print(f"{'='*50}")

    # 1. frame 분포
    print("\n[frame 분포]")
    fc = df["frame"].value_counts()
    for label in FRAME_LABELS:
        count = fc.get(label, 0)
        bar = "█" * int(count / max(fc) * 30)
        print(f"  {label:10s} {count:5d}건  {bar}")

    # 2. logic 분포
    print("\n[logic 분포]")
    lc = df["logic"].value_counts()
    for label in LOGIC_LABELS:
        count = lc.get(label, 0)
        bar = "█" * int(count / max(lc) * 30)
        print(f"  {label:12s} {count:5d}건  {bar}")

    # 3. stance_score 분포
    print("\n[stance_score 분포]")
    bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
    labels = ["강한비판", "비판", "중립", "우호", "강한옹호"]
    df["stance_bin"] = pd.cut(df["stance_score"], bins=bins, labels=labels)
    sb = df["stance_bin"].value_counts().sort_index()
    for label, count in sb.items():
        bar = "█" * int(count / len(df) * 50)
        print(f"  {label:8s} {count:5d}건  {bar}")

    # 4. 이상값 탐지
    print("\n[⚠ 이상값 탐지]")
    invalid_frame = df[~df["frame"].isin(FRAME_LABELS)]
    invalid_logic = df[~df["logic"].isin(LOGIC_LABELS)]
    invalid_stance = df[(df["stance_score"] < -1.0) | (df["stance_score"] > 1.0)]

    print(f"  frame 이상값:        {len(invalid_frame)}건")
    print(f"  logic 이상값:        {len(invalid_logic)}건")
    print(f"  stance_score 범위 벗어남: {len(invalid_stance)}건")

    # 5. 언론사별 stance 평균 (press 컬럼 있을 때)
    if "press" in df.columns and df["press"].notna().any():
        print("\n[언론사별 평균 stance_score (상위 10개)]")
        press_stance = df.groupby("press")["stance_score"].agg(["mean", "count"])
        press_stance = press_stance[press_stance["count"] >= 5].sort_values("mean")
        print(press_stance.head(10).to_string())

    # 6. frame × logic 교차표
    print("\n[frame × logic 교차표]")
    crosstab = pd.crosstab(df["frame"], df["logic"])
    print(crosstab.to_string())

    return df


def save_plots(df: pd.DataFrame, output_prefix: str = "label_stats"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # frame 분포
    fc = df["frame"].value_counts().reindex(FRAME_LABELS, fill_value=0)
    axes[0].bar(fc.index, fc.values, color="steelblue")
    axes[0].set_title("Frame 분포")
    axes[0].tick_params(axis='x', rotation=30)

    # logic 분포
    lc = df["logic"].value_counts().reindex(LOGIC_LABELS, fill_value=0)
    axes[1].bar(lc.index, lc.values, color="coral")
    axes[1].set_title("Logic 분포")
    axes[1].tick_params(axis='x', rotation=30)

    # stance_score 히스토그램
    axes[2].hist(df["stance_score"], bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].set_title("Stance Score 분포")
    axes[2].set_xlabel("score")
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plot_path = f"{output_prefix}_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n  📊 그래프 저장: {plot_path}")


def main(input_path: str):
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = print_stats(df)
    save_plots(df, output_prefix=input_path.replace(".csv", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="라벨링 결과 CSV 경로")
    args = parser.parse_args()
    main(args.input)