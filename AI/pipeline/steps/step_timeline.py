"""
Timeline Step
───────────────────────────────────────────────────────────────
① 기사 날짜별 TTP/ETP 계산 → importance 랭킹
② 상위 N개 시점 선정
③ 각 시점 대표 기사 결정 (article.id 기준)
④ queries_timeline 테이블 저장

반환:
    {
        "saved_count":  int,         # 저장된 타임라인 항목 수
        "timeline":     list[dict],  # 저장된 항목 상세
    }
"""
from __future__ import annotations

import sys
from pathlib import Path

# timeline 모듈 경로 등록
_TIMELINE_SRC = Path(__file__).resolve().parents[2] / "timeline" / "src"
if str(_TIMELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_TIMELINE_SRC))

import pandas as pd
from timeline.timepoint_rank import rank_timepoints        # type: ignore
from timeline.export_timeline import build_timeline        # type: ignore

from AI.pipeline.db.save_timeline import save_timeline_entry, clear_timeline_for_query


def _pick_rep_article_id(day_df: pd.DataFrame) -> int:
    """날짜 그룹에서 대표 기사 ID 반환 — 본문 가장 긴 것."""
    idx = day_df["body"].astype(str).str.len().idxmax()
    return int(day_df.loc[idx, "id"])


def run_timeline_step(
    query_id: int,
    query_text: str,
    articles: list[dict],
    top_timepoints: int = 8,
    overwrite: bool = True,
) -> dict:
    """
    Args:
        query_id:       queries 테이블 PK
        query_text:     검색어 (TF-IDF 점수 보정용)
        articles:       [{"id": int, "title": str, "body": str,
                          "published_at": str, ...}, ...]
        top_timepoints: 저장할 최대 날짜 수
        overwrite:      True면 기존 타임라인 삭제 후 재삽입

    Returns: 실행 결과 요약 dict
    """
    # 날짜 없는 기사 필터
    valid = [a for a in articles if a.get("published_at")]
    if not valid:
        print("  [Timeline] ⚠️  날짜 있는 기사 없음 — 스킵")
        return {"saved_count": 0, "timeline": []}

    # ── DataFrame 구성 ─────────────────────────────────────
    df = pd.DataFrame(valid)
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].astype(str)
    df["body"]  = df.get("body",  pd.Series()).fillna("").astype(str)
    df["title"] = df.get("title", pd.Series()).fillna("").astype(str)
    # url 컬럼: export_timeline.build_timeline 내부에서 사용
    if "url" not in df.columns:
        df["url"] = ""
    else:
        df["url"] = df["url"].fillna("").astype(str)

    if df.empty:
        print("  [Timeline] ⚠️  유효 날짜 없음 — 스킵")
        return {"saved_count": 0, "timeline": []}

    # ── Step 1: 시점 랭킹 ──────────────────────────────────
    print(f"  [Timeline] 시점 랭킹 계산 ({len(df)}건, {df['date'].nunique()}일)...")
    ranked = rank_timepoints(df)

    # ── Step 2: 타임라인 빌드 ─────────────────────────────
    timeline_entries = build_timeline(
        df,
        query=query_text,
        ranked_timepoints=ranked,
        top_timepoints=top_timepoints,
    )
    print(f"  [Timeline] 상위 {len(timeline_entries)}개 시점 선정")

    # ── Step 3: 기존 항목 삭제 (overwrite) ────────────────
    if overwrite:
        deleted = clear_timeline_for_query(query_id)
        if deleted:
            print(f"  [Timeline] 기존 항목 {deleted}건 삭제")

    # ── Step 4: DB 저장 ────────────────────────────────────
    saved: list[dict] = []
    for entry in timeline_entries:
        date_str = str(entry["date"])

        # 해당 날짜 기사들 중 대표 기사 ID (top_art)
        day_df = df[df["date"] == date_str]
        top_art = _pick_rep_article_id(day_df) if not day_df.empty else 0

        entry_id = save_timeline_entry(
            query_id=query_id,
            timeline_date=date_str,
            top_art=top_art,
        )
        saved.append({
            "id":            entry_id,
            "timeline_date": date_str,
            "top_art":       top_art,
            "importance":    entry["importance"],
            "count":         entry["count"],
        })
        print(f"  [Timeline] ✓  {date_str} | {entry['count']}건 | article_id={top_art}")

    return {
        "saved_count": len(saved),
        "timeline":    saved,
    }
