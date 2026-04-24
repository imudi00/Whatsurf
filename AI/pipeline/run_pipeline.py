#!/usr/bin/env python3
"""
AI 통합 파이프라인
══════════════════════════════════════════════════════════════
queries 테이블의 검색어 하나(query_id)를 받아 3단계 분석을 순서대로 실행.

실행 흐름:
    1. [Data]        Supabase2 articles 테이블 → 기사 로드
    2. [Clustering]  SBERT → UMAP → HDBSCAN → LLM 요약
                     → article_clusters 저장 + articles.cluster_label 업데이트
    3. [Feature Map] labeling 파이프라인(frame/logic/bias/omission/stance…)
                     → label_results/ 로컬 JSON 저장
                     → article_features upsert + comments upsert (자동)
    4. [Timeline]    TTP/ETP 랭킹 → 상위 시점 선정
                     → queries_timeline 저장

CLI 예시:
    # query_id=15 전체 실행
    python -m AI.pipeline.run_pipeline --query_id 15

    # 클러스터링 + 타임라인만
    python -m AI.pipeline.run_pipeline --query_id 15 --steps clustering timeline

    # 피처맵 피처 지정
    python -m AI.pipeline.run_pipeline --query_id 15 --features frame logic stance

    # DB 업로드 없이 로컬 JSON만 저장
    python -m AI.pipeline.run_pipeline --query_id 15 --no_upload

    # 타임라인 top_n 조정
    python -m AI.pipeline.run_pipeline --query_id 15 --top_timepoints 10
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 40.WhatSurf/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Windows cp949 UnicodeEncodeError 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 기사 로더 ───────────────────────────────────────────────

def _load_articles(query_id: int) -> list[dict]:
    """
    Supabase2 articles 테이블에서 query_id 기반 기사 로드.
    pipeline에서 필요한 모든 컬럼(id, title, body, published_at, url) 포함.
    """
    from AI.source.config.supabase2_client import supabase2

    PAGE_SIZE = 1000
    rows: list[dict] = []
    offset = 0

    while True:
        resp = (
            supabase2
            .table("articles")
            .select("id, query_id, title, body_text, published_at, url, source_id")
            .eq("query_id", query_id)
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        page = resp.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    # 컬럼 정규화: body_text → body
    for r in rows:
        r["body"] = r.pop("body_text", "") or ""

    return rows


def _fetch_query_text(query_id: int) -> str:
    """queries 테이블에서 query_text 조회."""
    from AI.source.config.supabase2_client import supabase2

    resp = (
        supabase2
        .table("queries")
        .select("query_text")
        .eq("id", query_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise ValueError(f"query_id={query_id} 가 queries 테이블에 없습니다.")
    return rows[0]["query_text"]


# ── 메인 파이프라인 ─────────────────────────────────────────

def run_pipeline(
    query_id: int,
    steps: list[str],
    *,
    features: list[str] | None = None,
    top_timepoints: int = 8,
    limit: int = 0,
    resume: bool = True,
    skip_comments: bool = False,
    out_dir: str | None = None,
    upload: bool = True,
) -> dict:
    """
    Args:
        query_id:       분석할 검색어 ID (queries.id)
        steps:          실행할 스텝 목록 ["clustering", "feature_map", "timeline"]
        features:       feature_map 에서 실행할 피처 (None=all)
        top_timepoints: timeline에서 저장할 최대 날짜 수
        limit:          기사 최대 로드 수 (0=전체)
        resume:         feature_map 중단 재개 여부
        skip_comments:  feature_map 댓글 건너뜀 여부
        out_dir:        feature_map 라벨 JSON 출력 디렉토리
        upload:         feature_map 완료 후 DB 자동 업로드 여부 (기본: True)

    Returns:
        각 스텝 결과 요약 dict
    """
    # out_dir: None이면 프로젝트 루트 기준 절대경로로 자동 설정
    # (app.py 실행 위치와 무관하게 항상 동일한 경로 사용)
    if out_dir is None:
        out_dir = str(_PROJECT_ROOT / "label_results")

    started_at = datetime.now(timezone.utc)
    results: dict = {}

    print("=" * 60)
    print(f"  AI 통합 파이프라인 시작")
    print(f"  query_id : {query_id}")
    print(f"  steps    : {steps}")
    print(f"  시작 시각 : {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # ── 검색어 조회 ─────────────────────────────────────────
    print("\n[0] 검색어 및 기사 로드...")
    query_text = _fetch_query_text(query_id)
    print(f"  query_text: '{query_text}'")

    articles = _load_articles(query_id)
    print(f"  기사 {len(articles)}건 로드 완료")

    if not articles:
        print("  ⚠️  기사가 없어 파이프라인을 종료합니다.")
        return {"error": "no_articles"}

    # ── Step 1: Clustering ──────────────────────────────────
    if "clustering" in steps:
        print("\n[1] Clustering 실행...")
        t0 = time.time()

        from AI.pipeline.steps.step_clustering import run_clustering_step
        results["clustering"] = run_clustering_step(
            query_id=query_id,
            query_text=query_text,
            articles=articles,
        )
        elapsed = time.time() - t0
        print(f"  → 완료 ({elapsed:.1f}s) | "
              f"클러스터 {results['clustering']['cluster_count']}개, "
              f"노이즈 {results['clustering']['noise_count']}건")

    # ── Step 2: Feature Map ─────────────────────────────────
    if "feature_map" in steps:
        print("\n[2] Feature Map 실행...")
        t0 = time.time()

        from AI.pipeline.steps.step_feature_map import run_feature_map_step
        results["feature_map"] = run_feature_map_step(
            query_id=query_id,
            features=features,
            limit=limit,
            resume=resume,
            skip_comments=skip_comments,
            out_dir=out_dir,
            upload=upload,
        )
        elapsed = time.time() - t0
        fm = results["feature_map"]
        status = "✓ 완료" if fm["success"] else "✕ 실패"
        upload_status = ""
        if upload:
            art = "✓" if fm["upload_articles"] else "✕"
            cmt = "skip" if skip_comments else ("✓" if fm["upload_comments"] else "✕")
            upload_status = f" | DB 업로드 article:{art} comment:{cmt}"
        print(f"  → {status} ({elapsed:.1f}s){upload_status}")

    # ── Step 3: Timeline ────────────────────────────────────
    if "timeline" in steps:
        print("\n[3] Timeline 실행...")
        t0 = time.time()

        from AI.pipeline.steps.step_timeline import run_timeline_step
        results["timeline"] = run_timeline_step(
            query_id=query_id,
            query_text=query_text,
            articles=articles,
            top_timepoints=top_timepoints,
        )
        elapsed = time.time() - t0
        print(f"  → 완료 ({elapsed:.1f}s) | "
              f"타임라인 {results['timeline']['saved_count']}개 저장")

    # ── 완료 요약 ───────────────────────────────────────────
    finished_at = datetime.now(timezone.utc)
    total_sec = (finished_at - started_at).total_seconds()

    print("\n" + "=" * 60)
    print(f"  파이프라인 완료 — 총 {total_sec:.1f}초")
    if "clustering" in results:
        c = results["clustering"]
        print(f"  [Clustering]  클러스터 {c['cluster_count']}개 | 노이즈 {c['noise_count']}건")
    if "feature_map" in results:
        fm = results["feature_map"]
        label_ok = "성공" if fm["success"] else "실패"
        if upload:
            art = "✓" if fm["upload_articles"] else "✕"
            cmt = "skip" if skip_comments else ("✓" if fm["upload_comments"] else "✕")
            print(f"  [FeatureMap]  라벨링:{label_ok} | DB article:{art} comment:{cmt} | 피처:{fm['features']}")
        else:
            print(f"  [FeatureMap]  {label_ok} | 피처: {fm['features']} (DB 업로드 스킵)")
    if "timeline" in results:
        tl = results["timeline"]
        print(f"  [Timeline]    {tl['saved_count']}개 날짜 저장")
    print("=" * 60)

    results["meta"] = {
        "query_id":    query_id,
        "query_text":  query_text,
        "article_cnt": len(articles),
        "steps":       steps,
        "started_at":  started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_sec": round(total_sec, 1),
    }
    return results


# ── CLI ─────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="AI 통합 파이프라인 — query_id 하나로 clustering/feature_map/timeline 실행"
    )
    ap.add_argument(
        "--query_id", type=int, required=True,
        help="queries 테이블 PK (분석할 검색어 ID)",
    )
    ap.add_argument(
        "--steps", nargs="+",
        default=["clustering", "feature_map", "timeline"],
        choices=["clustering", "feature_map", "timeline"],
        help="실행할 스텝 (기본: 전체)",
    )
    ap.add_argument(
        "--features", nargs="+", default=None,
        help="feature_map 에서 실행할 피처 (기본: all)",
    )
    ap.add_argument(
        "--top_timepoints", type=int, default=8,
        help="타임라인 상위 날짜 수 (기본: 8)",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="기사 최대 수 — feature_map 전용 (0=전체)",
    )
    ap.add_argument(
        "--no_resume", action="store_true",
        help="feature_map 중단 재개 비활성화",
    )
    ap.add_argument(
        "--skip_comments", action="store_true",
        help="feature_map 댓글 라벨링 건너뜀",
    )
    ap.add_argument(
        "--out_dir", default="./label_results",
        help="feature_map 라벨 JSON 출력 경로",
    )
    ap.add_argument(
        "--no_upload", action="store_true",
        help="feature_map 완료 후 DB 업로드 건너뜀 (로컬 JSON만 저장)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        query_id=args.query_id,
        steps=args.steps,
        features=args.features,
        top_timepoints=args.top_timepoints,
        limit=args.limit,
        resume=not args.no_resume,
        skip_comments=args.skip_comments,
        out_dir=args.out_dir,
        upload=not args.no_upload,
    )
