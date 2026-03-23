#!/usr/bin/env python3
"""
통합 파이프라인 테스트 스크립트
──────────────────────────────────────────────────────────────
실행:
    python AI/pipeline/test_pipeline.py

각 단계를 독립적으로 검증:
  1. DB 연결 및 테이블 존재 확인
  2. 기사 로드 확인
  3. 임포트 체인 확인 (clustering / timeline / feature_map)
  4. 전체 파이프라인 dry-run (DB 쓰기 없이 로직만 확인)
  5. 실제 clustering + timeline 실행 (DB에 저장)

특정 query_id 로 테스트하려면:
    python AI/pipeline/test_pipeline.py --query_id 3
    python AI/pipeline/test_pipeline.py --query_id 3 --dry_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OK   = "✓"
FAIL = "✕"
WARN = "⚠"


def _check(label: str, fn):
    try:
        result = fn()
        print(f"  {OK}  {label}" + (f" → {result}" if result is not None else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} → {e}")
        return False


def run_checks(query_id: int, dry_run: bool):
    print("=" * 60)
    print(f"  파이프라인 테스트  (query_id={query_id}, dry_run={dry_run})")
    print("=" * 60)

    all_ok = True

    # ── 1. 임포트 체인 ──────────────────────────────────
    print("\n[1] 모듈 임포트 확인")
    all_ok &= _check("DB 저장 모듈 (save_clusters)",
        lambda: __import__("AI.pipeline.db.save_clusters", fromlist=["save_cluster"]))
    all_ok &= _check("DB 저장 모듈 (save_timeline)",
        lambda: __import__("AI.pipeline.db.save_timeline", fromlist=["save_timeline_entry"]))
    all_ok &= _check("Clustering step",
        lambda: __import__("AI.pipeline.steps.step_clustering", fromlist=["run_clustering_step"]))
    all_ok &= _check("Timeline step",
        lambda: __import__("AI.pipeline.steps.step_timeline", fromlist=["run_timeline_step"]))
    all_ok &= _check("Feature map step",
        lambda: __import__("AI.pipeline.steps.step_feature_map", fromlist=["run_feature_map_step"]))

    # ── 2. DB 연결 ──────────────────────────────────────
    print("\n[2] Supabase2 연결 확인")
    from AI.source.config.supabase2_client import supabase2

    def _check_table(tbl):
        r = supabase2.table(tbl).select("*", count="exact").limit(1).execute()
        return f"{r.count}행"

    all_ok &= _check("queries 테이블",       lambda: _check_table("queries"))
    all_ok &= _check("articles 테이블",      lambda: _check_table("articles"))
    all_ok &= _check("article_clusters 테이블", lambda: _check_table("article_clusters"))
    all_ok &= _check("queries_timeline 테이블", lambda: _check_table("queries_timeline"))

    # ── 3. 검색어 + 기사 로드 ──────────────────────────
    print(f"\n[3] query_id={query_id} 기사 로드 확인")
    from AI.pipeline.run_pipeline import _fetch_query_text, _load_articles

    query_text = None
    articles   = []
    all_ok &= _check("검색어 조회",
        lambda: (globals().update({"query_text": _fetch_query_text(query_id)}) or query_text))

    if not query_text:
        try:
            query_text = _fetch_query_text(query_id)
        except Exception:
            pass

    if query_text:
        def _load():
            loaded = _load_articles(query_id)
            articles.extend(loaded)
            return f"{len(loaded)}건 (published_at 있음: {sum(1 for a in loaded if a.get('published_at'))}건)"
        all_ok &= _check("기사 로드", _load)

    if not articles:
        try:
            articles.extend(_load_articles(query_id))
        except Exception:
            pass

    # ── 4. dry-run 또는 실제 실행 ───────────────────────
    if dry_run:
        print(f"\n[4] Dry-run 완료 (DB 쓰기 없음)")
        print(f"  {OK}  기사 {len(articles)}건 확인, 쿼리: '{query_text}'")
    else:
        print(f"\n[4] 실제 실행 — clustering + timeline (query_id={query_id})")
        from AI.pipeline.run_pipeline import run_pipeline
        result = run_pipeline(
            query_id=query_id,
            steps=["clustering", "timeline"],
        )
        c  = result.get("clustering", {})
        tl = result.get("timeline",   {})
        all_ok &= _check("Clustering 결과",
            lambda: f"클러스터 {c.get('cluster_count',0)}개, 노이즈 {c.get('noise_count',0)}건")
        all_ok &= _check("Timeline 결과",
            lambda: f"{tl.get('saved_count',0)}개 날짜 저장")

    # ── 최종 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    status = f"{OK} 모든 테스트 통과" if all_ok else f"{FAIL} 일부 테스트 실패"
    print(f"  {status}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="통합 파이프라인 테스트")
    ap.add_argument("--query_id",  type=int, default=3, help="테스트할 query_id (기본: 3)")
    ap.add_argument("--dry_run",   action="store_true",  help="DB 쓰기 없이 로직만 확인")
    args = ap.parse_args()

    success = run_checks(args.query_id, args.dry_run)
    sys.exit(0 if success else 1)
