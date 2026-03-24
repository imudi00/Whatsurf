"""
Clustering Step
───────────────────────────────────────────────────────────────
① SBERT 임베딩 → UMAP 차원 축소 → HDBSCAN 클러스터링
② 각 클러스터: LLM으로 cluster_title / cluster_summary 생성
③ article_clusters 테이블 저장
④ articles.cluster_label 업데이트

반환:
    {
        "cluster_count": int,           # 유효 클러스터 수 (-1 노이즈 제외)
        "noise_count":   int,           # 노이즈로 분류된 기사 수
        "cluster_map":   {hdbscan_label: db_label},  # 레이블 매핑
        "labels":        list[int],     # 기사별 hdbscan 레이블
    }
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv  # [추가]

# clustering 모듈 경로 등록
_CLUSTERING_SRC = Path(__file__).resolve().parents[2] / "clustering" / "src"
if str(_CLUSTERING_SRC) not in sys.path:
    sys.path.insert(0, str(_CLUSTERING_SRC))

# joblib wmic WinError 방지 (Windows)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# HuggingFace 캐시를 프로젝트 루트로 고정
# (서버가 Administrator 등 다른 계정으로 실행될 때 Path.home() 접근 권한 오류 방지)
_HF_CACHE = str(Path(__file__).resolve().parents[3] / ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _HF_CACHE)

import hdbscan as _hdbscan

from clustering.embedding import sbert_embedding          # type: ignore
from clustering.reducer import reduce_dimension           # type: ignore
from clustering.clustering import density_cluster         # type: ignore


def _density_cluster_safe(reduced, n_articles: int):
    """
    기사 수에 따라 min_cluster_size를 동적 조정.
    HDBSCAN 최소 요구: n_samples >= min_samples + 1 (기본 min_samples = min_cluster_size).
    """
    # 기사 수 기반 동적 파라미터
    min_cs = max(3, min(10, n_articles // 5))   # 3 ~ 10
    min_ms = max(2, min_cs - 1)                 # min_samples = min_cluster_size - 1

    print(f"  [Clustering] HDBSCAN 파라미터: min_cluster_size={min_cs}, min_samples={min_ms}")

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=min_ms,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced)

from AI.llm.llm_client import call_llm_json
from AI.pipeline.db.save_clusters import save_cluster, update_article_cluster_label


# ── LLM 프롬프트 ────────────────────────────────────────────

def _cluster_summary_prompt(cluster_texts: list[str], query: str) -> str:
    sample = "\n".join(f"- {t[:200]}" for t in cluster_texts[:6])
    return f"""아래는 '{query}' 검색어로 수집된 뉴스 중 같은 군집으로 묶인 기사 일부입니다.

{sample}

다음 JSON 형식으로만 응답하세요 (코드블록 없이):
{{
  "title":   "이 군집의 핵심 주제를 15자 이내로 요약",
  "summary": "이 군집 기사들의 공통 내용을 2~3문장으로 요약"
}}"""


# ── 메인 함수 ────────────────────────────────────────────────

def run_clustering_step(
    query_id: int,
    query_text: str,
    articles: list[dict],
) -> dict:
    """
    Args:
        query_id:   queries 테이블 PK
        query_text: 검색어 (LLM 프롬프트용)
        articles:   [{"id": int, "title": str, "body": str, ...}, ...]

    Returns: 실행 결과 요약 dict
    """
    if not articles:
        print("  [Clustering] ⚠️  기사 없음 — 스킵")
        return {"cluster_count": 0, "noise_count": 0, "cluster_map": {}, "labels": []}

    # 최소 기사 수 체크 (HDBSCAN은 최소 5건 필요)
    MIN_ARTICLES = 5
    if len(articles) < MIN_ARTICLES:
        print(f"  [Clustering] ⚠️  기사 수({len(articles)})가 너무 적음 (최소 {MIN_ARTICLES}건) — 스킵")
        return {"cluster_count": 0, "noise_count": len(articles), "cluster_map": {}, "labels": [-1] * len(articles)}

    texts = [f"{a.get('title', '')} {a.get('body', '')}" for a in articles]

    # ── Step 1: 임베딩 ─────────────────────────────────────
    print(f"  [Clustering] SBERT 임베딩 ({len(texts)}건)...")
    embeddings = sbert_embedding(texts)

    # ── Step 2: 차원 축소 ──────────────────────────────────
    print("  [Clustering] UMAP 차원 축소...")
    reduced = reduce_dimension(embeddings)

    # ── Step 3: 클러스터링 ─────────────────────────────────
    print("  [Clustering] HDBSCAN 클러스터링...")
    raw_labels = _density_cluster_safe(reduced, len(articles))  # 기사 수 기반 동적 파라미터

    labels_list: list[int] = [int(l) for l in raw_labels]
    unique_clusters = sorted({l for l in labels_list if l != -1})
    noise_count = labels_list.count(-1)
    print(f"  [Clustering] 클러스터 {len(unique_clusters)}개, 노이즈 {noise_count}건")

    # ── Step 4: LLM 요약 + DB 저장 ────────────────────────
    cluster_map: dict[int, int] = {}   # hdbscan_label → db label(PK)

    for cl in unique_clusters:
        indices = [i for i, lab in enumerate(labels_list) if lab == cl]
        cluster_articles = [articles[i] for i in indices]
        cluster_texts    = [texts[i]    for i in indices]

        # 대표 기사: 본문 가장 긴 것
        rep = max(cluster_articles, key=lambda a: len(a.get("body", "")))
        rep_article_id = rep["id"]

        # LLM 요약
        try:
            result = call_llm_json(_cluster_summary_prompt(cluster_texts, query_text))
            cluster_title   = str(result.get("title",   f"클러스터 {cl}"))[:100]
            cluster_summary = str(result.get("summary", ""))
        except Exception as e:
            print(f"  [Clustering] ⚠️  LLM 오류 (cluster {cl}): {e}")
            cluster_title   = f"클러스터 {cl}"
            cluster_summary = ""

        # DB 저장
        db_label = save_cluster(
            query_id=query_id,
            cluster_title=cluster_title,
            cluster_summary=cluster_summary,
            rep_article_id=rep_article_id,
        )
        cluster_map[cl] = db_label
        print(f"  [Clustering] ✓  cluster {cl} → label {db_label}: {cluster_title}")

    # ── Step 5: articles.cluster_label 업데이트 ───────────
    updated = 0
    for i, article in enumerate(articles):
        hl = labels_list[i]
        if hl == -1:
            continue   # 노이즈 → null 유지
        update_article_cluster_label(article["id"], cluster_map[hl])
        updated += 1
    print(f"  [Clustering] articles.cluster_label 업데이트 {updated}건")

    return {
        "cluster_count": len(unique_clusters),
        "noise_count":   noise_count,
        "cluster_map":   cluster_map,
        "labels":        labels_list,
    }
