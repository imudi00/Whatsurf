"""
article_clusters 테이블 저장 + articles.cluster_label 업데이트
─────────────────────────────────────────────────────────────
article_clusters 컬럼:
  label (PK, auto)  query_id (FK)  frame_id (FK, nullable)
  cluster_title     cluster_summary  rep_article_id (FK)  created_at
"""
from __future__ import annotations

from datetime import datetime, timezone


def _sb2():
    from AI.source.config.supabase2_client import supabase2
    return supabase2


def save_cluster(
    *,
    query_id: int,
    cluster_title: str,
    cluster_summary: str,
    rep_article_id: int,
    frame_id: int | None = None,
) -> int:
    """
    article_clusters 에 클러스터 1건 저장.

    Returns:
        생성된 label (PK) int
    """
    row: dict = {
        "query_id":        query_id,
        "cluster_title":   cluster_title,
        "cluster_summary": cluster_summary,
        "rep_article_id":  rep_article_id,
        "created_at":      datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if frame_id is not None:
        row["frame_id"] = frame_id

    resp = _sb2().table("article_clusters").insert(row).execute()
    return int(resp.data[0]["label"])


def update_article_cluster_label(article_id: int, cluster_label: int) -> None:
    """
    articles 테이블의 cluster_label(FK) 컬럼 업데이트.
    클러스터링 결과 반영 시 호출.
    """
    _sb2().table("articles").update(
        {"cluster_label": cluster_label}
    ).eq("id", article_id).execute()
