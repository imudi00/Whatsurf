"""
queries_timeline 테이블 저장
──────────────────────────────────────────────────
queries_timeline 실제 컬럼 (DB 확인):
  id (PK, auto)  query_id (FK)
  timeline_date (timestamp)  top_art (FK → articles.id)  created_at

[주의] 수동 삽입된 더미 데이터로 인해 PK 시퀀스가 어긋날 수 있음.
       → _safe_insert() 가 23505(dup key) 발생 시 MAX(id)+1 로 자동 보정.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _sb2():
    from AI.source.config.supabase2_client import supabase2
    return supabase2


def _safe_insert(table: str, row: dict) -> dict:
    """
    INSERT 시도. PK 중복(23505) 발생하면 MAX(id)+1 로 재시도.
    Supabase 시퀀스가 기존 더미 데이터와 어긋날 때를 자동 처리.
    """
    sb2 = _sb2()
    try:
        resp = sb2.table(table).insert(row).execute()
        return resp.data[0]
    except Exception as e:
        if "23505" not in str(e):
            raise

        # 시퀀스 불일치 → MAX(id) 조회 후 수동 id 지정
        pk = "id"
        max_resp = sb2.table(table).select(pk).order(pk, desc=True).limit(1).execute()
        max_id = max_resp.data[0][pk] if max_resp.data else 0
        row[pk] = max_id + 1
        resp = sb2.table(table).insert(row).execute()
        return resp.data[0]


def save_timeline_entry(
    *,
    query_id: int,
    timeline_date: str,   # "YYYY-MM-DD" — DB가 timestamp이므로 ISO string 허용
    top_art: int,         # 대표 기사 articles.id
) -> int:
    """
    queries_timeline 에 타임라인 날짜 1건 저장.

    Returns:
        생성된 id (PK) int
    """
    row = {
        "query_id":      query_id,
        "timeline_date": timeline_date,
        "top_art":       top_art,
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }
    data = _safe_insert("queries_timeline", row)
    return int(data["id"])


def clear_timeline_for_query(query_id: int) -> int:
    """
    재실행 시 기존 타임라인 항목 삭제.

    Returns:
        삭제된 행 수
    """
    resp = (
        _sb2()
        .table("queries_timeline")
        .delete()
        .eq("query_id", query_id)
        .execute()
    )
    return len(resp.data or [])
