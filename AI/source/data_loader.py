import pandas as pd
from AI.source.config.supabase_client import supabase


# ──────────────────────────────────────────────────────────
# Supabase (기존 DB) — ai_test / news / press
# ──────────────────────────────────────────────────────────

def load_table_as_df(table_name: str, limit: int | None = None) -> pd.DataFrame:
    batch_size = 1000
    all_rows = []
    start = 0
    target_limit = limit if limit is not None else 1000000

    while start < target_limit:
        end = min(start + batch_size - 1, target_limit - 1)

        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(start, end)
            .execute()
        )

        rows = response.data or []
        if not rows:
            break

        all_rows.extend(rows)

        requested_count = end - start + 1
        if len(rows) < requested_count:
            break

        start += batch_size

    if limit is not None:
        all_rows = all_rows[:limit]

    return pd.DataFrame(all_rows)


def load_news_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("news", limit=limit)


def load_press_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("press", limit=limit)


def load_ai_test_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("ai_test", limit=limit)


# ──────────────────────────────────────────────────────────
# Supabase2 (신규 DB) — articles / article_features
# ──────────────────────────────────────────────────────────

def load_supabase2_articles(
    query_id: str | None = None,
    limit: int = 0,
) -> list[dict]:
    """
    Supabase2 'articles' 테이블에서 id, query_id, title, body_text 로드.

    Args:
        query_id: 필터링할 query_id (None 이면 전체)
        limit:    최대 건수 (0 = 전체)

    Returns:
        list[dict] — 각 행에 body / comments 키 정규화 완료
                     (body_text → body, comments=[] 추가)
    """
    from AI.source.config.supabase2_client import supabase2

    PAGE_SIZE = 1000
    rows: list[dict] = []
    offset = 0

    while True:
        fetch = PAGE_SIZE if limit == 0 else min(PAGE_SIZE, limit - len(rows))
        query = (
            supabase2
            .table("articles")
            .select("id, query_id, title, body_text")
        )
        if query_id is not None:
            query = query.eq("query_id", query_id)
        query = query.range(offset, offset + fetch - 1)

        resp = query.execute()
        page = resp.data or []
        rows.extend(page)

        if len(page) < fetch or (limit > 0 and len(rows) >= limit):
            break
        offset += fetch

    if limit > 0:
        rows = rows[:limit]

    # 컬럼 정규화: body_text → body, comments=[]
    for r in rows:
        r["body"]     = r.pop("body_text", "") or ""
        r["comments"] = []

    return rows


def load_supabase2_articles_df(
    query_id: str | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    """load_supabase2_articles 의 DataFrame 버전"""
    return pd.DataFrame(load_supabase2_articles(query_id=query_id, limit=limit))