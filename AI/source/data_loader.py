import pandas as pd
from AI.source.config.supabase_client import supabase


def load_table_as_df(table_name: str, limit: int | None = None) -> pd.DataFrame:
    batch_size = 1000
    all_rows = []
    start = 0
    target_limit = limit if limit is not None else 1000000

    print(f"[DEBUG] load_table_as_df(table_name={table_name}, limit={limit})")

    while start < target_limit:
        end = min(start + batch_size - 1, target_limit - 1)
        print(f"[DEBUG] requesting rows {start}~{end}")

        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(start, end)
            .execute()
        )

        rows = response.data or []
        print(f"[DEBUG] received {len(rows)} rows")

        if not rows:
            break

        all_rows.extend(rows)

        requested_count = end - start + 1
        if len(rows) < requested_count:
            break

        start += batch_size

    if limit is not None:
        all_rows = all_rows[:limit]

    print(f"[DEBUG] final loaded rows = {len(all_rows)}")
    return pd.DataFrame(all_rows)


def load_news_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("news", limit=limit)


def load_press_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("press", limit=limit)


def load_ai_test_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("ai_test", limit=limit)

def load_ai_test_body_map_by_ids(article_ids: list[str | int]) -> dict:
    """
    ai_test 테이블에서 article_ids에 해당하는 기사들의 id/title/body/published를 조회해서
    {id: {...}} 형태의 dict로 반환
    """
    if not article_ids:
        return {}

    ids = [str(x) for x in article_ids if str(x).strip()]
    if not ids:
        return {}

    response = (
        supabase
        .table("ai_test")
        .select("id,title,body,published")
        .in_("id", ids)
        .execute()
    )

    rows = response.data or []
    result = {}

    for row in rows:
        row_id = str(row.get("id"))
        result[row_id] = {
            "id": row.get("id"),
            "title": row.get("title", ""),
            "body": row.get("body", ""),
            "published": row.get("published", "")
        }

    return result