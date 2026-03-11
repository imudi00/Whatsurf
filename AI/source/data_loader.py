import pandas as pd
from config.supabase_client import supabase


def load_table_as_df(table_name: str, limit: int | None = None) -> pd.DataFrame:
    query = supabase.table(table_name).select("*")

    if limit is not None:
        query = query.limit(limit)

    response = query.execute()
    data = response.data or []

    return pd.DataFrame(data)


def load_news_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("news", limit=limit)


def load_press_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("press", limit=limit)


def load_ai_test_df(limit: int | None = None) -> pd.DataFrame:
    return load_table_as_df("ai_test", limit=limit)