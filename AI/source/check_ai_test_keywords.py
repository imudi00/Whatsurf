from collections import Counter
import pandas as pd

from AI.source.supabase_client import get_supabase_client

client = get_supabase_client()

table = "ai_test"
all_rows = []
page_size = 1000
start = 0

while True:
    resp = (
        client.table(table)
        .select("id,title,published,keyword")
        .range(start, start + page_size - 1)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        break

    all_rows.extend(rows)

    if len(rows) < page_size:
        break

    start += page_size

df = pd.DataFrame(all_rows)

print("전체 행 수:", len(df))
print("컬럼 목록:", df.columns.tolist())

if "keyword" in df.columns:
    print("\n[keyword별 개수]")
    print(df["keyword"].fillna("(null)").value_counts().to_string())

    print("\n[전체 unique keyword 목록]")
    for kw in sorted(df["keyword"].dropna().astype(str).unique()):
        print("-", kw)

else:
    print("keyword 컬럼이 없습니다.")