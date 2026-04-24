from config.supabase_client import supabase

def test_news_table():
    response = supabase.table("ai_test").select("*").limit(5).execute()

    print("조회 성공")
    print(response.data)

if __name__ == "__main__":
    test_news_table()