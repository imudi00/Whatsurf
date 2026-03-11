from data_loader import load_news_df

def main():
    df = load_news_df(limit=5)

    print("불러온 행 개수:", len(df))
    print("컬럼 목록:", list(df.columns))
    print(df.head())

if __name__ == "__main__":
    main()