from data_loader import load_news_df

def main():
    df = load_news_df(limit=100)

    print("입력 데이터 크기:", df.shape)
    print(df.head())

    # 여기부터 기존 감정분석 함수 연결
    # 예:
    # from AI.ai_emotion_test.src.emotion_model import run_emotion_pipeline
    # result = run_emotion_pipeline(df)

if __name__ == "__main__":
    main()