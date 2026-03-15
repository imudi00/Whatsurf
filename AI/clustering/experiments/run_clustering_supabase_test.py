from AI.source.data_loader import load_news_df
from AI.clustering.src.clustering.service_pipeline import run_service_clustering
from AI.clustering.src.clustering.clustering import print_cluster_samples


def main():
    # 1. Supabase에서 뉴스 데이터 불러오기
    df = load_news_df(limit=300)

    # 2. 클러스터링용 텍스트 생성 (제목 + 본문)
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["cluster_text"] = (df["title"] + " " + df["body"]).str.strip()

    # 3. 빈 문자열 제거
    texts = [text for text in df["cluster_text"].tolist() if text]

    print("불러온 기사 수:", len(df))
    print("클러스터링 대상 텍스트 수:", len(texts))

    # 4. 클러스터링 실행
    labels = run_service_clustering(texts)

    # 5. 결과 출력
    print("생성된 라벨 수:", len(labels))
    print_cluster_samples(texts, labels, max_samples=3)
    
    # 클러스터 결과 저장
    result_df = df.iloc[:len(labels)].copy()
    result_df["cluster_label"] = labels

    result_df[["id", "title", "cluster_label"]].to_csv(
    "cluster_results.csv",
    index=False,
    encoding="utf-8-sig"
    )

    print("클러스터 결과 저장 완료: cluster_results.csv")


if __name__ == "__main__":
    main()