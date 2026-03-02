from src.data_loader import load_naver_news
from src.service_pipeline import run_service_clustering

if __name__ == "__main__":

    texts = load_naver_news()
    print(f"Documents: {len(texts)}")

    labels = run_service_clustering(texts)

    print("\nCluster Results:")
    print(labels)