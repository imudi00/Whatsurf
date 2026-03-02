import time
import csv
from datetime import datetime

from src.clustering import print_cluster_samples
from src.data_loader import load_naver_news
from src.embedding import sbert_embedding
from src.tfidf_embedding import tfidf_embedding
from src.clustering import kmeans_cluster
from src.evaluation import evaluate_cluster


def save_log(model_name, emb_time, clust_time, score):
    with open("experiment_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            model_name,
            emb_time,
            clust_time,
            score
        ])


def run_experiment(model_type):
    print("Loading data...")
    texts = load_naver_news()
    print(f"Number of documents: {len(texts)}")

    print(f"Running {model_type}...")

    # Embedding
    start = time.time()

    if model_type == "SBERT":
        embeddings = sbert_embedding(texts)
    elif model_type == "TFIDF":
        embeddings = tfidf_embedding(texts)
    else:
        raise ValueError("Unknown model type")

    emb_time = round(time.time() - start, 2)
    print(f"Embedding Time: {emb_time} sec")

    # K 실험 반복
    for k in [5, 10, 15, 20]:
        print(f"\n--- K = {k} ---")

        start = time.time()
        labels = kmeans_cluster(embeddings, n_clusters=k)
        clust_time = round(time.time() - start, 2)
        print(f"Clustering Time: {clust_time} sec")

        score = evaluate_cluster(embeddings, labels)
        print(f"Silhouette Score: {score}")

        save_log(f"{model_type}_K{k}", emb_time, clust_time, score)

        if k == 20:
            print_cluster_samples(texts, labels)


if __name__ == "__main__":
    print("==== SBERT Experiment ====")
    run_experiment("SBERT")

    print("\n==== TF-IDF Experiment ====")
    run_experiment("TFIDF")