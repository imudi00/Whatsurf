# src 경로 설정
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows cp949 콘솔에서 한글/특수문자 UnicodeEncodeError 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # wmic WinError 억제


import time
import csv
from datetime import datetime

from src.clustering.clustering import print_cluster_samples, kmeans_cluster
from src.clustering.data_loader import load_naver_news
from src.clustering.embedding import sbert_embedding
from src.clustering.tfidf_embedding import tfidf_embedding
from src.clustering.evaluation import evaluate_cluster


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

    start = time.time()

    if model_type == "SBERT":
        embeddings = sbert_embedding(texts)
    elif model_type == "TFIDF":
        embeddings = tfidf_embedding(texts)
    else:
        raise ValueError("Unknown model type")

    emb_time = round(time.time() - start, 2)
    print(f"Embedding Time: {emb_time} sec")

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