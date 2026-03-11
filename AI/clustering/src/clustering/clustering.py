import hdbscan
from sklearn.cluster import KMeans


def kmeans_cluster(embeddings, n_clusters=10):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    return labels


def density_cluster(reduced_embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    return labels


def print_cluster_samples(texts, labels, max_samples=3):
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        print(f"\n===== Cluster {cluster_id} =====")

        cluster_docs = [text for text, label in zip(texts, labels) if label == cluster_id]

        for doc in cluster_docs[:max_samples]:
            print("-", doc[:120])