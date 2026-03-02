#Kmeans
#from sklearn.cluster import KMeans
#import time

#def kmeans_cluster(X, n_clusters=10):
    #start = time.time()
    #model = KMeans(n_clusters=n_clusters, random_state=42)
    #labels = model.fit_predict(X)
    #end = time.time()

    #print("Clustering Time:", round(end - start, 2), "sec")
    #return labels

#def print_cluster_samples(texts, labels, n=3):
    #from collections import defaultdict

    #clusters = defaultdict(list)

    #for text, label in zip(texts, labels):
        #clusters[label].append(text)

    #for label, items in clusters.items():
        #print(f"\n===== Cluster {label} =====")
        #for sample in items[:n]:
            #print("-", sample[:120])
            
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def hierarchical_cluster(embeddings):

    model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.25,
    metric="cosine",
    linkage="average"
)

    labels = model.fit_predict(embeddings)
    return labels


# 2. Issue Tree 생성
def build_issue_tree(texts, labels):
    tree = defaultdict(list)

    for text, label in zip(texts, labels):
        tree[label].append(text)

    return tree


# 3. Cluster centroid 계산
def compute_centroids(embeddings, labels):
    centroids = {}

    unique_labels = set(labels)

    for label in unique_labels:
        cluster_vectors = embeddings[labels == label]
        centroids[label] = np.mean(cluster_vectors, axis=0)

    return centroids

import hdbscan

def density_cluster(reduced_embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    return labels