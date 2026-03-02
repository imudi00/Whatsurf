from src.clustering.embedding import sbert_embedding
from src.clustering.tfidf_embedding import tfidf_embedding
from src.clustering.reducer import reduce_dimension as reduce_dim
from src.clustering.clustering import cluster_docs

def run_service_clustering(texts):

    print("Step 1: SBERT Embedding")
    embeddings = sbert_embedding(texts)

    print("Step 2: UMAP Reduction")
    reduced = reduce_dim(embeddings)

    print("Step 3: HDBSCAN Clustering")
    labels = cluster_docs(reduced, method="density")

    return labels