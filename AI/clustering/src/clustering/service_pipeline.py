from AI.clustering.src.clustering.embedding import sbert_embedding
from AI.clustering.src.clustering.reducer import reduce_dimension
from AI.clustering.src.clustering.clustering import density_cluster

def run_service_clustering(texts):
    print("Step 1: SBERT Embedding")
    embeddings = sbert_embedding(texts)

    print("Step 2: UMAP Reduction")
    reduced = reduce_dimension(embeddings)

    print("Step 3: HDBSCAN Clustering")
    labels = density_cluster(reduced)

    return labels