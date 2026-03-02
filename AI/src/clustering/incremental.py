import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_centroids(embeddings, labels):
    embeddings = np.array(embeddings)
    labels = np.array(labels)

def assign_new_article(new_embedding, centroids, threshold=0.75):
    best_label = None
    best_score = 0

    for label, centroid in centroids.items():
        score = cosine_similarity(
            new_embedding.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]

        if score > best_score:
            best_score = score
            best_label = label

    if best_score > 0.6:
        return best_label
    else:
        return "NEW_CLUSTER"