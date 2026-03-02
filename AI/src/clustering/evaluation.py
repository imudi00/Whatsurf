"""from sklearn.metrics import silhouette_score

def evaluate(X, labels):
    score = silhouette_score(X, labels)
    print("Silhouette Score:", round(score, 4))
    return score"""
    
from sklearn.metrics import silhouette_score


def evaluate_cluster(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    return round(score, 4)