import umap


def reduce_dimension(embeddings):
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced