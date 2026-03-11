from sentence_transformers import SentenceTransformer

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


def sbert_embedding(texts):
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings