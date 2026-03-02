from sentence_transformers import SentenceTransformer

def sbert_embedding(texts):
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings