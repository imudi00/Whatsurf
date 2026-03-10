from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_embedding(texts):
    vectorizer = TfidfVectorizer(
        max_features=10000
    )
    embeddings = vectorizer.fit_transform(texts)
    return embeddings