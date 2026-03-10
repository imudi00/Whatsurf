import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .text_utils import split_sentences_kor, eojel_count_kor, normalize_text

def _tokenize_simple(text: str):
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def score_sentences_tfidf(sentences, query, alpha=0.5):
    if not sentences:
        return []

    cleaned = [normalize_text(s) for s in sentences]
    vectorizer = TfidfVectorizer(tokenizer=_tokenize_simple, lowercase=False)
    X = vectorizer.fit_transform(cleaned)
    tfidf = X.toarray()
    vocab = vectorizer.get_feature_names_out()

    base_scores = tfidf.sum(axis=1)

    q_tokens = set(_tokenize_simple(query))
    if q_tokens:
        vocab_index = {t: i for i, t in enumerate(vocab)}
        bonus = np.zeros(len(sentences), dtype=float)
        for qt in q_tokens:
            if qt in vocab_index:
                bonus += tfidf[:, vocab_index[qt]]
        scores = base_scores + alpha * bonus  # 논문: alpha=0.5
    else:
        scores = base_scores

    return scores.tolist()

def summarize_timepoint(day_articles, query, max_eojel=17, top_sentences=3, alpha=0.5):
    sents = []
    for body in day_articles["body"].astype(str).tolist():
        sents.extend(split_sentences_kor(body))

    # 논문: 17 어절 이하만 사용
    cand = [s for s in sents if 1 <= eojel_count_kor(s) <= max_eojel]
    if not cand:
        cand = sents[:50]

    scores = score_sentences_tfidf(cand, query=query, alpha=alpha)
    if not scores:
        return ""

    idx = np.argsort(scores)[::-1]
    topk = min(top_sentences, len(idx))
    chosen = [cand[i] for i in idx[:topk]]

    # 중복 제거
    seen, unique = set(), []
    for s in chosen:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)

    return " ".join(unique)
