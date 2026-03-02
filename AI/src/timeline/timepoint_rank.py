import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def _tokenize_simple(text: str):
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def rank_timepoints(df: pd.DataFrame, top_keywords_per_day=20) -> pd.DataFrame:
    grouped = df.groupby("date")
    ttp = grouped.size().astype(int)  # TTP: 시점 빈도(날짜별 기사 수)

    etp_map = {}
    for date, g in grouped:
        day_text = " ".join(g["body"].astype(str).tolist())
        tokens = _tokenize_simple(day_text)
        if not tokens:
            etp_map[date] = 0
            continue

        docs = [" ".join(tokens)]
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
        X = vectorizer.fit_transform(docs)
        vocab = np.array(vectorizer.get_feature_names_out())
        scores = X.toarray()[0]

        if len(scores) == 0:
            etp_map[date] = 0
            continue

        topk = min(top_keywords_per_day, len(scores))
        top_idx = scores.argsort()[-topk:][::-1]
        top_terms = set(vocab[top_idx].tolist())

        # ETP 근사: 상위 키워드 등장 빈도 합
        freq = sum(1 for t in tokens if t in top_terms)
        etp_map[date] = int(freq)

    etp = pd.Series(etp_map).sort_index().astype(int)

    max_etp = max(etp.max(), 1)
    max_ttp = max(ttp.max(), 1)

    # 논문 식(1) 구조(근사): 정규화 곱
    importance = (etp / max_etp) * (ttp / max_ttp)

    out = pd.DataFrame({
        "date": ttp.index.astype(str),
        "TTP": ttp.values,
        "ETP": etp.reindex(ttp.index).values,
        "importance": importance.values
    }).sort_values(["importance", "TTP"], ascending=False).reset_index(drop=True)

    return out
