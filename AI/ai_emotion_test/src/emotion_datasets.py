from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from data_load import load_news_df

@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: Optional[List[str]] = None


def load_emotions_csv(path: str) -> TextLabelDataset:
    df = load_news_df()

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist() if "label" in df.columns else None

    return TextLabelDataset(texts=texts, labels=labels)


def load_bias_csv(path: str) -> pd.DataFrame:
    return load_news_df()


def load_loaded_words_csv(path: str) -> pd.DataFrame:
    return load_news_df()