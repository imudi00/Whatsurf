from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: Optional[List[str]] = None


def load_emotions_csv(path: str) -> TextLabelDataset:
    df = pd.read_csv(path)
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist() if "label" in df.columns else None
    return TextLabelDataset(texts=texts, labels=labels)


def load_bias_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: text, tone_dem, tone_rep, rationale(optional)
    return df


def load_loaded_words_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: text, is_biased, bias_terms
    return df