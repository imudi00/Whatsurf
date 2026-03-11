from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.data_loader import load_news_df
@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: Optional[List[str]] = None


def load_emotions_csv(path: str) -> TextLabelDataset:
    df = load_news_df()

    texts = df["body"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist() if "label" in df.columns else None

    return TextLabelDataset(texts=texts, labels=labels)


def load_bias_csv(path: str) -> pd.DataFrame:
    return load_news_df()


def load_loaded_words_csv(path: str) -> pd.DataFrame:
    return load_news_df()