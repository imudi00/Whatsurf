# io_utils.py
"""
감정 피처 데이터 입출력 유틸리티
"""
import os
import io
import json
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: Optional[List[str]] = None


# ──────────────────────────────────────────────
# CSV 로드
# ──────────────────────────────────────────────

def _decode_csv_bytes(raw: bytes) -> str:
    """인코딩 자동 감지 후 디코딩"""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Failed to decode CSV. Tried utf-8, cp949, euc-kr.")


def load_emotions_csv(path: str) -> TextLabelDataset:
    """감정 레이블 CSV 로드 → TextLabelDataset 반환"""
    raw = open(path, "rb").read()
    df = pd.read_csv(io.StringIO(_decode_csv_bytes(raw)))
    df.columns = [c.strip().lower() for c in df.columns]

    texts = df["body"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist() if "label" in df.columns else None
    return TextLabelDataset(texts=texts, labels=labels)


def load_bias_csv(path: str) -> pd.DataFrame:
    """편향 데이터 CSV 로드"""
    raw = open(path, "rb").read()
    df = pd.read_csv(io.StringIO(_decode_csv_bytes(raw)))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_loaded_words_csv(path: str) -> pd.DataFrame:
    """편향 단어 CSV 로드"""
    return load_bias_csv(path)


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
