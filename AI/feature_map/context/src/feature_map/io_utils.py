# io_utils.py
"""
맥락 피처 입출력 유틸리티
"""
import os
import io
import json
import pandas as pd


REQUIRED_COLS = ["body"]


def _decode_csv_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Failed to decode CSV.")


def load_news_csv(path: str) -> pd.DataFrame:
    """뉴스 CSV 로드 (body 컬럼 필수)"""
    raw = open(path, "rb").read()
    df = pd.read_csv(io.StringIO(_decode_csv_bytes(raw)))
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["body"] = df["body"].fillna("").astype(str)
    return df


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
