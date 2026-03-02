import os, json
import pandas as pd

REQUIRED_COLS = ["date", "title", "body"]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} (need {REQUIRED_COLS})")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    if "url" not in df.columns:
        df["url"] = ""
    else:
        df["url"] = df["url"].fillna("").astype(str)

    df = df[df["body"].str.strip().astype(bool)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
