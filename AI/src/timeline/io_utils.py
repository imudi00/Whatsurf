import os
import json
import io
import pandas as pd

REQUIRED_COLS = ["date", "title", "body"]

def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def _decode_csv_bytes(raw: bytes) -> str:
    """
    Windows/Excel/PowerShell 환경에서 흔한 UTF-8/UTF-8-SIG/CP949/EUC-KR 인코딩을
    안전하게 처리해서 '확실히' 올바른 문자열로 만든다.
    """
    # 1) UTF-8 BOM 있으면 확정
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    # 2) 후보 인코딩 순서대로 시도
    encodings_to_try = ["utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e

    # 3) 그래도 실패하면 마지막 에러로 종료
    raise UnicodeDecodeError(
        f"Failed to decode CSV bytes. Tried {encodings_to_try}. Last error: {last_err}"
    )

def load_news_csv(path: str) -> pd.DataFrame:
    # ✅ 핵심: pandas가 직접 파일을 열게 하지 말고, 우리가 바이트->문자열 디코딩을 확정한다.
    raw = open(path, "rb").read()
    text = _decode_csv_bytes(raw)

    # pandas에는 "문자열 스트림"으로 전달
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} (need {REQUIRED_COLS})")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    if "url" not in df.columns:
        df["url"] = ""
    else:
        df["url"] = df["url"].fillna("").astype(str)

    # date가 파싱 실패한 행 제거
    df = df[df["date"].notna()].copy()

    # body 빈 행 제거
    df = df[df["body"].str.strip().astype(bool)].copy()

    df.reset_index(drop=True, inplace=True)
    return df

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)