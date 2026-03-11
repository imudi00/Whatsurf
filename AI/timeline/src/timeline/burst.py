import pandas as pd

def compute_daily_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date").size().sort_index()

def detect_burst_points(counts: pd.Series, z_threshold=2.0, min_count=2):
    if len(counts) == 0:
        return []

    values = counts.values.astype(float)
    mu = values.mean()
    sigma = values.std(ddof=0) if values.std(ddof=0) > 0 else 1.0
    threshold = mu + z_threshold * sigma

    bursts = counts[(counts >= threshold) & (counts >= min_count)]
    return [str(d) for d in bursts.index]
