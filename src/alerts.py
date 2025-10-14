import os
import pandas as pd
from datetime import timedelta

from .config import PROCESSED_DIR, OUTPUT_DIR, NEGATIVE_SPIKE_THRESHOLD


def _load_nlp() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run NLP first to create reviews_nlp.parquet")
    return pd.read_parquet(path)


def detect_spikes() -> str:
    df = _load_nlp()
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["date"] = df["timestamp"].dt.date
    else:
        df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D").date

    # Compute negative sentiment flag for valid groupby aggregation
    df["neg_flag"] = (df.get("predicted_sentiment", "").astype(str) == "negative").astype(float)
    daily = (
        df.groupby("date")
        .agg(
            neg_rate=("neg_flag", "mean"),
            count=("review_text_clean", "count"),
        )
        .reset_index()
    )
    daily["spike"] = daily["neg_rate"].pct_change().fillna(0) > NEGATIVE_SPIKE_THRESHOLD

    out_path = os.path.join(OUTPUT_DIR, "alerts.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    print(detect_spikes())
