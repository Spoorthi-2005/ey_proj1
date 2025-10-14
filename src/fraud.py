import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import PROCESSED_DIR, OUTPUT_DIR, RANDOM_STATE


def _load_nlp() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run NLP first to create reviews_nlp.parquet")
    return pd.read_parquet(path)


def run_fraud() -> str:
    df = _load_nlp()
    # Build simple behavior features
    grp = df.groupby("user_id").agg(
        review_count=("review_text_clean", "count"),
        avg_len=("review_text_clean", lambda x: np.mean([len(t.split()) for t in x])),
        avg_rating=("rating", "mean") if "rating" in df.columns else ("sentiment_confidence", "mean")
    ).fillna(0)

    features = grp[["review_count", "avg_len", "avg_rating"]].values
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    grp["anomaly"] = iso.fit_predict(features)  # -1 anomalous

    outliers = grp[grp["anomaly"] == -1].reset_index()
    out_path = os.path.join(OUTPUT_DIR, "fake_reviewers.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outliers.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    print(run_fraud())
