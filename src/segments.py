import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, OUTPUT_DIR, RANDOM_STATE


def _load_nlp() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run NLP first to create reviews_nlp.parquet")
    return pd.read_parquet(path)


def run_segments(k: int = 4) -> str:
    df = _load_nlp()
    # Prepare columns for valid aggregations
    df["neg_flag"] = (df.get("predicted_sentiment", "").astype(str) == "negative").astype(float)
    if "verified_purchase" in df.columns and df["verified_purchase"].dtype != "O":
        ver_col = "verified_purchase"
    else:
        df["_verified"] = 0.0
        ver_col = "_verified"

    grp = (
        df.groupby("user_id")
        .agg(
            avg_rating=("rating", "mean") if "rating" in df.columns else ("neg_flag", "mean"),
            neg_rate=("neg_flag", "mean"),
            conf_var=("sentiment_confidence", "var") if "sentiment_confidence" in df.columns else ("neg_flag", "var"),
            verified_rate=(ver_col, "mean"),
        )
        .fillna(0)
    )

    feats = grp[["avg_rating", "neg_rate", "conf_var", "verified_rate"]].values
    X = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    grp["segment"] = km.fit_predict(X)

    out_path = os.path.join(OUTPUT_DIR, "segments.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grp.reset_index().to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    print(run_segments())
