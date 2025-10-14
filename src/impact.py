import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .config import PROCESSED_DIR, OUTPUT_DIR, RANDOM_STATE


def _load_nlp() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "reviews_nlp.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run NLP first to create reviews_nlp.parquet")
    return pd.read_parquet(path)


def synthesize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    # Group by week and create synthetic KPIs if not provided
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["week"] = df["timestamp"].dt.to_period("W").dt.start_time
    else:
        df["week"] = pd.date_range("2023-01-01", periods=len(df), freq="W")

    # Prepare indicator columns for valid groupby aggregations
    df["neg_flag"] = (df.get("predicted_sentiment", "").astype(str) == "negative").astype(float)
    df["anger_flag"] = (df.get("dominant_emotion", "").astype(str) == "anger").astype(float)

    agg_map = {
        "neg_sent_rate": ("neg_flag", "mean"),
        "anger_rate": ("anger_flag", "mean"),
        "count": ("review_text_clean", "count"),
    }
    if "rating" in df.columns:
        agg_map["avg_rating"] = ("rating", "mean")
    else:
        # If no rating column, synthesize a stable average ~4.0
        df["_synthetic_rating"] = 4.0
        agg_map["avg_rating"] = ("_synthetic_rating", "mean")

    weekly = df.groupby("week").agg(**agg_map).reset_index()

    # Synthetic sales/returns driven partly by sentiment
    rng = np.random.default_rng(RANDOM_STATE)
    base_sales = 1000 + rng.normal(0, 50, len(weekly))
    weekly["weekly_sales"] = base_sales * (1 - 0.4 * weekly["neg_sent_rate"]) + rng.normal(0, 20, len(weekly))
    weekly["returns"] = 50 + 200 * weekly["anger_rate"] + rng.normal(0, 5, len(weekly))
    return weekly


def run_impact() -> str:
    df = _load_nlp()
    weekly = synthesize_kpis(df)

    # Linear regression: sales vs negative sentiment
    X = weekly[["neg_sent_rate"]].values
    y = weekly["weekly_sales"].values
    lr = LinearRegression().fit(X, y)
    weekly["impact_score"] = lr.coef_[0]

    # Predict next 4 weeks with XGB (toy)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
    xgb.fit(X, y)
    weekly["pred_sales"] = xgb.predict(X)

    out_path = os.path.join(OUTPUT_DIR, "kpi_impact.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weekly.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    print(run_impact())
