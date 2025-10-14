import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for reliable imports
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import APP_TITLE, COLOR_PALETTE, PROCESSED_DIR, OUTPUT_DIR
# Import only lightweight modules at startup
from src import etl, nlp

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"## {APP_TITLE}")
st.caption("Advanced Multidimensional Review Intelligence & Sentiment Analytics")

# Sidebar actions
st.sidebar.title("Controls")

if st.sidebar.button("Run ETL"):
    p = etl.run_etl()
    try:
        st.toast(f"ETL complete → {p}", icon="✅")
    except Exception:
        pass
    st.success(f"ETL complete → {p}")

# ---------------- Live Review Analysis ----------------
## Dataset-only app (live text input removed by request)
    
# Performance controls
st.sidebar.subheader("Performance")
fast_mode = st.sidebar.checkbox("Fast mode (sample)", value=True)
sample_size = st.sidebar.number_input("Sample size", min_value=500, max_value=50000, value=3000, step=500)
batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=256, value=64, step=8)

# Sidebar buttons
if st.sidebar.button("Run NLP"):
    try:
        p = nlp.run_nlp(batch_size=int(batch_size), sample=int(sample_size) if fast_mode else None)
        st.success(f"NLP complete → {p}")
    except Exception as e:
        st.error(str(e))

if st.sidebar.button("Run Topics & Aspects"):
    try:
        import src.topic_aspect as topic_aspect
        t, a = topic_aspect.run_topics_and_aspects(sample=10000 if fast_mode else None)
        st.success(f"Topics → {t}; Aspects → {a}")
    except Exception as e:
        st.error(f"Topics/Aspects failed: {e}")

if st.sidebar.button("Run Impact"):
    try:
        import src.impact as impact
        p = impact.run_impact()
        st.success(f"Impact saved → {p}")
    except Exception as e:
        st.error(f"Impact failed: {e}")

if st.sidebar.button("Run Fraud Detection"):
    try:
        import src.fraud as fraud
        p = fraud.run_fraud()
        st.success(f"Fraud saved → {p}")
    except Exception as e:
        st.error(f"Fraud failed: {e}")

if st.sidebar.button("Run Segmentation"):
    try:
        import src.segments as segments
        p = segments.run_segments()
        st.success(f"Segments saved → {p}")
    except Exception as e:
        st.error(f"Segmentation failed: {e}")

if st.sidebar.button("Run Alerts"):
    try:
        import src.alerts as alerts
        p = alerts.detect_spikes()
        st.success(f"Alerts saved → {p}")
    except Exception as e:
        st.error(f"Alerts failed: {e}")

if st.sidebar.button("Build Weekly Report"):
    try:
        import src.report as report
        p = report.build_weekly_report()
        st.success(f"Report → {p}")
    except Exception as e:
        st.error(f"Report failed: {e}")

st.caption("Tip: Use sidebar to run each module. Fast mode gives instant results; turn it off for full data.")

# ===================== Main Visualizations =====================

def _load_parquet(path: str) -> pd.DataFrame | None:
    return pd.read_parquet(path) if os.path.exists(path) else None

# Try to load NLP data for overview & filters
df_nlp = _load_parquet(os.path.join(PROCESSED_DIR, "reviews_nlp.parquet"))

# Optional filters (only enabled if NLP data is present)
with st.sidebar:
    st.subheader("Filters")
    if df_nlp is not None:
        prod_opts = sorted([p for p in df_nlp.get("product_category", pd.Series([], dtype=str)).dropna().unique()])
        geo_opts = sorted([g for g in df_nlp.get("geo_location", pd.Series([], dtype=str)).dropna().unique()])
        product_filter = st.multiselect("Product Category", options=prod_opts, default=[])
        geo_filter = st.multiselect("Geo Location", options=geo_opts, default=[])
    else:
        st.info("Run NLP to enable filters and charts.")
        product_filter, geo_filter = [], []

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    out = df.copy()
    if product_filter:
        out = out[out.get("product_category", "").isin(product_filter)]
    if geo_filter:
        out = out[out.get("geo_location", "").isin(geo_filter)]
    return out

# -------- Overview & Trends --------
st.markdown("### Overview")
if df_nlp is not None:
    dfo = _apply_filters(df_nlp)
    if dfo is not None and len(dfo):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(dfo):,}")
        with col2:
            if "rating" in dfo.columns:
                st.metric("Avg Rating", f"{dfo['rating'].mean():.2f}")
            else:
                st.metric("Avg Rating", "N/A")
        with col3:
            pos = (dfo.get("predicted_sentiment", "") == "positive").mean() if "predicted_sentiment" in dfo else 0
            st.metric("Positivity", f"{pos:.0%}")
        with col4:
            st.metric("Data Window", f"{dfo.index.min() if 'timestamp' not in dfo else pd.to_datetime(dfo['timestamp']).min().date()} → {dfo.index.max() if 'timestamp' not in dfo else pd.to_datetime(dfo['timestamp']).max().date()}")

        # Sentiment daily trend
        if "timestamp" in dfo.columns:
            dfo["date"] = pd.to_datetime(dfo["timestamp"]).dt.date
            daily = dfo.groupby(["date", "predicted_sentiment"]).size().reset_index(name="count")
            fig = px.line(daily, x="date", y="count", color="predicted_sentiment", title="Daily Sentiment Counts")
            st.plotly_chart(fig, use_container_width=True)

        # Emotion distribution over time
        if "dominant_emotion" in dfo.columns and "timestamp" in dfo.columns:
            emo = dfo.groupby([pd.to_datetime(dfo["timestamp"]).dt.date, "dominant_emotion"]).size().reset_index(name="count")
            fig2 = px.area(emo, x="timestamp", y="count", color="dominant_emotion", title="Emotion Mix Over Time")
            fig2.update_traces(stackgroup="one")
            fig2.update_layout(xaxis_title="Date")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Run NLP to see Overview and Trends.")

# -------- Topics & Aspects (filter-aware) --------
st.markdown("### Topic Explorer & Aspects")
df_topics = _load_parquet(os.path.join(PROCESSED_DIR, "reviews_topics.parquet"))
df_aspects = _load_parquet(os.path.join(PROCESSED_DIR, "reviews_aspects.parquet"))

# Join topics/aspects back to NLP for filter columns
if df_nlp is not None:
    key = "review_text_clean"
    meta_cols = [c for c in ["product_category", "geo_location", key] if c in df_nlp.columns]
    if df_topics is not None and key in df_topics.columns:
        df_topics = df_topics.merge(df_nlp[meta_cols], left_on="text", right_on=key, how="left")
        df_topics = _apply_filters(df_topics)
    if df_aspects is not None and key in df_aspects.columns:
        df_aspects = df_aspects.merge(df_nlp[meta_cols], left_on="text", right_on=key, how="left")
        df_aspects = _apply_filters(df_aspects)

if df_topics is not None and len(df_topics):
    figt = px.histogram(df_topics, x="topic", title="Topic Distribution (filtered)")
    st.plotly_chart(figt, use_container_width=True)
if df_aspects is not None and len(df_aspects):
    text_blob = " ".join(df_aspects["aspects"].dropna().tolist())
    if text_blob:
        wc = WordCloud(width=900, height=300, background_color="white").generate(text_blob)
        fig_wc, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
if (df_topics is None or not len(df_topics)) and (df_aspects is None or not len(df_aspects)):
    st.info("Run Topics & Aspects to populate this section.")

# -------- Impact --------
st.markdown("### Business Impact")
# Recompute impact on the fly for the filtered view
if df_nlp is not None:
    dfi = _apply_filters(df_nlp)
    if dfi is not None and len(dfi):
        if "timestamp" in dfi.columns:
            dfi["week"] = pd.to_datetime(dfi["timestamp"]).dt.to_period("W").dt.start_time
        else:
            dfi["week"] = pd.date_range("2023-01-01", periods=len(dfi), freq="W")
        dfi["neg_flag"] = (dfi.get("predicted_sentiment", "").astype(str) == "negative").astype(float)
        weekly = dfi.groupby("week").agg(neg_sent_rate=("neg_flag", "mean"), count=("review_text_clean", "count")).reset_index()
        # synthetic sales dependent on negative rate
        base = 1000
        weekly["weekly_sales"] = base * (1 - 0.4 * weekly["neg_sent_rate"]) + np.random.normal(0, 10, len(weekly))
        figi = px.scatter(weekly, x="neg_sent_rate", y="weekly_sales", title="Weekly Sales vs Negative Sentiment (filtered)")
        try:
            x = weekly["neg_sent_rate"].to_numpy(); y = weekly["weekly_sales"].to_numpy()
            if x.size >= 2:
                coef = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 100); ys = coef[0] * xs + coef[1]
                figi.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Linear fit"))
        except Exception:
            pass
        st.plotly_chart(figi, use_container_width=True)
    else:
        st.info("Adjust filters or run NLP to view impact.")
else:
    st.info("Run NLP to generate KPI correlations.")

# -------- Segments --------
st.markdown("### Customer Segments")
df_seg = _load_parquet(os.path.join(OUTPUT_DIR, "segments.parquet"))
if df_seg is not None and len(df_seg) and "segment" in df_seg.columns:
    # If NLP is present and filters are applied, show only users in filtered set
    if df_nlp is not None:
        dff = _apply_filters(df_nlp)
        if dff is not None and "user_id" in dff.columns and "user_id" in df_seg.columns:
            df_seg = df_seg[df_seg["user_id"].isin(dff["user_id"].unique())]
    figs = px.histogram(df_seg, x="segment", title="Segment Counts (filtered)")
    st.plotly_chart(figs, use_container_width=True)
    st.dataframe(df_seg.head(20))
else:
    st.info("Run Segmentation to create customer groups.")

# -------- Alerts --------
st.markdown("### Alerts")
# Recompute alerts from filtered NLP for filter-aware view
if df_nlp is not None:
    dfa = _apply_filters(df_nlp)
    if dfa is not None and len(dfa):
        if "timestamp" in dfa.columns:
            dfa["date"] = pd.to_datetime(dfa["timestamp"]).dt.date
        else:
            dfa["date"] = pd.date_range("2023-01-01", periods=len(dfa), freq="D").date
        dfa["neg_flag"] = (dfa.get("predicted_sentiment", "").astype(str) == "negative").astype(float)
        daily = dfa.groupby("date").agg(neg_rate=("neg_flag", "mean"), total=("review_text_clean", "count")).reset_index()
        daily["spike"] = daily["neg_rate"].pct_change().fillna(0) > 0.2
        st.dataframe(daily.tail(20))
    else:
        st.info("Adjust filters or run NLP to view alerts.")
else:
    st.info("Run NLP to detect recent spikes in negative sentiment.")
