import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from wordcloud import WordCloud
import base64
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Ensure project root is on sys.path for reliable imports
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import APP_TITLE, COLOR_PALETTE, PROCESSED_DIR, OUTPUT_DIR
# Import only lightweight modules at startup
from src import etl, nlp

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

# Apply gold-focused palette to all charts (UI colors remain in CSS/theme)
_gold_seq = ["#F5C542", "#E0AA3E", "#2B2B2B", "#DADCE0"]
pio.templates["intellireview_gold_ui"] = go.layout.Template(
    layout=go.Layout(
        colorway=_gold_seq,
        paper_bgcolor="#F5F6FA",   # matches main background
        plot_bgcolor="#FFFFFF",    # white cards
        font=dict(color="#2B2B2B", family="Inter, Poppins, Roboto, Arial, sans-serif"),
        xaxis=dict(
            gridcolor="#000000", gridwidth=2.0,
            zerolinecolor="#000000",
            linecolor="#000000", linewidth=2.2, mirror=False,
            ticks="outside", tickwidth=2, tickcolor="#000000", ticklen=6,
            title=dict(font=dict(color="#000000", size=13)),
            tickfont=dict(color="#000000", size=12)
        ),
        yaxis=dict(
            gridcolor="#000000", gridwidth=2.0,
            zerolinecolor="#000000",
            linecolor="#000000", linewidth=2.2, mirror=False,
            ticks="outside", tickwidth=2, tickcolor="#000000", ticklen=6,
            title=dict(font=dict(color="#000000", size=13)),
            tickfont=dict(color="#000000", size=12)
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
)
px.defaults.template = "intellireview_gold_ui"
pio.templates.default = "intellireview_gold_ui"

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # JS helper to pulse-highlight the first visualization inside a section card
    st.markdown(
        """
        <script>
          window.pulseViz = function(anchorId){
            const tryPulse = (attempts=0)=>{
              const anchor = document.getElementById(anchorId);
              const card = anchor ? anchor.nextElementSibling : null;
              const viz = card ? card.querySelector('[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]') : null;
              if (viz){
                // retrigger animation
                viz.classList.remove('viz-pulse');
                void viz.offsetWidth;
                viz.classList.add('viz-pulse');
                card.scrollIntoView({behavior:'smooth', block:'start'});
              } else if (attempts < 20){
                setTimeout(()=>tryPulse(attempts+1), 120);
              }
            };
            tryPulse();
          }
        </script>
        """,
        unsafe_allow_html=True,
    )

# Focus/scroll state
if "focus" not in st.session_state:
    st.session_state["focus"] = None
if "pulse" not in st.session_state:
    st.session_state["pulse"] = None

_HERO_TITLE = "NLP-Powered Sentiment & Review Analytics"
st.markdown(
    """
    <div class="hero">
      <h1 class="app-title">NLP-Powered Sentiment & Review Analytics</h1>
      <div class="app-sub">Advanced Multidimensional Review Intelligence &amp; Sentiment Analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
        st.session_state["focus"] = "overview"
        st.session_state["pulse"] = "overview"
    except Exception as e:
        st.error(str(e))

if st.sidebar.button("Run Topics & Aspects"):
    try:
        import src.topic_aspect as topic_aspect
        t, a = topic_aspect.run_topics_and_aspects(sample=10000 if fast_mode else None)
        st.success(f"Topics → {t}; Aspects → {a}")
        st.session_state["focus"] = "topics"
        st.session_state["pulse"] = "topics"
    except Exception as e:
        st.error(f"Topics/Aspects failed: {e}")

if st.sidebar.button("Run Impact"):
    try:
        import src.impact as impact
        p = impact.run_impact()
        st.success(f"Impact saved → {p}")
        st.session_state["focus"] = "impact"
        st.session_state["pulse"] = "impact"
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
        st.session_state["focus"] = "segments"
        st.session_state["pulse"] = "segments"
    except Exception as e:
        st.error(f"Segmentation failed: {e}")

if st.sidebar.button("Run Alerts"):
    try:
        import src.alerts as alerts
        p = alerts.detect_spikes()
        st.success(f"Alerts saved → {p}")
        st.session_state["focus"] = "alerts"
        st.session_state["pulse"] = "alerts"
    except Exception as e:
        st.error(f"Alerts failed: {e}")

if st.sidebar.button("Build Weekly Report"):
    try:
        import src.report as report
        p = report.build_weekly_report()
        st.success(f"Report → {p}")
        st.session_state["report_path"] = p
        st.session_state["focus"] = "report"
        st.session_state["pulse"] = "report"
    except Exception as e:
        st.error(f"Report failed: {e}")


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

# -------- Helpers for nice display --------
def _fmt_date_window(df: pd.DataFrame) -> str:
    """Return a clean date window string for the metric without ellipsis."""
    try:
        if df is None or not len(df):
            return "No data"
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            start = ts.min()
            end = ts.max()
            if pd.isna(start) and pd.isna(end):
                return "No timestamp"
            if pd.isna(end) and not pd.isna(start):
                return f"Since {start.date()}"
            if pd.isna(start) and not pd.isna(end):
                return f"Until {end.date()}"
            return f"{start.date()} – {end.date()}"
        # Fall back to index if no timestamp
        try:
            start_idx = pd.to_datetime(df.index.min(), errors="coerce")
            end_idx = pd.to_datetime(df.index.max(), errors="coerce")
            if pd.isna(start_idx) and pd.isna(end_idx):
                return "No timestamp"
            if pd.isna(end_idx):
                return f"Since {start_idx.date()}"
            if pd.isna(start_idx):
                return f"Until {end_idx.date()}"
            return f"{start_idx.date()} – {end_idx.date()}"
        except Exception:
            return "No timestamp"
    except Exception:
        return "No timestamp"

# -------- Overview & Trends --------
focus = st.session_state.get("focus")
st.markdown('<div id="section-overview"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if focus=="overview" else ""}">', unsafe_allow_html=True)
st.markdown("### Overview")
if df_nlp is not None:
    dfo = _apply_filters(df_nlp)
    if dfo is not None and len(dfo):
        col1, col2, col3 = st.columns(3)
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

        # Sentiment daily trend
        if "timestamp" in dfo.columns:
            dfo["date"] = pd.to_datetime(dfo["timestamp"]).dt.date
            daily = dfo.groupby(["date", "predicted_sentiment"]).size().reset_index(name="count")
            fig = px.line(
                daily,
                x="date",
                y="count",
                color="predicted_sentiment",
                title="Daily Sentiment Counts",
            )
            fig.update_xaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                              ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                              title_font_color="#000", tickfont_color="#000")
            fig.update_yaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                              ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                              title_font_color="#000", tickfont_color="#000")
            st.plotly_chart(fig, use_container_width=True)

        # Emotion distribution over time
        if "dominant_emotion" in dfo.columns and "timestamp" in dfo.columns:
            emo = dfo.groupby([pd.to_datetime(dfo["timestamp"]).dt.date, "dominant_emotion"]).size().reset_index(name="count")
            fig2 = px.area(emo, x="timestamp", y="count", color="dominant_emotion", title="Emotion Mix Over Time")
            fig2.update_traces(stackgroup="one")
            fig2.update_layout(xaxis_title="Date")
            fig2.update_xaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                               ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                               title_font_color="#000", tickfont_color="#000")
            fig2.update_yaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                               ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                               title_font_color="#000", tickfont_color="#000")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Run NLP to see Overview and Trends.")
st.markdown('</div>', unsafe_allow_html=True)
if st.session_state.get("pulse") == "alerts":
    components.html("""
        <script>
          const card = document.getElementById('section-alerts')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stDataFrame"], [data-testid="stPlotlyChart"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None
if focus == "overview":
    components.html("""
        <script>
          const el = document.getElementById('section-overview');
          if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
        </script>
    """, height=0)
    st.session_state["focus"] = None
if st.session_state.get("pulse") == "overview":
    components.html("""
        <script>
          const card = document.getElementById('section-overview')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None

# -------- Topics & Aspects (filter-aware) --------
st.markdown('<div id="section-topics"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if focus=="topics" else ""}">', unsafe_allow_html=True)
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
    figt.update_xaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                      ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                      title_font_color="#000", tickfont_color="#000")
    figt.update_yaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                      ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                      title_font_color="#000", tickfont_color="#000")
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
st.markdown('</div>', unsafe_allow_html=True)
if focus == "topics":
    components.html("""
        <script>
          const el = document.getElementById('section-topics');
          if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
        </script>
    """, height=0)
    st.session_state["focus"] = None
if st.session_state.get("pulse") == "topics":
    components.html("""
        <script>
          const card = document.getElementById('section-topics')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None

## Weekly Report moved to the bottom after Alerts

# -------- Impact --------
st.markdown('<div id="section-impact"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if focus=="impact" else ""}">', unsafe_allow_html=True)
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
        figi.update_xaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                          ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                          title_font_color="#000", tickfont_color="#000")
        figi.update_yaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                          ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                          title_font_color="#000", tickfont_color="#000")
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
st.markdown('</div>', unsafe_allow_html=True)
# -------- Segments --------
st.markdown('<div id="section-segments"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if focus=="segments" else ""}">', unsafe_allow_html=True)
st.markdown("### Customer Segments")
df_seg = _load_parquet(os.path.join(OUTPUT_DIR, "segments.parquet"))
if df_seg is not None and len(df_seg) and "segment" in df_seg.columns:
    # If NLP is present and filters are applied, show only users in filtered set
    if df_nlp is not None:
        dff = _apply_filters(df_nlp)
        if dff is not None and "user_id" in dff.columns and "user_id" in df_seg.columns:
            df_seg = df_seg[df_seg["user_id"].isin(dff["user_id"].unique())]
    figs = px.histogram(df_seg, x="segment", title="Segment Counts (filtered)")
    figs.update_xaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                      ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                      title_font_color="#000", tickfont_color="#000")
    figs.update_yaxes(gridcolor="#000", gridwidth=2, zerolinecolor="#000", linecolor="#000", linewidth=2,
                      ticks="outside", tickwidth=2, tickcolor="#000", ticklen=6,
                      title_font_color="#000", tickfont_color="#000")
    st.plotly_chart(figs, use_container_width=True)
    # Preview table: drop first row, then add S.No starting at 1
    df_seg_view = df_seg.reset_index(drop=True).iloc[1:].copy()
    df_seg_view.insert(0, "s_no", range(1, len(df_seg_view) + 1))
    st.dataframe(df_seg_view.head(200), hide_index=True)
else:
    st.info("Run Segmentation to create customer groups.")
st.markdown('</div>', unsafe_allow_html=True)
if focus == "segments":
    components.html("""
        <script>
          const el = document.getElementById('section-segments');
          if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
        </script>
    """, height=0)
    st.session_state["focus"] = None
if st.session_state.get("pulse") == "segments":
    components.html("""
        <script>
          const card = document.getElementById('section-segments')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None

# -------- Alerts --------
st.markdown('<div id="section-alerts"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if focus=="alerts" else ""}">', unsafe_allow_html=True)
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
        df_alerts_view = daily.tail(20).reset_index(drop=True).iloc[1:].copy()
        df_alerts_view.insert(0, "s_no", range(1, len(df_alerts_view) + 1))
        st.dataframe(df_alerts_view, hide_index=True)
    else:
        st.info("Adjust filters or run NLP to view alerts.")
else:
    st.info("Run NLP to detect recent spikes in negative sentiment.")
st.markdown('</div>', unsafe_allow_html=True)

# Pulse for Impact
if st.session_state.get("pulse") == "impact":
    components.html("""
        <script>
          const card = document.getElementById('section-impact')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None
# -------- Weekly Report (download/preview) - LAST --------
st.markdown('<div id="section-report"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card {"highlight" if st.session_state.get("focus")=="report" else ""}">', unsafe_allow_html=True)
st.markdown("### Weekly Report")
report_path = st.session_state.get("report_path")
if not report_path:
    default_path = os.path.join(OUTPUT_DIR, "weekly_report.pdf")
    report_path = default_path if os.path.exists(default_path) else None

if report_path and os.path.exists(report_path):
    with open(report_path, "rb") as f:
        pdf_bytes = f.read()
    st.download_button(
        label="Download Weekly Report (PDF)",
        data=pdf_bytes,
        file_name="weekly_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    try:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        components.html(f"""
            <embed src='data:application/pdf;base64,{b64}' type='application/pdf' width='100%' height='380px' />
        """, height=400)
    except Exception:
        pass
else:
    st.info("Click 'Build Weekly Report' in the sidebar to generate the latest PDF.")
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get("focus") == "report":
    components.html("""
        <script>
          const el = document.getElementById('section-report');
          if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
        </script>
    """, height=0)
    st.session_state["focus"] = None
if st.session_state.get("pulse") == "report":
    components.html("""
        <script>
          const card = document.getElementById('section-report')?.nextElementSibling;
          if (card){
            const viz = card.querySelector('[data-testid="stDataFrame"], [data-testid="stPlotlyChart"]');
            if (viz){ viz.classList.add('viz-pulse'); card.scrollIntoView({behavior:'smooth', block:'start'}); }
          }
        </script>
    """, height=0)
    st.session_state["pulse"] = None
