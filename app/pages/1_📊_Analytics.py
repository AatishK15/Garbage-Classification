"""
Analytics Page — Prediction history and classification statistics.

Displays prediction history table, classification distribution charts,
daily trends, and export functionality.
"""

import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.helpers import get_class_info

st.set_page_config(
    page_title="📊 Analytics — Garbage Classification",
    page_icon="📊",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .analytics-header {
        background: linear-gradient(135deg, #1a73e8, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1d23, #22252b);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2ECC71;
    }
    .metric-label {
        color: #8899aa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class_info = get_class_info()


def get_db_path():
    return str(PROJECT_ROOT / "data" / "predictions" / "history.db")


def load_predictions() -> pd.DataFrame:
    """Load all predictions from database."""
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


# ─── Page Content ────────────────────────────────────────────────
st.markdown('<h1 class="analytics-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Track your classification history and waste distribution patterns.")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.info(
        "📭 No predictions yet. Upload and classify some images on the main page "
        "to see analytics here."
    )
    st.stop()

# ─── Key Metrics ─────────────────────────────────────────────────
st.markdown("### 📈 Key Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{len(df)}</div>'
        f'<div class="metric-label">Total Predictions</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    avg_conf = df["confidence"].mean() * 100
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{avg_conf:.1f}%</div>'
        f'<div class="metric-label">Avg Confidence</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    unique_classes = df["predicted_class"].nunique()
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{unique_classes}</div>'
        f'<div class="metric-label">Categories Found</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    most_common = df["predicted_class"].mode().iloc[0] if not df.empty else "N/A"
    icon = class_info.get(most_common, {}).get("icon", "❓")
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{icon}</div>'
        f'<div class="metric-label">Most Common: {most_common}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ─── Charts Row ──────────────────────────────────────────────────
st.markdown("### 📊 Classification Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Pie chart
    class_counts = df["predicted_class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]

    colors = [class_info.get(cls, {}).get("color", "#808080") for cls in class_counts["Class"]]

    fig_pie = px.pie(
        class_counts,
        values="Count",
        names="Class",
        title="Waste Category Distribution",
        color_discrete_sequence=colors,
        hole=0.4,
    )
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CCCCCC",
        title_font_size=16,
        title_font_color="#FFFFFF",
        legend_font_size=11,
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    # Bar chart
    fig_bar = px.bar(
        class_counts,
        x="Class",
        y="Count",
        title="Predictions per Category",
        color="Class",
        color_discrete_sequence=colors,
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CCCCCC",
        title_font_size=16,
        title_font_color="#FFFFFF",
        showlegend=False,
        xaxis=dict(title="", tickangle=45),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── Confidence Distribution ────────────────────────────────────
st.markdown("### 📉 Confidence Distribution")

fig_conf = px.histogram(
    df,
    x=df["confidence"] * 100,
    nbins=20,
    title="Prediction Confidence Distribution",
    labels={"x": "Confidence (%)", "count": "Number of Predictions"},
    color_discrete_sequence=["#2ECC71"],
)
fig_conf.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#CCCCCC",
    title_font_size=16,
    title_font_color="#FFFFFF",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)
st.plotly_chart(fig_conf, use_container_width=True)

# ─── Daily Trend ────────────────────────────────────────────────
if len(df) > 1:
    st.markdown("### 📅 Daily Prediction Trend")

    daily = df.groupby(df["timestamp"].dt.date).size().reset_index()
    daily.columns = ["Date", "Count"]

    fig_daily = px.line(
        daily,
        x="Date",
        y="Count",
        title="Predictions Over Time",
        markers=True,
        color_discrete_sequence=["#1a73e8"],
    )
    fig_daily.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#CCCCCC",
        title_font_size=16,
        title_font_color="#FFFFFF",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

# ─── Prediction History Table ────────────────────────────────────
st.markdown("### 📋 Prediction History")

display_df = df[["timestamp", "image_name", "predicted_class", "confidence", "inference_time_ms"]].copy()
display_df.columns = ["Timestamp", "Image", "Predicted Class", "Confidence", "Time (ms)"]
display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
display_df["Time (ms)"] = display_df["Time (ms)"].round(1)

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    column_config={
        "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
    },
)

# Export button
csv_data = display_df.to_csv(index=False)
st.download_button(
    label="📥 Export History as CSV",
    data=csv_data,
    file_name=f"garbage_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    type="secondary",
)
