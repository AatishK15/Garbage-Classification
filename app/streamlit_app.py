"""
Streamlit App — Main Garbage Classification Dashboard.

A premium, dark-themed web application for classifying waste images
with real-time predictions, confidence visualization, and disposal guidance.
"""

import io
import os
import sys
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="♻️ Garbage Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Dark Theme ──────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f9b58 0%, #1a73e8 50%, #00bcd4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .sub-header {
        text-align: center;
        color: #8899aa;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
        font-weight: 300;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1a1d23 0%, #252830 100%);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    .result-class {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }

    .result-confidence {
        font-size: 1.3rem;
        color: #2ECC71;
        font-weight: 600;
    }

    .result-low-confidence {
        font-size: 1.3rem;
        color: #E74C3C;
        font-weight: 600;
    }

    /* Disposal instruction card */
    .disposal-card {
        background: linear-gradient(145deg, #1a2332 0%, #1e2a3a 100%);
        border: 1px solid rgba(26, 115, 232, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    .disposal-title {
        color: #1a73e8;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }

    .disposal-text {
        color: #b0bec5;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Stats card */
    .stats-card {
        background: linear-gradient(145deg, #1a1d23 0%, #22252b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2ECC71, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stats-label {
        color: #8899aa;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(46, 204, 113, 0.4);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        background: rgba(46, 204, 113, 0.03);
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: rgba(46, 204, 113, 0.7);
        background: rgba(46, 204, 113, 0.06);
    }

    /* Confidence bar */
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        height: 12px;
        margin: 4px 0;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Prediction item */
    .prediction-item {
        display: flex;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Animated gradient border for main result */
    @keyframes gradient-shift {
        0% { border-color: #2ECC71; }
        33% { border-color: #1a73e8; }
        66% { border-color: #00bcd4; }
        100% { border-color: #2ECC71; }
    }

    .animated-border {
        animation: gradient-shift 4s ease infinite;
    }

    /* Pulse animation for icon */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    .pulse-icon {
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(46, 204, 113, 0.3), transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Database Helpers ────────────────────────────────────────────
def get_db_path() -> str:
    """Get the SQLite database path."""
    db_path = PROJECT_ROOT / "data" / "predictions" / "history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def init_db():
    """Initialize the predictions database."""
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_probabilities TEXT,
            inference_time_ms REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(result: dict, image_name: str):
    """Save a prediction to the database."""
    conn = sqlite3.connect(get_db_path())
    conn.execute(
        """INSERT INTO predictions (timestamp, image_name, predicted_class,
           confidence, all_probabilities, inference_time_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            image_name,
            result["predicted_class"],
            result["confidence"],
            json.dumps(result.get("all_probabilities", {})),
            result.get("inference_time_ms", 0),
        ),
    )
    conn.commit()
    conn.close()


def get_prediction_count() -> int:
    """Get total number of predictions."""
    try:
        conn = sqlite3.connect(get_db_path())
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_class_distribution() -> dict:
    """Get prediction counts per class."""
    try:
        conn = sqlite3.connect(get_db_path())
        rows = conn.execute(
            "SELECT predicted_class, COUNT(*) FROM predictions GROUP BY predicted_class"
        ).fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


# ─── Initialize ─────────────────────────────────────────────────
init_db()

# Load config
from src.utils.helpers import load_config, get_class_info
config = load_config()
class_info = get_class_info()


# ─── Model Loading (Cached) ─────────────────────────────────────
@st.cache_resource
def load_predictor(model_path: str):
    """Load the predictor with caching."""
    from src.inference.predictor import GarbagePredictor
    try:
        predictor = GarbagePredictor(model_path=model_path, config=config)
        return predictor
    except Exception as e:
        return None


def find_available_models() -> list:
    """Find all saved model files."""
    models_dir = PROJECT_ROOT / config.get("paths", {}).get("models_dir", "models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for ext in ["*.h5", "*.keras"]:
        models.extend(models_dir.glob(ext))
    return sorted(models, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♻️ Settings")
    st.markdown("---")

    # Model selection
    available_models = find_available_models()
    if available_models:
        model_names = [m.name for m in available_models]
        selected_model = st.selectbox(
            "🧠 Select Model",
            model_names,
            index=0,
            help="Choose a trained model for classification"
        )
        model_path = str(available_models[model_names.index(selected_model)])
        predictor = load_predictor(model_path)
    else:
        st.warning("⚠️ No trained models found.")
        st.info(
            "Train a model first:\n"
            "```bash\n"
            "python scripts/train.py --model mobilenetv2\n"
            "```"
        )
        predictor = None
        selected_model = None

    st.markdown("---")

    # Quick stats
    st.markdown("### 📊 Quick Stats")
    pred_count = get_prediction_count()
    dist = get_class_distribution()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predictions", pred_count)
    with col2:
        st.metric("Categories", len(dist) if dist else 7)

    if dist:
        st.markdown("#### Top Categories")
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_dist[:5]:
            info = class_info.get(cls, {})
            st.markdown(f"{info.get('icon', '❓')} **{cls}**: {count}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #556677; font-size: 0.8rem;'>"
        "Built with ❤️ using TensorFlow & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Main Content ───────────────────────────────────────────────
st.markdown('<h1 class="main-header">♻️ Garbage Classification System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered waste classification for a cleaner planet 🌍</p>',
    unsafe_allow_html=True,
)

# Horizontal divider
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Top stats row
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.markdown(
        f'<div class="stats-card">'
        f'<div class="stats-number">7</div>'
        f'<div class="stats-label">Waste Categories</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with stats_col2:
    st.markdown(
        f'<div class="stats-card">'
        f'<div class="stats-number">{pred_count}</div>'
        f'<div class="stats-label">Predictions Made</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with stats_col3:
    model_status = "✅" if predictor else "❌"
    st.markdown(
        f'<div class="stats-card">'
        f'<div class="stats-number">{model_status}</div>'
        f'<div class="stats-label">Model Status</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with stats_col4:
    accuracy_display = "85%+"
    st.markdown(
        f'<div class="stats-card">'
        f'<div class="stats-number">{accuracy_display}</div>'
        f'<div class="stats-label">Target Accuracy</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")
st.markdown("")

# ─── Image Upload Section ────────────────────────────────────────
st.markdown("### 📤 Upload Waste Image")

uploaded_file = st.file_uploader(
    "Drag and drop or click to upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP (max 10MB)",
    key="image_uploader",
)

if uploaded_file is not None:
    # Display upload info
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.caption(
        f"📁 **{uploaded_file.name}** — "
        f"{file_size_mb:.2f} MB — "
        f"{uploaded_file.type}"
    )

    # Layout: Image + Results side by side
    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.markdown("#### 🖼️ Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with result_col:
        if predictor is None:
            st.error(
                "⚠️ No model loaded. Please train a model first.\n\n"
                "```bash\npython scripts/train.py --model mobilenetv2\n```"
            )
        else:
            st.markdown("#### 🔍 Classification Result")

            # Run prediction with spinner
            with st.spinner("🔄 Analyzing image..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    result = predictor.predict_from_bytes(
                        file_bytes, filename=uploaded_file.name
                    )

                    # Primary result card
                    confidence_class = (
                        "result-confidence" if result["is_confident"]
                        else "result-low-confidence"
                    )

                    st.markdown(
                        f'<div class="result-card animated-border">'
                        f'<div class="pulse-icon" style="font-size: 3rem;">{result["icon"]}</div>'
                        f'<div class="result-class">{result["predicted_class"].upper()}</div>'
                        f'<div class="{confidence_class}">'
                        f'Confidence: {result["confidence_pct"]}'
                        f'</div>'
                        f'<div style="color: #8899aa; margin-top: 8px; font-size: 0.9rem;">'
                        f'⚡ Inference time: {result["inference_time_ms"]}ms'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if not result["is_confident"]:
                        st.warning(
                            "⚠️ Low confidence prediction. "
                            "The image might not contain clearly visible waste, "
                            "or the waste type might be ambiguous."
                        )

                    # Disposal instructions
                    st.markdown(
                        f'<div class="disposal-card">'
                        f'<div class="disposal-title">📋 Disposal Instructions</div>'
                        f'<div class="disposal-text">{result["instruction"]}</div>'
                        f'<div style="margin-top: 10px; color: #8899aa;">'
                        f'{result["bin"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Top predictions with confidence bars
                    st.markdown("#### 📊 Confidence Breakdown")

                    for pred in result["top_predictions"]:
                        conf_pct = pred["confidence"] * 100
                        bar_color = pred["color"]

                        col_icon, col_name, col_bar, col_pct = st.columns([0.5, 2, 4, 1])
                        with col_icon:
                            st.markdown(f"<span style='font-size: 1.5rem;'>{pred['icon']}</span>", unsafe_allow_html=True)
                        with col_name:
                            st.markdown(f"**{pred['class'].capitalize()}**")
                        with col_bar:
                            st.markdown(
                                f'<div class="confidence-bar-bg">'
                                f'<div class="confidence-bar-fill" '
                                f'style="width: {conf_pct}%; '
                                f'background: linear-gradient(90deg, {bar_color}, {bar_color}88);"></div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with col_pct:
                            st.markdown(f"**{pred['confidence_pct']}**")

                    # Save to history
                    st.markdown("")
                    if st.button("💾 Save to History", type="primary", use_container_width=True):
                        save_prediction(result, uploaded_file.name)
                        st.success("✅ Prediction saved to history!")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please ensure the uploaded file is a valid image.")

else:
    # Empty state
    st.markdown(
        '<div class="upload-area">'
        '<div style="font-size: 3rem; margin-bottom: 10px;">📸</div>'
        '<div style="font-size: 1.2rem; color: #aab; font-weight: 500;">'
        'Upload an image to classify waste</div>'
        '<div style="color: #667; margin-top: 8px; font-size: 0.9rem;">'
        'Supported: JPG, PNG, BMP, WEBP • Max 10MB</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Waste Category Guide ────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("### 🗂️ Waste Categories")

cat_cols = st.columns(4)
for i, (cls_name, info) in enumerate(sorted(class_info.items())):
    with cat_cols[i % 4]:
        st.markdown(
            f'<div class="stats-card" style="margin-bottom: 12px; text-align: left;">'
            f'<div style="font-size: 2rem; margin-bottom: 8px;">{info["icon"]}</div>'
            f'<div style="font-weight: 600; color: {info["color"]}; font-size: 1.05rem;">'
            f'{cls_name.capitalize()}</div>'
            f'<div style="color: #8899aa; font-size: 0.82rem; margin-top: 6px; line-height: 1.5;">'
            f'{info["instruction"]}</div>'
            f'<div style="color: #556677; font-size: 0.78rem; margin-top: 6px;">'
            f'{info["bin"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
