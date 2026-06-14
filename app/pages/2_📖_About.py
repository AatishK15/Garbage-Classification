"""
About Page — Project information, model details, and waste disposal guide.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils.helpers import get_class_info

st.set_page_config(
    page_title="📖 About — Garbage Classification",
    page_icon="📖",
    layout="wide",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .about-header {
        background: linear-gradient(135deg, #0f9b58, #2ECC71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .info-card {
        background: linear-gradient(145deg, #1a1d23, #22252b);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(46, 204, 113, 0.15);
        color: #2ECC71;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 3px;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class_info = get_class_info()

# ─── Header ──────────────────────────────────────────────────────
st.markdown('<h1 class="about-header">📖 About This Project</h1>', unsafe_allow_html=True)
st.markdown("---")

# ─── Project Overview ────────────────────────────────────────────
st.markdown("### 🌍 Project Overview")

st.markdown("""
<div class="info-card">
<p>The <strong>Garbage Classification System</strong> is an AI-powered web application
that uses deep learning to classify waste images into 7 categories, promoting
proper waste segregation and environmental awareness.</p>

<p>The system employs <strong>transfer learning</strong> with pre-trained convolutional neural
networks (CNNs) fine-tuned on a garbage classification dataset, achieving high accuracy
in identifying waste types from photographs.</p>
</div>
""", unsafe_allow_html=True)

# ─── Technology Stack ────────────────────────────────────────────
st.markdown("### 🛠️ Technology Stack")

tech_items = [
    "Python 3.10+", "TensorFlow / Keras", "MobileNetV2", "ResNet50",
    "EfficientNetB0", "OpenCV", "Streamlit", "Plotly",
    "scikit-learn", "NumPy", "Pandas", "Matplotlib", "Seaborn",
    "SQLite", "PIL / Pillow", "PyYAML",
]

tech_html = " ".join(f'<span class="tech-badge">{t}</span>' for t in tech_items)
st.markdown(f'<div class="info-card">{tech_html}</div>', unsafe_allow_html=True)

# ─── Model Architecture ─────────────────────────────────────────
st.markdown("### 🧠 Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Transfer Learning Pipeline:**

    1. **Base Model**: Pre-trained on ImageNet (1M+ images, 1000 classes)
    2. **Feature Extraction**: Freeze base layers, train custom head
    3. **Fine-tuning**: Unfreeze top 30% of base, train with low learning rate
    4. **Custom Head**: GlobalAvgPool → BatchNorm → Dense(256) → Dropout → Softmax

    **Training Strategy:**
    - Phase 1: Feature extraction (LR=0.001, 10 epochs)
    - Phase 2: Fine-tuning (LR=0.0001, 40 epochs)
    - Early stopping with patience=10
    - Class weight balancing for imbalanced data
    """)

with col2:
    st.markdown("""
    **Models Available:**

    | Model | Parameters | Strengths |
    |-------|-----------|-----------|
    | **MobileNetV2** | ~3.5M | Lightweight, fast inference |
    | **ResNet50** | ~25M | High capacity, deep features |
    | **EfficientNetB0** | ~5.3M | Balanced efficiency |
    | **Custom CNN** | ~2M | Baseline comparison |

    **Data Augmentation:**
    - Random rotation (±20°)
    - Width/height shift (±20%)
    - Zoom (±15%), Shear (±15%)
    - Horizontal flip
    - Brightness adjustment (0.8–1.2×)
    """)

# ─── Dataset Information ─────────────────────────────────────────
st.markdown("### 📦 Dataset Information")

st.markdown("""
<div class="info-card">
<p><strong>Source:</strong> Kaggle Garbage Classification Dataset (Extended)</p>
<p><strong>Original Classes:</strong> 12 categories mapped to 7 target classes</p>
<p><strong>Split:</strong> 70% Train / 15% Validation / 15% Test (stratified)</p>
<p><strong>Preprocessing:</strong> Resized to 224×224, normalized to [0, 1]</p>
</div>
""", unsafe_allow_html=True)

# ─── Waste Disposal Guide ────────────────────────────────────────
st.markdown("### 🗂️ Complete Waste Disposal Guide")

for cls_name, info in sorted(class_info.items()):
    with st.expander(f"{info['icon']}  {cls_name.upper()}", expanded=False):
        st.markdown(f"**Disposal Instruction:** {info['instruction']}")
        st.markdown(f"**Bin Type:** {info['bin']}")

        # Add examples based on class
        examples = {
            "cardboard": "📋 Boxes, packaging, cereal boxes, shipping materials",
            "glass": "🍾 Bottles, jars, containers, broken glassware",
            "metal": "🥫 Cans, aluminum foil, metal lids, wire",
            "organic": "🍌 Food scraps, coffee grounds, yard clippings, leaves",
            "paper": "📰 Newspapers, magazines, printer paper, mail",
            "plastic": "🧴 Bottles, containers, bags, packaging, utensils",
            "trash": "🗑️ Mixed/contaminated waste, styrofoam, ceramics, diapers",
        }
        st.markdown(f"**Common Examples:** {examples.get(cls_name, '')}")

# ─── Environmental Impact ────────────────────────────────────────
st.markdown("### 🌱 Why Waste Classification Matters")

impact_cols = st.columns(3)

with impact_cols[0]:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
    <div style="font-size: 2.5rem;">🌊</div>
    <h4>Ocean Protection</h4>
    <p style="color: #8899aa; font-size: 0.9rem;">
    Proper waste sorting prevents 8M+ tons of plastic
    from entering oceans annually.
    </p>
    </div>
    """, unsafe_allow_html=True)

with impact_cols[1]:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
    <div style="font-size: 2.5rem;">♻️</div>
    <h4>Resource Recovery</h4>
    <p style="color: #8899aa; font-size: 0.9rem;">
    Recycling 1 ton of paper saves 17 trees,
    7,000 gallons of water, and 3 cubic yards of landfill.
    </p>
    </div>
    """, unsafe_allow_html=True)

with impact_cols[2]:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
    <div style="font-size: 2.5rem;">🌡️</div>
    <h4>Climate Action</h4>
    <p style="color: #8899aa; font-size: 0.9rem;">
    Proper composting of organic waste reduces methane
    emissions from landfills by up to 50%.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #556677; padding: 20px;'>"
    "♻️ Garbage Classification System v1.0.0 — "
    "Built with TensorFlow, Streamlit & ❤️ for the planet"
    "</div>",
    unsafe_allow_html=True,
)
