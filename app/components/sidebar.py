"""
Sidebar — Reusable sidebar component for the Streamlit dashboard.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils.helpers import get_class_info


def render_sidebar(
    available_models: list,
    prediction_count: int,
    class_distribution: dict,
) -> dict:
    """
    Render the sidebar with model selection and quick stats.

    Args:
        available_models: List of model file paths.
        prediction_count: Total predictions made.
        class_distribution: Dict of class → count.

    Returns:
        Dict with sidebar selections (e.g., selected_model).
    """
    class_info = get_class_info()
    result = {"model_path": None, "model_name": None}

    with st.sidebar:
        st.markdown("## ♻️ Settings")
        st.markdown("---")

        # Model selection
        if available_models:
            model_names = [Path(m).name for m in available_models]
            selected = st.selectbox(
                "🧠 Select Model",
                model_names,
                index=0,
                help="Choose a trained model for classification",
            )
            idx = model_names.index(selected)
            result["model_path"] = str(available_models[idx])
            result["model_name"] = selected
        else:
            st.warning("⚠️ No trained models found.")
            st.code("python scripts/train.py --model mobilenetv2", language="bash")

        st.markdown("---")

        # Quick stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predictions", prediction_count)
        with col2:
            st.metric("Categories", len(class_distribution) if class_distribution else 7)

        if class_distribution:
            st.markdown("#### Top Categories")
            sorted_dist = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
            for cls, count in sorted_dist[:5]:
                info = class_info.get(cls, {})
                st.markdown(f"{info.get('icon', '❓')} **{cls}**: {count}")

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #556677; font-size: 0.8rem;'>"
            "Built with ❤️ using TensorFlow & Streamlit</div>",
            unsafe_allow_html=True,
        )

    return result
