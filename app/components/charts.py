"""
Charts — Reusable Plotly chart components for the dashboard.
"""

from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go


# Common dark theme layout
DARK_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#CCCCCC",
    title_font_color="#FFFFFF",
    title_font_size=16,
)


def create_confidence_chart(
    probabilities: Dict[str, float],
    class_colors: Optional[Dict[str, str]] = None,
    title: str = "Classification Confidence",
) -> go.Figure:
    """
    Create a horizontal bar chart showing confidence scores per class.

    Args:
        probabilities: Dict mapping class name to probability.
        class_colors: Optional dict mapping class to color.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]

    colors = []
    if class_colors:
        colors = [class_colors.get(cls, "#2ECC71") for cls in classes]
    else:
        colors = ["#2ECC71"] * len(classes)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis=dict(title="Confidence (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="", autorange="reversed"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_distribution_pie(
    class_counts: Dict[str, int],
    class_colors: Optional[Dict[str, str]] = None,
    title: str = "Classification Distribution",
) -> go.Figure:
    """
    Create a donut chart showing classification distribution.

    Args:
        class_counts: Dict mapping class to count.
        class_colors: Optional dict mapping class to color.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = None
    if class_colors:
        colors = [class_colors.get(cls, "#808080") for cls in classes]

    fig = px.pie(
        values=counts,
        names=classes,
        title=title,
        color_discrete_sequence=colors,
        hole=0.4,
    )

    fig.update_layout(**DARK_LAYOUT, legend_font_size=11)
    fig.update_traces(textinfo="percent+label", textfont_size=11)

    return fig


def create_trend_line(
    dates: List,
    counts: List[int],
    title: str = "Predictions Over Time",
) -> go.Figure:
    """
    Create a line chart showing prediction trends over time.

    Args:
        dates: List of dates.
        counts: List of prediction counts.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=counts,
            mode="lines+markers",
            line=dict(color="#1a73e8", width=2),
            marker=dict(size=8, color="#1a73e8"),
            fill="tozeroy",
            fillcolor="rgba(26, 115, 232, 0.1)",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
    )

    return fig
