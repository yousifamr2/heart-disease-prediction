"""
chart_service.py
─────────────────────────────────────────────────────────────────────────
Visualization Layer — generates all charts as base64 PNG strings.

Rules:
  - All functions accept hashable args (tuples) for lru_cache compatibility
  - All functions return base64-encoded PNG as a data-URI string
  - No temp files, no disk I/O — everything lives in memory
  - Charts are cached per unique shap_data; same data = instant return
"""

import io
import base64
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Consistent dark style ────────────────────────────────────────────────────
_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#e2e8f0",
    "axes.labelcolor":  "#334155",
    "axes.titlecolor":  "#0f172a",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#334155",
    "grid.color":       "#e2e8f0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64 data-URI PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@lru_cache(maxsize=128)
def generate_feature_importance_chart(shap_tuple: tuple) -> str:
    """
    Top-5 feature importance bar chart (horizontal).

    Parameters
    ----------
    shap_tuple : tuple of (feature_name, shap_value) pairs
        Must be hashable for lru_cache. Convert dict with:
        tuple(sorted(shap_data.items()))

    Returns
    -------
    str : base64 data-URI PNG string ready for <img src="...">
    """
    # Sort by absolute impact, take top 5
    items = sorted(shap_tuple, key=lambda x: abs(x[1]), reverse=True)[:5]
    features = [item[0] for item in items]
    values   = [item[1] for item in items]
    colors   = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.barh(features, [abs(v) for v in values], color=colors,
                       edgecolor="white", linewidth=0.5, height=0.55)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{abs(val):.3f}", va="center", fontsize=9, color="#475569")

        ax.set_xlabel("SHAP Impact Score", fontsize=10)
        ax.set_title("Top Features Influencing This Prediction", fontsize=12,
                     fontweight="bold", pad=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)

        # Legend
        pos_patch = mpatches.Patch(color="#ef4444", label="↑ Increases Risk")
        neg_patch = mpatches.Patch(color="#3b82f6", label="↓ Decreases Risk")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9,
                  framealpha=0.8)

        fig.tight_layout()
        return _fig_to_b64(fig)


@lru_cache(maxsize=128)
def generate_shap_waterfall_chart(shap_tuple: tuple) -> str:
    """
    SHAP waterfall-style chart — shows cumulative feature contributions.

    Parameters
    ----------
    shap_tuple : tuple of (feature_name, shap_value) pairs

    Returns
    -------
    str : base64 data-URI PNG string
    """
    # Sort by value (descending) for waterfall visual
    items    = sorted(shap_tuple, key=lambda x: x[1], reverse=True)
    features = [item[0] for item in items]
    values   = [item[1] for item in items]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))

        # Cumulative baseline
        running = 0.0
        bar_starts = []
        for v in values:
            bar_starts.append(running)
            running += v

        bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

        for i, (start, val, color) in enumerate(zip(bar_starts, values, bar_colors)):
            ax.barh(i, val, left=start, color=color, edgecolor="white",
                    linewidth=0.5, height=0.55, alpha=0.88)
            ax.text(start + val / 2, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="-")
        ax.set_xlabel("Cumulative SHAP Contribution", fontsize=10)
        ax.set_title("SHAP Waterfall — Feature Contributions", fontsize=12,
                     fontweight="bold", pad=10)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.invert_yaxis()

        pos_patch = mpatches.Patch(color="#ef4444", label="↑ Toward Disease")
        neg_patch = mpatches.Patch(color="#3b82f6", label="↓ Away From Disease")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9,
                  framealpha=0.8)

        fig.tight_layout()
        return _fig_to_b64(fig)


def clear_chart_cache() -> None:
    """Manually clear the lru_cache for both chart functions."""
    generate_feature_importance_chart.cache_clear()
    generate_shap_waterfall_chart.cache_clear()
