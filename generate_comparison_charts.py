"""
Generate Architectural Comparison Charts for Khet Guard Report
----------------------------------------------------------------
1. model_comparison_metrics.png: Grouped bar chart comparing Primary Metric & Macro-F1.
2. accuracy_vs_latency_tradeoff.png: Scatter plot of Accuracy/mAP vs. GPU Latency with parameter bubbles.

Output directory: model/exports/model_comparison_charts/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

out_dir = Path("model/exports/model_comparison_charts")
out_dir.mkdir(parents=True, exist_ok=True)

# Data from live benchmarks & evaluation metrics
architectures = ["Arch 1\n(EfficientNet-B4)", "Arch 2\n(Swin-B Transformer)", "Arch 3\n(RT-DETR Detector)"]

# Primary Metrics (Accuracy % for Arch 1 & 2; mAP50 % for Arch 3)
primary_metrics = [97.19, 98.83, 47.57]  # %
primary_labels = ["97.2% Acc", "98.8% Acc", "47.6% mAP50"]

# Macro-F1
macro_f1 = [0.9712, 0.9881, 0.0]  # 0.0 for Arch 3 (N/A)
f1_labels = ["0.971 F1", "0.988 F1", "N/A (Detection)"]

# Latency (GPU ms) & Parameters (M)
gpu_latencies = [54.80, 78.90, 104.07]  # ms
param_counts = [18.06, 91.28, 32.97]    # Millions

# Colors matching project palette
purple_dark = "#4a148c"
purple_light = "#7b1fa2"
teal_accent = "#00897b"
amber_accent = "#f57c00"

# ==============================================================================
# Chart 1: Grouped Bar Chart of Primary Metrics & Macro-F1
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

x = np.arange(len(architectures))
width = 0.35

rects1 = ax1.bar(x - width/2, primary_metrics, width, label="Primary Metric (Acc / mAP50 %)", color="#6a1b9a", alpha=0.9)

# Secondary axis for F1
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, [f1 * 100 for f1 in macro_f1], width, label="Macro-F1 Score", color="#00897b", alpha=0.85)

ax1.set_ylabel("Primary Metric Score (%)", color="#6a1b9a", fontweight="bold")
ax2.set_ylabel("Macro-F1 Score (%)", color="#00897b", fontweight="bold")
ax1.set_title("Khet Guard Architectural Metric Comparison", fontsize=14, fontweight="bold", pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(architectures, fontweight="bold")
ax1.set_ylim(0, 115)
ax2.set_ylim(0, 115)

# Value annotations
for rect, label in zip(rects1, primary_labels):
    h = rect.get_height()
    ax1.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4a148c")

for idx, (rect, label) in enumerate(zip(rects2, f1_labels)):
    if idx == 2:  # Arch 3 N/A flag
        ax2.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, 5), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#d32f2f")
    else:
        h = rect.get_height()
        ax2.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#004d40")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="none")

plt.figtext(0.5, 0.01, "*Note: Arch 1 & 2 evaluated on PlantVillage classification (Acc/F1). Arch 3 evaluated on PlantDoc detection (mAP50).",
            ha="center", fontsize=8, fontstyle="italic", color="#555555")

fig.tight_layout()
chart1_path = out_dir / "model_comparison_metrics.png"
fig.savefig(chart1_path, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved Chart 1: {chart1_path}")


# ==============================================================================
# Chart 2: Accuracy / Primary Metric vs. GPU Latency Scatter Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

colors = [purple_dark, "#00695c", amber_accent]
markers = ["o", "s", "^"]
bubble_sizes = [p * 15 for p in param_counts]  # Size proportional to params

for i, (lat, met, name, params, c, m, size) in enumerate(zip(gpu_latencies, primary_metrics, architectures, param_counts, colors, markers, bubble_sizes)):
    ax.scatter(lat, met, s=size, color=c, alpha=0.7, edgecolors="black", linewidth=1.5, label=f"{name.splitlines()[0]} ({params:.1f}M params)", zorder=3)
    
    # Label placement offsets
    offset_y = 3 if i != 2 else -7
    offset_x = 2 if i == 0 else (-12 if i == 1 else 2)
    
    ax.annotate(
        f"{name.replace(chr(10), ' ')}\n({met:.1f}%, {lat:.1f} ms)",
        xy=(lat, met),
        xytext=(lat + offset_x, met + offset_y),
        fontsize=9,
        fontweight="bold",
        color=c,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, lw=1, alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=c, lw=1.2)
    )

ax.set_xlabel("GPU Inference Latency (ms / image)", fontsize=11, fontweight="bold")
ax.set_ylabel("Primary Metric (Accuracy / mAP50 %)", fontsize=11, fontweight="bold")
ax.set_title("Khet Guard Accuracy vs. Latency Pareto Trade-Off", fontsize=14, fontweight="bold", pad=15)

ax.set_xlim(40, 120)
ax.set_ylim(40, 105)
ax.grid(True, linestyle="--", alpha=0.5)

# Annotation box explaining Pareto frontier
ax.text(0.03, 0.05, "Bubble size = Parameter Count\nUpper-Left = Ideal (Fast & Accurate)",
        transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))

ax.legend(loc="upper right", frameon=True, facecolor="white")

fig.tight_layout()
chart2_path = out_dir / "accuracy_vs_latency_tradeoff.png"
fig.savefig(chart2_path, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved Chart 2: {chart2_path}")
