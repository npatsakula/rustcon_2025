#!/usr/bin/env python3
"""
Generate histogram for ML Frameworks LOCs comparison
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use SVG backend for high-quality vector output
matplotlib.use("SVG")

# Data from presentation_plan.md
frameworks = [
    "TensorFlow\n(2015)",
    "PyTorch\n(2016)",
    "JAX\n(2021)",
    "GGML\n(2022)",
    "MLX\n(2023)",
    "Tinygrad\n(2023)",
]

python_locs = [458145, 873835, 222913, 0, 7625, 24978]
cxx_locs = [1611200, 640493, 53121, 142918, 58317, 0]

# Set up the bar positions
x = np.arange(len(frameworks))
width = 0.35

# Create figure with dark background to match presentation theme
# Using 16:9 aspect ratio for better fit on slides
fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")

# Create bars
bars1 = ax.bar(
    x - width / 2,
    python_locs,
    width,
    label="Python LOCs",
    color="#ff6b35",
    edgecolor="#000",
    linewidth=0.5,
)
bars2 = ax.bar(
    x + width / 2,
    cxx_locs,
    width,
    label="C++ LOCs",
    color="#f7931e",
    edgecolor="#000",
    linewidth=0.5,
)

# Customize the plot
ax.set_xlabel(
    "Framework (Release Year)", fontsize=15, color="#d0d0d0", fontweight="bold"
)
ax.set_ylabel("Lines of Code", fontsize=15, color="#d0d0d0", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(frameworks, fontsize=13, color="#c0c0c0", fontweight="medium")
ax.tick_params(axis="y", labelcolor="#c0c0c0", labelsize=12)
ax.tick_params(axis="x", colors="#606060")
ax.tick_params(axis="y", colors="#606060")

# Format y-axis to show numbers with commas
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

# Add legend
ax.legend(
    fontsize=12,
    loc="upper right",
    facecolor="#2d2d2d",
    edgecolor="#404040",
    labelcolor="#f0f0f0",
)

# Add grid for better readability
ax.grid(axis="y", alpha=0.2, color="#404040", linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)

# Customize spines
for spine in ax.spines.values():
    spine.set_edgecolor("#404040")
    spine.set_linewidth(1)


# Add value labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only show label if there's a value
            ax.annotate(
                f"{int(height):,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#f0f0f0",
                rotation=0,
            )


autolabel(bars1)
autolabel(bars2)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save as SVG
plt.savefig(
    "assets/frameworks_histogram.svg",
    format="svg",
    facecolor="#1a1a1a",
    edgecolor="none",
    bbox_inches="tight",
)
print("✓ Generated: frameworks_histogram.svg")
