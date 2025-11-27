#!/usr/bin/env python3
"""
Plot performance comparison between Morok and PyTorch
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_performance_data(filepath="assets/performance.csv"):
    """Load performance data from CSV."""
    df = pd.read_csv(filepath)
    return df


def plot_performance(df, output_file="assets/performance_comparison.png"):
    """
    Create a grouped bar chart comparing Morok and PyTorch performance.
    """
    # Create figure with matching background color
    # Using wider aspect ratio since it will have a header on the slide
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Set dark theme to match presentation
    plt.style.use("dark_background")

    # Prepare data
    tasks = df["Task"].values
    morok_times = df["Morok"].values
    torch_times = df["Torch"].values

    # Set up bar positions
    x = np.arange(len(tasks))
    width = 0.35

    # Define colors matching the presentation theme
    morok_color = "#ff6b35"  # Orange-red (primary color)
    torch_color = "#f7931e"  # Orange (secondary color)

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        morok_times,
        width,
        label="Morok",
        color=morok_color,
        edgecolor="#000",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        torch_times,
        width,
        label="PyTorch",
        color=torch_color,
        edgecolor="#000",
        linewidth=0.5,
    )

    # Customize the plot
    ax.set_xlabel("Task", fontsize=15, color="#d0d0d0", fontweight="bold")
    ax.set_ylabel("Benchmark Score", fontsize=15, color="#d0d0d0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=14, color="#c0c0c0", fontweight="medium")
    ax.tick_params(axis="y", labelcolor="#c0c0c0", labelsize=12)
    ax.tick_params(axis="x", colors="#606060")
    ax.tick_params(axis="y", colors="#606060")

    # Add legend
    ax.legend(
        fontsize=12,
        loc="upper left",
        facecolor="#2d2d2d",
        edgecolor="#404040",
        labelcolor="#f0f0f0",
    )

    # Add grid for better readability
    ax.grid(axis="y", alpha=0.2, color="#404040", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Customize spines to match histogram style
    for spine in ax.spines.values():
        spine.set_edgecolor("#404040")
        spine.set_linewidth(1)

    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#f0f0f0",
                fontweight="bold",
            )

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    # Save with dark background
    fig.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="#1a1a1a", edgecolor="none"
    )
    print(f"✓ Plot saved to {output_file}")

    plt.show()


def print_summary(df):
    """Print performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS")
    print("=" * 70)

    for _, row in df.iterrows():
        task = row["Task"]
        morok = row["Morok"]
        torch = row["Torch"]

        if morok < torch:
            speedup = torch / morok
            print(f"  {task:20s}: Morok is {speedup:.2f}x FASTER")
        elif morok > torch:
            slowdown = morok / torch
            print(f"  {task:20s}: Morok is {slowdown:.2f}x SLOWER")
        else:
            print(f"  {task:20s}: Same performance")

    print("\n" + "=" * 70)
    avg_morok = df["Morok"].mean()
    avg_torch = df["Torch"].mean()
    overall_speedup = avg_torch / avg_morok
    print(f"OVERALL: Morok is {overall_speedup:.2f}x faster on average")
    print("=" * 70)


def main():
    print("Loading performance data from performance.csv...")

    df = load_performance_data("assets/performance.csv")

    print(f"✓ Loaded {len(df)} benchmark tasks")

    print_summary(df)

    print("\nGenerating plot...")
    plot_performance(df)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
