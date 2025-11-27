#!/usr/bin/env python3
"""
Plot deep learning framework popularity from Google Trends data.
"""

import matplotlib.pyplot as plt
import pandas as pd


def load_and_clean_data(filepath="popularity.csv"):
    """
    Load Google Trends data and clean it.

    Handles '<1' values by converting them to 0.5
    """
    # Skip the first row (Category: All categories)
    df = pd.read_csv(filepath, skiprows=1)

    # Rename columns to remove the ': (Worldwide)' suffix
    df.columns = [
        col.split(":")[0].strip() if ":" in col else col for col in df.columns
    ]

    # Convert Week to datetime
    df["Week"] = pd.to_datetime(df["Week"])

    # Replace '<1' with 0.5 and convert to float
    for col in df.columns:
        if col != "Week":
            df[col] = df[col].replace("<1", 0.5).astype(float)

    return df


def plot_trends(df, output_file="assets/frameworks_trends.png"):
    """
    Create a line plot of framework popularity over time.
    """
    # Create figure with matching background color
    # Using wider aspect ratio since it will have a header on the slide
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Set dark theme to match presentation
    plt.style.use("dark_background")

    # Define colors for each framework (matching generate_histogram.py)
    colors = {
        "PyTorch": "#ff6b35",  # Orange-red (primary color)
        "TensorFlow": "#f7931e",  # Orange (secondary color)
        "JAX": "#ffd93d",  # Yellow
        "MLX": "#6bcf7f",  # Green
        "Tinygrad": "#4ecdc4",  # Cyan
    }

    # Plot each framework
    for col in df.columns:
        if col != "Week":
            ax.plot(
                df["Week"],
                df[col],
                label=col,
                linewidth=2.5,
                color=colors.get(col, None),
            )

    ax.set_xlabel("Date", fontsize=14, fontweight="bold", color="#d0d0d0")
    ax.set_ylabel(
        "Search Interest (Relative)", fontsize=14, fontweight="bold", color="#d0d0d0"
    )
    ax.legend(
        fontsize=12,
        loc="upper left",
        framealpha=0.9,
        facecolor="#2d2d2d",
        edgecolor="#404040",
        labelcolor="#f0f0f0",
    )
    ax.grid(True, alpha=0.2, linestyle="--", color="#404040")
    ax.tick_params(axis="both", labelcolor="#c0c0c0", colors="#606060")

    # Customize spines to match histogram style
    for spine in ax.spines.values():
        spine.set_edgecolor("#404040")
        spine.set_linewidth(1)

    plt.tight_layout()

    # Save with dark background
    fig.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="#1a1a1a", edgecolor="none"
    )
    print(f"Plot saved to {output_file}")

    plt.show()


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Get stats for non-Week columns
    stats_df = df.drop(columns=["Week"]).describe()
    print(stats_df.round(2))

    print("\n" + "=" * 70)
    print("LATEST VALUES (Most Recent Week)")
    print("=" * 70)
    latest = df.iloc[-1]
    print(f"Week: {latest['Week'].strftime('%Y-%m-%d')}")
    for col in df.columns:
        if col != "Week":
            print(f"  {col:12s}: {latest[col]:5.1f}")

    print("\n" + "=" * 70)
    print("PEAK VALUES")
    print("=" * 70)
    for col in df.columns:
        if col != "Week":
            max_val = df[col].max()
            max_date = df.loc[df[col].idxmax(), "Week"].strftime("%Y-%m-%d")
            print(f"  {col:12s}: {max_val:5.1f} (on {max_date})")


def main():
    print("Loading Google Trends data from popularity.csv...")

    df = load_and_clean_data("assets/popularity.csv")

    print(f"✓ Loaded {len(df)} weeks of data")
    print(
        f"  Date range: {df['Week'].min().strftime('%Y-%m-%d')} to {df['Week'].max().strftime('%Y-%m-%d')}"
    )

    print_summary(df)

    print("\nGenerating plot...")
    plot_trends(df)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
