"""
Results Visualization.
Generates all plots for Chapter 6 (Preliminary Results):
- Learning trajectories (mAP@50 vs cycles)
- AUC comparison bar chart
- Precision/Recall comparison
- Latency summary
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


COLORS = {
    "random": "#95A5A6",
    "deterministic": "#E67E22",
    "bald_only": "#9B59B6",
    "bald_diversity": "#3498DB",
}
LABELS = {
    "random": "Random Sampling",
    "deterministic": "Deterministic Uncertainty",
    "bald_only": "BALD-only (MC Dropout)",
    "bald_diversity": "BALD + Diversity (Proposed)",
}


def load_results(results_dir):
    """Load all results JSON files from directory."""
    results = []
    rdir = Path(results_dir)

    # Try combined file first
    combined = rdir / "all_results.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)

    # Otherwise scan subdirectories
    for exp_dir in sorted(rdir.iterdir()):
        rfile = exp_dir / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                results.append(json.load(f))
    return results


def plot_learning_trajectories(results, output_dir, l0_filter=None):
    """Plot mAP@50 vs cumulative labelled samples for each strategy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for r in results:
        if l0_filter and r["l0_size"] != l0_filter:
            continue
        strategy = r["strategy"]
        traj = r["trajectory"]
        x = [t["n_labelled"] for t in traj]
        y = [t["mAP50"] for t in traj]
        ax.plot(x, y, '-o', color=COLORS.get(strategy, '#333'),
                label=LABELS.get(strategy, strategy), linewidth=2, markersize=6)

    ax.set_xlabel("Cumulative Labelled Samples", fontsize=12)
    ax.set_ylabel("mAP@50", fontsize=12)
    title = "Learning Trajectory"
    if l0_filter: title += f" (L₀ = {l0_filter})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fname = f"trajectory_L0{l0_filter}.pdf" if l0_filter else "trajectory.pdf"
    plt.savefig(Path(output_dir) / fname, bbox_inches='tight')
    plt.savefig(Path(output_dir) / fname.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_auc_comparison(results, output_dir):
    """Bar chart of AUC across strategies and L0 sizes."""
    data = []
    for r in results:
        data.append({
            "Strategy": LABELS.get(r["strategy"], r["strategy"]),
            "L0": r["l0_size"],
            "AUC": r.get("AUC", 0),
            "strategy_key": r["strategy"],
        })
    df = pd.DataFrame(data)

    if df.empty:
        print("  No AUC data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    l0_values = sorted(df["L0"].unique())
    n_strategies = df["strategy_key"].nunique()
    bar_width = 0.18
    x = np.arange(len(l0_values))

    for i, (strat_key, strat_label) in enumerate(LABELS.items()):
        subset = df[df["strategy_key"] == strat_key]
        if subset.empty: continue
        auc_vals = [subset[subset["L0"] == l0]["AUC"].values[0]
                    if len(subset[subset["L0"] == l0]) > 0 else 0
                    for l0 in l0_values]
        ax.bar(x + i * bar_width, auc_vals, bar_width,
               label=strat_label, color=COLORS.get(strat_key, '#333'), alpha=0.85)

    ax.set_xlabel("Initial Labelled Set Size (L₀)", fontsize=12)
    ax.set_ylabel("AUC (Area Under Learning Curve)", fontsize=12)
    ax.set_title("Annotation Efficiency: AUC Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
    ax.set_xticklabels([f"L₀={v}" for v in l0_values])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "auc_comparison.pdf", bbox_inches='tight')
    plt.savefig(Path(output_dir) / "auc_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved auc_comparison.pdf")


def plot_final_metrics(results, output_dir):
    """Grouped bar chart of final precision, recall, mAP@50."""
    data = []
    for r in results:
        final = r["trajectory"][-1]
        data.append({
            "Strategy": LABELS.get(r["strategy"], r["strategy"]),
            "strategy_key": r["strategy"],
            "L0": r["l0_size"],
            "mAP@50": final["mAP50"],
            "Precision": final["precision"],
            "Recall": final["recall"],
        })
    df = pd.DataFrame(data)

    for l0 in sorted(df["L0"].unique()):
        subset = df[df["L0"] == l0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, metric in zip(axes, ["mAP@50", "Precision", "Recall"]):
            values = []
            labels = []
            colors = []
            for strat_key in LABELS:
                row = subset[subset["strategy_key"] == strat_key]
                if not row.empty:
                    values.append(row[metric].values[0])
                    labels.append(LABELS[strat_key])
                    colors.append(COLORS[strat_key])

            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.85)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(metric, fontsize=11)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=9)

        fig.suptitle(f"Final Metrics (L₀ = {l0})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"final_metrics_L0{l0}.pdf", bbox_inches='tight')
        plt.savefig(Path(output_dir) / f"final_metrics_L0{l0}.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved final_metrics_L0{l0}.pdf")


def plot_latency(results, output_dir):
    """Bar chart of inference latency."""
    data = []
    for r in results:
        if "inference_latency_ms" in r:
            data.append({
                "Strategy": LABELS.get(r["strategy"], r["strategy"]),
                "Latency (ms)": r["inference_latency_ms"],
                "strategy_key": r["strategy"],
            })
    if not data:
        print("  No latency data")
        return

    df = pd.DataFrame(data).groupby("Strategy").mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(k, '#333') for k in
              [d["strategy_key"] for d in data[:len(df)]]]
    bars = ax.bar(df["Strategy"], df["Latency (ms)"], color=colors[:len(df)], alpha=0.85)
    ax.set_ylabel("Inference Latency (ms/image)", fontsize=12)
    ax.set_title("Operational Feasibility: Inference Latency", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{bar.get_height():.1f}', ha='center', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "latency.pdf", bbox_inches='tight')
    plt.savefig(Path(output_dir) / "latency.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved latency.pdf")


def generate_summary_table(results, output_dir):
    """Generate LaTeX summary table."""
    rows = []
    for r in results:
        final = r["trajectory"][-1]
        rows.append({
            "Strategy": LABELS.get(r["strategy"], r["strategy"]),
            "$L_0$": r["l0_size"],
            "mAP@50": f"{final['mAP50']:.4f}",
            "Precision": f"{final['precision']:.4f}",
            "Recall": f"{final['recall']:.4f}",
            "AUC": f"{r.get('AUC', 0):.2f}",
            "Latency (ms)": f"{r.get('inference_latency_ms', 0):.1f}",
        })

    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, escape=False)

    with open(Path(output_dir) / "summary_table.tex", "w") as f:
        f.write(latex)
    df.to_csv(Path(output_dir) / "summary_table.csv", index=False)
    print("  Saved summary_table.tex/csv")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} experiments\n")

    if not results:
        print("No results found!")
        return

    print("Generating plots...")
    l0_values = set(r["l0_size"] for r in results)
    for l0 in sorted(l0_values):
        plot_learning_trajectories(results, output_dir, l0_filter=l0)

    plot_auc_comparison(results, output_dir)
    plot_final_metrics(results, output_dir)
    plot_latency(results, output_dir)
    generate_summary_table(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
