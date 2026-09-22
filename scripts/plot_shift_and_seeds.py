"""
Plot Generation for Multi-Seed and Distribution Shift Results.

Generates thesis-quality figures for:
  - H1/H3: Learning curves with mean ± std confidence bands (multi-seed)
  - H2:    Recall curves under covariate shift, stability comparison
  - H4:    Latency box-plots across seeds

Usage
-----
# After running run_multi_seed.py:
python scripts/plot_shift_and_seeds.py \\
    --multi_seed_results results/multi_seed/all_aggregated.json \\
    --shift_results results/shift/all_shift_aggregated.json \\
    --output figures/

Outputs (all PDF + PNG for thesis):
  h1_learning_curves_with_bands.pdf   — mAP@50 mean±std across seeds (H1)
  h2_shift_recall_curves.pdf          — recall under shift, all strategies (H2)
  h2_recall_variance_barplot.pdf      — inter-cycle recall variance (H2 stability)
  h3_auc_comparison_with_ci.pdf       — AUC barplot with error bars (H3)
  h4_latency_boxplot.pdf              — per-strategy latency distribution (H4)
  results_table_multiseed.tex         — LaTeX table with mean±std for thesis
  h2_shift_table.tex                  — LaTeX H2 shift results table
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "random":       "Random",
    "deterministic": "Det. Uncertainty",
    "bald_only":    "BALD-only",
    "bald_diversity": "BALD+Diversity (Proposed)",
}

STRATEGY_COLORS = {
    "random":        "#999999",
    "deterministic": "#4878CF",
    "bald_only":     "#D65F5F",
    "bald_diversity": "#6ACC65",
}

STRATEGY_MARKERS = {
    "random":        "o",
    "deterministic": "s",
    "bald_only":     "^",
    "bald_diversity": "D",
}

L0_LINESTYLES = {100: "-", 200: "--", 300: "-.", 400: ":"}

THESIS_FIGSIZE = (7, 4.5)


def setup_style():
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
    })


# ---------------------------------------------------------------------------
# H1/H3: multi-seed learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves_with_bands(all_agg: list, output_dir: Path, l0_filter: int = 100):
    """
    Plot mAP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L0 size (or use l0_filter for a single plot).
    """
    if not HAS_MPL:
        print("[plot] matplotlib not available — skipping figure generation")
        return

    filtered = [a for a in all_agg if a.get("l0_size") == l0_filter]
    if not filtered:
        print(f"[plot] No results found for L0={l0_filter}")
        return

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)

    for a in filtered:
        strategy = a.get("strategy", "")
        mean_traj = np.array(a.get("trajectory_mAP50_mean", []))
        std_traj  = np.array(a.get("trajectory_mAP50_std",  []))
        x_mean    = np.array(a.get("trajectory_n_labels_mean", list(range(len(mean_traj)))))

        if len(mean_traj) == 0:
            continue

        color  = STRATEGY_COLORS.get(strategy, "#333333")
        marker = STRATEGY_MARKERS.get(strategy, "o")
        label  = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(x_mean, mean_traj, color=color, marker=marker,
                markevery=2, label=label)
        ax.fill_between(x_mean,
                         mean_traj - std_traj,
                         mean_traj + std_traj,
                         color=color, alpha=0.15)

    n_seeds = filtered[0].get("n_runs", "?") if filtered else "?"
    ax.set_xlabel("Cumulative labelled samples")
    ax.set_ylabel("mAP@50")
    ax.set_title(f"H1: Learning curves (L0={l0_filter}, {n_seeds} seeds, mean ± 1 std)")
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)

    out_pdf = output_dir / f"h1_learning_curves_L0{l0_filter}_bands.pdf"
    out_png = output_dir / f"h1_learning_curves_L0{l0_filter}_bands.png"
    fig.tight_layout()
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[plot] Saved: {out_pdf}")


def plot_auc_comparison(all_agg: list, output_dir: Path, l0_filter: int = 100):
    """Bar plot: AUC mean ± std per strategy (H1/H3 comparison)."""
    if not HAS_MPL:
        return

    filtered = [a for a in all_agg if a.get("l0_size") == l0_filter]
    strategies = ["random", "deterministic", "bald_only", "bald_diversity"]

    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    means  = []
    stds   = []
    colors = []

    for s in strategies:
        match = [a for a in filtered if a.get("strategy") == s]
        if match:
            means.append(match[0].get("AUC_mean", 0))
            stds.append(match[0].get("AUC_std", 0))
        else:
            means.append(0)
            stds.append(0)
        colors.append(STRATEGY_COLORS.get(s, "#333333"))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(strategies))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("AUC (mAP@50 × labels)")
    n_seeds = filtered[0].get("n_runs", "?") if filtered else "?"
    ax.set_title(f"H1/H3: AUC comparison (L0={l0_filter}, {n_seeds} seeds, mean ± std)")

    out_pdf = output_dir / f"h3_auc_comparison_L0{l0_filter}.pdf"
    out_png = output_dir / f"h3_auc_comparison_L0{l0_filter}.png"
    fig.tight_layout()
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[plot] Saved: {out_pdf}")


# ---------------------------------------------------------------------------
# H2: shift recall curves and stability
# ---------------------------------------------------------------------------

def plot_shift_recall_curves(shift_agg: list, output_dir: Path, l0_filter: int = 100):
    """
    Plot recall learning curves under covariate shift (H2).
    Compares Bayesian vs Deterministic stability.
    """
    if not HAS_MPL:
        return

    filtered = [a for a in shift_agg if a.get("l0_size") == l0_filter]
    if not filtered:
        return

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)

    for a in filtered:
        strategy = a.get("strategy", "")
        mean_traj = np.array(a.get("trajectory_recall_mean", []))
        std_traj  = np.array(a.get("trajectory_recall_std",  []))
        x         = np.arange(len(mean_traj))

        if len(mean_traj) == 0:
            continue

        color  = STRATEGY_COLORS.get(strategy, "#333333")
        marker = STRATEGY_MARKERS.get(strategy, "o")
        label  = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(x, mean_traj, color=color, marker=marker,
                markevery=1, label=label)
        ax.fill_between(x,
                         mean_traj - std_traj,
                         mean_traj + std_traj,
                         color=color, alpha=0.15)

    n_seeds = filtered[0].get("n_seeds", "?") if filtered else "?"
    shift_info = filtered[0].get("shift_setup", {}) if filtered else {}
    intensity = shift_info.get("shift_intensity", "")
    ax.set_xlabel("Adaptation cycle")
    ax.set_ylabel("Recall (on target domain)")
    ax.set_title(
        f"H2: Recall under covariate shift — L0={l0_filter}, {n_seeds} seeds "
        f"({'simulated: ' + intensity if intensity else 'real cross-dataset'})"
    )
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    out_pdf = output_dir / f"h2_shift_recall_L0{l0_filter}.pdf"
    out_png = output_dir / f"h2_shift_recall_L0{l0_filter}.png"
    fig.tight_layout()
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[plot] Saved: {out_pdf}")


def plot_recall_variance_barplot(shift_agg: list, output_dir: Path, l0_filter: int = 100):
    """
    Bar plot of inter-cycle recall variance per strategy (H2 stability metric).
    Lower variance = more stable adaptation under shift.
    """
    if not HAS_MPL:
        return

    filtered = [a for a in shift_agg if a.get("l0_size") == l0_filter]
    if not filtered:
        return

    strategies = ["random", "deterministic", "bald_only", "bald_diversity"]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    means  = []
    stds   = []
    colors = []

    for s in strategies:
        match = [a for a in filtered if a.get("strategy") == s]
        if match:
            means.append(match[0].get("recall_variance_mean", 0))
            stds.append(match[0].get("recall_variance_std", 0))
        else:
            means.append(0)
            stds.append(0)
        colors.append(STRATEGY_COLORS.get(s, "#333333"))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(strategies))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
           error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Recall variance (lower = more stable)")
    n_seeds = filtered[0].get("n_seeds", "?") if filtered else "?"
    ax.set_title(f"H2: Recall stability under shift (L0={l0_filter}, {n_seeds} seeds)")

    out_pdf = output_dir / f"h2_recall_variance_L0{l0_filter}.pdf"
    out_png = output_dir / f"h2_recall_variance_L0{l0_filter}.png"
    fig.tight_layout()
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[plot] Saved: {out_pdf}")


# ---------------------------------------------------------------------------
# H4: latency box-plots
# ---------------------------------------------------------------------------

def plot_latency_boxplot(all_agg: list, output_dir: Path):
    """Box plot of inference latency across seeds (H4 operational feasibility)."""
    if not HAS_MPL:
        return

    strategies = ["random", "deterministic", "bald_only", "bald_diversity"]
    data_by_strategy = {s: [] for s in strategies}

    for a in all_agg:
        s = a.get("strategy", "")
        if s not in data_by_strategy:
            continue
        for run in a.get("runs", []):
            lat = run.get("inference_latency_ms")
            if lat is not None:
                data_by_strategy[s].append(lat)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_data  = [data_by_strategy[s] for s in strategies]
    box_colors = [STRATEGY_COLORS[s] for s in strategies]
    labels     = [STRATEGY_LABELS[s] for s in strategies]

    bp = ax.boxplot(plot_data, patch_artist=True, labels=labels,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Inference latency (ms/image)")
    ax.set_title("H4: Inference latency distribution across seeds")
    ax.set_xticklabels(labels, rotation=15, ha="right")

    out_pdf = output_dir / "h4_latency_boxplot.pdf"
    out_png = output_dir / "h4_latency_boxplot.png"
    fig.tight_layout()
    fig.savefig(str(out_pdf))
    fig.savefig(str(out_png))
    plt.close(fig)
    print(f"[plot] Saved: {out_pdf}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def build_latex_table_multiseed(all_agg: list, l0_sizes: list = None) -> str:
    """
    Build a LaTeX table of multi-seed results (thesis Table format).
    Format: mean ± std for AUC, recall, mAP50, latency.
    """
    if l0_sizes is None:
        l0_sizes = sorted({a["l0_size"] for a in all_agg})

    strategies = ["random", "deterministic", "bald_only", "bald_diversity"]
    s_labels   = {
        "random":        "Random",
        "deterministic": "Det. Unc.",
        "bald_only":     "BALD-only",
        "bald_diversity":"BALD+Div (proposed)",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Multi-seed results (mean $\pm$ std, $N=5$ seeds). "
        r"AUC: area under mAP@50 curve. Recall and mAP@50 at final cycle. "
        r"Latency: ms per image (deterministic inference).}",
        r"\label{tab:multiseed_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lc rr rr rr r}",
        r"\toprule",
        r"Strategy & $L_0$ & AUC $\uparrow$ & $\pm$ & Recall $\uparrow$ & $\pm$ "
        r"& mAP@50 $\uparrow$ & $\pm$ & Latency (ms) \\",
        r"\midrule",
    ]

    for l0 in l0_sizes:
        first = True
        for s in strategies:
            match = [a for a in all_agg if a.get("l0_size") == l0 and a.get("strategy") == s]
            if not match:
                continue
            a = match[0]
            auc_m  = a.get("AUC_mean",            0.0)
            auc_s  = a.get("AUC_std",              0.0)
            rec_m  = a.get("final_recall_mean",    0.0)
            rec_s  = a.get("final_recall_std",     0.0)
            map_m  = a.get("final_mAP50_mean",     0.0)
            map_s  = a.get("final_mAP50_std",      0.0)
            lat_m  = a.get("inference_latency_ms_mean", 0.0)
            lat_s  = a.get("inference_latency_ms_std",  0.0)
            l0_str = str(l0) if first else ""
            first  = False
            # Bold the proposed method
            name = s_labels.get(s, s)
            if s == "bald_diversity":
                name = r"\textbf{" + name + r"}"
            lines.append(
                f"  {name} & {l0_str} "
                f"& {auc_m:.2f} & {auc_s:.2f} "
                f"& {rec_m:.4f} & {rec_s:.4f} "
                f"& {map_m:.4f} & {map_s:.4f} "
                f"& {lat_m:.1f} $\\pm$ {lat_s:.1f} \\\\"
            )
        if l0 != l0_sizes[-1]:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_latex_h2_table(shift_agg: list, l0_sizes: list = None) -> str:
    """Build LaTeX table for H2 shift stability comparison."""
    if l0_sizes is None:
        l0_sizes = sorted({a.get("l0_size", 0) for a in shift_agg})

    strategies = ["random", "deterministic", "bald_only", "bald_diversity"]
    s_labels   = {
        "random":        "Random",
        "deterministic": "Det. Unc.",
        "bald_only":     "BALD-only",
        "bald_diversity":"BALD+Div (proposed)",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{H2: Recall stability under covariate shift (mean $\pm$ std, $N=5$ seeds). "
        r"Recall Var.: inter-cycle variance of recall on target domain (lower = more stable). "
        r"Mono.: monotonicity fraction ($\uparrow$ = more monotone improvement).}",
        r"\label{tab:h2_shift_results}",
        r"\begin{tabular}{lc rr rr rr}",
        r"\toprule",
        r"Strategy & $L_0$ & Recall $\uparrow$ & $\pm$ & Recall Var. $\downarrow$ & $\pm$ "
        r"& Mono. $\uparrow$ & $\pm$ \\",
        r"\midrule",
    ]

    for l0 in l0_sizes:
        first = True
        for s in strategies:
            match = [a for a in shift_agg if a.get("l0_size") == l0 and a.get("strategy") == s]
            if not match:
                continue
            a = match[0]
            rec_m  = a.get("final_recall_mean",      0.0)
            rec_s  = a.get("final_recall_std",        0.0)
            var_m  = a.get("recall_variance_mean",    0.0)
            var_s  = a.get("recall_variance_std",     0.0)
            mon_m  = a.get("monotonicity_mean",       0.0)
            mon_s  = a.get("monotonicity_std",        0.0)
            l0_str = str(l0) if first else ""
            first  = False
            name = s_labels.get(s, s)
            if s == "bald_diversity":
                name = r"\textbf{" + name + r"}"
            lines.append(
                f"  {name} & {l0_str} "
                f"& {rec_m:.4f} & {rec_s:.4f} "
                f"& {var_m:.6f} & {var_s:.6f} "
                f"& {mon_m:.2f} & {mon_s:.2f} \\\\"
            )
        if l0 != l0_sizes[-1]:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot multi-seed and shift results")
    parser.add_argument("--multi_seed_results", type=str, default=None,
                        help="Path to all_aggregated.json from run_multi_seed.py")
    parser.add_argument("--shift_results", type=str, default=None,
                        help="Path to all_shift_aggregated.json from run_shift_experiments.py")
    parser.add_argument("--output", type=str, default="./figures",
                        help="Output directory for figures")
    parser.add_argument("--l0", type=int, default=100,
                        help="L0 size to plot (default: 100)")
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Multi-seed results (H1/H3/H4)
    multi_agg = []
    if args.multi_seed_results:
        with open(args.multi_seed_results) as f:
            multi_agg = json.load(f)
        print(f"[plot] Loaded {len(multi_agg)} multi-seed result combinations")

        plot_learning_curves_with_bands(multi_agg, output_dir, l0_filter=args.l0)
        plot_auc_comparison(multi_agg, output_dir, l0_filter=args.l0)
        plot_latency_boxplot(multi_agg, output_dir)

        latex_table = build_latex_table_multiseed(multi_agg)
        table_path = output_dir / "results_table_multiseed.tex"
        table_path.write_text(latex_table)
        print(f"[plot] Saved: {table_path}")

    # Shift results (H2)
    shift_agg = []
    if args.shift_results:
        with open(args.shift_results) as f:
            shift_agg = json.load(f)
        print(f"[plot] Loaded {len(shift_agg)} shift result combinations")

        plot_shift_recall_curves(shift_agg, output_dir, l0_filter=args.l0)
        plot_recall_variance_barplot(shift_agg, output_dir, l0_filter=args.l0)

        latex_h2 = build_latex_h2_table(shift_agg)
        h2_path = output_dir / "h2_shift_table.tex"
        h2_path.write_text(latex_h2)
        print(f"[plot] Saved: {h2_path}")

    if not multi_agg and not shift_agg:
        print("[plot] No results provided. Pass --multi_seed_results and/or --shift_results.")


if __name__ == "__main__":
    main()
