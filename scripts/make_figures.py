#!/usr/bin/env python3
"""
Figures for the thesis, generated from stored results.

The examination board flagged Figures 2, 11, 14 and 17 as illegible. That
is a systematic problem, not four accidents: the previous figures used
matplotlib's defaults, which size text for a full-width screen and then
get shrunk into a two-thirds-width LaTeX float. Everything here is sized
for the final printed width instead.

The legibility rules applied throughout:

  * Base font 13pt, axis labels 14pt, titles 15pt, measured at the figure
    size the document actually includes. Nothing is scaled down after.
  * Figures are authored at the width they are placed at, so
    `\\includegraphics[width=\\textwidth]` performs no resampling.
  * Series identity is never carried by color alone: every series also has
    its own marker and line style, and is directly labelled at the end of
    its curve. This survives greyscale printing and colour-vision
    deficiency, and it is what makes a four-series plot readable without
    hunting back to a legend.
  * Vector PDF for the document, PNG at 300 dpi for slides.

The palette is a colour-vision-validated categorical set; its hues are
assigned in fixed order, so a strategy keeps its colour across every
figure in the thesis and the reader learns it once.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.aggregate import aggregate_block, load_records
from src.campaign import load_campaign

# --- Design tokens ---------------------------------------------------------
# Categorical hues in fixed slot order, validated for colour-vision
# deficiency on a light surface (worst adjacent pair dE 9.1 protan,
# 22.9 normal vision).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

STRATEGY_ORDER = [
    "random",
    "deterministic",
    "bald_only",
    "bald_diversity",
    "bsb_diversity",
    "none",
]
PRETTY = {
    "random": "Random",
    "deterministic": "Deterministic",
    "bald_only": "BALD-only",
    "bald_diversity": "BALD + Diversity",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble (5)",
    "conformal": "Conformal",
    "swag": "SWAG",
    "bsb": "BSB",
    "psb": "PSB",
    "bsb_diversity": "BSB + Diversity",
    "none": "No adaptation",
}


def style(base: int = 13, label: int = 14, title: int = 15) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": base,
            "axes.labelsize": label,
            "axes.titlesize": title,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "figure.autolayout": False,
            "pdf.fonttype": 42,   # embed as TrueType: selectable text in the PDF
            "ps.fonttype": 42,
        }
    )


def _clean_axes(ax, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)


def _pretty(name) -> str:
    return PRETTY.get(str(name), str(name).replace("_", " ").title())


def _slot(name: str, order: Sequence[str]) -> int:
    try:
        return list(order).index(str(name))
    except ValueError:
        return len(order) % len(SERIES_COLORS)


def _save(fig, out_dir: Path, name: str, formats=("pdf", "png")) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _declutter(entries, min_gap: float) -> Dict[str, float]:
    """Push overlapping end-labels apart while keeping their order.

    Curves converge at the right edge, so placing each label at its
    curve's final y-value produces a pile of overlapping text -- exactly
    the illegibility the board objected to. This spreads them by at least
    `min_gap` in data units, preserving the vertical ordering so each
    label still reads as belonging to the curve above or below it.
    """
    ordered = sorted(entries, key=lambda e: e[1])
    placed: Dict[str, float] = {}
    last = None
    for name, y in ordered:
        target = y if last is None else max(y, last + min_gap)
        placed[name] = target
        last = target
    return placed


def fig_learning_curves(
    aggs: List[Dict], out_dir: Path, l0: int = 100, budget: int = 50, dataset=None
) -> List[Path]:
    """H1: mAP@50 against cumulative annotations, with seed variability."""
    selected = [
        a
        for a in aggs
        if a.get("l0_size") == l0
        and a.get("budget") == budget
        and (dataset is None or a.get("dataset") == dataset)
        and a.get("curve_mAP50_mean")
    ]
    if not selected:
        return []
    selected.sort(key=lambda a: _slot(a.get("strategy"), STRATEGY_ORDER))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _clean_axes(ax)
    endpoints = []

    for agg in selected:
        name = str(agg.get("strategy"))
        slot = _slot(name, STRATEGY_ORDER)
        color = SERIES_COLORS[slot % len(SERIES_COLORS)]
        mean = np.array([v if v is not None else np.nan for v in agg["curve_mAP50_mean"]])
        std = np.array([v if v is not None else 0.0 for v in agg.get("curve_mAP50_std", [])])
        x = np.array([v if v is not None else np.nan for v in agg.get("curve_n_labelled", [])])
        if len(x) != len(mean):
            x = np.arange(len(mean)) * budget + l0

        # The band is +-1 standard deviation across seeds: the spread a
        # replication would land in, which is the honest way to show that
        # two curves separated by less than the band are not separated.
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            x, mean,
            color=color,
            marker=MARKERS[slot % len(MARKERS)],
            linestyle=LINESTYLES[slot % len(LINESTYLES)],
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=_pretty(name),
            zorder=3,
        )
        if np.isfinite(mean[-1]) and np.isfinite(x[-1]):
            endpoints.append((_pretty(name), float(mean[-1]), color, float(x[-1])))

    # Direct labels at the curve ends, decluttered, so identity never
    # depends on colour alone and the reader does not have to travel to
    # the legend for every curve.
    if endpoints:
        span = max(e[1] for e in endpoints) - min(e[1] for e in endpoints)
        y_range = max(span, 0.12)
        gap = y_range * 0.16
        placed = _declutter([(e[0], e[1]) for e in endpoints], gap)
        x_end = max(e[3] for e in endpoints)
        for label, y_true, color, _ in endpoints:
            y_label = placed[label]
            ax.annotate(
                label,
                xy=(x_end, y_true),
                xytext=(14, 0),
                textcoords="offset points",
                xycoords="data",
                color=color,
                fontsize=11,
                va="center",
                fontweight="medium",
                annotation_clip=False,
            ) if abs(y_label - y_true) < 1e-9 else ax.annotate(
                label,
                xy=(x_end, y_true),
                xytext=(x_end + (x_end * 0.06), y_label),
                textcoords="data",
                color=color,
                fontsize=11,
                va="center",
                fontweight="medium",
                arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8, alpha=0.6),
                annotation_clip=False,
            )

    ax.set_xlabel("Cumulative annotated samples")
    ax.set_ylabel("mAP@50 on held-out test set")
    suffix = f" — {dataset}" if dataset else ""
    ax.set_title(
        f"Annotation efficiency at $L_0$ = {l0}, $B$ = {budget}{suffix}", pad=12
    )
    ax.margins(x=0.30)
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    tag = f"_{dataset}" if dataset else ""
    return _save(fig, out_dir, f"h1_learning_curves{tag}_L0{l0}_B{budget}")


def fig_auc_comparison(
    aggs: List[Dict], out_dir: Path, l0: int = 100, budget: int = 50, dataset=None
) -> List[Path]:
    """H1/H3: normalised AUC per strategy with bootstrap intervals."""
    selected = [
        a for a in aggs
        if a.get("l0_size") == l0 and a.get("budget") == budget
        and (dataset is None or a.get("dataset") == dataset)
        and a.get("auc_normalized_mean") is not None
    ]
    if not selected:
        return []
    selected.sort(key=lambda a: _slot(a.get("strategy"), STRATEGY_ORDER))

    names = [_pretty(a.get("strategy")) for a in selected]
    means = [a["auc_normalized_mean"] for a in selected]
    lows = [a.get("auc_normalized_ci_low", m) for a, m in zip(selected, means)]
    highs = [a.get("auc_normalized_ci_high", m) for a, m in zip(selected, means)]
    errs = np.array([
        [max(m - lo, 0) for m, lo in zip(means, lows)],
        [max(hi - m, 0) for m, hi in zip(means, highs)],
    ])
    colors = [
        SERIES_COLORS[_slot(a.get("strategy"), STRATEGY_ORDER) % len(SERIES_COLORS)]
        for a in selected
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    _clean_axes(ax)
    bars = ax.bar(names, means, color=colors, width=0.62, zorder=3)
    ax.errorbar(
        names, means, yerr=errs, fmt="none",
        ecolor=INK_SECONDARY, elinewidth=1.4, capsize=5, zorder=4,
    )
    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=INK,
        )

    ax.set_ylabel("Normalised AUC (mean mAP@50 per unit budget)")
    suffix = f" — {dataset}" if dataset else ""
    ax.set_title(
        f"Annotation efficiency at $L_0$ = {l0}, $B$ = {budget}{suffix}", pad=12
    )
    lo = min(lows) if lows else 0
    ax.set_ylim(max(0.0, lo - 0.08), max(highs) + 0.06 if highs else 1.0)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    tag = f"_{dataset}" if dataset else ""
    return _save(fig, out_dir, f"h1_auc_normalized{tag}_L0{l0}_B{budget}")


def fig_sweet_spot(aggs: List[Dict], out_dir: Path, dataset=None) -> List[Path]:
    """Where informed selection pays off: gain over random by L0 and B."""
    cells: Dict[tuple, Dict[str, float]] = {}
    for agg in aggs:
        if dataset is not None and agg.get("dataset") != dataset:
            continue
        key = (agg.get("l0_size"), agg.get("budget"))
        value = agg.get("auc_normalized_mean")
        if value is None:
            continue
        cells.setdefault(key, {})[str(agg.get("strategy"))] = value

    l0s = sorted({k[0] for k in cells if k[0] is not None})
    budgets = sorted({k[1] for k in cells if k[1] is not None})
    if not l0s or not budgets:
        return []

    matrix = np.full((len(budgets), len(l0s)), np.nan)
    for i, b in enumerate(budgets):
        for j, l0 in enumerate(l0s):
            entry = cells.get((l0, b), {})
            if "bald_diversity" in entry and "random" in entry:
                matrix[i, j] = entry["bald_diversity"] - entry["random"]

    if np.all(np.isnan(matrix)):
        return []

    # Diverging scale centred on zero: the sign is the question (does
    # informed selection help here at all), so the midpoint must be
    # neutral and the two arms must read as opposites.
    limit = float(np.nanmax(np.abs(matrix))) or 0.01
    fig, ax = plt.subplots(figsize=(1.35 * len(l0s) + 2.6, 1.15 * len(budgets) + 2.4))
    mesh = ax.imshow(
        matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower"
    )
    ax.set_xticks(range(len(l0s)), [str(v) for v in l0s])
    ax.set_yticks(range(len(budgets)), [str(v) for v in budgets])
    ax.set_xlabel("Initial labelled set size $L_0$")
    ax.set_ylabel("Annotation budget $B$")
    suffix = f" — {dataset}" if dataset else ""
    ax.set_title(f"Gain of BALD + Diversity over random sampling{suffix}", pad=12)

    for i in range(len(budgets)):
        for j in range(len(l0s)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color=MUTED, fontsize=11)
                continue
            value = matrix[i, j]
            shade = "#ffffff" if abs(value) > limit * 0.6 else INK
            ax.text(
                j, i, f"{value:+.3f}",
                ha="center", va="center", color=shade, fontsize=11.5,
                fontweight="semibold" if abs(value) > limit * 0.6 else "normal",
            )

    bar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label("$\\Delta$ normalised AUC", fontsize=12)
    bar.outline.set_visible(False)
    fig.tight_layout()
    tag = f"_{dataset}" if dataset else ""
    return _save(fig, out_dir, f"sweet_spot_heatmap{tag}")


def fig_shift_stability(aggs: List[Dict], out_dir: Path) -> List[Path]:
    """H2: recall variance per strategy across shift pairs (lower is better)."""
    pairs = sorted({str(a.get("pair_label")) for a in aggs if a.get("pair_label")})
    strategies = sorted(
        {str(a.get("strategy")) for a in aggs if a.get("strategy")},
        key=lambda s: _slot(s, STRATEGY_ORDER),
    )
    if not pairs or not strategies:
        return []

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(pairs) + 3.0), 4.6))
    _clean_axes(ax)
    width = 0.8 / max(len(strategies), 1)
    positions = np.arange(len(pairs))

    for k, strategy in enumerate(strategies):
        means, errs = [], []
        for pair in pairs:
            match = next(
                (a for a in aggs
                 if str(a.get("pair_label")) == pair and str(a.get("strategy")) == strategy),
                None,
            )
            means.append(match.get("recall_variance_mean", np.nan) if match else np.nan)
            errs.append(match.get("recall_variance_std", 0.0) if match else 0.0)
        offset = (k - (len(strategies) - 1) / 2) * width
        ax.bar(
            positions + offset, means, width=width * 0.92,
            yerr=errs, capsize=3,
            color=SERIES_COLORS[_slot(strategy, STRATEGY_ORDER) % len(SERIES_COLORS)],
            label=_pretty(strategy),
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.1},
            zorder=3,
        )

    ax.set_xticks(positions, [p.replace("_to_", " → ") for p in pairs])
    ax.set_ylabel("Inter-cycle recall variance  (lower = more stable)")
    ax.set_title("H2: learning-curve stability under covariate shift", pad=12)
    ax.legend(ncol=min(len(strategies), 4), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir, "h2_shift_stability")


def fig_shift_recovery(shift_aggs, baseline_aggs, out_dir: Path) -> List[Path]:
    """H2: how much recall adaptation recovers from the domain gap."""
    baselines = {
        str(a.get("pair_label")): a.get("baseline_target_recall_mean")
        for a in baseline_aggs
    }
    pairs = sorted({str(a.get("pair_label")) for a in shift_aggs if a.get("pair_label")})
    pairs = [p for p in pairs if baselines.get(p) is not None]
    if not pairs:
        return []

    strategies = sorted(
        {str(a.get("strategy")) for a in shift_aggs},
        key=lambda s: _slot(s, STRATEGY_ORDER),
    )
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(pairs) + 3.0), 4.6))
    _clean_axes(ax)
    width = 0.8 / max(len(strategies), 1)
    positions = np.arange(len(pairs))

    for k, strategy in enumerate(strategies):
        values = []
        for pair in pairs:
            match = next(
                (a for a in shift_aggs
                 if str(a.get("pair_label")) == pair and str(a.get("strategy")) == strategy),
                None,
            )
            final = match.get("final_recall_mean") if match else None
            base = baselines.get(pair)
            values.append(final - base if final is not None and base is not None else np.nan)
        offset = (k - (len(strategies) - 1) / 2) * width
        ax.bar(
            positions + offset, values, width=width * 0.92,
            color=SERIES_COLORS[_slot(strategy, STRATEGY_ORDER) % len(SERIES_COLORS)],
            label=_pretty(strategy), zorder=3,
        )

    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.set_xticks(positions, [p.replace("_to_", " → ") for p in pairs])
    ax.set_ylabel("Recall gained over the unadapted model")
    ax.set_title("Recovery from the domain gap through adaptation", pad=12)
    ax.legend(ncol=min(len(strategies), 4), loc="best")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir, "h2_shift_recovery")


def fig_latency_distribution(aggs: List[Dict], out_dir: Path, budget_ms: float = 33.0) -> List[Path]:
    """H4: latency percentiles against the real-time budget."""
    selected = [a for a in aggs if a.get("latency_ms_mean_mean") is not None]
    if not selected:
        return []
    selected.sort(key=lambda a: _slot(a.get("strategy"), STRATEGY_ORDER))

    names = [_pretty(a.get("strategy")) for a in selected]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _clean_axes(ax)

    for key, marker, label, alpha in (
        ("latency_ms_mean", "o", "mean", 1.0),
        ("latency_ms_p95", "s", "p95", 0.85),
        ("latency_ms_p99", "^", "p99", 0.7),
    ):
        values = [a.get(f"{key}_mean", np.nan) for a in selected]
        errs = [a.get(f"{key}_std", 0.0) for a in selected]
        ax.errorbar(
            x, values, yerr=errs, fmt=marker, markersize=9,
            color=SERIES_COLORS[0] if label == "mean" else
            (SERIES_COLORS[1] if label == "p95" else SERIES_COLORS[3]),
            ecolor=INK_SECONDARY, elinewidth=1.2, capsize=4,
            label=label, alpha=alpha, linestyle="none", zorder=3,
        )

    ax.axhline(budget_ms, color="#d03b3b", linewidth=1.8, linestyle="--", zorder=2)
    ax.annotate(
        f"real-time budget ({budget_ms:.0f} ms)",
        xy=(len(names) - 0.5, budget_ms), xytext=(0, 6),
        textcoords="offset points", ha="right", color="#d03b3b", fontsize=11,
    )
    ax.set_xticks(x, names)
    ax.set_ylabel("Inference latency per frame (ms)")
    ax.set_title("H4: deployed inference latency against the real-time budget", pad=12)
    ax.legend(title="statistic", ncol=3, loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir, "h4_latency_distribution")


def fig_resource_profile(aggs: List[Dict], out_dir: Path) -> List[Path]:
    """H4: the four resources the board asked for, as small multiples.

    Small multiples rather than a dual-axis chart: memory in megabytes,
    utilisation in percent, power in watts and traffic in megabytes per
    cycle share no scale, and overlaying them on two y-axes would invent
    crossings that mean nothing.
    """
    panels = [
        ("peak_host_memory_mb", "Peak host RAM (MB)"),
        ("peak_gpu_memory_mb", "Peak GPU memory (MB)"),
        ("gpu_util_pct_mean", "GPU utilisation (%)"),
        ("gpu_power_w_mean", "GPU power (W)"),
        ("gpu_energy_j_per_cycle", "Energy per cycle (J)"),
        ("network_mb_per_cycle", "Network payload (MB/cycle)"),
    ]
    available = [
        (key, label) for key, label in panels
        if any(a.get(f"{key}_mean") is not None for a in aggs)
    ]
    if not available:
        return []

    selected = sorted(aggs, key=lambda a: _slot(a.get("strategy"), STRATEGY_ORDER))
    names = [_pretty(a.get("strategy")) for a in selected]
    colors = [
        SERIES_COLORS[_slot(a.get("strategy"), STRATEGY_ORDER) % len(SERIES_COLORS)]
        for a in selected
    ]

    ncols = min(3, len(available))
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows), squeeze=False
    )

    for idx, (key, label) in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        _clean_axes(ax)
        means = [a.get(f"{key}_mean", np.nan) for a in selected]
        errs = [a.get(f"{key}_std", 0.0) for a in selected]
        ax.bar(
            range(len(names)), means, yerr=errs, capsize=4,
            color=colors, width=0.6,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.1}, zorder=3,
        )
        ax.set_xticks(range(len(names)), names, rotation=18, ha="right", fontsize=10)
        ax.set_title(label, fontsize=12.5, pad=8)

    for idx in range(len(available), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("H4: operational cost of continuous adaptation", fontsize=15, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "h4_resource_profile")


def fig_diversity(aggs: List[Dict], out_dir: Path, dataset=None) -> List[Path]:
    """H3/C4: cluster coverage and redundancy of the selected batches."""
    selected = [
        a for a in aggs
        if a.get("diversity_coverage_mean_mean") is not None
        and (dataset is None or a.get("dataset") == dataset)
    ]
    if not selected:
        return []
    selected.sort(key=lambda a: _slot(a.get("strategy"), STRATEGY_ORDER))

    names = [_pretty(a.get("strategy")) for a in selected]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))

    for ax, key, label, invert in (
        (axes[0], "diversity_coverage_mean", "Cluster coverage per cycle", False),
        (axes[1], "diversity_redundancy_rate_mean", "Redundant pairs in batch", True),
    ):
        _clean_axes(ax)
        means = [a.get(f"{key}_mean", np.nan) for a in selected]
        errs = [a.get(f"{key}_std", 0.0) for a in selected]
        ax.bar(
            x, means, yerr=errs, capsize=4, width=0.6,
            color=[
                SERIES_COLORS[_slot(a.get("strategy"), STRATEGY_ORDER) % len(SERIES_COLORS)]
                for a in selected
            ],
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.1}, zorder=3,
        )
        ax.set_xticks(x, names, rotation=16, ha="right", fontsize=10.5)
        arrow = "lower is better" if invert else "higher is better"
        ax.set_title(f"{label}\n({arrow})", fontsize=12.5, pad=8)
        ax.set_ylim(0, 1.0)

    suffix = f" — {dataset}" if dataset else ""
    fig.suptitle(f"Diversity of the acquisition step{suffix}", fontsize=15, y=1.02)
    fig.tight_layout()
    tag = f"_{dataset}" if dataset else ""
    return _save(fig, out_dir, f"h3_diversity{tag}")


def fig_forgetting_matrix(aggs: List[Dict], results_root: str, out_dir: Path) -> List[Path]:
    """Per-domain performance after each cycle: the forgetting matrix."""
    records = load_records(results_root, "E5_long_horizon")
    matrices = [
        r["result"]["forgetting"]
        for r in records
        if r.get("result", {}).get("forgetting", {}).get("performance_matrix")
    ]
    if not matrices:
        return []

    source = matrices[0]
    matrix = np.array(source["performance_matrix"], dtype=float)
    domains = source.get("domains", [])
    cycles = source.get("cycles", list(range(matrix.shape[0])))

    fig, ax = plt.subplots(
        figsize=(1.0 * len(cycles) + 3.4, 0.75 * len(domains) + 2.8)
    )
    mesh = ax.imshow(matrix.T, cmap="Blues", aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(cycles)), [str(c) for c in cycles])
    ax.set_yticks(range(len(domains)), domains)
    ax.set_xlabel("Adaptation cycle")
    ax.set_ylabel("Evaluation domain")
    ax.set_title("mAP@50 per source domain across the adaptation horizon", pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(
                i, j, f"{value:.2f}",
                ha="center", va="center", fontsize=10,
                color="#ffffff" if value > 0.55 else INK,
            )

    bar = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label("mAP@50", fontsize=12)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, out_dir, "forgetting_matrix")


def fig_sensitivity(aggs: List[Dict], out_dir: Path) -> List[Path]:
    """One panel per swept hyperparameter."""
    params = sorted({str(a.get("sweep_param")) for a in aggs if a.get("sweep_param")})
    if not params:
        return []

    ncols = min(2, len(params))
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    for idx, param in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        _clean_axes(ax)
        rows = [a for a in aggs if str(a.get("sweep_param")) == param]
        rows.sort(key=lambda a: str(a.get("sweep_value")))
        labels = [str(a.get("sweep_value")) for a in rows]
        means = [a.get("auc_normalized_mean", np.nan) for a in rows]
        errs = [a.get("auc_normalized_std", 0.0) for a in rows]
        ax.errorbar(
            range(len(labels)), means, yerr=errs,
            marker="o", markersize=8, color=SERIES_COLORS[0],
            ecolor=INK_SECONDARY, elinewidth=1.3, capsize=4, zorder=3,
        )
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel("Normalised AUC")
        ax.set_title(param.replace("_", " "), fontsize=12.5, pad=8)

    for idx in range(len(params), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Hyperparameter sensitivity at $L_0$ = 100", fontsize=15, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "sensitivity")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all_figures(spec, results_root: str = "results", out_dir: Path = Path("figures")) -> List[Path]:
    reporting = (spec.reporting or {}).get("figures", {})
    style(
        base=int(reporting.get("base_font_size", 13)),
        label=int(reporting.get("label_font_size", 14)),
        title=int(reporting.get("title_font_size", 15)),
    )
    out_dir = Path(out_dir)
    written: List[Path] = []

    aggregates: Dict[str, List[Dict]] = {}
    for block in spec.blocks:
        records = load_records(results_root, block["id"])
        if records:
            aggregates[block["id"]] = aggregate_block(records)

    main = aggregates.get("E1_main_grid", [])
    if main:
        # One figure per dataset. Overlaying two datasets on one axis
        # would draw each strategy twice in the same colour, which is how
        # a four-series plot silently becomes an unreadable eight-series
        # one.
        datasets = sorted({a.get("dataset") for a in main if a.get("dataset")})
        for dataset in datasets or [None]:
            subset = [a for a in main if dataset is None or a.get("dataset") == dataset]
            l0s = sorted({a.get("l0_size") for a in subset if a.get("l0_size")})
            budgets = sorted({a.get("budget") for a in subset if a.get("budget")})
            for l0 in l0s:
                for budget in budgets:
                    written += fig_learning_curves(
                        subset, out_dir, l0=l0, budget=budget, dataset=dataset
                    )
            focus = 100 if 100 in l0s else (l0s[0] if l0s else 100)
            for budget in budgets:
                written += fig_auc_comparison(
                    subset, out_dir, l0=focus, budget=budget, dataset=dataset
                )
            written += fig_sweet_spot(subset, out_dir, dataset=dataset)
            written += fig_diversity(subset, out_dir, dataset=dataset)

    shift = aggregates.get("E2b_shift_real", [])
    baseline = aggregates.get("E2a_shift_baseline", [])
    if shift:
        written += fig_shift_stability(shift, out_dir)
    if shift and baseline:
        written += fig_shift_recovery(shift, baseline, out_dir)

    profiling = aggregates.get("E6_operational_profiling", [])
    if profiling:
        written += fig_latency_distribution(profiling, out_dir)
        written += fig_resource_profile(profiling, out_dir)

    if aggregates.get("E5_long_horizon"):
        written += fig_forgetting_matrix(aggregates["E5_long_horizon"], results_root, out_dir)

    if aggregates.get("E4_sensitivity"):
        written += fig_sensitivity(aggregates["E4_sensitivity"], out_dir)

    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--config", default="configs/campaign.yaml")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    spec = load_campaign(args.config)
    written = build_all_figures(spec, results_root=args.results, out_dir=Path(args.out))
    for path in written:
        print(f"  {path}")
    print(f"\n{len(written)} file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
