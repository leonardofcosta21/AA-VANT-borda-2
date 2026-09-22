"""
Statistical validation for the experimental campaign.

The examination board asked for "mais execucoes experimentais e validacao
estatistica robusta", and the thesis (Section 5.3.6.2) promises bootstrap
confidence intervals at n = 1000, alpha = 0.05. This module implements
that promise and adds the pieces a reviewer will expect alongside it:
paired non-parametric tests, multiple-comparison correction, and effect
sizes.

Why non-parametric
------------------
With five seeds per condition, a normality assumption is untestable. The
default comparison is therefore the Wilcoxon signed-rank test on seed-
paired values (each seed runs every strategy on identical splits, so the
pairing is real), with the exact distribution used at these sample sizes.
The bootstrap CI is reported regardless, because it communicates the
magnitude of the uncertainty rather than only a binary verdict.

Why the effect size matters here
--------------------------------
At n = 5 a real difference can fail to reach significance. Reporting
Cliff's delta next to the p-value lets the thesis say "the difference is
large but the sample is too small to exclude chance", which is an honest
and defensible claim, instead of silently dropping the comparison.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is in requirements, but the module must import without it
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

__all__ = [
    "BootstrapCI",
    "ComparisonResult",
    "bootstrap_ci",
    "bootstrap_diff_ci",
    "paired_test",
    "cliffs_delta",
    "cohens_d",
    "holm_bonferroni",
    "compare_strategies",
    "significance_table",
]

DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    n: int
    n_boot: int
    alpha: float

    def as_dict(self) -> Dict:
        return asdict(self)

    def latex(self, fmt: str = "{:.3f}") -> str:
        return (
            f"{fmt.format(self.mean)} "
            f"[{fmt.format(self.lower)}, {fmt.format(self.upper)}]"
        )


@dataclass
class ComparisonResult:
    metric: str
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    difference: float
    diff_ci: Tuple[float, float]
    p_value: float
    p_adjusted: Optional[float] = None
    test: str = "wilcoxon"
    effect_size: float = 0.0
    effect_name: str = "cliffs_delta"
    effect_magnitude: str = "negligible"
    n_pairs: int = 0
    significant: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
    statistic=np.mean,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for a statistic.

    The seed is fixed so that re-running the reporting step on stored
    results reproduces the exact interval printed in the thesis.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), 0, n_boot, alpha)
    if n == 1:
        v = float(arr[0])
        return BootstrapCI(v, v, v, 1, n_boot, alpha)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(arr[idx], axis=1)
    lower = float(np.percentile(boot, 100 * alpha / 2))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return BootstrapCI(float(statistic(arr)), lower, upper, n, n_boot, alpha)


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    paired: bool = True,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean difference ``a - b``.

    When ``paired`` (the default, since every seed runs every strategy on
    the same split) the resampling is over seed indices, which preserves
    the pairing and gives a much tighter interval than resampling the two
    groups independently.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)

    if paired:
        n = min(len(arr_a), len(arr_b))
        if n == 0:
            return float("nan"), float("nan"), float("nan")
        diffs = arr_a[:n] - arr_b[:n]
        ci = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
        return ci.mean, ci.lower, ci.upper

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(arr_a, size=len(arr_a), replace=True)
        sb = rng.choice(arr_b, size=len(arr_b), replace=True)
        boots[i] = sa.mean() - sb.mean()
    return (
        float(arr_a.mean() - arr_b.mean()),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> Tuple[float, str, List[str]]:
    """Wilcoxon signed-rank on paired samples, with honest fallbacks.

    Returns ``(p_value, test_name, notes)``. The notes carry the caveats
    that belong in the thesis text rather than being swallowed: too few
    pairs for the test to ever reach significance, all-zero differences,
    or scipy being unavailable.
    """
    notes: List[str] = []
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n < 2:
        return float("nan"), "none", ["fewer than 2 paired observations"]

    diffs = arr_a[:n] - arr_b[:n]
    if np.allclose(diffs, 0):
        return 1.0, "wilcoxon", ["all paired differences are zero"]

    if n < 6:
        notes.append(
            f"n={n} pairs: the minimum attainable two-sided p is "
            f"{2 ** (1 - n):.3f}; interpret with the effect size"
        )

    if _scipy_stats is None:
        notes.append("scipy unavailable, fell back to a sign test")
        n_pos = int((diffs > 0).sum())
        n_eff = int((diffs != 0).sum())
        p = _binomial_two_sided(n_pos, n_eff)
        return p, "sign_test", notes

    try:
        result = _scipy_stats.wilcoxon(
            arr_a[:n], arr_b[:n], alternative=alternative, mode="exact"
        )
    except TypeError:  # newer scipy renamed mode -> method
        result = _scipy_stats.wilcoxon(
            arr_a[:n], arr_b[:n], alternative=alternative, method="exact"
        )
    except ValueError as exc:  # pragma: no cover
        notes.append(f"wilcoxon failed ({exc}); fell back to a sign test")
        n_pos = int((diffs > 0).sum())
        n_eff = int((diffs != 0).sum())
        return _binomial_two_sided(n_pos, n_eff), "sign_test", notes

    return float(result.pvalue), "wilcoxon", notes


def _binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p under p=0.5, without scipy."""
    if n == 0:
        return 1.0
    from math import comb

    def pmf(i: int) -> float:
        return comb(n, i) * 0.5 ** n

    observed = pmf(k)
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12)
    return float(min(1.0, total))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Cliff's delta and its conventional magnitude label.

    Non-parametric, bounded in [-1, 1], and meaningful at n = 5, which is
    exactly the regime this campaign operates in. Thresholds follow
    Romano et al. (2006): 0.147 negligible, 0.33 small, 0.474 medium.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0, "undefined"

    comparisons = np.sign(arr_a[:, None] - arr_b[None, :])
    delta = float(comparisons.mean())

    magnitude = abs(delta)
    if magnitude < 0.147:
        label = "negligible"
    elif magnitude < 0.33:
        label = "small"
    elif magnitude < 0.474:
        label = "medium"
    else:
        label = "large"
    return delta, label


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d, reported only as a secondary descriptor."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n < 2:
        return float("nan")
    diffs = arr_a[:n] - arr_b[:n]
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diffs) / sd)


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni step-down correction.

    Chosen over plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate, which matters when the campaign runs
    six strategies over several conditions and the raw comparison count
    climbs into the dozens.
    """
    values = list(p_values)
    n = len(values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: (np.isnan(values[i]), values[i]))
    adjusted = [float("nan")] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = values[idx]
        if np.isnan(p):
            adjusted[idx] = float("nan")
            continue
        adj = min(1.0, (n - rank) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    rejected = [
        (not np.isnan(adjusted[i])) and adjusted[i] < alpha for i in range(n)
    ]
    return adjusted, rejected


# ---------------------------------------------------------------------------
# High-level comparison driver
# ---------------------------------------------------------------------------

def compare_strategies(
    per_strategy_values: Dict[str, Sequence[float]],
    metric: str,
    reference: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    higher_is_better: bool = True,
) -> List[ComparisonResult]:
    """Compare strategies on one metric, corrected for multiplicity.

    Parameters
    ----------
    per_strategy_values
        ``{strategy: [value_seed0, value_seed1, ...]}``. Seed order must
        be consistent across strategies for the pairing to be valid; the
        campaign runner guarantees this by using ``base_seed + k``
        everywhere.
    reference
        When given, only ``reference`` vs each other strategy is tested
        (the comparison the hypotheses actually make). When ``None``,
        every unordered pair is tested, which inflates the correction.
    """
    names = list(per_strategy_values)
    if reference and reference in names:
        pairs = [(reference, other) for other in names if other != reference]
    else:
        pairs = list(itertools.combinations(names, 2))

    results: List[ComparisonResult] = []
    for a_name, b_name in pairs:
        a_vals = list(per_strategy_values[a_name])
        b_vals = list(per_strategy_values[b_name])
        n_pairs = min(len(a_vals), len(b_vals))

        p, test_name, notes = paired_test(a_vals, b_vals)
        delta, magnitude = cliffs_delta(a_vals, b_vals)
        diff, lo, hi = bootstrap_diff_ci(
            a_vals, b_vals, paired=True, n_boot=n_boot, alpha=alpha, seed=seed
        )
        if not higher_is_better:
            notes.append("lower is better for this metric")

        results.append(
            ComparisonResult(
                metric=metric,
                group_a=a_name,
                group_b=b_name,
                mean_a=float(np.mean(a_vals)) if a_vals else float("nan"),
                mean_b=float(np.mean(b_vals)) if b_vals else float("nan"),
                difference=diff,
                diff_ci=(lo, hi),
                p_value=p,
                test=test_name,
                effect_size=delta,
                effect_magnitude=magnitude,
                n_pairs=n_pairs,
                notes=notes,
            )
        )

    adjusted, rejected = holm_bonferroni([r.p_value for r in results], alpha=alpha)
    for result, p_adj, is_sig in zip(results, adjusted, rejected):
        result.p_adjusted = p_adj
        result.significant = bool(is_sig)
    return results


def significance_table(results: Sequence[ComparisonResult]) -> List[Dict]:
    """Flatten comparison results into rows ready for CSV or LaTeX."""
    rows = []
    for r in results:
        rows.append(
            {
                "metric": r.metric,
                "comparison": f"{r.group_a} vs {r.group_b}",
                "mean_a": r.mean_a,
                "mean_b": r.mean_b,
                "difference": r.difference,
                "ci_low": r.diff_ci[0],
                "ci_high": r.diff_ci[1],
                "test": r.test,
                "p_value": r.p_value,
                "p_holm": r.p_adjusted,
                "significant": r.significant,
                "cliffs_delta": r.effect_size,
                "effect": r.effect_magnitude,
                "n_pairs": r.n_pairs,
                "notes": "; ".join(r.notes),
            }
        )
    return rows
