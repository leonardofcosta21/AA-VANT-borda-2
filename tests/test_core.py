"""
Unit tests for the parts of the campaign that a mock run cannot prove.

The mock backend proves the plumbing connects. These tests prove the
numbers are right: that AUC integrates what it claims to, that the
statistical machinery behaves at the sample sizes this campaign actually
has, that the uncertainty prefilter narrows the pool instead of silently
disabling itself, and that the forgetting metrics detect forgetting.

Run with: pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition import apply_uncertainty_prefilter
from src.diversity_metrics import analyse_selection, cluster_coverage, selection_entropy
from src.forgetting import ForgettingTracker
from src.metrics import (
    auc_learning_curve,
    auc_normalized,
    label_efficiency,
    max_drawdown,
    monotonicity,
    recall_variance,
)
from src.stats import bootstrap_ci, cliffs_delta, holm_bonferroni, paired_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestAUC:
    def test_constant_curve_integrates_to_rectangle(self):
        # A detector stuck at 0.5 over 400 labels encloses 0.5 * 400.
        assert auc_learning_curve([0.5] * 5, [0, 100, 200, 300, 400]) == pytest.approx(200.0)

    def test_normalisation_recovers_the_mean_level(self):
        # This is the property the board asked for: the normalised value
        # must read directly as "average mAP@50 sustained".
        assert auc_normalized([0.5] * 5, [0, 100, 200, 300, 400]) == pytest.approx(0.5)

    def test_normalised_auc_is_budget_invariant(self):
        # Same learning behaviour, different budget: the raw AUC doubles
        # while the normalised value must not move.
        small = auc_normalized([0.4, 0.6], [0, 200])
        large = auc_normalized([0.4, 0.6], [0, 400])
        assert small == pytest.approx(large)
        assert auc_learning_curve([0.4, 0.6], [0, 400]) == pytest.approx(
            2 * auc_learning_curve([0.4, 0.6], [0, 200])
        )

    def test_explicit_l_max_overrides_observed_span(self):
        # A run whose pool dried up early must still be normalised by the
        # nominal budget, or it would be flattered relative to its peers.
        value = auc_normalized([0.5, 0.5], [0, 200], l_max=400)
        assert value == pytest.approx(0.25)

    def test_degenerate_inputs_do_not_raise(self):
        assert auc_learning_curve([], []) == 0.0
        assert auc_learning_curve([0.5], [0]) == 0.0
        assert auc_normalized([0.5, 0.6], [100, 100]) == 0.0


class TestStability:
    def test_monotonic_curve_scores_one(self):
        assert monotonicity([0.1, 0.2, 0.3, 0.4]) == 1.0

    def test_single_regression_is_counted(self):
        assert monotonicity([0.1, 0.3, 0.2, 0.4]) == pytest.approx(2 / 3)

    def test_drawdown_finds_the_worst_collapse_not_the_last(self):
        # Variance would be dominated by the overall rise; drawdown must
        # report the 0.4 cliff, which is the operational risk.
        assert max_drawdown([0.2, 0.7, 0.3, 0.8]) == pytest.approx(0.4)

    def test_drawdown_is_zero_for_a_monotone_curve(self):
        assert max_drawdown([0.1, 0.2, 0.3]) == 0.0

    def test_recall_variance_ignores_missing_cycles(self):
        assert recall_variance([0.5, np.nan, 0.5]) == pytest.approx(0.0)


class TestLabelEfficiency:
    def test_interpolates_between_bracketing_cycles(self):
        # 0.5 sits halfway between 0.4 at 100 labels and 0.6 at 200.
        assert label_efficiency([0.4, 0.6], [100, 200], target=0.5) == pytest.approx(150.0)

    def test_unreached_target_returns_none_not_the_budget(self):
        # Returning the budget would silently claim the target was met.
        assert label_efficiency([0.1, 0.2], [100, 200], target=0.9) is None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 0.05, 20).tolist()
        ci = bootstrap_ci(values, n_boot=500)
        assert ci.lower <= ci.mean <= ci.upper

    def test_is_deterministic_across_calls(self):
        # The thesis prints these numbers; re-running the report must not
        # change them.
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert bootstrap_ci(values).as_dict() == bootstrap_ci(values).as_dict()

    def test_single_observation_degenerates_safely(self):
        ci = bootstrap_ci([0.42])
        assert ci.mean == ci.lower == ci.upper == pytest.approx(0.42)


class TestPairedTest:
    def test_five_seeds_warn_about_attainable_p(self):
        # The campaign runs 5 seeds; the test must say so rather than let
        # a 0.0625 be read as "no effect".
        p, test, notes = paired_test([0.6] * 5, [0.5] * 5)
        assert test == "wilcoxon"
        assert any("minimum attainable" in n for n in notes)
        assert p >= 0.0625

    def test_identical_samples_give_p_one(self):
        p, _, notes = paired_test([0.5] * 5, [0.5] * 5)
        assert p == 1.0
        assert any("zero" in n for n in notes)

    def test_too_few_pairs_is_reported_not_guessed(self):
        p, test, notes = paired_test([0.5], [0.4])
        assert np.isnan(p) and test == "none" and notes


class TestEffectSize:
    def test_complete_dominance_is_delta_one(self):
        delta, label = cliffs_delta([1, 2, 3], [0.1, 0.2, 0.3])
        assert delta == pytest.approx(1.0)
        assert label == "large"

    def test_identical_distributions_are_negligible(self):
        _, label = cliffs_delta([1, 2, 3], [1, 2, 3])
        assert label == "negligible"

    def test_sign_follows_argument_order(self):
        forward, _ = cliffs_delta([1, 2], [3, 4])
        backward, _ = cliffs_delta([3, 4], [1, 2])
        assert forward == -backward


class TestHolm:
    def test_is_less_conservative_than_bonferroni(self):
        # The whole reason for choosing Holm: same family-wise error
        # rate, more power.
        p_values = [0.01, 0.02, 0.03]
        adjusted, _ = holm_bonferroni(p_values)
        assert adjusted[0] == pytest.approx(0.03)      # 3 * 0.01
        assert all(a <= b for a, b in zip(adjusted, [3 * p for p in p_values]))

    def test_adjusted_values_are_monotone(self):
        adjusted, _ = holm_bonferroni([0.001, 0.04, 0.5])
        assert adjusted == sorted(adjusted)

    def test_nan_p_values_do_not_reject(self):
        adjusted, rejected = holm_bonferroni([float("nan"), 0.001])
        assert np.isnan(adjusted[0]) and rejected[0] is False


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

class TestUncertaintyPrefilter:
    def test_narrows_the_pool_while_funding_the_budget(self):
        # The regression this guards: with a fixed alpha the old code
        # produced fewer survivors than the budget and then threw the
        # filter away, so the documented two-stage selection never ran.
        scores = np.linspace(0, 1, 350)
        filtered = apply_uncertainty_prefilter(list(range(350)), scores, budget=50, alpha=0.1)
        assert len(filtered) >= 50, "the filter must always fund the budget"
        assert len(filtered) < 350, "the filter must actually narrow the pool"

    def test_keeps_the_most_uncertain_candidates(self):
        scores = np.linspace(0, 1, 200)
        filtered = apply_uncertainty_prefilter(list(range(200)), scores, budget=20, alpha=0.1)
        assert min(filtered) > 100, "survivors must come from the high-uncertainty tail"

    def test_small_pool_passes_through_untouched(self):
        scores = np.linspace(0, 1, 30)
        assert apply_uncertainty_prefilter(list(range(30)), scores, 50, 0.1) == list(range(30))

    def test_missing_scores_are_a_no_op(self):
        assert apply_uncertainty_prefilter([1, 2, 3], None, 2, 0.1) == [1, 2, 3]

    def test_constant_scores_still_fund_the_budget(self):
        # Every candidate ties at the threshold: the top-up path must fire.
        scores = np.ones(200)
        filtered = apply_uncertainty_prefilter(list(range(200)), scores, budget=50, alpha=0.1)
        assert len(filtered) >= 50


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_coverage_detects_concentration(self):
        labels = [0, 0, 1, 1, 2, 2]
        assert cluster_coverage(labels, [0, 2, 4])["coverage"] == pytest.approx(1.0)
        assert cluster_coverage(labels, [0, 1])["coverage"] == pytest.approx(1 / 3)

    def test_entropy_separates_even_from_lopsided(self):
        labels = [0, 0, 0, 1, 1, 1]
        even = selection_entropy(labels, [0, 3])
        lopsided = selection_entropy(labels, [0, 1])
        assert even > lopsided
        assert lopsided == pytest.approx(0.0)

    def test_spread_selection_beats_clustered_selection(self):
        # Three tight blobs; picking one from each must score higher
        # coverage than picking three from one blob. This is the property
        # the H3 argument rests on.
        rng = np.random.default_rng(0)
        pool = np.vstack([
            rng.normal(loc, 0.05, size=(10, 4)) for loc in (0.0, 5.0, 10.0)
        ])
        spread = analyse_selection(pool, [0, 10, 20], n_clusters=3, seed=0)
        clustered = analyse_selection(pool, [0, 1, 2], n_clusters=3, seed=0)
        assert spread["coverage"] > clustered["coverage"]
        assert spread["mean_pairwise_distance"] > clustered["mean_pairwise_distance"]


# ---------------------------------------------------------------------------
# Forgetting
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_detects_degradation_on_a_prior_domain(self):
        tracker = ForgettingTracker(domains=["source", "target"])
        tracker.record(0, {"source": 0.80, "target": 0.20})
        tracker.record(1, {"source": 0.70, "target": 0.50})
        tracker.record(2, {"source": 0.60, "target": 0.65})
        summary = tracker.summary(adapted_domain="target", source_domain="source")

        assert summary["backward_transfer"] < 0, "losing 0.20 on source is forgetting"
        assert summary["forgetting_measure"] == pytest.approx(0.20)
        assert summary["source_retention"] == pytest.approx(0.75)
        assert summary["plasticity"] == pytest.approx(0.45)

    def test_stable_prior_domain_shows_no_forgetting(self):
        tracker = ForgettingTracker(domains=["source", "target"])
        tracker.record(0, {"source": 0.80, "target": 0.20})
        tracker.record(1, {"source": 0.80, "target": 0.60})
        summary = tracker.summary(adapted_domain="target", source_domain="source")
        assert summary["forgetting_measure"] == pytest.approx(0.0)
        assert summary["source_retention"] == pytest.approx(1.0)

    def test_missing_evaluations_are_skipped_not_counted_as_zero(self):
        # A domain evaluated on a coarser schedule must not look like a
        # catastrophic drop to 0.
        tracker = ForgettingTracker(domains=["a"])
        tracker.record(0, {"a": 0.8})
        tracker.record(1, {})
        tracker.record(2, {"a": 0.8})
        summary = tracker.summary()
        assert summary["forgetting_measure"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Campaign expansion
# ---------------------------------------------------------------------------

class TestCampaign:
    @pytest.fixture(scope="class")
    def spec(self):
        from src.campaign import load_campaign

        return load_campaign("configs/campaign.yaml")

    def test_expansion_is_deterministic(self, spec):
        from src.campaign import expand_blocks

        first = [r.run_id for r in expand_blocks(spec)]
        second = [r.run_id for r in expand_blocks(spec)]
        assert first == second, "resume depends on stable run ids"

    def test_run_ids_are_unique(self, spec):
        from src.campaign import expand_blocks

        ids = [r.run_id for r in expand_blocks(spec)]
        assert len(ids) == len(set(ids)), "a collision would overwrite a result"

    def test_grid_size_matches_the_declaration(self, spec):
        from src.campaign import expand_blocks

        block = spec.block("E1_main_grid")
        expected = (
            (1 + len(block.get("also_on", [])))
            * len(block["strategies"])
            * len(block["l0_sizes"])
            * len(block["budgets"])
            * block["n_seeds"]
        )
        assert len(expand_blocks(spec, ["E1_main_grid"])) == expected

    def test_seeds_are_shared_across_strategies(self, spec):
        from src.campaign import expand_blocks

        runs = expand_blocks(spec, ["E1_main_grid"])
        by_strategy = {}
        for run in runs:
            if run.params.get("l0_size") == 100 and run.params.get("budget") == 50:
                by_strategy.setdefault(run.params["strategy"], set()).add(run.seed)
        seed_sets = list(by_strategy.values())
        assert all(s == seed_sets[0] for s in seed_sets), (
            "paired statistical tests require identical seeds per strategy"
        )

    def test_fingerprint_changes_when_a_parameter_changes(self, spec):
        from src.campaign import expand_blocks

        run = expand_blocks(spec, ["E1_main_grid"])[0]
        before = run.fingerprint()
        run.params["budget"] = 999
        assert run.fingerprint() != before, (
            "resume must re-run a config that changed under the same id"
        )

    def test_every_block_has_a_rationale_and_addresses(self, spec):
        # The matrix document is generated from these fields; an empty one
        # would produce a silently incomplete mapping.
        for block in spec.blocks:
            assert block.get("addresses"), f"{block['id']} addresses nothing"
            assert block.get("rationale", "").strip(), f"{block['id']} has no rationale"


class TestMockBackend:
    def test_produces_every_metric_the_tables_consume(self):
        from src.campaign import RunSpec, _mock_backend

        run = RunSpec(
            run_id="t", block_id="E1_main_grid", kind="al_run", seed=42,
            params={
                "strategy": "bald_diversity", "l0_size": 100, "budget": 50,
                "cycles": 8, "collect_diversity_metrics": True,
                "collect_profiling": True,
            },
        )
        result = _mock_backend(run)
        for key in ("auc", "auc_normalized", "final_mAP50", "final_recall",
                    "recall_variance", "trajectory", "profiling"):
            assert key in result, f"mock is missing {key}"
        assert result["mock"] is True, "mock output must be self-identifying"
        assert len(result["trajectory"]) == 9
        assert 0.0 <= result["auc_normalized"] <= 1.0
