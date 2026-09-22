#!/usr/bin/env python3
"""
Single entry point for the whole experimental campaign.

    python run_campaign.py plan                      # what would run, and how much
    python run_campaign.py run --block E1_main_grid  # run one block
    python run_campaign.py run --all                 # everything, by priority
    python run_campaign.py run --block E1_main_grid --mock   # validate wiring
    python run_campaign.py status                    # progress per block
    python run_campaign.py report                    # aggregate, test, tabulate
    python run_campaign.py matrix                    # board item -> experiment map

Every block is resumable: re-running skips what already completed, so an
interrupted campaign costs only the remainder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.campaign import CampaignRunner, expand_blocks, load_campaign


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_minutes(run) -> float:
    """Rough wall-clock estimate per run on a single modern GPU.

    Deliberately crude: it exists so `plan` can answer "is this three
    hours or three weeks" before committing the machine, not to be
    accurate. The constants come from the timings in the previous
    campaign's result files (initial training dominates, then cycles).
    """
    params = run.params
    cycles = int(params.get("cycles", 8))
    l0 = int(params.get("l0_size", 100))
    budget = int(params.get("budget", 50))
    epochs_init = int(params.get("epochs_initial", 30))
    epochs_cycle = int(params.get("epochs_per_cycle", 5))

    initial = 0.015 * epochs_init * max(l0, 25) / 100
    per_cycle = 0.012 * epochs_cycle * (l0 + budget * cycles) / 100
    scoring = 0.3 * cycles * (1 if "bald" in str(params.get("strategy")) else 0.15)
    total = initial + per_cycle * cycles + scoring

    if params.get("method") == "deep_ensemble" or params.get("estimator") == "deep_ensemble":
        total *= int(params.get("ensemble_members", 5))
    if run.kind in ("shift_run", "shift_baseline_run"):
        total *= 1.2  # two workspaces, two evaluations per cycle
    if params.get("collect_per_domain_eval"):
        total *= 1.4
    return max(total, 1.0)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_plan(args) -> int:
    spec = load_campaign(args.config)
    blocks = args.block or None
    runs = expand_blocks(spec, blocks)

    by_block = {}
    for run in runs:
        entry = by_block.setdefault(run.block_id, {"n": 0, "minutes": 0.0, "priority": run.priority})
        entry["n"] += 1
        entry["minutes"] += estimate_minutes(run)

    print(f"\nCampaign: {spec.name}  ({spec.path})")
    print(f"{'block':<28}{'runs':>7}{'est. hours':>13}{'priority':>10}")
    print("-" * 58)
    total_runs = total_hours = 0
    for block_id, entry in sorted(by_block.items(), key=lambda kv: kv[1]["priority"]):
        hours = entry["minutes"] / 60
        total_runs += entry["n"]
        total_hours += hours
        print(f"{block_id:<28}{entry['n']:>7}{hours:>13.1f}{entry['priority']:>10}")
    print("-" * 58)
    print(f"{'TOTAL':<28}{total_runs:>7}{total_hours:>13.1f}")
    print(
        "\nEstimates assume one GPU and no parallelism. Run the priority-1 "
        "blocks first: they cover H1-H4 and every board item that blocks the "
        "defence.\n"
    )
    return 0


def cmd_run(args) -> int:
    spec = load_campaign(args.config)
    blocks = args.block or None
    if not blocks and not args.all:
        print(
            "Refusing to run the entire campaign implicitly.\n"
            "Pass --block <id> for one block, or --all to run everything.\n"
            "Use `plan` first to see the cost."
        )
        return 2

    runs = expand_blocks(spec, blocks)
    if not runs:
        print(f"No runs matched {blocks}. Known blocks: "
              f"{[b['id'] for b in spec.blocks]}")
        return 2

    runner = CampaignRunner(
        spec,
        backend="mock" if args.mock else "real",
        output_root=args.output,
        dry_run=args.dry_run,
        continue_on_error=not args.fail_fast,
    )
    summary = runner.execute(runs, force=args.force, limit=args.limit)
    return 0 if summary.get("failed", 0) == 0 else 1


def cmd_status(args) -> int:
    spec = load_campaign(args.config)
    runs = expand_blocks(spec, args.block or None)
    runner = CampaignRunner(spec, backend="mock", output_root=args.output)
    status = runner.status(runs)

    print(f"\n{'block':<28}{'done':>8}{'failed':>9}{'pending':>10}{'total':>8}")
    print("-" * 63)
    for block_id, counts in sorted(status.items()):
        print(
            f"{block_id:<28}{counts['completed']:>8}{counts['failed']:>9}"
            f"{counts['pending']:>10}{counts['total']:>8}"
        )
    print()
    incomplete = [b for b, c in status.items() if c["pending"] or c["failed"]]
    if incomplete:
        print("Not finished: " + ", ".join(sorted(incomplete)))
    else:
        print("All planned runs are complete.")
    return 0


def cmd_report(args) -> int:
    from scripts.make_tables import build_all_tables
    from scripts.make_figures import build_all_figures

    spec = load_campaign(args.config)
    out_tables = Path(args.tables)
    out_figures = Path(args.figures)

    written = build_all_tables(spec, results_root=args.output, out_dir=out_tables)
    print(f"\n[report] {len(written)} table file(s) written to {out_tables}")

    if not args.no_figures:
        figures = build_all_figures(spec, results_root=args.output, out_dir=out_figures)
        print(f"[report] {len(figures)} figure(s) written to {out_figures}")

    write_matrix(spec, Path("docs/EXPERIMENT_MATRIX.md"))
    print("[report] docs/EXPERIMENT_MATRIX.md refreshed")
    return 0


def write_matrix(spec, path: Path) -> None:
    """Regenerate the board-item -> experiment mapping from the config.

    Generated rather than hand-written so it cannot drift from the
    experiments that actually run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment matrix",
        "",
        "Generated by `python run_campaign.py report`. Do not edit by hand:",
        "the source of truth is `configs/campaign.yaml`.",
        "",
        "Each row maps an item from the examination board's revision plan",
        "(*Planejamento de ajustes na tese*, May 2026) to the experiment that",
        "answers it.",
        "",
    ]
    for block in spec.blocks:
        runs = expand_blocks(spec, [block["id"]])
        lines += [
            f"## {block['id']} — {block.get('title', '')}",
            "",
            f"**Kind:** `{block.get('kind')}`  ",
            f"**Planned runs:** {len(runs)}  ",
            f"**Priority:** {block.get('priority', 5)}",
            "",
            "**Addresses:**",
            "",
        ]
        for item in block.get("addresses", []):
            lines.append(f"- {item}")
        lines += ["", "**Why this experiment:**", "", block.get("rationale", "").strip(), ""]
    path.write_text("\n".join(lines))


def cmd_matrix(args) -> int:
    spec = load_campaign(args.config)
    write_matrix(spec, Path(args.out))
    print(f"Wrote {args.out}")
    return 0


def cmd_list(args) -> int:
    spec = load_campaign(args.config)
    runs = expand_blocks(spec, args.block or None)
    for run in runs:
        print(run.run_id)
    print(f"\n{len(runs)} run(s)")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental campaign orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/campaign.yaml")
    parser.add_argument("--output", default="results")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="show planned runs and cost estimate")
    p_plan.add_argument("--block", nargs="+")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="execute runs")
    p_run.add_argument("--block", nargs="+")
    p_run.add_argument("--all", action="store_true", help="run every block")
    p_run.add_argument("--mock", action="store_true",
                       help="synthetic backend: validates wiring without a GPU")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--force", action="store_true",
                       help="re-run runs already marked complete")
    p_run.add_argument("--limit", type=int, help="stop after N runs")
    p_run.add_argument("--fail-fast", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="progress per block")
    p_status.add_argument("--block", nargs="+")
    p_status.set_defaults(func=cmd_status)

    p_report = sub.add_parser("report", help="aggregate, test and tabulate")
    p_report.add_argument("--tables", default="tables")
    p_report.add_argument("--figures", default="figures")
    p_report.add_argument("--no-figures", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_matrix = sub.add_parser("matrix", help="regenerate the experiment matrix doc")
    p_matrix.add_argument("--out", default="docs/EXPERIMENT_MATRIX.md")
    p_matrix.set_defaults(func=cmd_matrix)

    p_list = sub.add_parser("list", help="list run ids")
    p_list.add_argument("--block", nargs="+")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
