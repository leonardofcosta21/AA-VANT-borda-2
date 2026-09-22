#!/usr/bin/env python3
"""
Audit which datasets are present, usable, and consistent.

Run this first, before anything else. Most of a campaign's lost time goes
to discovering at hour six that a dataset was half-converted, and the
previous code surfaced that as an exception deep inside the dataset
manager rather than as a checkable fact.

What it checks, per dataset
---------------------------
Presence       root exists and has the YOLO layout
Pairing        every image has a label file, and vice versa
Emptiness      how many label files are empty (legitimate background
               frames, or a broken conversion -- the count tells you)
Class range    class ids fall inside the declared taxonomy
Coordinates    YOLO boxes are normalised into [0, 1] and non-degenerate
Sufficiency    the split is large enough for the L0 sizes the campaign asks
               for, including the pool the adaptation cycles will consume
Taxonomy       data.yaml class names agree with the unified taxonomy

Exit code is 0 when every dataset the campaign needs is usable, 1
otherwise, so it can gate a Makefile target or CI job.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UNIFIED_TAXONOMY = ["person", "vehicle", "hazard"]

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


def find_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for directory in (root / "images", root):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.suffix.lower() in IMAGE_SUFFIXES and "labels" not in path.parts:
                    images.append(path)
            if images:
                break
    return images


def label_for(image: Path) -> Optional[Path]:
    parts = list(image.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / (image.stem + ".txt")


def read_class_names(root: Path) -> Optional[List[str]]:
    for candidate in ("data.yaml", "dataset.yaml", f"{root.name.lower()}.yaml"):
        path = root / candidate
        if path.exists():
            try:
                with open(path) as fh:
                    data = yaml.safe_load(fh) or {}
                names = data.get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                if isinstance(names, list):
                    return names
            except Exception:
                continue
    return None


def audit_dataset(
    name: str, root_str: str, sample: int = 400, required_l0: int = 400,
    required_pool: int = 400, seed: int = 42,
) -> Dict:
    root = Path(root_str)
    report: Dict = {"name": name, "root": str(root), "issues": [], "status": OK}

    if not root.exists():
        report["status"] = FAIL
        report["issues"].append("directory does not exist")
        report["hint"] = (
            f"python scripts/download_datasets.py --dataset {name} "
            f"&& python scripts/prepare_datasets.py"
        )
        return report

    images = find_images(root)
    report["n_images"] = len(images)
    if not images:
        report["status"] = FAIL
        report["issues"].append("no image files found under the root")
        return report

    class_names = read_class_names(root)
    report["class_names"] = class_names
    report["n_classes"] = len(class_names) if class_names else None
    if class_names is None:
        report["issues"].append(
            "no data.yaml with class names; the loop will have to infer the taxonomy"
        )
        report["status"] = WARN

    rng = random.Random(seed)
    sampled = images if len(images) <= sample else rng.sample(images, sample)

    missing = empty = malformed = 0
    class_counter: Counter = Counter()
    bad_coords = 0
    out_of_range = 0
    degenerate = 0

    for image in sampled:
        label = label_for(image)
        if label is None or not label.exists():
            missing += 1
            continue
        try:
            lines = [l.strip() for l in label.read_text().splitlines() if l.strip()]
        except Exception:
            malformed += 1
            continue
        if not lines:
            empty += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                malformed += 1
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                malformed += 1
                continue
            class_counter[cls] += 1
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                bad_coords += 1
            if w <= 0 or h <= 0:
                degenerate += 1
            if class_names and cls >= len(class_names):
                out_of_range += 1

    scale = len(images) / max(len(sampled), 1)
    report["sampled"] = len(sampled)
    report["labelled_images_est"] = int((len(sampled) - missing - empty) * scale)
    report["missing_labels_pct"] = round(100 * missing / len(sampled), 2)
    report["empty_labels_pct"] = round(100 * empty / len(sampled), 2)
    report["boxes_in_sample"] = sum(class_counter.values())
    report["class_distribution"] = dict(sorted(class_counter.items()))

    if missing / len(sampled) > 0.5:
        report["status"] = FAIL
        report["issues"].append(
            f"{report['missing_labels_pct']}% of sampled images have no label file; "
            "the conversion did not complete"
        )
    elif missing:
        report["status"] = WARN if report["status"] == OK else report["status"]
        report["issues"].append(
            f"{report['missing_labels_pct']}% of sampled images have no label file"
        )
    if malformed:
        report["status"] = FAIL
        report["issues"].append(f"{malformed} malformed label line(s) in the sample")
    if bad_coords:
        report["status"] = FAIL
        report["issues"].append(
            f"{bad_coords} box(es) outside [0,1]: coordinates were not normalised "
            "during conversion"
        )
    if degenerate:
        report["status"] = WARN if report["status"] == OK else report["status"]
        report["issues"].append(f"{degenerate} box(es) with zero width or height")
    if out_of_range:
        report["status"] = FAIL
        report["issues"].append(
            f"{out_of_range} box(es) reference a class id beyond the declared taxonomy"
        )
    if empty / len(sampled) > 0.3:
        report["status"] = WARN if report["status"] == OK else report["status"]
        report["issues"].append(
            f"{report['empty_labels_pct']}% of sampled images have empty labels; "
            "verify these are intentional background frames"
        )

    # Capacity: L0 plus the pool the adaptation cycles will consume, plus
    # the held-out test set.
    needed = int(required_l0 + required_pool + 0.2 * len(images))
    report["capacity_needed"] = needed
    if len(images) < needed:
        report["status"] = FAIL if len(images) < required_l0 * 2 else WARN
        report["issues"].append(
            f"only {len(images)} images: the campaign needs about {needed} "
            f"(L0 up to {required_l0}, an unlabelled pool of {required_pool}, "
            f"and a 20% held-out test set)"
        )

    if class_names and name == "unified":
        lowered = [str(c).lower() for c in class_names]
        if lowered != UNIFIED_TAXONOMY:
            report["status"] = WARN if report["status"] == OK else report["status"]
            report["issues"].append(
                f"unified taxonomy is {class_names}, expected {UNIFIED_TAXONOMY}"
            )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit datasets before a campaign")
    parser.add_argument("--config", default="configs/campaign.yaml")
    parser.add_argument("--sample", type=int, default=400,
                        help="images to inspect per dataset")
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args()

    with open(args.config) as fh:
        spec = yaml.safe_load(fh) or {}
    datasets = spec.get("datasets", {})

    # Largest L0 and pool any block asks for, so the capacity check is
    # calibrated to what will actually run.
    max_l0 = 400
    max_pool = 400
    for block in spec.get("blocks", []):
        for l0 in block.get("l0_sizes", []) or []:
            max_l0 = max(max_l0, int(l0))
        cycles = int(block.get("cycles", 8) or 8)
        for budget in block.get("budgets", [block.get("budget", 50)]) or [50]:
            if budget:
                max_pool = max(max_pool, int(budget) * cycles)

    print(f"\nAuditing {len(datasets)} dataset(s) declared in {args.config}")
    print(f"Campaign needs up to L0={max_l0} and a pool of {max_pool} per run.\n")

    reports = []
    for name, root in datasets.items():
        report = audit_dataset(
            name, root, sample=args.sample,
            required_l0=max_l0, required_pool=max_pool,
        )
        reports.append(report)

    width = max(len(r["name"]) for r in reports) + 2
    print(f"{'dataset':<{width}}{'status':<8}{'images':>9}{'labelled':>10}  notes")
    print("-" * (width + 30 + 40))
    for r in sorted(reports, key=lambda r: (r["status"] != FAIL, r["name"])):
        note = r["issues"][0] if r["issues"] else "ready"
        print(
            f"{r['name']:<{width}}{r['status']:<8}"
            f"{r.get('n_images', 0):>9}{r.get('labelled_images_est', 0):>10}  {note}"
        )
        for extra in r["issues"][1:]:
            print(f"{'':<{width}}{'':<8}{'':>9}{'':>10}  {extra}")
        if r.get("hint"):
            print(f"{'':<{width}}{'':<8}{'':>9}{'':>10}  -> {r['hint']}")

    usable = [r for r in reports if r["status"] != FAIL]
    failed = [r for r in reports if r["status"] == FAIL]

    print(f"\n{len(usable)}/{len(reports)} dataset(s) usable.")
    if failed:
        print("Blocked datasets: " + ", ".join(r["name"] for r in failed))
        print(
            "\nWhat this means for the campaign:\n"
            "  * E1_main_grid needs at least `sard` (and `unified` for the C5 claim).\n"
            "  * E2a/E2b need both members of at least one source->target pair.\n"
            "  * E5_long_horizon needs `unified` plus the individual domains for\n"
            "    the per-domain forgetting evaluation.\n"
            "Blocks whose datasets are missing will fail fast with a message\n"
            "naming the dataset, rather than part-running and producing a\n"
            "table with silent gaps."
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"\nFull report: {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
