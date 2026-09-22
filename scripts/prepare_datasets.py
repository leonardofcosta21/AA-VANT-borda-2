#!/usr/bin/env python3
"""
prepare_datasets.py
===================
Converte os datasets da tese para formato YOLO padronizado e
constrói o dataset unificado (3 superclasses: person, vehicle, hazard).

Estruturas detectadas:
  SARD     → datasets/SARD/Sard.v17-tiles_3x3.yolov8/{train,valid,test}/{images,labels}/
  VisDrone → datasets/VisDrone/VisDrone2019-DET-{train,val} (3)/{images,annotations}/
  UAVDT    → datasets/UAVDT/uavdt-DatasetNinja/{train,test}/{img,ann}/*.json
  FloodNet → datasets/FloodNet/FloodNet-Supervised_v1.0/{train,val,test}/{*-org-img,*-label-img}/
  AIDER    → datasets/AIDER/AIDER/{collapsed_building,fire,flooded_areas,normal,traffic_incident}/
  LADI     → vazio (sem imagens ainda)

Uso:
    python prepare_datasets.py --datasets_dir ./datasets
    python prepare_datasets.py --datasets_dir ./datasets --skip_unified
"""

import os, sys, json, shutil, struct, argparse, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import yaml

# ---------------------------------------------------------------------------
# Taxonomia unificada
# ---------------------------------------------------------------------------
UNIFIED_CLASSES = ["person", "vehicle", "hazard"]

# Mapeamento classe_original → superclasse unificada
# SARD: 0=person, 1=vehicle (já YOLO, só remap)
SARD_MAP = {0: 0, 1: 1}

# VisDrone: 0=pedestrian,1=person,2=bicycle,3=car,4=van,
#           5=truck,6=tricycle,7=awning-tricycle,8=bus,9=motor
VISDRONE_MAP = {0:0, 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}

# UAVDT (DatasetNinja classTitle): car→vehicle, bus→vehicle, truck→vehicle
UAVDT_TITLE_MAP = {"car": 1, "bus": 1, "truck": 1}

# FloodNet: pixel value → superclasse
# 0=background, 1=building-flooded, 2=building-non-flooded,
# 3=road-flooded, 4=road-non-flooded, 5=water,
# 6=tree, 7=vehicle, 8=pool, 9=grass
FLOODNET_MAP = {1:2, 2:2, 3:2, 4:2, 5:2, 7:1, 8:2}
# 0,6,9 descartados (background, tree, grass)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_yolo_label(path: Path, anns: List[Tuple]):
    with open(path, "w") as f:
        for a in anns:
            f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")

def write_data_yaml(dst: Path, class_names: List[str]):
    data = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(dst / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def get_img_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if HAS_PIL:
        try:
            with Image.open(str(path)) as im:
                return im.size  # (W, H)
        except Exception:
            pass
    if HAS_CV2:
        im = cv2.imread(str(path))
        if im is not None:
            return im.shape[1], im.shape[0]
    # PNG header
    try:
        with open(path, "rb") as f:
            h = f.read(24)
        if h[:8] == b"\x89PNG\r\n\x1a\n":
            W = struct.unpack(">I", h[16:20])[0]
            H = struct.unpack(">I", h[20:24])[0]
            return W, H
    except Exception:
        pass
    return None, None

def copy_file(src: Path, dst: Path):
    if not dst.exists():
        shutil.copy2(str(src), str(dst))

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return (
        min(max(cx, 0), 1), min(max(cy, 0), 1),
        min(max(w,  0), 1), min(max(h,  0), 1),
    )

def xywh_to_yolo(x, y, w, h, W, H):
    return xyxy_to_yolo(x, y, x+w, y+h, W, H)

# ---------------------------------------------------------------------------
# 1. SARD  (já está em formato YOLO)
# ---------------------------------------------------------------------------

def prepare_sard(datasets_dir: Path) -> Optional[Path]:
    raw = datasets_dir / "SARD"
    dst = datasets_dir / "_prepared" / "SARD"

    # Encontrar a subpasta do Roboflow (qualquer nome)
    candidates = [d for d in raw.iterdir() if d.is_dir()] if raw.exists() else []
    if not candidates:
        print("  [SARD] Pasta não encontrada — pulando.")
        return None
    src = candidates[0]
    print(f"  [SARD] Fonte: {src.name}")

    split_map = {"train": "train", "valid": "val", "test": "val"}
    count = {"train": 0, "val": 0}

    for src_split, dst_split in split_map.items():
        img_dir = src / src_split / "images"
        lbl_dir = src / src_split / "labels"
        if not img_dir.exists():
            continue
        dst_img = mkdir(dst / "images" / dst_split)
        dst_lbl = mkdir(dst / "labels" / dst_split)
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            copy_file(img, dst_img / img.name)
            if lbl.exists():
                # Remap classes
                lines = lbl.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    new_cls = SARD_MAP.get(cls, cls)
                    new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
                (dst_lbl / (img.stem + ".txt")).write_text("\n".join(new_lines) + "\n")
            else:
                (dst_lbl / (img.stem + ".txt")).write_text("")
            count[dst_split] += 1

    write_data_yaml(dst, UNIFIED_CLASSES)
    print(f"  [SARD] train={count['train']} val={count['val']} → {dst}")
    return dst

# ---------------------------------------------------------------------------
# 2. VisDrone-DET
# ---------------------------------------------------------------------------

def prepare_visdrone(datasets_dir: Path) -> Optional[Path]:
    raw = datasets_dir / "VisDrone"
    dst = datasets_dir / "_prepared" / "VisDrone"

    if not raw.exists():
        print("  [VisDrone] Pasta não encontrada — pulando.")
        return None

    # Mapear splits: pegar a versão com annotations (3)
    # train → "VisDrone2019-DET-train (3)"
    # val   → "VisDrone2019-DET-val"
    # test  → "VisDrone2019-DET-test-dev (3)"
    split_candidates = {
        "train": ["VisDrone2019-DET-train (3)", "VisDrone2019-DET-train"],
        "val":   ["VisDrone2019-DET-val",        "VisDrone2019-DET-val (3)"],
    }

    count = {"train": 0, "val": 0}

    for dst_split, names in split_candidates.items():
        src_dir = None
        for name in names:
            candidate = raw / name
            if candidate.exists() and (candidate / "annotations").exists():
                src_dir = candidate
                break
        if src_dir is None:
            # Tentar qualquer pasta com annotations
            for d in raw.iterdir():
                if d.is_dir() and dst_split.replace("val","val") in d.name.lower() \
                        and (d / "annotations").exists():
                    src_dir = d
                    break
        if src_dir is None:
            print(f"  [VisDrone] Split '{dst_split}' não encontrado.")
            continue

        img_dir = src_dir / "images"
        ann_dir = src_dir / "annotations"
        print(f"  [VisDrone] {dst_split} ← {src_dir.name}")

        dst_img = mkdir(dst / "images" / dst_split)
        dst_lbl = mkdir(dst / "labels" / dst_split)

        for ann_file in ann_dir.glob("*.txt"):
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                c = img_dir / (ann_file.stem + ext)
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                continue

            W, H = get_img_size(img_path)
            if W is None:
                continue

            copy_file(img_path, dst_img / img_path.name)

            anns = []
            for line in ann_file.read_text().strip().splitlines():
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cat = int(parts[5]) - 1  # VisDrone é 1-indexed
                if w <= 0 or h <= 0:
                    continue
                new_cls = VISDRONE_MAP.get(cat)
                if new_cls is None:
                    continue
                cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, W, H)
                anns.append((new_cls, cx, cy, nw, nh))

            write_yolo_label(dst_lbl / (ann_file.stem + ".txt"), anns)
            count[dst_split] += 1

    write_data_yaml(dst, UNIFIED_CLASSES)
    print(f"  [VisDrone] train={count['train']} val={count['val']} → {dst}")
    return dst

# ---------------------------------------------------------------------------
# 3. UAVDT  (DatasetNinja JSON)
# ---------------------------------------------------------------------------

def prepare_uavdt(datasets_dir: Path) -> Optional[Path]:
    raw = datasets_dir / "UAVDT"
    dst = datasets_dir / "_prepared" / "UAVDT"

    ninja_dir = raw / "uavdt-DatasetNinja"
    if not ninja_dir.exists():
        print("  [UAVDT] Pasta não encontrada — pulando.")
        return None

    split_map = {"train": "train", "test": "val"}
    count = {"train": 0, "val": 0}

    for src_split, dst_split in split_map.items():
        img_dir = ninja_dir / src_split / "img"
        ann_dir = ninja_dir / src_split / "ann"
        if not img_dir.exists():
            continue

        dst_img = mkdir(dst / "images" / dst_split)
        dst_lbl = mkdir(dst / "labels" / dst_split)

        for ann_file in ann_dir.glob("*.json"):
            img_name = ann_file.stem  # e.g. "M0101_img000001.jpg"
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            with open(ann_file) as f:
                data = json.load(f)

            W = data["size"]["width"]
            H = data["size"]["height"]

            anns = []
            for obj in data.get("objects", []):
                title = obj.get("classTitle", "").lower()
                new_cls = UAVDT_TITLE_MAP.get(title)
                if new_cls is None:
                    continue
                pts = obj["points"]["exterior"]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                if x2 <= x1 or y2 <= y1:
                    continue
                cx, cy, nw, nh = xyxy_to_yolo(x1, y1, x2, y2, W, H)
                anns.append((new_cls, cx, cy, nw, nh))

            copy_file(img_path, dst_img / img_name)
            write_yolo_label(dst_lbl / (Path(img_name).stem + ".txt"), anns)
            count[dst_split] += 1

    write_data_yaml(dst, UNIFIED_CLASSES)
    print(f"  [UAVDT] train={count['train']} val={count['val']} → {dst}")
    return dst

# ---------------------------------------------------------------------------
# 4. FloodNet  (segmentação → bboxes por contorno)
# ---------------------------------------------------------------------------

def prepare_floodnet(datasets_dir: Path) -> Optional[Path]:
    raw = datasets_dir / "FloodNet" / "FloodNet-Supervised_v1.0"
    dst = datasets_dir / "_prepared" / "FloodNet"

    if not raw.exists():
        print("  [FloodNet] Pasta não encontrada — pulando.")
        return None

    if not HAS_CV2:
        print("  [FloodNet] opencv-python não instalado.")
        print("             Execute: pip install opencv-python")
        print("             Sem ele as máscaras não são convertidas em bboxes.")
        return None

    split_map = {
        "train": ("train-org-img", "train-label-img"),
        "val":   ("val-org-img",   "val-label-img"),
        "test":  ("test-org-img",  "test-label-img"),
    }
    dst_split_map = {"train": "train", "val": "val", "test": "val"}
    count = {"train": 0, "val": 0}

    for src_split, (img_folder, lbl_folder) in split_map.items():
        img_dir = raw / src_split / img_folder
        lbl_dir = raw / src_split / lbl_folder
        if not img_dir.exists():
            continue

        dst_split = dst_split_map[src_split]
        dst_img = mkdir(dst / "images" / dst_split)
        dst_lbl = mkdir(dst / "labels" / dst_split)

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            # Máscara: nome_lab.png
            mask_path = lbl_dir / (img_path.stem + "_lab.png")
            if not mask_path.exists():
                mask_path = lbl_dir / (img_path.stem + ".png")

            copy_file(img_path, dst_img / img_path.name)

            anns = []
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    H, W = mask.shape
                    for pixel_val, new_cls in FLOODNET_MAP.items():
                        binary = ((mask == pixel_val) * 255).astype("uint8")
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w * h < 200:  # descarta objetos minúsculos
                                continue
                            cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, W, H)
                            anns.append((new_cls, cx, cy, nw, nh))

            write_yolo_label(dst_lbl / (img_path.stem + ".txt"), anns)
            count[dst_split] += 1

    write_data_yaml(dst, UNIFIED_CLASSES)
    print(f"  [FloodNet] train={count['train']} val={count['val']} → {dst}")
    return dst

# ---------------------------------------------------------------------------
# 5. AIDER  (sem bbox → pool não-anotado)
# ---------------------------------------------------------------------------

def prepare_aider(datasets_dir: Path) -> Optional[Path]:
    raw = datasets_dir / "AIDER" / "AIDER"
    dst = datasets_dir / "_prepared" / "AIDER"

    if not raw.exists():
        print("  [AIDER] Pasta não encontrada — pulando.")
        return None

    unlab = mkdir(dst / "unlabelled" / "images")
    cats = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
    count = 0

    for cat in cats:
        cat_dir = raw / cat
        if not cat_dir.exists():
            continue
        for img in cat_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                copy_file(img, unlab / f"{cat}_{img.name}")
                count += 1

    (dst / "README.txt").write_text(
        "AIDER — sem bounding boxes.\n"
        f"Pool não-anotado: {count} imagens em unlabelled/images/\n"
    )
    print(f"  [AIDER] {count} imagens não-anotadas → {dst}")
    return dst

# ---------------------------------------------------------------------------
# 6. Dataset Unificado
# ---------------------------------------------------------------------------

def build_unified(datasets_dir: Path, val_frac: float = 0.2, seed: int = 42):
    prepared = datasets_dir / "_prepared"
    unified  = datasets_dir / "unified"

    dst_img_t = mkdir(unified / "images"     / "train")
    dst_lbl_t = mkdir(unified / "labels"     / "train")
    dst_img_v = mkdir(unified / "images"     / "val")
    dst_lbl_v = mkdir(unified / "labels"     / "val")
    dst_unlab = mkdir(unified / "unlabelled" / "images")

    sources_bbox = ["SARD", "VisDrone", "UAVDT", "FloodNet"]
    all_items = []  # (img_path, lbl_path, src_name)

    for src in sources_bbox:
        src_dir = prepared / src
        if not src_dir.exists():
            print(f"  [unified] {src} não preparado — pulando.")
            continue
        for split in ["train", "val"]:
            img_d = src_dir / "images" / split
            lbl_d = src_dir / "labels" / split
            if not img_d.exists():
                continue
            for img in img_d.iterdir():
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = lbl_d / (img.stem + ".txt")
                all_items.append((img, lbl, src))

    # Shuffle e split
    rng = random.Random(seed)
    rng.shuffle(all_items)
    n_val = max(50, int(len(all_items) * val_frac))
    val_items   = all_items[:n_val]
    train_items = all_items[n_val:]

    def process(items, dst_img, dst_lbl):
        for img_path, lbl_path, src in items:
            prefix = f"{src}_{img_path.name}"
            copy_file(img_path, dst_img / prefix)
            lbl_dst = dst_lbl / (Path(prefix).stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(lbl_dst))
            else:
                lbl_dst.write_text("")

    process(train_items, dst_img_t, dst_lbl_t)
    process(val_items,   dst_img_v, dst_lbl_v)

    # Pool não-anotado (AIDER)
    n_unlab = 0
    aider_unlab = prepared / "AIDER" / "unlabelled" / "images"
    if aider_unlab.exists():
        for img in aider_unlab.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                copy_file(img, dst_unlab / f"AIDER_{img.name}")
                n_unlab += 1

    write_data_yaml(unified, UNIFIED_CLASSES)

    meta = {
        "classes": UNIFIED_CLASSES,
        "split": {
            "train": len(train_items),
            "val":   len(val_items),
            "unlabelled": n_unlab,
        },
        "sources": sources_bbox,
    }
    with open(unified / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [unified] train={len(train_items)} | val={len(val_items)} | "
          f"unlabelled={n_unlab}")
    print(f"  [unified] Salvo em: {unified}")
    print(f"  [unified] data.yaml: {unified / 'data.yaml'}")
    return unified

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------

def print_summary(datasets_dir: Path):
    print("\n" + "="*65)
    print("  RESUMO")
    print("="*65)
    prepared = datasets_dir / "_prepared"
    for name in ["SARD", "VisDrone", "UAVDT", "FloodNet", "AIDER"]:
        d = prepared / name
        if not d.exists():
            print(f"  {name:<12} ❌  não preparado")
            continue
        n_t = len(list((d/"images"/"train").glob("*"))) if (d/"images"/"train").exists() else 0
        n_v = len(list((d/"images"/"val").glob("*")))   if (d/"images"/"val").exists()   else 0
        n_u = len(list((d/"unlabelled"/"images").glob("*"))) if (d/"unlabelled"/"images").exists() else 0
        print(f"  {name:<12} ✅  train={n_t} | val={n_v} | unlabelled={n_u}")

    u = datasets_dir / "unified"
    if u.exists():
        n_t = len(list((u/"images"/"train").glob("*")))
        n_v = len(list((u/"images"/"val").glob("*")))
        n_u = len(list((u/"unlabelled"/"images").glob("*"))) if (u/"unlabelled"/"images").exists() else 0
        print(f"  {'unified':<12} ✅  train={n_t} | val={n_v} | unlabelled={n_u}")
    print("="*65)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepara datasets para formato YOLO unificado"
    )
    parser.add_argument("--datasets_dir", default="./datasets",
                        help="Pasta raiz dos datasets (default: ./datasets)")
    parser.add_argument("--skip_unified", action="store_true",
                        help="Não construir o dataset unificado")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+",
                        choices=["sard","visdrone","uavdt","floodnet","aider","unified"],
                        help="Preparar apenas datasets específicos")
    args = parser.parse_args()

    base = Path(args.datasets_dir)
    only = set(args.only) if args.only else None

    print(f"\nDiretório base: {base.resolve()}")
    print(f"opencv disponível: {HAS_CV2} (necessário para FloodNet)")
    print(f"Pillow disponível: {HAS_PIL}")
    print()

    if only is None or "sard"     in only:
        print("[1/5] SARD")
        prepare_sard(base)

    if only is None or "visdrone" in only:
        print("[2/5] VisDrone")
        prepare_visdrone(base)

    if only is None or "uavdt"    in only:
        print("[3/5] UAVDT")
        prepare_uavdt(base)

    if only is None or "floodnet" in only:
        print("[4/5] FloodNet")
        prepare_floodnet(base)

    if only is None or "aider"    in only:
        print("[5/5] AIDER")
        prepare_aider(base)

    if not args.skip_unified and (only is None or "unified" in only):
        print("\n[6/6] Dataset Unificado")
        build_unified(base, args.val_frac, args.seed)

    print_summary(base)

    print("\nPróximos passos:")
    print(f"  # Experimento com SARD (como nos resultados preliminares):")
    print(f"  python run_experiment.py --data_root {base}/_prepared/SARD --run_all")
    print(f"  # Experimento com dataset unificado (6 datasets, C5):")
    print(f"  python run_multi_seed.py --data_root {base}/unified --n_seeds 5")
    print(f"  # Experimento de shift VisDrone→FloodNet (H2):")
    print(f"  python run_shift_experiments.py \\")
    print(f"    --source {base}/_prepared/VisDrone \\")
    print(f"    --target {base}/_prepared/FloodNet --n_seeds 5")

if __name__ == "__main__":
    main()
