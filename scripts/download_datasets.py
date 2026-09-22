#!/usr/bin/env python3
"""
download_datasets.py
====================
Download e preparação dos 7 datasets utilizados na tese:

  1. SARD         — Search and Rescue Dataset (Roboflow)
  2. VisDrone-DET — VisDrone 2019 Detection (GitHub release)
  3. VisDrone-VID — VisDrone 2019 Video (GitHub release)
  4. UAVDT        — UAV Detection and Tracking (Baidu Pan / mirror)
  5. AIDER        — Aerial Image Dataset for Emergency Response (GitHub)
  6. FloodNet     — Post-flood imagery Hurricane Harvey (IEEE DataPort / GitHub)
  7. LADI v2      — Low Altitude Disaster Imagery v2 (AWS S3 / Hugging Face)

Após o download cada dataset é convertido para o formato YOLO e organizado
na estrutura esperada pelo pipeline:

  datasets/
    SARD/
      images/train/   images/val/
      labels/train/   labels/val/
      data.yaml
    VisDrone/
      images/train/   images/val/
      labels/train/   labels/val/
      data.yaml
    ... (idem para os demais)
    unified/          ← dataset unificado (3 superclasses) gerado ao final

Uso
---
  # Baixar tudo:
  python download_datasets.py --output_dir ./datasets --all

  # Apenas datasets específicos:
  python download_datasets.py --output_dir ./datasets --datasets sard visdrone floodnet

  # Pular download (já baixado) e só converter:
  python download_datasets.py --output_dir ./datasets --all --skip_download

  # Apenas construir o dataset unificado a partir dos já convertidos:
  python download_datasets.py --output_dir ./datasets --build_unified_only

Dependências
------------
  pip install roboflow requests tqdm opencv-python Pillow pyyaml

  Para SARD via Roboflow, defina a variável de ambiente:
    export ROBOFLOW_API_KEY="sua_chave_aqui"
  (obtenha em https://app.roboflow.com → Settings → API Keys)

Notas de licença
----------------
  SARD        : CC BY 4.0
  VisDrone    : Pesquisa não-comercial (ver LICENSE nos arquivos originais)
  UAVDT       : Pesquisa não-comercial
  AIDER       : MIT
  FloodNet    : CC BY 4.0
  LADI v2     : MIT
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import argparse
import random
import yaml
import struct
import zlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import requests
    from tqdm import tqdm
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[AVISO] 'requests' e/ou 'tqdm' não instalados. "
          "Execute: pip install requests tqdm")

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

# ---------------------------------------------------------------------------
# Taxonomia unificada (3 superclasses)
# ---------------------------------------------------------------------------

UNIFIED_CLASSES = ["person", "vehicle", "hazard"]

# Mapeamento de classes originais → superclasse unificada
# Qualquer classe não mapeada é descartada (sem bounding-box útil)
CLASS_MAP = {
    # SARD (classes 0=person, 1=vehicle)
    "sard": {0: 0, 1: 1},

    # VisDrone (10 classes)
    # 0=pedestrian,1=person,2=bicycle,3=car,4=van,5=truck,
    # 6=tricycle,7=awning-tricycle,8=bus,9=motor
    "visdrone": {
        0: 0, 1: 0,                      # person
        2: 1, 3: 1, 4: 1, 5: 1,          # vehicle
        6: 1, 7: 1, 8: 1, 9: 1,          # vehicle
    },

    # UAVDT (3 classes: 0=car, 1=bus, 2=truck)
    "uavdt": {0: 1, 1: 1, 2: 1},

    # AIDER — sem bounding boxes; imagens vão para pool não-anotado
    "aider": {},

    # FloodNet (segmentação convertida para bbox)
    # Superclasses: vehicle=vehicle, tudo mais=hazard
    # 0=background,1=building-flooded,2=building-non-flooded,3=road-flooded,
    # 4=road-non-flooded,5=water,6=tree,7=vehicle,8=pool,9=grass
    "floodnet": {
        7: 1,                             # vehicle
        1: 2, 2: 2, 3: 2, 4: 2,          # hazard (estruturas/estradas)
        5: 2, 8: 2,                       # hazard (água/piscina)
        # background, tree, grass descartados
    },

    # LADI v2 — anotações multi-label; sem bounding-box → pool não-anotado
    "ladi": {},
}

# ---------------------------------------------------------------------------
# Helpers gerais
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _yolo_yaml(output_dir: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    p = output_dir / "data.yaml"
    _write_yaml(p, data)
    return p


def _download_file(url: str, dest: Path, desc: str = ""):
    """Download com barra de progresso."""
    if not HAS_REQUESTS:
        print(f"  [ERRO] requests não disponível. Baixe manualmente: {url}")
        return False
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERRO] Download falhou: {e}")
        return False


def _extract(archive: Path, dest: Path):
    """Extrai zip ou tar.gz."""
    dest.mkdir(parents=True, exist_ok=True)
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif str(archive).endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dest)
    elif str(archive).endswith(".tar"):
        with tarfile.open(archive, "r:") as t:
            t.extractall(dest)
    print(f"  Extraído em: {dest}")


def _copy_image(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(str(src), str(dst))


def _write_label(label_path: Path, annotations: List[Tuple]):
    """Escreve arquivo YOLO .txt com lista de (class_id, cx, cy, w, h)."""
    with open(label_path, "w") as f:
        for ann in annotations:
            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")


def _train_val_split(
    items: List, val_frac: float = 0.2, seed: int = 42
) -> Tuple[List, List]:
    r = random.Random(seed)
    shuffled = items.copy()
    r.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# 1. SARD — via Roboflow API
# ---------------------------------------------------------------------------

SARD_ROBOFLOW_URL = "https://universe.roboflow.com/datasets-pdabr/sard-8xjhy"

def download_sard(output_dir: Path, skip_download: bool = False) -> Path:
    """
    Baixa SARD via Roboflow.
    Requer: pip install roboflow
    Requer: variável de ambiente ROBOFLOW_API_KEY
    """
    dst = output_dir / "SARD"
    if (dst / "data.yaml").exists() and skip_download:
        print(f"  [SARD] Já convertido em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  SARD — Search and Rescue Dataset")
    print(f"  Fonte: {SARD_ROBOFLOW_URL}")
    print("="*60)

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("  [AVISO] ROBOFLOW_API_KEY não definida.")
        print("  Opções:")
        print("  1) export ROBOFLOW_API_KEY='sua_chave'  e rode novamente")
        print("  2) Baixe manualmente em https://universe.roboflow.com/datasets-pdabr/sard-8xjhy")
        print("     → Export → YOLOv8 → extraia em datasets/SARD/")
        _write_manual_instructions(dst, SARD_ROBOFLOW_URL, "YOLOv8 format")
        return dst

    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [ERRO] roboflow não instalado. Execute: pip install roboflow")
        _write_manual_instructions(dst, SARD_ROBOFLOW_URL, "YOLOv8 format")
        return dst

    _mkdir(dst)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("datasets-pdabr").project("sard-8xjhy")
    dataset = project.version(1).download("yolov8", location=str(dst))

    # Roboflow já gera estrutura YOLO — renomear se necessário
    _normalise_roboflow_structure(dst)
    print(f"  [SARD] Pronto em: {dst}")
    return dst


def _normalise_roboflow_structure(dst: Path):
    """Garante estrutura images/train, images/val, labels/train, labels/val."""
    # Roboflow usa train/ valid/ test/ no mesmo nível
    for src_split, dst_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        for kind in ["images", "labels"]:
            src = dst / src_split / kind
            if not src.exists():
                # Tenta estrutura alternativa Roboflow
                src = dst / src_split
                if not (src / kind).exists():
                    continue
                src = src / kind
            dst_d = dst / kind / dst_split
            _mkdir(dst_d)
            for f in src.iterdir():
                target = dst_d / f.name
                if not target.exists():
                    shutil.copy2(str(f), str(target))


# ---------------------------------------------------------------------------
# 2 + 3. VisDrone-DET e VisDrone-VID
# ---------------------------------------------------------------------------

VISDRONE_DET_URLS = {
    "train": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip",
    "val":   "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip",
    "test":  "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip",
}

VISDRONE_VID_URLS = {
    "train": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-train.zip",
    "val":   "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-val.zip",
}

# Nota: os links do GitHub Release são os oficiais mas o arquivo é grande.
# Como fallback, o script imprime instruções para download manual do
# mirror oficial da AISKYEYE Lab (Tianjin University).

VISDRONE_MANUAL_URL = "https://github.com/VisDrone/VisDrone-Dataset"

def download_visdrone(output_dir: Path, skip_download: bool = False) -> Path:
    """Baixa e converte VisDrone-DET e VisDrone-VID."""
    dst = output_dir / "VisDrone"
    raw = output_dir / "_raw" / "VisDrone"

    if (dst / "data.yaml").exists() and skip_download:
        print(f"  [VisDrone] Já convertido em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  VisDrone 2019 (DET + VID)")
    print(f"  Fonte: {VISDRONE_MANUAL_URL}")
    print("="*60)

    _mkdir(raw)
    _mkdir(dst)

    # Tentar download automático (arquivos grandes ~1-3 GB cada)
    success = {}
    for split, url in VISDRONE_DET_URLS.items():
        archive = raw / f"VisDrone2019-DET-{split}.zip"
        if archive.exists():
            print(f"  [VisDrone-DET-{split}] Arquivo já existe, pulando download.")
            success[split] = archive
        elif _download_file(url, archive, f"VisDrone-DET-{split}"):
            success[split] = archive
        else:
            print(f"  [AVISO] Falha no download automático de VisDrone-DET-{split}.")
            print(f"  Baixe manualmente de: {url}")
            print(f"  Salve em: {archive}")

    # Converter DET para YOLO
    for split, archive in success.items():
        extract_dir = raw / f"det_{split}"
        if not extract_dir.exists():
            _extract(archive, extract_dir)
        dst_split = "val" if split in ("val", "test") else "train"
        _convert_visdrone_det(extract_dir, dst, dst_split)

    # VisDrone-VID (frames)
    for split, url in VISDRONE_VID_URLS.items():
        archive = raw / f"VisDrone2019-VID-{split}.zip"
        if not archive.exists():
            print(f"\n  [VisDrone-VID-{split}] Download: {url}")
            _download_file(url, archive, f"VisDrone-VID-{split}")

        if archive.exists():
            extract_dir = raw / f"vid_{split}"
            if not extract_dir.exists():
                _extract(archive, extract_dir)
            dst_split = "val" if split == "val" else "train"
            _convert_visdrone_vid(extract_dir, dst, dst_split, fps_sample=1)

    _yolo_yaml(dst, ["pedestrian","person","bicycle","car","van",
                      "truck","tricycle","awning-tricycle","bus","motor"])
    print(f"  [VisDrone] Pronto em: {dst}")
    return dst


def _convert_visdrone_det(src_root: Path, dst: Path, split: str):
    """
    Converte anotações VisDrone-DET para YOLO.
    Formato original: x_tl, y_tl, w, h, score, category, truncation, occlusion
    """
    img_dir = None
    ann_dir = None
    # Descoberta de estrutura (pode variar entre train/val/test)
    for d in src_root.rglob("images"):
        if d.is_dir():
            img_dir = d
            break
    for d in src_root.rglob("annotations"):
        if d.is_dir():
            ann_dir = d
            break
    if not img_dir or not ann_dir:
        print(f"  [AVISO] Estrutura não reconhecida em {src_root}")
        return

    dst_img = _mkdir(dst / "images" / split)
    dst_lbl = _mkdir(dst / "labels" / split)

    for ann_file in sorted(ann_dir.glob("*.txt")):
        img_name = ann_file.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            img_name = ann_file.stem + ".png"
            img_path = img_dir / img_name
        if not img_path.exists():
            continue

        # Obter dimensões da imagem
        W, H = _get_image_size(img_path)
        if W is None:
            continue

        annotations = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cat = int(parts[5]) - 1  # VisDrone é 1-indexed
                if w <= 0 or h <= 0:
                    continue
                # Converter para YOLO normalizado
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                cx, cy, nw, nh = (
                    min(max(cx, 0), 1), min(max(cy, 0), 1),
                    min(max(nw, 0), 1), min(max(nh, 0), 1),
                )
                annotations.append((cat, cx, cy, nw, nh))

        _copy_image(img_path, dst_img)
        _write_label(dst_lbl / (ann_file.stem + ".txt"), annotations)


def _convert_visdrone_vid(src_root: Path, dst: Path, split: str, fps_sample: int = 1):
    """
    Extrai frames de VisDrone-VID e converte anotações para YOLO.
    fps_sample: selecionar 1 frame a cada N (default=1 = 1fps de um vídeo a 25fps → 1 em 25).
    """
    dst_img = _mkdir(dst / "images" / split)
    dst_lbl = _mkdir(dst / "labels" / split)

    # VisDrone-VID: sequences/uav000XXXX/
    seq_dirs = sorted(src_root.rglob("sequences"))
    ann_dirs = sorted(src_root.rglob("annotations"))

    for seq_dir in seq_dirs:
        for seq in sorted(seq_dir.iterdir()):
            if not seq.is_dir():
                continue
            seq_name = seq.name
            # Encontrar arquivo de anotação correspondente
            ann_file = None
            for ad in ann_dirs:
                candidate = ad / (seq_name + ".txt")
                if candidate.exists():
                    ann_file = candidate
                    break
            if ann_file is None:
                continue

            frames = sorted(seq.glob("*.jpg")) + sorted(seq.glob("*.png"))
            # Amostrar frames
            sampled = frames[::fps_sample * 25] if fps_sample > 0 else frames

            # Carregar anotações (frame_idx, x, y, w, h, score, cat, trunc, occ)
            ann_by_frame: Dict[int, List] = {}
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 7:
                        continue
                    fidx = int(parts[0])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    cat = int(parts[7]) - 1
                    if fidx not in ann_by_frame:
                        ann_by_frame[fidx] = []
                    ann_by_frame[fidx].append((cat, x, y, w, h))

            for frame_path in sampled:
                # VisDrone-VID: frame 0000001.jpg → index 1
                try:
                    fidx = int(frame_path.stem)
                except ValueError:
                    continue

                W, H = _get_image_size(frame_path)
                if W is None:
                    continue

                frame_key = f"{seq_name}_{frame_path.stem}"
                img_dst = dst_img / (frame_key + frame_path.suffix)
                lbl_dst = dst_lbl / (frame_key + ".txt")

                if not img_dst.exists():
                    shutil.copy2(str(frame_path), str(img_dst))

                annotations = []
                for (cat, x, y, w, h) in ann_by_frame.get(fidx, []):
                    if w <= 0 or h <= 0:
                        continue
                    cx = (x + w / 2) / W
                    cy = (y + h / 2) / H
                    nw = w / W
                    nh = h / H
                    annotations.append((cat, cx, cy, nw, nh))
                _write_label(lbl_dst, annotations)


# ---------------------------------------------------------------------------
# 4. UAVDT
# ---------------------------------------------------------------------------

UAVDT_MANUAL_URL = "https://sites.google.com/view/grli-uavdt/dataset"
UAVDT_DRIVE_URL  = "https://drive.google.com/drive/folders/1zAqRR9Yw0sTyZH3cXcEZxNJuWC1NxYDi"

def download_uavdt(output_dir: Path, skip_download: bool = False) -> Path:
    """
    UAVDT requer download manual via Google Drive.
    O script verifica se o arquivo já existe e converte.
    """
    dst = output_dir / "UAVDT"
    raw = output_dir / "_raw" / "UAVDT"

    if (dst / "data.yaml").exists() and skip_download:
        print(f"  [UAVDT] Já convertido em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  UAVDT — UAV Detection and Tracking Dataset")
    print(f"  Fonte: {UAVDT_MANUAL_URL}")
    print("="*60)
    print("  UAVDT requer download manual (Google Drive):")
    print(f"  1) Acesse: {UAVDT_DRIVE_URL}")
    print("  2) Baixe: UAV-benchmark-M.zip  e  GT.zip")
    print(f"  3) Salve em: {raw}/")
    print(f"  4) Execute novamente com --datasets uavdt --skip_download=false")
    _mkdir(raw)

    # Procurar arquivos já baixados
    benchmarks = list(raw.glob("UAV-benchmark*.zip")) + list(raw.glob("UAV-benchmark*.tar*"))
    gts        = list(raw.glob("GT*.zip")) + list(raw.glob("GT*.tar*"))

    if not benchmarks or not gts:
        _write_manual_instructions(dst, UAVDT_MANUAL_URL,
            "Baixe UAV-benchmark-M.zip e GT.zip do Google Drive")
        return dst

    _mkdir(dst)

    # Extrair
    for arc in benchmarks + gts:
        extract_dir = raw / arc.stem
        if not extract_dir.exists():
            _extract(arc, extract_dir)

    # Converter para YOLO
    _convert_uavdt(raw, dst)
    _yolo_yaml(dst, ["car", "bus", "truck"])
    print(f"  [UAVDT] Pronto em: {dst}")
    return dst


def _convert_uavdt(raw: Path, dst: Path, fps_sample: int = 2):
    """
    UAVDT: frames em UAV-benchmark-M/data/seqXXX/img000001.jpg
    Anotações em GT/seqXXX_gt_whole.txt (frame,id,x,y,w,h,out-of-view,occ,vehicle-type)
    """
    dst_img_t = _mkdir(dst / "images" / "train")
    dst_lbl_t = _mkdir(dst / "labels" / "train")
    dst_img_v = _mkdir(dst / "images" / "val")
    dst_lbl_v = _mkdir(dst / "labels" / "val")

    data_dirs = list(raw.rglob("data"))
    gt_dirs   = list(raw.rglob("GT"))

    if not data_dirs or not gt_dirs:
        print("  [UAVDT] Estrutura de diretórios não encontrada. Verifique a extração.")
        return

    data_dir = data_dirs[0]
    gt_dir   = gt_dirs[0]

    seqs = sorted(data_dir.iterdir())
    val_seqs = set(s.name for s in seqs[-max(1, len(seqs)//5):])  # ~20% como val

    for seq in seqs:
        if not seq.is_dir():
            continue
        split = "val" if seq.name in val_seqs else "train"
        dst_img = dst_img_v if split == "val" else dst_img_t
        dst_lbl = dst_lbl_v if split == "val" else dst_lbl_t

        gt_file = gt_dir / f"{seq.name}_gt_whole.txt"
        if not gt_file.exists():
            continue

        # Carregar GT
        ann_by_frame: Dict[int, List] = {}
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                fidx = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                vtype = int(parts[8]) - 1  # 1=car,2=bus,3=truck → 0,1,2
                vtype = max(0, min(vtype, 2))
                if fidx not in ann_by_frame:
                    ann_by_frame[fidx] = []
                ann_by_frame[fidx].append((vtype, x, y, w, h))

        frames = sorted(seq.glob("img*.jpg"))
        sampled = frames[::fps_sample]

        for frame_path in sampled:
            try:
                fidx = int(frame_path.stem.replace("img", ""))
            except ValueError:
                continue

            W, H = _get_image_size(frame_path)
            if W is None:
                continue

            frame_key = f"{seq.name}_{frame_path.stem}"
            img_dst = dst_img / (frame_key + ".jpg")
            lbl_dst = dst_lbl / (frame_key + ".txt")

            if not img_dst.exists():
                shutil.copy2(str(frame_path), str(img_dst))

            annotations = []
            for (cat, x, y, w, h) in ann_by_frame.get(fidx, []):
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw, nh = w / W, h / H
                annotations.append((cat, cx, cy, nw, nh))
            _write_label(lbl_dst, annotations)


# ---------------------------------------------------------------------------
# 5. AIDER — sem bounding boxes
# ---------------------------------------------------------------------------

AIDER_GITHUB_URL = "https://github.com/ckyrkou/AIDER"
AIDER_GDRIVE_URL = "https://drive.google.com/drive/folders/1pMy1haaiFCxk-8-dZXAL_1bvvJD5qECM"

def download_aider(output_dir: Path, skip_download: bool = False) -> Path:
    """
    AIDER: somente imagens de cena (sem bounding box).
    Copiadas para pool não-anotado (unlabelled/aider/).
    """
    dst = output_dir / "AIDER"
    raw = output_dir / "_raw" / "AIDER"

    if (dst / "README.txt").exists() and skip_download:
        print(f"  [AIDER] Já processado em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  AIDER — Aerial Image Dataset for Emergency Response")
    print(f"  Fonte: {AIDER_GITHUB_URL}")
    print("="*60)
    print("  AIDER requer download manual (Google Drive):")
    print(f"  1) Acesse: {AIDER_GDRIVE_URL}")
    print("  2) Baixe toda a pasta AIDER")
    print(f"  3) Extraia em: {raw}/")
    _mkdir(raw)

    # Verificar se imagens existem
    img_exts = {".jpg", ".jpeg", ".png"}
    found = list(raw.rglob("*.jpg")) + list(raw.rglob("*.png"))

    if not found:
        _write_manual_instructions(dst, AIDER_GDRIVE_URL,
            "Baixe do Google Drive e extraia em datasets/_raw/AIDER/")
        return dst

    _mkdir(dst)
    # AIDER vai para pool não-anotado (sem bounding-box)
    unlabelled_dir = _mkdir(dst / "unlabelled" / "images")

    categories = ["fire_smoke", "flood", "collapsed_building", "traffic_accident", "normal"]
    cat_counts = {}

    for cat in categories:
        cat_dir = None
        for d in raw.rglob(cat):
            if d.is_dir():
                cat_dir = d
                break
        if cat_dir is None:
            # Tentar variações de nome
            for d in raw.iterdir():
                if cat.replace("_", "") in d.name.lower().replace("_", ""):
                    cat_dir = d
                    break
        if cat_dir is None:
            continue
        imgs = [f for f in cat_dir.rglob("*") if f.suffix.lower() in img_exts]
        for img in imgs:
            dst_img = unlabelled_dir / f"{cat}_{img.name}"
            if not dst_img.exists():
                shutil.copy2(str(img), str(dst_img))
        cat_counts[cat] = len(imgs)

    # Se não encontrou por categoria, copiar tudo
    if not any(cat_counts.values()):
        for img in found:
            dst_img = unlabelled_dir / img.name
            if not dst_img.exists():
                shutil.copy2(str(img), str(dst_img))
        cat_counts["all"] = len(found)

    readme = (
        "AIDER — Aerial Image Dataset for Emergency Response\n"
        "====================================================\n"
        "Este dataset NÃO possui bounding-box annotations.\n"
        "As imagens estão em unlabelled/images/ para uso como\n"
        "pool não-anotado no pipeline de Active Learning.\n\n"
        "Contagem por categoria:\n" +
        "\n".join(f"  {k}: {v}" for k, v in cat_counts.items()) + "\n"
    )
    (dst / "README.txt").write_text(readme)
    print(f"  [AIDER] {sum(cat_counts.values())} imagens em: {dst}/unlabelled/")
    return dst


# ---------------------------------------------------------------------------
# 6. FloodNet
# ---------------------------------------------------------------------------

FLOODNET_GITHUB_URL = "https://github.com/BinaLab/FloodNet-Supervised_v1.0"
FLOODNET_IEEE_URL   = "https://ieee-dataport.org/open-access/floodnet-high-resolution-aerial-imagery-dataset-post-flood-scene-understanding"

def download_floodnet(output_dir: Path, skip_download: bool = False) -> Path:
    """
    FloodNet: segmentação semântica → bounding boxes por contorno.
    Requer opencv-python para a conversão de máscara → bbox.
    """
    dst = output_dir / "FloodNet"
    raw = output_dir / "_raw" / "FloodNet"

    if (dst / "data.yaml").exists() and skip_download:
        print(f"  [FloodNet] Já convertido em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  FloodNet — Hurricane Harvey Post-Flood Dataset")
    print(f"  Fonte: {FLOODNET_IEEE_URL}")
    print("="*60)
    print("  FloodNet requer download manual (IEEE DataPort):")
    print(f"  1) Acesse: {FLOODNET_IEEE_URL}")
    print("  2) Faça login com IEEE Account (gratuito)")
    print("  3) Baixe o dataset completo")
    print(f"  4) Extraia em: {raw}/")
    print("  Alternativa (sem conta IEEE): baixe via repositório Hugging Face")
    print("    pip install huggingface_hub")
    print("    python -c \"from huggingface_hub import snapshot_download; "
          "snapshot_download(repo_id='Roboflow-100/floodnet-v1', repo_type='dataset', "
          f"local_dir='{raw}')\"")
    _mkdir(raw)

    # Tentar Hugging Face automaticamente
    _try_hf_download("Roboflow-100/floodnet-v1", raw, "dataset")

    # Verificar estrutura
    img_dirs = list(raw.rglob("images")) + list(raw.rglob("Images"))
    mask_dirs = list(raw.rglob("masks")) + list(raw.rglob("Masks")) + \
                list(raw.rglob("labels")) + list(raw.rglob("Labels"))

    if not img_dirs:
        _write_manual_instructions(dst, FLOODNET_IEEE_URL,
            "Baixe do IEEE DataPort e extraia em datasets/_raw/FloodNet/")
        return dst

    _mkdir(dst)

    if not HAS_CV2:
        print("  [AVISO] opencv-python não instalado. Usando conversão simplificada.")

    _convert_floodnet(raw, dst)
    _yolo_yaml(dst, ["building-flooded","building-non-flooded","road-flooded",
                      "road-non-flooded","water","tree","vehicle","pool","grass"])
    print(f"  [FloodNet] Pronto em: {dst}")
    return dst


def _convert_floodnet(raw: Path, dst: Path):
    """
    Converte máscaras de segmentação FloodNet em bounding boxes YOLO.
    Classes (pixel value = class):
      0=background, 1=building-flooded, 2=building-non-flooded,
      3=road-flooded, 4=road-non-flooded, 5=water, 6=tree,
      7=vehicle, 8=pool, 9=grass
    """
    splits_map = [("train", "train"), ("val", "val"), ("test", "val")]

    for src_split, dst_split in splits_map:
        # Descoberta de diretórios de imagem e máscara
        img_dir = _find_dir(raw, ["image", src_split])
        msk_dir = _find_dir(raw, ["label", src_split]) or _find_dir(raw, ["mask", src_split])

        if not img_dir:
            continue

        dst_img = _mkdir(dst / "images" / dst_split)
        dst_lbl = _mkdir(dst / "labels" / dst_split)

        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        for img_path in img_files:
            # Encontrar máscara correspondente
            mask_path = None
            if msk_dir:
                for ext in [".png", "_lab.png", ".jpg"]:
                    candidate = msk_dir / (img_path.stem + ext)
                    if candidate.exists():
                        mask_path = candidate
                        break

            _copy_image(img_path, dst_img)

            if mask_path is None or not HAS_CV2:
                # Sem máscara ou sem cv2: criar label vazio
                _write_label(dst_lbl / (img_path.stem + ".txt"), [])
                continue

            annotations = _mask_to_bboxes(mask_path)
            _write_label(dst_lbl / (img_path.stem + ".txt"), annotations)


def _mask_to_bboxes(mask_path: Path) -> List[Tuple]:
    """Converte máscara semântica em lista de bboxes YOLO."""
    if not HAS_CV2:
        return []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    H, W = mask.shape
    annotations = []
    for class_id in range(1, 10):  # ignorar background (0)
        binary = (mask == class_id).astype("uint8") * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 100:  # descartar objetos muito pequenos
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            annotations.append((class_id - 1, cx, cy, nw, nh))
    return annotations


# ---------------------------------------------------------------------------
# 7. LADI v2
# ---------------------------------------------------------------------------

LADI_HF_REPO   = "MITLL/ladi-v2-dataset"
LADI_ARXIV_URL = "https://arxiv.org/abs/2406.02780"

def download_ladi(output_dir: Path, skip_download: bool = False) -> Path:
    """
    LADI v2: anotações multi-label, sem bounding box.
    Vai para pool não-anotado.
    """
    dst = output_dir / "LADI"
    raw = output_dir / "_raw" / "LADI"

    if (dst / "README.txt").exists() and skip_download:
        print(f"  [LADI v2] Já processado em {dst} — pulando.")
        return dst

    print("\n" + "="*60)
    print("  LADI v2 — Low Altitude Disaster Imagery v2")
    print(f"  Fonte: {LADI_HF_REPO}")
    print("="*60)

    _mkdir(raw)

    # Tentar Hugging Face
    success = _try_hf_download(LADI_HF_REPO, raw, "dataset")

    if not success:
        print("  Download manual:")
        print(f"  1) pip install huggingface_hub")
        print(f"  2) python -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download(repo_id='{LADI_HF_REPO}', repo_type='dataset', "
              f"local_dir='{raw}')\"")
        print(f"  3) Execute novamente com --datasets ladi")
        _write_manual_instructions(dst, f"https://huggingface.co/datasets/{LADI_HF_REPO}",
            "Baixe via Hugging Face Hub (snapshot_download)")
        return dst

    _mkdir(dst)
    unlabelled_dir = _mkdir(dst / "unlabelled" / "images")

    img_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [f for f in raw.rglob("*") if f.suffix.lower() in img_exts]

    count = 0
    for img in all_imgs:
        dst_img = unlabelled_dir / img.name
        if not dst_img.exists():
            shutil.copy2(str(img), str(dst_img))
            count += 1

    readme = (
        "LADI v2 — Low Altitude Disaster Imagery v2\n"
        "===========================================\n"
        "Multi-label dataset sem bounding boxes.\n"
        "Imagens em unlabelled/images/ para uso como pool não-anotado.\n"
        f"Total de imagens: {count}\n"
        f"Referência: Boussioux et al. (2024), arXiv:2406.02780\n"
    )
    (dst / "README.txt").write_text(readme)
    print(f"  [LADI v2] {count} imagens em: {dst}/unlabelled/")
    return dst


# ---------------------------------------------------------------------------
# Dataset Unificado (3 superclasses)
# ---------------------------------------------------------------------------

def build_unified_dataset(output_dir: Path, val_frac: float = 0.2, seed: int = 42) -> Path:
    """
    Constrói o dataset unificado com taxonomia de 3 superclasses:
      0=person, 1=vehicle, 2=hazard

    Combina todos os datasets com bounding boxes disponíveis,
    remapeia classes, e partilha em train/val.

    Datasets sem bounding box (AIDER, LADI) contribuem apenas ao
    pool não-anotado (unlabelled/).
    """
    unified = output_dir / "unified"
    _mkdir(unified)

    print("\n" + "="*60)
    print("  Construindo dataset unificado (3 superclasses)")
    print("="*60)

    dst_img_t = _mkdir(unified / "images" / "train")
    dst_lbl_t = _mkdir(unified / "labels" / "train")
    dst_img_v = _mkdir(unified / "images" / "val")
    dst_lbl_v = _mkdir(unified / "labels" / "val")
    dst_unlab = _mkdir(unified / "unlabelled" / "images")

    # Fontes com bounding box
    bbox_sources = {
        "sard":      (output_dir / "SARD",     CLASS_MAP["sard"]),
        "visdrone":  (output_dir / "VisDrone",  CLASS_MAP["visdrone"]),
        "uavdt":     (output_dir / "UAVDT",     CLASS_MAP["uavdt"]),
        "floodnet":  (output_dir / "FloodNet",  CLASS_MAP["floodnet"]),
    }

    stats = {}
    all_items = []  # (img_path, lbl_path, source_name, class_map)

    for src_name, (src_dir, cmap) in bbox_sources.items():
        if not src_dir.exists():
            print(f"  [AVISO] {src_name} não encontrado em {src_dir} — pulando.")
            continue

        for split in ["train", "val"]:
            img_dir = src_dir / "images" / split
            lbl_dir = src_dir / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    all_items.append((img_path, lbl_path, src_name, cmap))

        stats[src_name] = len(all_items)

    # Embaralhar e dividir
    set_seed(seed)
    r = random.Random(seed)
    r.shuffle(all_items)
    n_val = max(50, int(len(all_items) * val_frac))
    val_items  = all_items[:n_val]
    train_items = all_items[n_val:]

    print(f"  Total com bbox: {len(all_items)} | train: {len(train_items)} | val: {len(val_items)}")

    def _process_items(items, dst_img, dst_lbl, split_name):
        count = 0
        for img_path, lbl_path, src_name, cmap in items:
            prefix = f"{src_name}_{img_path.stem}"
            img_dst = dst_img / (prefix + img_path.suffix)
            lbl_dst = dst_lbl / (prefix + ".txt")

            if not img_dst.exists():
                shutil.copy2(str(img_path), str(img_dst))

            # Remap classes
            new_anns = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    orig_cls = int(parts[0])
                    if orig_cls not in cmap:
                        continue  # descarta classes sem mapeamento
                    new_cls = cmap[orig_cls]
                    new_anns.append((new_cls,) + tuple(float(x) for x in parts[1:5]))
            _write_label(lbl_dst, new_anns)
            count += 1
        return count

    n_train = _process_items(train_items, dst_img_t, dst_lbl_t, "train")
    n_val   = _process_items(val_items,   dst_img_v, dst_lbl_v, "val")

    # Pool não-anotado (AIDER + LADI)
    n_unlab = 0
    for src_name, src_dir in [("AIDER", output_dir / "AIDER"),
                                ("LADI",  output_dir / "LADI")]:
        unlab_src = src_dir / "unlabelled" / "images"
        if not unlab_src.exists():
            continue
        for img in unlab_src.glob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dst_img = dst_unlab / f"{src_name}_{img.name}"
                if not dst_img.exists():
                    shutil.copy2(str(img), str(dst_img))
                    n_unlab += 1

    print(f"  Pool não-anotado: {n_unlab} imagens")

    # Gerar data.yaml
    _yolo_yaml(unified, UNIFIED_CLASSES)

    # Gerar metadata.json
    meta = {
        "description": "Unified multi-source UAV dataset (3-class taxonomy)",
        "classes": UNIFIED_CLASSES,
        "class_map": {k: {str(ki): vi for ki, vi in v.items()} for k, v in CLASS_MAP.items()},
        "sources": {k: str(v[0]) for k, v in bbox_sources.items()},
        "split": {"train": n_train, "val": n_val, "unlabelled": n_unlab},
        "total_annotated": n_train + n_val,
    }
    with open(unified / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Dataset unificado: {n_train+n_val} imagens anotadas + {n_unlab} não-anotadas")
    print(f"  Salvo em: {unified}")
    print(f"  data.yaml: {unified / 'data.yaml'}")
    return unified


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _get_image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Retorna (W, H) de uma imagem sem carregar pixels completos."""
    if HAS_PIL:
        try:
            with Image.open(str(path)) as img:
                return img.size
        except Exception:
            pass
    if HAS_CV2:
        img = cv2.imread(str(path))
        if img is not None:
            return img.shape[1], img.shape[0]
    # Leitura manual de header PNG/JPEG
    try:
        with open(path, "rb") as f:
            header = f.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            W = struct.unpack(">I", header[16:20])[0]
            H = struct.unpack(">I", header[20:24])[0]
            return W, H
        if header[:2] == b"\xff\xd8":  # JPEG
            f2 = open(path, "rb")
            f2.read(2)
            while True:
                marker, = struct.unpack(">H", f2.read(2))
                length, = struct.unpack(">H", f2.read(2))
                if marker in (0xFFC0, 0xFFC1, 0xFFC2):
                    f2.read(1)
                    H, W = struct.unpack(">HH", f2.read(4))
                    f2.close()
                    return W, H
                f2.read(length - 2)
            f2.close()
    except Exception:
        pass
    return None, None


def _find_dir(root: Path, keywords: List[str]) -> Optional[Path]:
    """Encontra diretório que contém todos os keywords no caminho."""
    for d in root.rglob("*"):
        if d.is_dir():
            path_lower = str(d).lower()
            if all(kw.lower() in path_lower for kw in keywords):
                return d
    return None


def _try_hf_download(repo_id: str, local_dir: Path, repo_type: str = "dataset") -> bool:
    """Tenta baixar dataset do Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"  Baixando {repo_id} via Hugging Face Hub...")
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            ignore_patterns=["*.parquet"],  # ignorar parquet se não necessário
        )
        return True
    except ImportError:
        print("  [INFO] huggingface_hub não instalado. "
              "Execute: pip install huggingface_hub")
    except Exception as e:
        print(f"  [AVISO] Hugging Face download falhou: {e}")
    return False


def _write_manual_instructions(dst: Path, url: str, instructions: str):
    """Escreve arquivo com instruções de download manual."""
    _mkdir(dst)
    txt = (
        f"DOWNLOAD MANUAL NECESSÁRIO\n"
        f"==========================\n"
        f"URL: {url}\n\n"
        f"Instruções:\n{instructions}\n\n"
        f"Após o download, execute:\n"
        f"  python download_datasets.py --output_dir ./datasets --all\n"
    )
    (dst / "DOWNLOAD_INSTRUCTIONS.txt").write_text(txt)
    print(f"  Instruções salvas em: {dst / 'DOWNLOAD_INSTRUCTIONS.txt'}")


def print_summary(output_dir: Path):
    """Imprime resumo do estado de cada dataset."""
    print("\n" + "="*70)
    print("  RESUMO DOS DATASETS")
    print("="*70)

    datasets = {
        "SARD":     ("BBox", "data.yaml"),
        "VisDrone": ("BBox", "data.yaml"),
        "UAVDT":    ("BBox", "data.yaml"),
        "AIDER":    ("Sem bbox", "README.txt"),
        "FloodNet": ("BBox", "data.yaml"),
        "LADI":     ("Sem bbox", "README.txt"),
        "unified":  ("3 superclasses", "data.yaml"),
    }

    for name, (dtype, marker) in datasets.items():
        d = output_dir / name
        if not d.exists():
            status = "❌ NÃO ENCONTRADO"
        elif (d / marker).exists():
            # Contar imagens
            n_train = len(list((d / "images" / "train").glob("*"))) if (d / "images" / "train").exists() else 0
            n_val   = len(list((d / "images" / "val").glob("*")))   if (d / "images" / "val").exists()   else 0
            n_unlab = len(list((d / "unlabelled" / "images").glob("*"))) if (d / "unlabelled" / "images").exists() else 0
            status = f"✅  train={n_train} | val={n_val} | unlabelled={n_unlab}"
        elif (d / "DOWNLOAD_INSTRUCTIONS.txt").exists():
            status = "⚠️  Download manual pendente"
        else:
            status = "⚠️  Parcialmente convertido"

        print(f"  {name:<12} [{dtype:<15}]  {status}")

    print("="*70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_DATASETS = ["sard", "visdrone", "uavdt", "aider", "floodnet", "ladi"]

DOWNLOAD_FNS = {
    "sard":     download_sard,
    "visdrone": download_visdrone,
    "uavdt":    download_uavdt,
    "aider":    download_aider,
    "floodnet": download_floodnet,
    "ladi":     download_ladi,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download e preparação dos datasets da tese (SARD, VisDrone, UAVDT, AIDER, FloodNet, LADI v2)"
    )
    parser.add_argument("--output_dir", type=str, default="./datasets",
                        help="Diretório raiz para salvar os datasets (default: ./datasets)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=ALL_DATASETS + ["all"],
                        default=None,
                        help="Datasets para baixar. 'all' para todos.")
    parser.add_argument("--all", action="store_true",
                        help="Baixar todos os 7 datasets")
    parser.add_argument("--skip_download", action="store_true",
                        help="Pular download (usar arquivos já baixados)")
    parser.add_argument("--build_unified", action="store_true",
                        help="Construir dataset unificado após download")
    parser.add_argument("--build_unified_only", action="store_true",
                        help="Apenas construir dataset unificado (sem download)")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fração de validação para o dataset unificado (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_unified_only:
        build_unified_dataset(output_dir, args.val_frac, args.seed)
        print_summary(output_dir)
        return

    # Determinar quais datasets baixar
    if args.all or (args.datasets and "all" in args.datasets):
        to_download = ALL_DATASETS
    elif args.datasets:
        to_download = args.datasets
    else:
        parser.print_help()
        print("\n  Use --all para baixar todos os datasets ou --datasets <lista>")
        return

    print(f"\n  Datasets selecionados: {', '.join(to_download)}")
    print(f"  Diretório de saída: {output_dir.resolve()}")
    print(f"  Skip download: {args.skip_download}")

    # Download e conversão
    for ds in to_download:
        fn = DOWNLOAD_FNS.get(ds)
        if fn:
            fn(output_dir, skip_download=args.skip_download)

    # Dataset unificado
    if args.build_unified or args.all:
        build_unified_dataset(output_dir, args.val_frac, args.seed)

    print_summary(output_dir)
    print("\nPróximos passos:")
    print(f"  python run_experiment.py --data_root {output_dir}/SARD --run_all")
    print(f"  python run_multi_seed.py --data_root {output_dir}/unified --n_seeds 5")
    print(f"  python run_shift_experiments.py --source {output_dir}/VisDrone --target {output_dir}/FloodNet --n_seeds 5")


if __name__ == "__main__":
    main()
