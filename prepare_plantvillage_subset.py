"""
Download a small real PlantVillage *color* subset and write the NPZ
layout expected by train_pytorch_fusion.py:

  data/processed/plantvillage_color_dataset.npz
    images   uint8  (N, H, W, 3)
    metadata float32 (N, 16)  — zeros: PlantVillage has no soil/weather fields
    labels   int64  (N,)
    class_names  unicode array

This is real leaf photographs, not create_mock_datasets.py noise.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# 10 visually distinct crop/disease classes used in PlantVillage papers.
TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

MAX_PER_CLASS = 220
IMAGE_SIZE = 224
SEED = 42
OUTPUT_NPZ = Path("data/processed/plantvillage_color_dataset.npz")
OUTPUT_META = Path("data/processed/plantvillage_color_subset_meta.json")
RAW_DIR = Path("data/raw/plantvillage_color_subset")


def _to_uint8_rgb(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _save_npz(images, labels, class_names):
    images = np.stack(images, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    metadata = np.zeros((len(labels), 16), dtype=np.float32)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(labels))
    images, labels, metadata = images[order], labels[order], metadata[order]

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        images=images,
        metadata=metadata,
        labels=labels,
        class_names=np.array(class_names),
    )
    counts = {class_names[i]: int((labels == i).sum()) for i in range(len(class_names))}
    meta = {
        "source": "PlantVillage color (real photographs)",
        "npz_path": str(OUTPUT_NPZ).replace("\\", "/"),
        "num_images": int(len(labels)),
        "num_classes": len(class_names),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "max_per_class": MAX_PER_CLASS,
        "class_names": class_names,
        "counts": counts,
        "metadata_note": "16-dim metadata is zeros; PlantVillage images have no soil/weather records.",
        "mock_data": False,
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_NPZ}  images={images.shape}  classes={len(class_names)}")
    for name, n in counts.items():
        print(f"  {n:4d}  {name}")
    return meta


def download_via_huggingface() -> bool:
    from datasets import load_dataset

    wanted = set(TARGET_CLASSES)
    buckets: dict[str, list] = defaultdict(list)

    print("Loading Hugging Face mohanty/PlantVillage (color, streaming)...")
    ds = load_dataset("mohanty/PlantVillage", "color", split="train", streaming=True)

    for example in ds:
        if all(len(buckets[c]) >= MAX_PER_CLASS for c in TARGET_CLASSES):
            break
        label = example.get("label")
        if hasattr(label, "item"):
            # ClassLabel may already be decoded depending on the loader.
            pass
        if not isinstance(label, str):
            # Some configs store integer class ids; skip if we cannot map.
            crop = example.get("crop")
            disease = example.get("disease")
            if crop and disease:
                label = f"{crop}___{str(disease).replace(' ', '_')}"
            else:
                continue
        if label not in wanted:
            continue
        if len(buckets[label]) >= MAX_PER_CLASS:
            continue
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        buckets[label].append(_to_uint8_rgb(image))

    missing = [c for c in TARGET_CLASSES if len(buckets[c]) == 0]
    if missing:
        print(f"Hugging Face stream missed classes: {missing}")
        return False

    class_names = [c for c in TARGET_CLASSES if len(buckets[c]) > 0]
    images, labels = [], []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(class_names):
        class_dir = RAW_DIR / name
        class_dir.mkdir(parents=True, exist_ok=True)
        for j, arr in enumerate(buckets[name]):
            images.append(arr)
            labels.append(idx)
            Image.fromarray(arr).save(class_dir / f"{j:04d}.jpg", quality=90)
    _save_npz(images, labels, class_names)
    return True


def download_via_github_sparse() -> bool:
    """Fallback: sparse-checkout only the needed color class folders."""
    import subprocess
    import shutil

    repo_dir = Path("data/raw/PlantVillage-Dataset")
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    print("Falling back to GitHub sparse checkout of color class folders...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/spMohanty/PlantVillage-Dataset.git",
            str(repo_dir),
        ]
    )
    sparse_paths = [f"raw/color/{c}" for c in TARGET_CLASSES]
    subprocess.check_call(["git", "sparse-checkout", "set", *sparse_paths], cwd=repo_dir)

    class_names = []
    images, labels = [], []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in TARGET_CLASSES:
        src = repo_dir / "raw" / "color" / name
        if not src.is_dir():
            print(f"Missing folder {src}")
            continue
        files = sorted(
            [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        random.Random(SEED).shuffle(files)
        files = files[:MAX_PER_CLASS]
        if not files:
            continue
        class_idx = len(class_names)
        class_names.append(name)
        dest = RAW_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        for j, path in enumerate(files):
            arr = _to_uint8_rgb(Image.open(path))
            images.append(arr)
            labels.append(class_idx)
            Image.fromarray(arr).save(dest / f"{j:04d}.jpg", quality=90)

    if len(class_names) < 8:
        return False
    _save_npz(images, labels, class_names)
    return True


def main():
    random.seed(SEED)
    try:
        ok = download_via_huggingface()
    except Exception as exc:
        print(f"Hugging Face download failed: {exc}")
        ok = False
    if not ok:
        ok = download_via_github_sparse()
    if not ok:
        raise SystemExit("Could not download a real PlantVillage color subset.")


if __name__ == "__main__":
    main()
