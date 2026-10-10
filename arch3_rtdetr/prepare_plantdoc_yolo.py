"""
PlantDoc Dataset Downloader & YOLO Format Converter
---------------------------------------------------
Downloads the agyaatcoder/PlantDoc dataset from Hugging Face and converts it to
the standard YOLO / RT-DETR dataset directory structure & YAML format expected by
Ultralytics RT-DETR.

Dataset source:
    Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection," CoDS-COMAD 2020.
    Hugging Face: agyaatcoder/PlantDoc (~2,578 images, ~8,900 bounding boxes)

Output Directory Structure:
    arch3_rtdetr/data/plantdoc/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── plantdoc.yaml
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def prepare_plantdoc(output_root: str = "arch3_rtdetr/data/plantdoc") -> str:
    root = Path(output_root).resolve()
    images_train_dir = root / "images" / "train"
    images_val_dir = root / "images" / "val"
    labels_train_dir = root / "labels" / "train"
    labels_val_dir = root / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[+] Loading agyaatcoder/PlantDoc dataset from Hugging Face...")
    ds = load_dataset("agyaatcoder/PlantDoc")

    train_split = ds["train"]
    # HF PlantDoc uses 'test' for the evaluation set
    val_split = ds["test"] if "test" in ds else ds["validation"]

    # 1. Collect all unique category names to form a consistent class index
    all_categories = set()
    for split in [train_split, val_split]:
        for sample in split:
            objs = sample.get("objects", {})
            cats = objs.get("category", [])
            for c in cats:
                all_categories.add(c.strip())

    sorted_categories = sorted(list(all_categories))
    cat2id = {cat: idx for idx, cat in enumerate(sorted_categories)}

    print(f"[+] Found {len(sorted_categories)} unique disease/plant classes in PlantDoc:")
    for idx, name in enumerate(sorted_categories):
        print(f"    {idx:2d}: {name}")

    # 2. Helper function to process each split
    def process_split(dataset_split, img_dir: Path, label_dir: Path, split_name: str):
        print(f"\n[+] Processing {split_name} split ({len(dataset_split)} images)...")
        converted_count = 0
        bbox_count = 0

        for idx, sample in enumerate(tqdm(dataset_split, desc=f"Converting {split_name}")):
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            
            w, h = img.size
            if w <= 0 or h <= 0:
                continue

            img_filename = f"{split_name}_{idx:05d}.jpg"
            label_filename = f"{split_name}_{idx:05d}.txt"

            img_path = img_dir / img_filename
            label_path = label_dir / label_filename

            # Save image
            img.convert("RGB").save(img_path, quality=95)

            # Convert bounding boxes [x_min, y_min, box_w, box_h] -> YOLO [class_id, x_center, y_center, norm_w, norm_h]
            objs = sample.get("objects", {})
            bboxes = objs.get("bbox", [])
            categories = objs.get("category", [])

            yolo_lines = []
            for bbox, cat_name in zip(bboxes, categories):
                cat_clean = cat_name.strip()
                if cat_clean not in cat2id:
                    continue
                cid = cat2id[cat_clean]

                x_min, y_min, bw, bh = bbox
                # Clamp coordinates to image boundaries
                x_min = max(0.0, min(float(x_min), float(w)))
                y_min = max(0.0, min(float(y_min), float(h)))
                bw = max(0.0, min(float(bw), float(w - x_min)))
                bh = max(0.0, min(float(bh), float(h - y_min)))

                if bw <= 1e-4 or bh <= 1e-4:
                    continue

                x_center = (x_min + bw / 2.0) / float(w)
                y_center = (y_min + bh / 2.0) / float(h)
                norm_w = bw / float(w)
                norm_h = bh / float(h)

                yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                bbox_count += 1

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            converted_count += 1

        print(f"[OK] {split_name} complete: {converted_count} images, {bbox_count} bounding boxes.")

    process_split(train_split, images_train_dir, labels_train_dir, "train")
    process_split(val_split, images_val_dir, labels_val_dir, "val")

    # 3. Write plantdoc.yaml config file
    yaml_path = root / "plantdoc.yaml"
    names_dict = {idx: name for idx, name in enumerate(sorted_categories)}

    yaml_content = f"""# PlantDoc Object Detection Dataset Configuration for RT-DETR / YOLO
path: {root.as_posix()}
train: images/train
val: images/val

names:
"""
    for idx, name in names_dict.items():
        yaml_content += f"  {idx}: \"{name}\"\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[OK] Created dataset config YAML: {yaml_path}")
    
    # Save category mapping JSON for reference
    with open(root / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(cat2id, f, indent=2)

    return str(yaml_path)


if __name__ == "__main__":
    prepare_plantdoc()
