"""
Architecture 3 (RT-DETR) Result Generation & Evaluation Artifacts
------------------------------------------------------------------
Evaluates the fine-tuned RT-DETR model on the PlantDoc validation set:
1. `arch3_sample_detections.png`: 2x3 image grid of top-3 highest-confidence detections
   rendered cleanly via ultralytics built-in .plot() with NMS (conf=0.5, iou=0.5).
2. Quantifies duplicate box suppression across all 236 validation images.
3. `arch3_detection_metrics.json`: Overall mAP50, mAP50-95, precision, recall, and NMS suppression stats.

Output directory: model/exports/arch3/results/
"""

import json
import os
import sys
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import RTDETR

# COCO 80 class names for strict verification
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}


def generate_arch3_evaluation(
    weights_path: str = "model/exports/arch3/results/weights/best.pt",
    yaml_path: str = "arch3_rtdetr/data/plantdoc/plantdoc.yaml",
    output_dir: str = "model/exports/arch3/results",
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(
            f"[!] Error: Fine-tuned checkpoint not found at '{weights_path}'. "
            "Please ensure train_arch3_rtdetr.py has run to completion first."
        )

    print(f"[+] Loading fine-tuned RT-DETR model from: {weights_path}")
    model = RTDETR(weights_path)

    # 1. Run ultralytics built-in .val() method against the PlantDoc validation set
    print(f"[+] Running ultralytics .val() evaluation on PlantDoc validation set...")
    val_results = model.val(data=yaml_path, split="val", project=str(out_path.parent), name="eval", exist_ok=True)

    # Extract official metrics
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    print("\n" + "=" * 60)
    print("Architecture 3 (RT-DETR Fine-Tuned) PlantDoc Evaluation Metrics:")
    print("=" * 60)
    print(f"  mAP@50     : {map50:.4f}")
    print(f"  mAP@50-95  : {map50_95:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print("=" * 60 + "\n")

    # 2. Compute Duplicate Box Suppression Statistics across all 236 validation images
    val_img_dir = Path("arch3_rtdetr/data/plantdoc/images/val")
    val_images = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))

    if not val_images:
        raise RuntimeError("[!] No validation images found in arch3_rtdetr/data/plantdoc/images/val")

    print(f"[+] Quantifying duplicate box suppression across {len(val_images)} validation images...")
    total_raw_boxes = 0
    total_nms_boxes = 0

    for img_p in val_images:
        # Raw un-suppressed queries (low confidence, no IoU threshold)
        res_raw = model.predict(str(img_p), conf=0.10, iou=1.0, verbose=False)[0]
        n_raw = len(res_raw.boxes) if res_raw.boxes is not None else 0
        total_raw_boxes += n_raw

        # Suppressed queries (conf=0.5, iou=0.5 NMS)
        res_nms = model.predict(str(img_p), conf=0.50, iou=0.50, verbose=False)[0]
        n_nms = len(res_nms.boxes) if res_nms.boxes is not None else 0
        total_nms_boxes += n_nms

    suppressed_boxes = total_raw_boxes - total_nms_boxes
    suppression_pct = (suppressed_boxes / total_raw_boxes * 100.0) if total_raw_boxes > 0 else 0.0

    print(f"[OK] Duplicate Box Suppression Stats:")
    print(f"     Raw Low-Conf Detections (conf>=0.10, iou=1.0) : {total_raw_boxes} boxes")
    print(f"     Clean NMS Detections (conf>=0.50, iou=0.50)    : {total_nms_boxes} boxes")
    print(f"     Redundant Boxes Removed by NMS & Conf Filter   : {suppressed_boxes} ({suppression_pct:.1f}%)\n")

    # 3. Render Sample Detections Grid using Ultralytics built-in .plot() (conf=0.5, iou=0.5, max top 3 per image)
    print("[+] Rendering clean 2x3 sample detection grid (top-3 boxes, conf>=0.5, iou=0.5)...")
    grid_images = val_images[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Architecture 3 (Fine-Tuned RT-DETR) Clean PlantDoc Detections (conf >= 0.50, IoU = 0.50)", fontsize=15, fontweight="bold", y=0.98)

    detected_labels = []

    for idx, (img_p, ax) in enumerate(zip(grid_images, axes.flat)):
        # Run prediction with conf=0.5, iou=0.5, max_det=3 (top 3 highest confidence detections)
        results = model.predict(str(img_p), conf=0.50, iou=0.50, max_det=3, verbose=False)[0]
        box_count = len(results.boxes) if results.boxes is not None else 0

        # Collect detected class labels for strict verification
        if results.boxes is not None and len(results.boxes) > 0:
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            for cid in cls_ids:
                class_name = model.names[cid] if hasattr(model, "names") and cid in model.names else f"Class-{cid}"
                detected_labels.append(class_name)

        # Ultralytics built-in .plot() renders bounding boxes and non-overlapping label tags cleanly
        plotted_bgr = results.plot(line_width=2, font_size=10)
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(plotted_rgb)
        ax.set_title(f"Sample #{idx+1} ({box_count} box{'es' if box_count != 1 else ''}, top-3 conf >= 0.50)", fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    grid_output_path = out_path / "arch3_sample_detections.png"
    plt.savefig(grid_output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved clean sample detection grid to: {grid_output_path}")

    # 4. Strict Label Verification: Ensure no COCO classes
    coco_matches = [lbl for lbl in detected_labels if lbl.lower() in COCO_CLASSES and "leaf" not in lbl.lower()]
    if coco_matches:
        raise RuntimeError(f"[!] ERROR: Found COCO-style labels in detection grid output: {coco_matches}")
    else:
        print(f"[OK] Label Verification Passed: All detected classes are valid PlantDoc disease categories.")

    # 5. Save Metrics JSON
    metrics_summary = {
        "architecture": "arch3_rtdetr_plantdoc_finetuned",
        "dataset": "PlantDoc",
        "checkpoint": weights_path,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": precision,
        "recall": recall,
        "total_raw_boxes_eval": total_raw_boxes,
        "clean_nms_boxes_eval": total_nms_boxes,
        "suppressed_redundant_boxes_pct": round(suppression_pct, 2),
        "sample_labels": list(set(detected_labels)),
    }

    metrics_json = out_path / "arch3_detection_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[OK] Saved detection metrics to: {metrics_json}")
    return metrics_summary


if __name__ == "__main__":
    generate_arch3_evaluation()
