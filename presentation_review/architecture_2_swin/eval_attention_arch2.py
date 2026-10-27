"""
generate_fusion_results_arch2.py
--------------------------------

Generates evaluation artifacts for Architecture 2 (Swin-B + Cross-Attention):
  1) Validation classification report (precision, recall, F1 per class) -> arch2_classification_report.json
  2) Validation confusion matrix -> arch2_confusion_matrix.png
  3) Training & validation learning curves -> arch2_learning_curves.png
  4) Sample validation predictions grid -> arch2_sample_predictions.png
  5) Multi-head attention-rollout grid -> arch2_attention_rollout_grid.png
  6) Detailed metrics JSON -> arch2_val_metrics.json

Architecture 1 files (DO NOT EDIT):
  generate_fusion_results.py — Architecture 1 evaluation script
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from fusion_model_arch2_swin import SwinFusionModel
from train_arch2_swin import ARCH2_CONFIG, ImageMetadataDataset, val_transform

OUT_DIR = Path("model/exports/arch2/results")
CKPT_PATH = Path("model/exports/arch2/best_model_disease.pth")
CSV_PATH = Path("model/exports/arch2/metrics_arch2_disease.csv")
NPZ_PATH = Path(ARCH2_CONFIG["models"]["disease"]["data_path"])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(img_chw: torch.Tensor) -> np.ndarray:
    """Convert normalized (C, H, W) tensor back to RGB [0, 1] numpy array."""
    arr = img_chw.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def overlay_attention_map(rgb01: np.ndarray, attn_map: np.ndarray) -> np.ndarray:
    """
    Overlay a 2D attention spatial heatmap (e.g. 7x7) onto an RGB image (H, W, 3).
    """
    h, w, _ = rgb01.shape
    v_min, v_max = attn_map.min(), attn_map.max()
    if v_max - v_min > 1e-8:
        norm_map = (attn_map - v_min) / (v_max - v_min)
    else:
        norm_map = np.ones_like(attn_map) * 0.5

    resized = Image.fromarray((norm_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    resized_arr = np.asarray(resized).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heat = cmap(resized_arr)[..., :3]
    return np.clip(0.4 * rgb01 + 0.6 * heat, 0, 1)


def evaluate_arch2(ckpt_path: Path = CKPT_PATH, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ckpt_path.exists():
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}")
        print("[!] Using random weights for Architecture 2 evaluation dry-run...")
        num_classes = 38
        class_names = [f"Class_{i:02d}" for i in range(num_classes)]
        ckpt_meta = {"epoch": 0, "val_acc": 0.0, "val_loss": 0.0}
        model = SwinFusionModel(num_classes=num_classes, pretrained=False).to(device)
    else:
        print(f"[+] Loading Architecture 2 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        num_classes = int(ckpt.get("num_classes", 38))
        class_names = ckpt.get("class_names")
        ckpt_meta = {
            "epoch": ckpt.get("epoch", 0),
            "val_acc": ckpt.get("val_acc", 0.0),
            "val_loss": ckpt.get("val_loss", 0.0),
        }
        if not class_names:
            data = np.load(NPZ_PATH, allow_pickle=True)
            class_names = [str(x) for x in data["class_names"].tolist()]

        model = SwinFusionModel(
            num_classes=num_classes,
            num_attn_heads=ckpt.get("num_attn_heads", 8),
            pretrained=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Dataset NPZ file missing: {NPZ_PATH}")

    dataset = ImageMetadataDataset(str(NPZ_PATH), transform=val_transform, train=False, seed=42)
    print(f"[+] Evaluating on validation set ({len(dataset)} samples)...")

    y_true, y_pred, y_conf = [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            image, meta, label = dataset[i]
            img_tensor = image.unsqueeze(0).to(device, dtype=torch.float)
            meta_tensor = torch.as_tensor(meta, dtype=torch.float).unsqueeze(0).to(device)
            logits = model(img_tensor, meta_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())

            y_true.append(int(label))
            y_pred.append(pred)
            y_conf.append(float(probs[pred].item()))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Save classification report & metrics
    (out_dir / "arch2_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "arch2_val_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path).replace("\\", "/"),
                "val_accuracy": acc,
                "n_val": int(len(dataset)),
                "ckpt_val_acc": ckpt_meta["val_acc"],
                "ckpt_val_loss": ckpt_meta["val_loss"],
                "ckpt_epoch": ckpt_meta["epoch"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 1. Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_title(f"Architecture 2 (Swin-B) Confusion Matrix (val acc = {acc:.1%}, n = {len(dataset)})")
    short_names = [n.replace("___", "\n") for n in class_names]
    ax.set_xticks(range(num_classes), short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_classes), short_names, fontsize=8)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    fig.colorbar(im, ax=ax, fraction=0.046)

    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "arch2_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # 2. Sample predictions grid
    rng = np.random.default_rng(42)
    wrong = np.where(y_true != y_pred)[0]
    right = np.where(y_true == y_pred)[0]

    idxs = []
    if len(wrong):
        idxs.extend(rng.choice(wrong, size=min(4, len(wrong)), replace=False).tolist())
    if len(right):
        idxs.extend(rng.choice(right, size=min(4, len(right)), replace=False).tolist())
    idxs = idxs[:8]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()

    for ax, idx in zip(axes, idxs):
        image, meta, label = dataset[idx]
        rgb = denormalize(image)
        true_n = class_names[int(label)]
        pred_n = class_names[y_pred[idx]]
        ok = "CORRECT" if y_pred[idx] == int(label) else "MISCLASSIFIED"

        ax.imshow(rgb)
        ax.set_title(
            f"[{ok}] {y_conf[idx]:.0%}\nTrue: {true_n}\nPred: {pred_n}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[len(idxs) :]:
        ax.axis("off")

    fig.suptitle("Architecture 2 (Swin-B) Sample Predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "arch2_sample_predictions.png", dpi=160)
    plt.close(fig)

    # 3. Attention-rollout grid
    print("[+] Generating Swin-B cross-attention rollout grid...")
    rollout_idxs = idxs[:6]
    fig, axes = plt.subplots(len(rollout_idxs), 2, figsize=(8, 3 * len(rollout_idxs)))
    if len(rollout_idxs) == 1:
        axes = np.array([axes])

    for row, idx in enumerate(rollout_idxs):
        image, meta, label = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device, dtype=torch.float)
        meta_tensor = torch.as_tensor(meta, dtype=torch.float).unsqueeze(0).to(device)

        attn_map = model.attention_rollout(img_tensor, meta_tensor)  # returns np.ndarray (7, 7)

        rgb = denormalize(image)
        overlay = overlay_attention_map(rgb, attn_map)

        pred = y_pred[idx]
        ok = "CORRECT" if pred == int(label) else "WRONG"

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Original (True: {class_names[int(label)]})", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Cross-Attn Rollout [{ok}] Pred: {class_names[pred]} ({y_conf[idx]:.0%})", fontsize=8
        )
        axes[row, 1].axis("off")

        Image.fromarray((overlay * 255).astype(np.uint8)).save(
            out_dir / f"arch2_attention_rollout_{row:02d}.png"
        )

    fig.suptitle("Architecture 2 (Swin-B + Cross-Attention) Attention Rollout Grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "arch2_attention_rollout_grid.png", dpi=160)
    plt.close(fig)

    # 4. Learning Curves (if CSV exists)
    if CSV_PATH.exists():
        import csv

        rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        train_acc = [float(r["train_acc"]) for r in rows]
        val_acc_list = [float(r["val_acc"]) for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, marker="o", label="Train Loss", color="indigo")
        axes[0].plot(epochs, val_loss, marker="o", label="Val Loss", color="mediumpurple")
        axes[0].set_title("Architecture 2 Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(epochs, train_acc, marker="o", label="Train Acc", color="teal")
        axes[1].plot(epochs, val_acc_list, marker="o", label="Val Acc", color="lightseagreen")
        axes[1].set_title("Architecture 2 Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Architecture 2 (Swin-B) Training & Validation Curves", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "arch2_learning_curves.png", dpi=160)
        plt.close(fig)
        print(f"[+] Wrote learning curves to {out_dir / 'arch2_learning_curves.png'}")

    print(f"\n[+] Architecture 2 evaluation complete. Output folder: {out_dir}")
    print(f"    Validation Accuracy : {acc:.4f} ({len(dataset)} samples)")


if __name__ == "__main__":
    evaluate_arch2()
