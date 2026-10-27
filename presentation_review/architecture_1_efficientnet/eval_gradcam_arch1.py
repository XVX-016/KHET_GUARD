"""
Generate Review-2 intermediate results from a trained fusion checkpoint:
  1) validation confusion matrix
  2) sample predictions (true vs predicted + confidence)
  3) Grad-CAM overlays
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

from train_pytorch_fusion import CONFIG, FusionModel, ImageMetadataDataset, val_transform

OUT_DIR = Path("model/exports/review2_results")
CKPT_PATH = Path("model/exports/best_model_disease.pth")
NPZ_PATH = Path(CONFIG["models"]["disease"]["data_path"])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


class GradCAM:
    def __init__(self, model: FusionModel):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.backbone.features[-1]
        target._forward_hooks.clear()
        target._backward_hooks.clear()
        if hasattr(target, "_full_backward_hooks"):
            target._full_backward_hooks.clear()
        # FusionModel registers a deprecated non-full backward hook; reset the flag
        # so we can attach a full backward hook for Grad-CAM.
        target._is_full_backward_hook = None
        self._fh = target.register_forward_hook(self._on_forward)
        self._bh = target.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module, inp, out):
        self.activations = out

    def _on_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, image, metadata, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image, metadata)
        score = logits[0, class_idx]
        score.backward()
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

    def close(self):
        self._fh.remove()
        self._bh.remove()


def overlay_cam(rgb01: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (rgb01.shape[1], rgb01.shape[0]), Image.BILINEAR
    )
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    heat = cmap(cam_arr)[..., :3]
    return np.clip(0.45 * rgb01 + 0.55 * heat, 0, 1)


def main():
    if not CKPT_PATH.exists():
        raise SystemExit(f"Checkpoint not found: {CKPT_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    class_names = ckpt.get("class_names")
    if not class_names:
        data = np.load(NPZ_PATH, allow_pickle=True)
        class_names = [str(x) for x in data["class_names"].tolist()]

    dataset = ImageMetadataDataset(str(NPZ_PATH), transform=val_transform, train=False, seed=42)
    model = FusionModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred, y_conf = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, meta, label = dataset[i]
            logits = model(
                image.unsqueeze(0).to(device, dtype=torch.float),
                torch.as_tensor(meta, dtype=torch.float).unsqueeze(0).to(device),
            )
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
        y_true, y_pred, labels=list(range(num_classes)), target_names=class_names, output_dict=True
    )
    (OUT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "val_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(CKPT_PATH).replace("\\", "/"),
                "val_accuracy": acc,
                "n_val": int(len(dataset)),
                "ckpt_val_acc": ckpt.get("val_acc"),
                "ckpt_val_loss": ckpt.get("val_loss"),
                "ckpt_epoch": ckpt.get("epoch"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Validation confusion matrix  (acc={acc:.1%}, n={len(dataset)})")
    short = [n.replace("___", "\n") for n in class_names]
    ax.set_xticks(range(num_classes), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_classes), short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Pick a mix of correct and incorrect samples, diverse classes.
    rng = np.random.default_rng(42)
    idxs = []
    wrong = np.where(y_true != y_pred)[0]
    right = np.where(y_true == y_pred)[0]
    if len(wrong):
        idxs.extend(rng.choice(wrong, size=min(3, len(wrong)), replace=False).tolist())
    if len(right):
        idxs.extend(rng.choice(right, size=min(5, len(right)), replace=False).tolist())
    idxs = idxs[:8]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, idx in zip(axes, idxs):
        image, meta, label = dataset[idx]
        rgb = denormalize(image)
        true_n = class_names[int(label)]
        pred_n = class_names[y_pred[idx]]
        ok = "OK" if y_pred[idx] == int(label) else "WRONG"
        ax.imshow(rgb)
        ax.set_title(
            f"{ok}  {y_conf[idx]:.0%}\ntrue: {true_n}\npred: {pred_n}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes[len(idxs) :]:
        ax.axis("off")
    fig.suptitle("Sample validation predictions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)

    cam_engine = GradCAM(model)
    cam_idxs = idxs[:6]
    fig, axes = plt.subplots(len(cam_idxs), 2, figsize=(8, 3 * len(cam_idxs)))
    if len(cam_idxs) == 1:
        axes = np.array([axes])
    for row, idx in enumerate(cam_idxs):
        image, meta, label = dataset[idx]
        img_b = image.unsqueeze(0).to(device, dtype=torch.float).requires_grad_(True)
        meta_b = torch.as_tensor(meta, dtype=torch.float).unsqueeze(0).to(device)
        pred = int(y_pred[idx])
        cam = cam_engine(img_b, meta_b, pred)
        rgb = denormalize(image)
        overlay = overlay_cam(rgb, cam)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"true: {class_names[int(label)]}", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM pred: {class_names[pred]} ({y_conf[idx]:.0%})", fontsize=8)
        axes[row, 1].axis("off")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(OUT_DIR / f"gradcam_{row:02d}.png")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gradcam_grid.png", dpi=160)
    plt.close(fig)
    cam_engine.close()

    csv_path = Path("model/exports/metrics_disease.csv")
    if csv_path.exists():
        import csv as _csv

        rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
        epochs = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", label="train")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], marker="o", label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(epochs, [float(r["train_acc"]) for r in rows], marker="o", label="train")
        axes[1].plot(epochs, [float(r["val_acc"]) for r in rows], marker="o", label="val")
        axes[1].set_title("Accuracy (%)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        fig.suptitle("Reduced-scale PlantVillage fusion run (6 epochs)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "learning_curves.png", dpi=160)
        plt.close(fig)

    print(f"Wrote results to {OUT_DIR}")
    print(f"Validation accuracy: {acc:.4f} on {len(dataset)} images")


if __name__ == "__main__":
    main()
