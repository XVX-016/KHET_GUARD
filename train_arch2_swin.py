"""
train_arch2_swin.py
-------------------

Training script for Architecture 2 — Swin-B + cross-attention metadata fusion.

Architecture 1 files (DO NOT EDIT):
  fusion_model.py          — TF/Keras CNN fusion model
  train_pytorch_fusion.py  — EfficientNet-B4 PyTorch training script
  train_with_gradcam.py    — TF-based training + Grad-CAM

This script is intentionally written to mirror the conventions of
train_pytorch_fusion.py so metrics CSVs and checkpoints are directly
comparable. Key intentional differences are marked with "ARCH2:".

Verified hardware characteristics (RTX 4060 Laptop, 8 GB VRAM, AMP on):
  Arch 1 (EfficientNet-B4): 334 ms/batch, 1.96 GB VRAM peak
  Arch 2 (Swin-B):          179 ms/batch, 3.53 GB VRAM peak

Time estimates for the disease dataset (2132 images, batch=16):
  Arch 1 — 40 s/epoch  → 6 epochs ≈  4 min
  Arch 2 — 22 s/epoch  → 6 epochs ≈  2.2 min   (Swin-B is 1.86x faster/batch)

Honest assessment (read before running):
  See the printed WARNING block at the bottom of main().

Author: Khet Guard ML Team
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# ARCH2: Use the modern torch.amp API (torch >= 2.0) to avoid deprecation warnings
# that appear when using torch.cuda.amp.autocast in torch 2.8.
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Reuse the dataset class and transforms verbatim from Architecture 1.
# This guarantees the same split, the same augmentation, and the same
# normalisation — so any accuracy difference is purely due to the backbone.
from train_pytorch_fusion import (
    CONFIG as ARCH1_CONFIG,
    ImageMetadataDataset,
    train_transform,
    val_transform,
)
from fusion_model_arch2_swin import SwinFusionModel, create_swin_fusion_model

# ---------------------------------------------------------------------------
# Logging — identical format to train_pytorch_fusion.py
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# ARCH2: Configuration
# Mirrors ARCH1_CONFIG structure exactly so both can be fed to the same
# plotting / comparison scripts.  Only output_dir differs to guarantee
# checkpoints never overwrite Architecture 1 artefacts.
# ===========================================================================
ARCH2_CONFIG: dict = {
    "models": {
        # Same paths as Arch 1 — we share datasets, not models
        "disease": {
            "num_classes": 10,   # actual class count in the current subset
                                 # (will be overridden at runtime by label scan)
            "data_path": ARCH1_CONFIG["models"]["disease"]["data_path"],
        },
        "pest": {
            "num_classes": 20,
            "data_path": ARCH1_CONFIG["models"]["pest"]["data_path"],
        },
        "cattle": {
            "num_classes": 41,
            "data_path": ARCH1_CONFIG["models"]["cattle"]["data_path"],
        },
    },
    "training": {
        # ------- identical to Arch 1 so runs are directly comparable -------
        "batch_size": 16,   # fits in 3.53 GB VRAM with Swin-B + AMP
        "epochs": 6,        # same short run — see honest_assessment() below
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dropout_rate": 0.3,
        "patience": 10,
        "use_amp": True,
        "num_workers": 0,   # Windows-safe (no fork on Windows)
        "seed": 42,
        # ------- Arch-2 specific additions ---------------------------------
        "num_attn_heads": 8,   # cross-attention heads; must divide 1024
        "pretrained": True,    # load ImageNet-1k Swin-B weights
        "freeze_backbone_epochs": 2,  # train only the fusion head + metadata
                                       # branch for the first N epochs so the
                                       # randomly-initialised cross-attn layer
                                       # does not corrupt pretrained features
    },
    # ARCH2: Separate output dir — will never shadow Arch 1 files
    "output_dir": "model/exports/arch2",
}


# ===========================================================================
# Honest Assessment
# (printed by main() before training starts so you can stop if needed)
# ===========================================================================
HONEST_ASSESSMENT = """
+------------------------------------------------------------------------------+
|  HONEST ASSESSMENT -- read before committing to a training run               |
+------------------------------------------------------------------------------+
|  Hardware  : RTX 4060 Laptop, 8 GB VRAM                                     |
|  Benchmark : Swin-B  179 ms/batch  3.53 GB VRAM   (AMP, batch=16)          |
|              EfficientNet-B4  334 ms/batch  1.96 GB VRAM                    |
|  Estimate  : Swin-B  ~22 s/epoch  ->  6 epochs approx 2.2 minutes total      |
|              EfficientNet-B4  ~40 s/epoch  ->  6 epochs approx 4 minutes total|
|                                                                              |
|  WHY SWIN-B MAY SCORE LOWER THAN ARCH 1 AT ONLY 6 EPOCHS:                  |
|                                                                              |
|  1. Dataset is tiny for a ViT (2132 images, 10 classes).                    |
|     EfficientNet-B4 has stronger CNN inductive biases (locality,            |
|     translation equivariance) that help when fine-tuning data is scarce.   |
|     Vision Transformers generally need more data or more epochs to           |
|     match CNNs on small datasets -- this is well-established in the          |
|     ViT literature (Dosovitskiy et al., 2020; Touvron et al. DeiT).        |
|                                                                              |
|  2. The cross-attention fusion head starts from random weights.             |
|     freeze_backbone_epochs=2 mitigates this, but 6 epochs is still         |
|     short for the new attention weights to converge.                         |
|                                                                              |
|  3. Swin-B has 91M params vs ~19M for EfficientNet-B4.  With only          |
|     1705 training samples, the larger model is at greater risk of           |
|     over-fitting -- especially after unfreezing (epoch 3+).                  |
|                                                                              |
|  WHAT TO EXPECT:                                                             |
|  * Arch 1 will likely show higher val_acc at epoch 6 on this dataset.       |
|  * Arch 2's advantage shows at 20-50 epochs with data augmentation,         |
|    or with more data (the full PlantVillage = 54,000 images).               |
|  * This 6-epoch run is still valid as a controlled comparison baseline.     |
|                                                                              |
|  MITIGATION ALREADY BAKED IN:                                                |
|  * freeze_backbone_epochs=2 -> head warms up before backbone fine-tunes     |
|  * weight_decay=1e-4 -> L2 regularisation against over-fitting              |
|  * AMP (bfloat16) -> faster, lower VRAM, lets you run more epochs if        |
|    you increase ARCH2_CONFIG["training"]["epochs"] before running           |
+------------------------------------------------------------------------------+
"""


# ===========================================================================
# Training function
# (same structure as train_model() in train_pytorch_fusion.py)
# ===========================================================================

def train_model(model_name: str, config: dict) -> tuple[Path | None, Path | None]:
    """
    Train SwinFusionModel for a single task (disease / pest / cattle).

    Parameters
    ----------
    model_name : one of "disease", "pest", "cattle"
    config     : ARCH2_CONFIG dict

    Returns
    -------
    (checkpoint_path, onnx_path) — both as Path objects (or None on failure)
    """
    logger.info(f"[arch2/{model_name}] Starting training ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[arch2/{model_name}] Device: {device}")

    data_path = config["models"][model_name]["data_path"]
    if not os.path.exists(data_path):
        logger.warning(
            f"[arch2/{model_name}] Dataset not found: {data_path}. Skipping."
        )
        return None, None

    seed = config["training"]["seed"]

    # ------------------------------------------------------------------
    # Datasets — ARCH2: reuse ImageMetadataDataset + transforms from Arch 1
    # so the exact same 80/20 stratified split is used.
    # ------------------------------------------------------------------
    train_dataset = ImageMetadataDataset(
        data_path, transform=train_transform, train=True, seed=seed
    )
    val_dataset = ImageMetadataDataset(
        data_path, transform=val_transform, train=False, seed=seed
    )

    # Determine actual class count from labels (same logic as Arch 1)
    num_classes = int(
        max(train_dataset.labels.max(), val_dataset.labels.max()) + 1
    )
    logger.info(
        f"[arch2/{model_name}] {len(train_dataset)} train / "
        f"{len(val_dataset)} val  |  {num_classes} classes"
    )

    workers = config["training"]["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=workers,
    )

    # ------------------------------------------------------------------
    # ARCH2: Model
    # ------------------------------------------------------------------
    model = create_swin_fusion_model(
        num_classes=num_classes,
        metadata_dim=16,
        dropout_rate=config["training"]["dropout_rate"],
        num_attn_heads=config["training"]["num_attn_heads"],
        pretrained=config["training"]["pretrained"],
    ).to(device)

    logger.info(f"[arch2/{model_name}] Model info: {model.model_info()}")
    params = model.count_parameters()
    logger.info(
        f"[arch2/{model_name}] Parameters: "
        f"backbone={params['swin_backbone']:,}  "
        f"meta={params['metadata_branch']:,}  "
        f"attn={params['cross_attention']:,}  "
        f"head={params['classifier_head']:,}  "
        f"total={params['total']:,}"
    )

    # ------------------------------------------------------------------
    # ARCH2: Freeze backbone for first N epochs so the randomly-initialised
    # cross-attention layer warms up without corrupting pretrained features.
    # Backbone is unfrozen in the training loop below.
    # ------------------------------------------------------------------
    freeze_epochs = config["training"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        for p in model.swin_features.parameters():
            p.requires_grad = False
        for p in model.swin_norm.parameters():
            p.requires_grad = False
        logger.info(
            f"[arch2/{model_name}] Backbone frozen for first {freeze_epochs} epoch(s)."
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ARCH2: Use torch.amp API (not deprecated torch.cuda.amp)
    use_amp = config["training"]["use_amp"] and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------
    # Output paths — ARCH2: all under model/exports/arch2/
    # so they can NEVER overwrite Architecture 1 checkpoints in model/exports/
    # ------------------------------------------------------------------
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / f"best_model_{model_name}.pth"
    # ARCH2: CSV has same columns as metrics_disease.csv produced by Arch 1
    # Columns: epoch, train_loss, train_acc, val_loss, val_acc, lr
    csv_path = out_dir / f"metrics_arch2_{model_name}.csv"
    log_dir = out_dir / f"logs_{model_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None

    # ------------------------------------------------------------------
    # Training loop — mirrors train_pytorch_fusion.py exactly
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(config["training"]["epochs"]):

        # ARCH2: Unfreeze backbone after freeze_backbone_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for p in model.swin_features.parameters():
                p.requires_grad = True
            for p in model.swin_norm.parameters():
                p.requires_grad = True
            # Rebuild optimizer so newly unfrozen params get a learning rate
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            logger.info(
                f"[arch2/{model_name}] Epoch {epoch+1}: backbone unfrozen, "
                f"optimizer reset."
            )

        # ---- Train phase ------------------------------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, metadata, labels in train_loader:
            images   = images.to(device, dtype=torch.float)
            metadata = metadata.to(device, dtype=torch.float)
            labels   = labels.to(device, dtype=torch.long)  # CrossEntropyLoss requires long

            optimizer.zero_grad()

            if scaler is not None:
                # ARCH2: device_type arg required by torch.amp.autocast
                with autocast(device_type="cuda"):
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # ---- Validation phase -------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images   = images.to(device, dtype=torch.float)
                metadata = metadata.to(device, dtype=torch.float)
                labels   = labels.to(device, dtype=torch.long)

                if scaler is not None:
                    with autocast(device_type="cuda"):
                        outputs = model(images, metadata)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # ---- Metrics — same formula as Arch 1 ---------------------------
        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        train_acc   = 100 * train_correct / train_total
        val_acc     = 100 * val_correct   / val_total

        scheduler.step(val_loss)

        # ---- TensorBoard ------------------------------------------------
        if writer is not None:
            writer.add_scalar("Loss/Train",       train_loss, epoch)
            writer.add_scalar("Loss/Validation",  val_loss,   epoch)
            writer.add_scalar("Accuracy/Train",   train_acc,  epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # ---- CSV row — identical columns to metrics_disease.csv ----------
        row = {
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        # ---- Logging — same format as Arch 1 ----------------------------
        logger.info(
            f"[arch2/{model_name}] Epoch {epoch+1}/{config['training']['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # ---- Best-model checkpoint (on val_loss, same as Arch 1) --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":           val_loss,
                    "val_acc":            val_acc,
                    "num_classes":        num_classes,
                    "class_names":        train_dataset.class_names,
                    "data_path":          data_path,
                    # ARCH2: Store arch metadata in checkpoint for eval scripts
                    "arch":               "arch2_swin_fusion",
                    "backbone":           "swin_b",
                    "num_attn_heads":     config["training"]["num_attn_heads"],
                },
                checkpoint_path,
            )
            # Also save class-name lookup (same pattern as Arch 1)
            if train_dataset.class_names:
                labels_path = out_dir / f"{model_name}_labels.json"
                labels_path.write_text(
                    json.dumps(train_dataset.class_names, indent=2),
                    encoding="utf-8",
                )
            logger.info(
                f"[arch2/{model_name}] Saved best checkpoint → {checkpoint_path} "
                f"(val_loss={val_loss:.4f})"
            )
        else:
            patience_counter += 1

        # ---- Early stopping ---------------------------------------------
        if patience_counter >= config["training"]["patience"]:
            logger.info(
                f"[arch2/{model_name}] Early stopping after {epoch+1} epochs."
            )
            break

    # ------------------------------------------------------------------
    # ONNX export — same opset and dynamic-axes as Arch 1
    # ------------------------------------------------------------------
    model.eval()
    dummy_image    = torch.randn(1, 3, 224, 224).to(device)
    dummy_metadata = torch.randn(1, 16).to(device)
    onnx_path = out_dir / f"{model_name}_arch2_model.onnx"

    try:
        torch.onnx.export(
            model,
            (dummy_image, dummy_metadata),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image", "metadata"],
            output_names=["output"],
            dynamic_axes={
                "image":    {0: "batch_size"},
                "metadata": {0: "batch_size"},
                "output":   {0: "batch_size"},
            },
        )
        logger.info(f"[arch2/{model_name}] Exported ONNX → {onnx_path}")
    except Exception as exc:
        logger.error(f"[arch2/{model_name}] ONNX export failed: {exc}")
        onnx_path = None

    # ------------------------------------------------------------------
    # Write CSV metrics — identical columns to metrics_disease.csv
    # Columns: epoch, train_loss, train_acc, val_loss, val_acc, lr
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(history)
    logger.info(f"[arch2/{model_name}] Per-epoch metrics → {csv_path}")

    if writer is not None:
        writer.close()

    logger.info(
        f"[arch2/{model_name}] Done. "
        f"Best val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.2f}%"
    )
    return checkpoint_path, onnx_path


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Print the honest assessment FIRST so the user can stop if needed
    print(HONEST_ASSESSMENT)

    logger.info("Starting Architecture 2 (Swin-B) training pipeline ...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {p.name}  |  VRAM: {p.total_memory/1024**3:.1f} GB")
        torch.cuda.empty_cache()

    out_dir = Path(ARCH2_CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    trained: dict = {}

    for model_name in ARCH2_CONFIG["models"]:
        try:
            ckpt, onnx_file = train_model(model_name, ARCH2_CONFIG)
            trained[model_name] = {
                "checkpoint": str(ckpt)  if ckpt      else None,
                "onnx":       str(onnx_file) if onnx_file else None,
            }
        except Exception as exc:
            logger.error(f"[arch2/{model_name}] Training failed: {exc}")
            trained[model_name] = {"checkpoint": None, "onnx": None}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = out_dir / "training_summary_arch2.json"
    summary_path.write_text(json.dumps(trained, indent=2), encoding="utf-8")
    logger.info(f"Summary → {summary_path}")
    logger.info(
        "TensorBoard: tensorboard --logdir=model/exports/arch2 "
        "--host=127.0.0.1 --port=6007"
    )
    logger.info(
        "To compare Arch 1 vs Arch 2 metrics side-by-side:\n"
        "  Arch 1 CSV: model/exports/metrics_disease.csv\n"
        f"  Arch 2 CSV: {out_dir}/metrics_arch2_disease.csv\n"
        "  (Same column names: epoch, train_loss, train_acc, val_loss, val_acc, lr)"
    )


if __name__ == "__main__":
    main()
