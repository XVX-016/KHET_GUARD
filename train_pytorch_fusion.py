"""
train_pytorch_fusion.py

Unified PyTorch training for Crop Disease + Pest + Cattle models with:
- Image + metadata fusion
- Grad-CAM support
- Best-model checkpointing
- TensorBoard logging
- ONNX export
- Automatic Mixed Precision (AMP) for GPU memory optimization
- Fixed label types for CrossEntropyLoss

Author: Khet Guard ML Team
"""

import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
import logging
from sklearn.model_selection import train_test_split

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard extra not installed
    SummaryWriter = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Config
# ======================
CONFIG = {
    "models": {
        "disease": {"num_classes": 38, "data_path": "data/processed/plantvillage_color_dataset.npz"},
        "pest": {"num_classes": 20, "data_path": "data/processed/pest_dataset.npz"},
        "cattle": {"num_classes": 41, "data_path": "data/processed/cattle_dataset.npz"}
    },
    "training": {
        "batch_size": 16,  # Reduced for GPU memory safety
        "epochs": 6,       # Short demonstration run for Review-2
        "learning_rate": 1e-4,
        "dropout_rate": 0.3,
        "patience": 10,
        "use_amp": True,  # Enable Automatic Mixed Precision
        "num_workers": 0,  # Windows-safe
        "seed": 42,
    },
    "output_dir": "model/exports"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ======================
# Dataset Loader
# ======================
class ImageMetadataDataset(Dataset):
    def __init__(self, npz_path, transform=None, train=True, seed=42):
        data = np.load(npz_path, allow_pickle=True)
        images = data["images"]
        metadata = data["metadata"] if "metadata" in data.files else np.zeros((len(images), 16), dtype=np.float32)
        labels = data["labels"].astype(np.int64)
        self.class_names = (
            [str(x) for x in data["class_names"].tolist()]
            if "class_names" in data.files
            else None
        )
        self.transform = transform

        # Stratified split so a class-concatenated NPZ cannot leak last classes into val-only.
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
        keep = train_idx if train else val_idx
        self.images = images[keep]
        self.metadata = metadata[keep]
        self.labels = labels[keep]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        meta = self.metadata[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, meta, label

# Data transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# Model Definition (Image + Metadata Fusion)
# ======================
class FusionModel(nn.Module):
    def __init__(self, num_classes, metadata_dim=16, dropout_rate=0.3):
        super().__init__()
        # Use EfficientNet-B4 as backbone
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Metadata branch
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Grad-CAM hooks
        self.gradients = None
        self.activation = None
        self.backbone.features[-1].register_forward_hook(self._save_activation)
        self.backbone.features[-1].register_backward_hook(self._save_gradient)

    def forward(self, image, metadata):
        x_img = self.backbone(image)
        x_meta = self.metadata_fc(metadata)
        x = torch.cat([x_img, x_meta], dim=1)
        out = self.classifier(x)
        return out

    # Grad-CAM hooks
    def _save_activation(self, module, input, output):
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_gradcam(self, image, metadata, class_idx=None):
        """Generate Grad-CAM visualization"""
        if self.activation is None or self.gradients is None:
            raise ValueError("Grad-CAM not available. Run forward pass first.")
        
        # Get gradients for the target class
        if class_idx is None:
            class_idx = torch.argmax(self.forward(image, metadata), dim=1)
        
        # Global average pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the feature maps
        for i in range(self.activation.size(1)):
            self.activation[:, i, :, :] *= pooled_grads[i]
        
        # Generate heatmap
        heatmap = torch.mean(self.activation, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.cpu().numpy()

# ======================
# Training Function
# ======================
def train_model(model_name, config):
    logger.info(f"Starting training for {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_path = config["models"][model_name]["data_path"]

    # Check if data exists
    if not os.path.exists(data_path):
        logger.warning(f"Dataset not found: {data_path}. Skipping {model_name} training.")
        return None, None

    seed = config["training"].get("seed", 42)
    # Create datasets
    train_dataset = ImageMetadataDataset(data_path, transform=train_transform, train=True, seed=seed)
    val_dataset = ImageMetadataDataset(data_path, transform=val_transform, train=False, seed=seed)
    num_classes = int(max(train_dataset.labels.max(), val_dataset.labels.max()) + 1)
    logger.info(
        f"{model_name}: {len(train_dataset)} train / {len(val_dataset)} val images, {num_classes} classes"
    )

    workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=workers
    )

    # Create model
    model = FusionModel(num_classes=num_classes, dropout_rate=config["training"]["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize AMP scaler if using mixed precision
    scaler = GradScaler() if config["training"]["use_amp"] and device.type == 'cuda' else None

    # TensorBoard logging (optional)
    log_dir = Path(config["output_dir"]) / f"logs_{model_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None
    csv_path = Path(config["output_dir"]) / f"metrics_{model_name}.csv"
    history = []

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = Path(config["output_dir"]) / f"best_model_{model_name}.pth"

    for epoch in range(config["training"]["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, metadata, labels in train_loader:
            images = images.to(device, dtype=torch.float)
            metadata = metadata.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)  # Fix: ensure long tensor for CrossEntropyLoss

            optimizer.zero_grad()
            
            # Use AMP if enabled
            if scaler is not None:
                with autocast():
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
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device, dtype=torch.float)
                metadata = metadata.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)  # Fix: ensure long tensor for CrossEntropyLoss

                # Use AMP for validation if enabled
                if scaler is not None:
                    with autocast():
                        outputs = model(images, metadata)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images, metadata)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        logger.info(f"[{model_name}] Epoch {epoch+1}/{config['training']['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': num_classes,
                'class_names': train_dataset.class_names,
                'data_path': data_path,
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["training"]["patience"]:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Export to ONNX
    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_metadata = torch.randn(1, 16).to(device)
    onnx_path = Path(config["output_dir"]) / f"{model_name}_model.onnx"
    
    try:
        torch.onnx.export(
            model, 
            (dummy_image, dummy_metadata), 
            onnx_path, 
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['image', 'metadata'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'metadata': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"Exported {model_name} model to ONNX: {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export {model_name} to ONNX: {e}")
        onnx_path = None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=history[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(history)
    logger.info(f"Wrote per-epoch metrics to {csv_path}")

    if writer is not None:
        writer.close()
    return checkpoint_path, onnx_path

# ======================
# Main
# ======================
def main():
    logger.info("Starting PyTorch training pipeline...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set memory management for better GPU utilization
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    trained_models = {}
    
    for model_name in CONFIG["models"].keys():
        try:
            logger.info(f"Starting training for {model_name}...")
            ckpt, onnx_file = train_model(model_name, CONFIG)
            trained_models[model_name] = {
                "checkpoint": str(ckpt) if ckpt else None,
                "onnx": str(onnx_file) if onnx_file else None
            }
            logger.info(f"{model_name} training finished. Checkpoint: {ckpt}, ONNX: {onnx_file}")
            
            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            trained_models[model_name] = {"checkpoint": None, "onnx": None}
            
            # Clear GPU memory even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save training summary
    summary_path = Path(CONFIG["output_dir"]) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(trained_models, f, indent=2)
    
    logger.info(f"Training pipeline complete. Summary saved to {summary_path}")
    logger.info("Run TensorBoard with: tensorboard --logdir=model/exports --host=127.0.0.1 --port=6006")

if __name__ == "__main__":
    main()
