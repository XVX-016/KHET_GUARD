"""
Architecture 3 (RT-DETR) Fine-Tuning Pipeline
----------------------------------------------
Fine-tunes a pretrained RT-DETR object detection model on the PlantDoc dataset
with bounding-box annotations for plant disease regions.

Dataset:
    PlantDoc (Singh et al., CoDS-COMAD 2020)
    ~2,578 images, ~8,900 bounding boxes, 27–30 plant disease classes.

Output Directory:
    model/exports/arch3/
    ├── weights/
    │   └── best.pt
    └── results/
        ├── results.csv
        └── confusion_matrix.png
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import RTDETR

# Ensure arch3_rtdetr is on python path
sys.path.append(str(Path(__file__).parent.resolve()))
from prepare_plantdoc_yolo import prepare_plantdoc


def train_arch3_rtdetr(
    epochs: int = 5,
    imgsz: int = 640,
    batch_size: int = 8,
    lr0: float = 1e-4,
    device_id: str = "0",
) -> None:
    yaml_path = Path("arch3_rtdetr/data/plantdoc/plantdoc.yaml")
    
    # 1. Prepare dataset if not already converted
    if not yaml_path.exists():
        print("[!] PlantDoc dataset YAML not found. Running dataset conversion script...")
        yaml_path = Path(prepare_plantdoc())
    else:
        print(f"[OK] Found existing dataset configuration: {yaml_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print("Architecture 3 (RT-DETR) Fine-Tuning Specification & Timing")
    print("=" * 60)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Model Checkpoint     : rtdetr-l.pt (Pretrained COCO)")
    print(f"Dataset              : PlantDoc (~2,342 train, ~236 val images)")
    print(f"Epoch Budget         : {epochs} epochs")
    print(f"Batch Size           : {batch_size}")
    print(f"Image Resolution     : {imgsz} x {imgsz}")
    print(f"Initial Learning Rate: {lr0}")
    print("-" * 60)
    print("Realistic Time / Hardware Expectations:")
    print("  * CUDA GPU (RTX series / T4): ~2.5 - 3.5 minutes per epoch.")
    print(f"  * Expected Total Training Time ({epochs} epochs): ~12 - 18 minutes total.")
    print("  * Note: PlantDoc is a noisy real-world dataset. mAP50 benchmarks typically")
    print("    range around 0.40 - 0.65 mAP (unlike PlantVillage 97%+ classification).")
    print("=" * 60 + "\n")

    # 2. Load Pretrained RT-DETR
    print("[+] Loading pretrained RT-DETR model (rtdetr-l.pt)...")
    model = RTDETR("rtdetr-l.pt")

    # 3. Output directories
    project_dir = Path("model/exports/arch3").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # 4. Fine-tuning execution
    print(f"[+] Starting fine-tuning for {epochs} epochs...")
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        lr0=lr0,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_dir),
        name="results",
        exist_ok=True,
        save=True,
        plots=True,
        workers=2,
    )

    print("\n" + "=" * 60)
    print("[OK] Architecture 3 RT-DETR Training Complete!")
    print(f"[OK] Checkpoints & Evaluation saved to: {project_dir / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    train_arch3_rtdetr(epochs=5, imgsz=640, batch_size=8, lr0=1e-4)
