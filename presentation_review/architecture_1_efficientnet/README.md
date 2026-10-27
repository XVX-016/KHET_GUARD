# Architecture 1: EfficientNet-B4 + Late Concatenation Fusion

## Overview
- **Type**: Single-label image classification with multimodal tabular fusion
- **Visual Backbone**: Pretrained EfficientNet-B4 (1,792-dimensional visual embedding)
- **Metadata Fusion**: 16-D agronomic metadata (crop type, stage, soil, weather) projected through a 2-layer MLP (64 -> 32) and concatenated with image features (`[1792 + 32 = 1824-D]`).
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) on the final convolutional layer.

---

## Performance Metrics

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **97.19%** | Evaluated on PlantVillage validation split |
| **Macro-F1 Score** | **0.9712** | Balanced across all 10 evaluated crop disease classes |
| **Validation Loss** | **0.0713** | CrossEntropyLoss at optimal checkpoint |
| **Parameters** | **18.06 M** | EfficientNet-B4 backbone + MLP head |
| **Input Resolution** | **224 × 224** | Standard input size |
| **E2E GPU Latency** | **31.11 ms** | Total pipeline: Preproc + Forward + Classification |
| **Pure Forward Latency** | **47.41 ms** | Raw model forward pass on CUDA |

---

## Directory Contents
- `model_arch1_efficientnet.py`: Complete architecture definition, multimodal dataset loader, and PyTorch training loop.
- `eval_gradcam_arch1.py`: Evaluation pipeline generating confusion matrix, class metrics, and Grad-CAM attention visualizations.
- `results/confusion_matrix.png`: Validation confusion matrix across all classes.
- `results/gradcam_grid.png`: Grad-CAM heatmaps showing where the CNN focuses on diseased leaves.
- `results/learning_curves.png`: Loss and accuracy curves across training epochs.
- `results/val_metrics.json`: Exported validation metrics JSON.
