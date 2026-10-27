# Architecture 2: Swin-B Transformer + Cross-Attention Fusion

## Overview
- **Type**: Single-label image classification with cross-attention multimodal fusion
- **Visual Backbone**: Swin-B (Shifted Window Vision Transformer, 1024-dimensional patch representations)
- **Metadata Fusion**: Multi-head Cross-Attention (8 heads). The 49 image patch tokens act as **queries**, and the projected 16-D metadata token acts as **key & value**. This allows the network to learn *spatially where* metadata context matters most.
- **Explainability**: Attention-Rollout mapping generating 7×7 spatial importance overlays across transformer blocks.

---

## Performance Metrics

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **98.83%** | **Highest accuracy** across all evaluated architectures |
| **Macro-F1 Score** | **0.9881** | Exceptional precision and recall across fine-grained leaf spots |
| **Validation Loss** | **0.0366** | CrossEntropyLoss at optimal checkpoint |
| **Parameters** | **91.28 M** | Swin-B backbone + Cross-Attention module |
| **Input Resolution** | **224 × 224** | 7×7 patch token grid (window size 7) |
| **E2E GPU Latency** | **47.42 ms** | Preprocessing + Forward + Classification |
| **Pure Forward Latency** | **80.08 ms** | CUDA-synchronized Swin-B forward pass |

---

## Directory Contents
- `model_arch2_swin.py`: Model architecture definition, CrossAttentionFusion module, and Attention Rollout implementation.
- `train_arch2_swin.py`: PyTorch training routine with Cosine Annealing, AdamW, and Mixed Precision.
- `eval_attention_arch2.py`: Generates confusion matrix, metrics, and spatial attention rollout heatmaps.
- `results/arch2_confusion_matrix.png`: Confusion matrix on validation set.
- `results/arch2_attention_rollout_grid.png`: Spatial attention rollout maps showing cross-attention highlights on symptoms.
- `results/arch2_learning_curves.png`: Training convergence curves.
- `results/arch2_val_metrics.json`: Validation metrics summary.
