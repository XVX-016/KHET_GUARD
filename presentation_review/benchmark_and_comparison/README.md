# 3-Architecture Comparison & Benchmark Summary

This folder contains cross-model comparative benchmarks, tradeoff visualizations, and execution scripts to present to reviewers.

---

## Direct Architectural Comparison

| Dimension | Architecture 1 (EfficientNet-B4) | Architecture 2 (Swin-B Transformer) | Architecture 3 (RT-DETR) |
| :--- | :--- | :--- | :--- |
| **Task Paradigm** | Image Classification | Image Classification | Multi-Lesion Object Detection |
| **Model Type** | Convolutional Neural Network (CNN) | Shifted-Window Vision Transformer | Real-Time Detection Transformer |
| **Input Resolution** | 224 × 224 | 224 × 224 | 640 × 640 |
| **Parameter Count** | **18.06 M** (Lightest) | 91.28 M (Heaviest) | 32.97 M (Medium) |
| **Primary Metric** | **97.19%** Accuracy | **98.83%** Accuracy *(Best)* | **47.57%** mAP@50 (PlantDoc) |
| **Macro-F1 Score** | 0.9712 | **0.9881** *(Best)* | N/A (Object Detection) |
| **Fusion Mechanism** | Late Concatenation `[1792 + 32]` | Multi-Head Cross-Attention (8 heads) | Metadata Query Injection into 300 Queries |
| **Explainability** | Grad-CAM Heatmaps | Attention Rollout (7×7 spatial) | Bounding Box Localizations |
| **Headline E2E Latency** | **31.11 ms** *(Fastest)* | 47.42 ms | 128.72 ms |
| **Pure Forward Latency**| **47.41 ms** | 80.08 ms | 80.23 ms |
| **Best Used For** | Fast edge/mobile triage | Maximum classification accuracy | Localizing co-occurring infections |

---

## Generated Comparison Charts
- `model_comparison_metrics.png`: Side-by-side grouped bar chart highlighting primary accuracy and Macro-F1 across all architectures.
- `accuracy_vs_latency_tradeoff.png`: Scatter plot illustrating the accuracy vs. latency frontier annotated with parameter counts and resolutions.

## Scripts
- `generate_comparison_charts.py`: Reusable script used to render the publication-quality comparison charts.
- `benchmark_latency_rigorous.py`: Authoritative latency benchmarking harness with GPU synchronization and warmups.
