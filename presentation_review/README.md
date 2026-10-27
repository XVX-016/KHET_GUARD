# Presentation Review: 3 Agricultural AI Architectures

This directory organizes all **code**, **evaluations**, and **visual results** for the 3 evaluated architectures into clean, self-contained folders ready for presentation.

---

## Folder Layout

```
presentation_review/
│
├── architecture_1_efficientnet/       # CNN baseline + Late Fusion
│   ├── model_arch1_efficientnet.py   # Model architecture & PyTorch dataset/training
│   ├── eval_gradcam_arch1.py         # Evaluation & Grad-CAM visualizer
│   ├── README.md                     # Architecture specs & metrics summary
│   └── results/                      # Confusion matrix, Grad-CAM grids, curves
│
├── architecture_2_swin/               # Vision Transformer + Cross-Attention Fusion
│   ├── model_arch2_swin.py           # Swin-B + Multi-Head Cross-Attention module
│   ├── train_arch2_swin.py           # Training pipeline
│   ├── eval_attention_arch2.py       # Evaluation & Spatial Attention Rollout
│   ├── README.md                     # Architecture specs & metrics summary
│   └── results/                      # Confusion matrix, attention maps, curves
│
├── architecture_3_rtdetr/             # Transformer Object Detector + Query Fusion
│   ├── model_arch3_rtdetr.py         # RT-DETR model + Metadata Query Injection
│   ├── train_arch3_rtdetr.py         # Detection fine-tuning on PlantDoc
│   ├── eval_rtdetr_arch3.py          # Evaluation & bounding box visualizer
│   ├── README.md                     # Architecture specs & metrics summary
│   └── results/                      # Sample detections, PR curve, confusion matrix
│
└── benchmark_and_comparison/          # Cross-Architectural Comparison & Latency
    ├── model_comparison_metrics.png   # Grouped bar chart (Accuracy & F1)
    ├── accuracy_vs_latency_tradeoff.png # Tradeoff scatter plot
    ├── generate_comparison_charts.py  # Script generating the comparison figures
    ├── benchmark_latency_rigorous.py  # GPU-synchronized latency benchmark script
    └── README.md                     # Comprehensive comparison matrix & tradeoffs
```

---

## Executive Summary for Reviewers

| Architecture | Backbone | Task | Key Metric | E2E Latency | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arch 1** | EfficientNet-B4 | Classification | **97.19% Acc** (0.971 F1) | **31.11 ms** | 18.06 M |
| **Arch 2** | Swin-B ViT | Classification | **98.83% Acc** (0.988 F1) | **47.42 ms** | 91.28 M |
| **Arch 3** | RT-DETR-L | Object Detection | **47.57% mAP@50** | **128.72 ms** | 32.97 M |

> All models fuse 16-D agronomic metadata (crop, weather, soil) with visual features, demonstrating significant domain-informed precision gains across both leaf classification and spatial lesion detection.
