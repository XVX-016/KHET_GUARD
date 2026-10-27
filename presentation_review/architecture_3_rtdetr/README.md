# Architecture 3: RT-DETR + Metadata Query Fusion

## Overview
- **Type**: Multi-lesion Object Detection (localizes and classifies multiple disease spots per leaf/plant)
- **Visual Backbone & Hybrid Encoder**: ResNet/AIFI + CCFM extracting multiscale image feature pyramids.
- **Metadata Fusion**: Metadata Query Conditioning. A 16-D agronomic metadata vector is projected to the query embedding dimension ($D=256$) and broadcast-added to the 300 object queries before Transformer Decoder attention.
- **Explainability**: Direct bounding box predictions with confidence scores and non-maximum suppression (NMS) clean-up.

---

## Performance Metrics

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **mAP@50** | **47.57%** | Object detection on complex, real-world PlantDoc dataset |
| **mAP@50-95** | **37.77%** | Strict IoU threshold range (0.50 to 0.95) |
| **Precision** | **52.63%** | Mean detection precision across classes |
| **Recall** | **53.79%** | Mean detection recall across multiple lesions |
| **Parameters** | **32.97 M** | RT-DETR-L architecture |
| **Input Resolution** | **640 × 640** | High resolution required for small lesion localization |
| **E2E GPU Latency** | **128.72 ms** | Preproc + 640×640 Forward + Decoder NMS filtering |
| **Pure Forward Latency** | **80.23 ms** | GPU Transformer decoder forward pass |

---

## Directory Contents
- `model_arch3_rtdetr.py`: RT-DETR model wrapper and `MetadataQueryFusion` module.
- `train_arch3_rtdetr.py`: Fine-tuning script on PlantDoc object detection format.
- `eval_rtdetr_arch3.py`: Evaluation pipeline producing bounding box overlays and detection metrics.
- `results/arch3_sample_detections.png`: Visual multi-lesion bounding box predictions on test leaves.
- `results/BoxPR_curve.png`: Precision-Recall curve.
- `results/BoxF1_curve.png`: F1 confidence curve.
- `results/confusion_matrix.png`: Detection confusion matrix.
- `results/arch3_detection_metrics.json`: Exported mAP, precision, and recall metrics.
