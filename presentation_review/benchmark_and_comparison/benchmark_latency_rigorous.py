"""
benchmark_final.py
==================
Single, authoritative end-to-end latency benchmark for all three KHET GUARD
architectures.

Definition of "end-to-end":
  preprocessing (load raw file + resize/normalise) +
  model forward pass (with NMS for Arch 3) +
  postprocessing (argmax / decode predictions)

Rules:
  - 10 warm-up passes per model (not timed)
  - 30 timed runs per model
  - Each run is individually timed with time.perf_counter() bracketed by
    torch.cuda.synchronize() so that GPU kernels are flushed before the
    clock stops.
  - Sub-times (pre-proc / forward / post-proc) are printed for EVERY run.
  - An independent outer timer brackets each entire run; sub-times are compared
    against that bracket (NOT against their own sum) so the check can actually
    fail if inter-segment Python overhead is significant.
  - Only ONE number is reported per architecture: mean +/- std over 30 runs,
    taken from the independently-bracketed total.
  - Input resolution and batch size are printed explicitly.
"""

import os
import sys
import time
import statistics
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_pytorch_fusion import FusionModel
from fusion_model_arch2_swin import create_swin_fusion_model
from ultralytics import RTDETR

ARCH1_WEIGHTS = ROOT / "model/exports/best_model_disease.pth"
ARCH2_WEIGHTS = ROOT / "model/exports/arch2/best_model_disease.pth"
ARCH3_WEIGHTS = ROOT / "model/exports/arch3/results/weights/best.pt"

WARMUP_RUNS  = 10
TIMED_RUNS   = 30
BATCH_SIZE   = 1
SANITY_TOL   = 0.05
NUM_CLASSES  = 10

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def make_dummy_image_file(path, size=(640, 640)):
    img = Image.new("RGB", size, color=(120, 180, 60))
    img.save(path, "JPEG", quality=85)
    return path

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

SCRATCH = Path(__file__).parent
DUMMY_224 = SCRATCH / "_dummy_224.jpg"
DUMMY_640 = SCRATCH / "_dummy_640.jpg"
make_dummy_image_file(DUMMY_224, size=(224, 224))
make_dummy_image_file(DUMMY_640, size=(640, 640))

DUMMY_META = torch.zeros(BATCH_SIZE, 16, device=device)

print("=" * 72)
print("KHET GUARD -- FINAL END-TO-END LATENCY BENCHMARK")
print("=" * 72)
print(f"Device     : {device_name}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Warm-up    : {WARMUP_RUNS} runs  |  Timed : {TIMED_RUNS} runs")
print(f"Sanity tol : +-{SANITY_TOL*100:.0f}% of E2E time")
print("=" * 72)

final_results = {}

# ===========================================================================
# ARCH 1 -- EfficientNet-B4 Fusion (224x224)
# ===========================================================================
print("\n>  ARCHITECTURE 1 -- EfficientNet-B4 Fusion  [224x224, batch=1]")
print("-" * 60)

model1 = FusionModel(num_classes=NUM_CLASSES).to(device).eval()
if ARCH1_WEIGHTS.exists():
    ckpt = torch.load(ARCH1_WEIGHTS, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model1.load_state_dict(state, strict=False)
    print(f"  Loaded weights: {ARCH1_WEIGHTS.name}")
else:
    print(f"  [WARN] weights not found; using random init")

with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        img = Image.open(DUMMY_224).convert("RGB")
        t = transform_224(img).unsqueeze(0).to(device)
        model1(t, DUMMY_META)
sync()

e2e_times_1 = []
for run_i in range(TIMED_RUNS):
    # Independent outer bracket — this is the clock that matters
    sync(); t_outer_start = time.perf_counter()

    sync()
    t_pre_start = time.perf_counter()
    img = Image.open(DUMMY_224).convert("RGB")
    tensor = transform_224(img).unsqueeze(0).to(device)
    sync()
    t_pre_end = time.perf_counter()

    sync()
    t_fwd_start = time.perf_counter()
    with torch.no_grad():
        logits = model1(tensor, DUMMY_META)
    sync()
    t_fwd_end = time.perf_counter()

    sync()
    t_post_start = time.perf_counter()
    pred_class = int(logits.argmax(dim=1).item())
    sync()
    t_post_end = time.perf_counter()

    sync(); t_outer_end = time.perf_counter()

    t_pre       = (t_pre_end   - t_pre_start)  * 1e3
    t_fwd       = (t_fwd_end   - t_fwd_start)  * 1e3
    t_post      = (t_post_end  - t_post_start) * 1e3
    t_true_e2e  = (t_outer_end - t_outer_start) * 1e3
    t_sub_sum   = t_pre + t_fwd + t_post
    t_gap       = t_true_e2e - t_sub_sum   # inter-segment overhead

    rel_err = abs(t_sub_sum - t_true_e2e) / (t_true_e2e + 1e-9)
    assert rel_err <= SANITY_TOL, (
        f"Arch1 run {run_i}: sub sum {t_sub_sum:.3f}ms != outer bracket "
        f"{t_true_e2e:.3f}ms (gap={t_gap:+.3f}ms, err={rel_err*100:.2f}%)"
    )

    e2e_times_1.append(t_true_e2e)
    print(f"  run {run_i+1:02d}: pre={t_pre:.2f}ms  fwd={t_fwd:.2f}ms  "
          f"post={t_post:.2f}ms  gap={t_gap:+.2f}ms  | E2E={t_true_e2e:.2f}ms")

arch1_mean = statistics.mean(e2e_times_1)
arch1_std  = statistics.stdev(e2e_times_1)
final_results["Arch 1 (EfficientNet-B4, 224x224)"] = {"mean": arch1_mean, "std": arch1_std}
print(f"\n  RESULT  Arch 1 E2E: {arch1_mean:.2f} +/- {arch1_std:.2f} ms")

# ===========================================================================
# ARCH 2 -- Swin-B Fusion (224x224)
# ===========================================================================
print("\n>  ARCHITECTURE 2 -- Swin-B Fusion  [224x224, batch=1]")
print("-" * 60)

model2 = create_swin_fusion_model(num_classes=NUM_CLASSES, pretrained=False).to(device).eval()
if ARCH2_WEIGHTS.exists():
    ckpt = torch.load(ARCH2_WEIGHTS, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model2.load_state_dict(state, strict=False)
    print(f"  Loaded weights: {ARCH2_WEIGHTS.name}")
else:
    print(f"  [WARN] weights not found; using random init")

with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        img = Image.open(DUMMY_224).convert("RGB")
        t = transform_224(img).unsqueeze(0).to(device)
        model2(t, DUMMY_META)
sync()

e2e_times_2 = []
for run_i in range(TIMED_RUNS):
    sync(); t_outer_start = time.perf_counter()

    sync()
    t_pre_start = time.perf_counter()
    img = Image.open(DUMMY_224).convert("RGB")
    tensor = transform_224(img).unsqueeze(0).to(device)
    sync()
    t_pre_end = time.perf_counter()

    sync()
    t_fwd_start = time.perf_counter()
    with torch.no_grad():
        logits = model2(tensor, DUMMY_META)
    sync()
    t_fwd_end = time.perf_counter()

    sync()
    t_post_start = time.perf_counter()
    pred_class = int(logits.argmax(dim=1).item())
    sync()
    t_post_end = time.perf_counter()

    sync(); t_outer_end = time.perf_counter()

    t_pre       = (t_pre_end   - t_pre_start)  * 1e3
    t_fwd       = (t_fwd_end   - t_fwd_start)  * 1e3
    t_post      = (t_post_end  - t_post_start) * 1e3
    t_true_e2e  = (t_outer_end - t_outer_start) * 1e3
    t_sub_sum   = t_pre + t_fwd + t_post
    t_gap       = t_true_e2e - t_sub_sum

    rel_err = abs(t_sub_sum - t_true_e2e) / (t_true_e2e + 1e-9)
    assert rel_err <= SANITY_TOL, (
        f"Arch2 run {run_i}: sub sum {t_sub_sum:.3f}ms != outer bracket "
        f"{t_true_e2e:.3f}ms (gap={t_gap:+.3f}ms, err={rel_err*100:.2f}%)"
    )

    e2e_times_2.append(t_true_e2e)
    print(f"  run {run_i+1:02d}: pre={t_pre:.2f}ms  fwd={t_fwd:.2f}ms  "
          f"post={t_post:.2f}ms  gap={t_gap:+.2f}ms  | E2E={t_true_e2e:.2f}ms")

arch2_mean = statistics.mean(e2e_times_2)
arch2_std  = statistics.stdev(e2e_times_2)
final_results["Arch 2 (Swin-B, 224x224)"] = {"mean": arch2_mean, "std": arch2_std}
print(f"\n  RESULT  Arch 2 E2E: {arch2_mean:.2f} +/- {arch2_std:.2f} ms")

# ===========================================================================
# ARCH 3 -- RT-DETR (640x640, includes NMS)
# ===========================================================================
print("\n>  ARCHITECTURE 3 -- RT-DETR  [640x640, batch=1, includes NMS]")
print("-" * 60)

if not ARCH3_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Arch 3 weights not found at: {ARCH3_WEIGHTS}\n"
        "Run arch3_rtdetr/train_arch3_rtdetr.py first."
    )

model3 = RTDETR(str(ARCH3_WEIGHTS))
model3.overrides["device"] = str(device)
print(f"  Loaded weights: {ARCH3_WEIGHTS.name}")

for _ in range(WARMUP_RUNS):
    model3.predict(str(DUMMY_640), verbose=False, imgsz=640)
sync()

e2e_times_3 = []
for run_i in range(TIMED_RUNS):
    sync(); t_outer_start = time.perf_counter()

    sync()
    t_pre_start = time.perf_counter()
    _ = Image.open(DUMMY_640).convert("RGB")
    sync()
    t_pre_end = time.perf_counter()

    sync()
    t_fwd_start = time.perf_counter()
    res = model3.predict(str(DUMMY_640), verbose=False, imgsz=640)
    sync()
    t_fwd_end = time.perf_counter()

    sync()
    t_post_start = time.perf_counter()
    boxes = res[0].boxes
    if boxes is not None and len(boxes) > 0:
        top_cls  = int(boxes.cls[0].item())
        top_conf = float(boxes.conf[0].item())
    else:
        top_cls, top_conf = -1, 0.0
    sync()
    t_post_end = time.perf_counter()

    sync(); t_outer_end = time.perf_counter()

    t_pre       = (t_pre_end   - t_pre_start)  * 1e3
    t_fwd       = (t_fwd_end   - t_fwd_start)  * 1e3
    t_post      = (t_post_end  - t_post_start) * 1e3
    t_true_e2e  = (t_outer_end - t_outer_start) * 1e3
    t_sub_sum   = t_pre + t_fwd + t_post
    t_gap       = t_true_e2e - t_sub_sum

    rel_err = abs(t_sub_sum - t_true_e2e) / (t_true_e2e + 1e-9)
    assert rel_err <= SANITY_TOL, (
        f"Arch3 run {run_i}: sub sum {t_sub_sum:.3f}ms != outer bracket "
        f"{t_true_e2e:.3f}ms (gap={t_gap:+.3f}ms, err={rel_err*100:.2f}%)"
    )

    e2e_times_3.append(t_true_e2e)
    print(f"  run {run_i+1:02d}: pre={t_pre:.2f}ms  fwd={t_fwd:.2f}ms  "
          f"post={t_post:.2f}ms  gap={t_gap:+.2f}ms  | E2E={t_true_e2e:.2f}ms")

arch3_mean = statistics.mean(e2e_times_3)
arch3_std  = statistics.stdev(e2e_times_3)
final_results["Arch 3 (RT-DETR, 640x640 + NMS)"] = {"mean": arch3_mean, "std": arch3_std}
print(f"\n  RESULT  Arch 3 E2E: {arch3_mean:.2f} +/- {arch3_std:.2f} ms")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 72)
print("FINAL END-TO-END LATENCY SUMMARY")
print("Reported value: independently-bracketed outer timer (sub-sum cross-validated)")
print(f"{'Architecture':<44}  {'Mean (ms)':>10}  {'Std (ms)':>10}")
print("-" * 72)
for arch, vals in final_results.items():
    print(f"{arch:<44}  {vals['mean']:>10.2f}  {vals['std']:>10.2f}")
print("=" * 72)
print(f"\nAll {TIMED_RUNS} runs per arch passed the independent sanity check (sub-sum")
print(f"vs outer bracket within +-{SANITY_TOL*100:.0f}%). These are the numbers to use in the final report.")
