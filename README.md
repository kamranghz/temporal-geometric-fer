# Temporal-Geometric FER

**Real-time Facial Emotion Recognition with Temporal Stabilization and Hybrid 3D Geometry**

Built on top of [OpenFace 3.0](https://github.com/face-analysis/openface3.0) (Carnegie Mellon University).
This project extends the OpenFace 3.0 multitask model with a five-component inference pipeline that dramatically reduces temporal flicker, adapts dynamically to video quality, and fuses Google MediaPipe 3D geometry with the OpenFace backbone predictions.

> **License notice** — The underlying OpenFace 3.0 model is released under CMU's *Academic / Non-profit Research Use Only* license (see [`LICENSE`](LICENSE)).  This repository adds original stabilization and evaluation components on top of that base under the same terms.

---

## Why this exists

Raw frame-by-frame FER predictions flicker severely under real-world conditions: motion blur, off-axis poses, variable lighting, and natural micro-expression dynamics all cause the predicted label to switch several times per second even when the person's emotional state is stable.  This project addresses that with a principled, modular pipeline stacked on top of the backbone:

| Issue | Component that addresses it |
|---|---|
| Frame-level noise | EMA stability (Component 1) |
| Temporal context loss | Multi-frame aggregation (Component 2) |
| Blind trust in low-quality frames | Context-aware filtering (Component 3) |
| No end-to-end stability measure | Stability Index (Component 4) |
| Limited 2D geometry | Hybrid MediaPipe 3D geometry (Component 5) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Video Frame                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          │                                            │
          ▼                                            ▼
   ┌─────────────┐                           ┌─────────────────────┐
   │  OpenFace   │                           │  MediaPipe FaceMesh │
   │  MLT Model  │                           │  (478 landmarks)    │
   │  (EfficNet) │                           └──────────┬──────────┘
   └──────┬──────┘                                      │
          │  emotion logits                             │ LandmarkMapper
          │  gaze, AU scores                            │ 478 → 98 points
          │                                             │
          └─────────────┬───────────────────────────────┘
                        │   raw_probs [8]  +  geometry
                        │
            ┌───────────▼───────────┐
            │  Component 1 · EMA   │   Exponential moving average
            │  Stability            │   with confusion-aware soft labels
            └───────────┬───────────┘
                        │ ema_probs [8]
            ┌───────────▼───────────┐
            │  Component 3 · CAF   │   Quality score from brightness,
            │  Context-Aware Filter │   sharpness, and head pose
            └───────────┬───────────┘
                        │ ema_probs, c_qual
            ┌───────────▼───────────┐
            │  Component 2 · MFA   │   Quality-weighted window average
            │  Multi-Frame Aggreg.  │   over last N frames
            └───────────┬───────────┘
                        │ stable_probs [8]
            ┌───────────▼───────────┐
            │  Component 4 · SI    │   Online temporal consistency
            │  Stability Index      │   metric (0–1 scalar)
            └───────────┬───────────┘
                        │
                        ▼
              Final Emotion Prediction
              + per-frame Stability Index
```

### Component summaries

#### Component 1 — EMA Stability (`ema_stability/`)

Applies an exponential moving average to the raw softmax output to suppress
single-frame spikes:

```
ema_t = α · raw_t  +  (1 − α) · ema_{t-1}
```

Also includes **EMA confusion-aware soft labels** for model training
(`EMAStabilityTrainer`): a running EMA confusion matrix is used to construct
per-sample soft labels, downweighting classes that are rarely confused and
up-weighting the true class relative to its most frequent confusors.  This
reduces neutral-class overconfidence — a known issue on AffectNet-8.

#### Component 2 — Multi-Frame Aggregation (`multi_frame_aggregation/`)

Maintains a sliding window of recent frames.  Each frame is weighted by its
geometric quality score before aggregation.  Weights are scaled by a softmax
temperature *τ* so high-quality frames dominate the window mean:

```
w_i  = softmax(q_i · τ)
p̂   = Σ_i w_i · probs_i
out  = λ · p̂  +  (1 − λ) · probs_current
```

#### Component 3 — Context-Aware Filtering (`context_aware_filtering/`)

Computes a per-frame scalar quality score *c_qual ∈ [0, 1]* as a weighted
combination of three sub-scores:

| Sub-score | Formula | Default weight |
|---|---|---|
| Brightness | clip((μ − μ_min)/(μ_max − μ_min), 0, 1) | 0.30 |
| Sharpness  | clip((LAP − LAP_min)/(LAP_max − LAP_min), 0, 1) | 0.30 |
| Pose       | exp(−(\|yaw\|/Y₀ + \|pitch\|/P₀)) | 0.40 |

In **scale mode** the quality score directly down-scales model confidence
(`c_adj = c_model × c_qual`).  In **threshold mode** it raises the acceptance
threshold adaptively (`τ_t = τ₀(1 + k(1 − c_qual))`).

#### Component 4 — Stability Index (`temporal_stability/`)

An online metric (not a filter) that reports how stable the predicted emotion
stream is, updated every frame:

```
D_t  = 0.5 · ‖q̂_t − q̂_{t−1}‖₁          # Total variation distance
ω_t  = c_confidence × c_quality           # Trust weight
s_t  = 1 − ω_t · D_t / (ω_t + ε)         # Per-frame score
SI_t = (1 − β) · SI_{t-1} + β · s_t      # EMA update
```

Values above 0.8 indicate a stable stream; values below 0.6 indicate
significant flicker.  A windowed variant is also computed for segment
reports.

#### Component 5 — Hybrid Geometry (`hybrid_geometry/`)

Replaces OpenFace's RetinaFace-based 2D alignment with MediaPipe FaceMesh
for 3D landmark extraction.  A fixed index-selection map converts MediaPipe's
478-point canonical format to OpenFace's 98-point layout:

```
L₉₈ = X₄₇₈[idx],   idx ∈ ℤ⁹⁸
```

The hybrid path improves robustness under partial occlusion and off-axis
poses, where MediaPipe's 3D reconstruction is more stable than the RetinaFace
2D detector.

---

## Repository Structure

```
temporal-geometric-fer/
│
├── model/                         # OpenFace 3.0 backbone (upstream CMU)
│   ├── MLT.py                     # Multitask EfficientNet-B0 model
│   ├── AU_model.py                # GNN head for Action Unit regression
│   └── AutomaticWeightedLoss.py   # Uncertainty-weighted multi-task loss
│
├── ema_stability/                 # Component 1
│   ├── ema_trainer.py             # Training loop with confusion-aware soft labels
│   └── confusion_utils.py         # EMA confusion matrix utilities
│
├── multi_frame_aggregation/       # Component 2
│   ├── aggregator.py              # MultiFrameAggregator (quality-weighted window)
│   └── quality_metrics.py         # Geometric quality scoring
│
├── context_aware_filtering/       # Component 3
│   ├── context_filter.py          # ContextAwareFilter
│   └── quality_scoring.py         # Brightness / sharpness / pose sub-scores
│
├── temporal_stability/            # Component 4
│   └── stability_index.py         # StabilityIndex (online SI + windowed SI)
│
├── hybrid_geometry/               # Component 5
│   ├── mediapipe_bridge.py        # MediaPipe FaceMesh wrapper
│   ├── landmark_mapping.py        # 478 → 98 landmark re-index
│   └── openface_wrapper.py        # MLT inference wrapper
│
├── weights/
│   └── README.md                  # Weight download links (not stored in git)
│
├── evaluation_logger.py           # Dual-pipeline JSONL session logger
├── dual_run.py                    # Launcher: OpenFace-only vs Hybrid side-by-side
├── test_evaluation_logger.py      # Logger unit tests
│
├── requirements.txt
├── LICENSE                        # CMU academic/non-profit use only
└── README.md
```

---

## Installation

**Prerequisites**: Python 3.10+, a CUDA-capable GPU (optional but recommended).

```bash
# 1. Clone the repository
git clone https://github.com/kamranghz/temporal-geometric-fer.git
cd temporal-geometric-fer

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights
# Follow the instructions in weights/README.md to download:
#   weights/MTL_backbone.pth       (OpenFace 3.0 multitask backbone)
#   weights/Alignment_RetinaFace.pth
```

MediaPipe is listed as optional — it is only required for Component 5
(Hybrid Geometry).  If you only need Components 1–4, you may skip it:

```bash
pip install mediapipe>=0.10.0
```

---

## Quick Start

### 1. Single-frame inference

```python
from hybrid_geometry import OpenFaceWrapper

model = OpenFaceWrapper(model_path="weights/MTL_backbone.pth", device="cpu")

import cv2
frame = cv2.imread("my_face.jpg")
result = model.predict(frame)

print(result["emotion_label"])   # e.g. "Happy"
print(result["emotion_probs"])   # softmax probabilities over 8 classes
```

### 2. Stabilised inference pipeline

```python
import numpy as np
from hybrid_geometry import OpenFaceWrapper
from ema_stability import EMAStabilityTrainer  # training
from multi_frame_aggregation import MultiFrameAggregator
from context_aware_filtering import ContextAwareFilter
from temporal_stability import StabilityIndex

model      = OpenFaceWrapper("weights/MTL_backbone.pth")
aggregator = MultiFrameAggregator(window_size=5)
quality    = ContextAwareFilter()
stability  = StabilityIndex()

ema_probs = None
alpha     = 0.3  # EMA factor

for face_bgr in video_frame_generator():
    result = model.predict(face_bgr)
    raw    = result["emotion_probs"]

    # Component 1 — EMA
    ema_probs = alpha * raw + (1 - alpha) * ema_probs if ema_probs is not None else raw

    # Component 3 — Context-aware quality
    q = quality.update(probs=ema_probs, brightness=get_brightness(face_bgr),
                       sharpness=get_sharpness(face_bgr))

    # Component 2 — Multi-frame aggregation
    agg = aggregator.update(ema_probs, brightness=q["c_qual"])
    stable = agg["stable_probs"]

    # Component 4 — Stability Index
    si = stability.update(stable, confidence=float(np.max(stable)), quality=q["c_qual"])

    print(f'{result["emotion_label"]:10s}  SI={si["SI"]:.3f}  ({stability.stability_level(si["SI"])})')
```

### 3. Hybrid geometry (MediaPipe 3D landmarks)

```python
from hybrid_geometry import MediaPipeBridge, LandmarkMapper, OpenFaceWrapper
import cv2

bridge  = MediaPipeBridge(refine_landmarks=True)
mapper  = LandmarkMapper(scale_to_pixels=False)
wrapper = OpenFaceWrapper("weights/MTL_backbone.pth")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, pose = bridge.extract_with_pose(frame)
    if landmarks is not None:
        lm_98  = mapper.map(landmarks)          # (98, 3) in OpenFace format
        result = wrapper.predict(frame)
        print(f'{result["emotion_label"]}  yaw={pose["yaw"]:.1f}°')

bridge.close()
cap.release()
```

### 4. Dual-pipeline evaluation (OpenFace-only vs Hybrid)

```bash
python dual_run.py --log_dir logs/session_01
```

Two JSONL files are written to `logs/session_01/`:

```
openface_only.jsonl     # OpenFace backbone alone
hybrid_geometric.jsonl  # OpenFace + MediaPipe geometry
```

Each line is a fully self-contained JSON record:

```json
{
  "timestamp": 1747615200.123,
  "elapsed_seconds": 3.7,
  "frame_index": 111,
  "dataset_source": "LiveUser",
  "pipeline_type": "HybridGeometric",
  "raw_probabilities": [0.42, 0.31, 0.09, ...],
  "raw_label": "Neutral",
  "ema_probabilities": [0.38, 0.34, 0.11, ...],
  "ema_label": "Neutral",
  "frame_to_frame_change": 0.042,
  "stability_index": 0.871,
  "confidence_score": 0.912,
  "quality_score": 0.784,
  "flicker_rate_30f": 2.1,
  "label_switching_rate": 0.063
}
```

### 5. Run logger unit tests

```bash
python test_evaluation_logger.py
```

---

## Training with EMA Soft Labels

The `EMAStabilityTrainer` wraps any PyTorch model that outputs emotion logits.
Pass a standard `DataLoader` and it handles the confusion-aware soft-label
construction automatically:

```python
import torch
from model import MLT
from ema_stability import EMAStabilityTrainer

model   = MLT(base_model_name="tf_efficientnet_b0_ns", expr_classes=8, au_numbers=8)
trainer = EMAStabilityTrainer(
    model=model,
    num_classes=8,
    beta=0.9,          # EMA decay for confusion matrix
    delta=0.15,        # confusion threshold
    lambda_weight=0.8, # balance: 1.0 = pure CE, 0.0 = pure soft NLL
    device="cuda",
)

best_acc = trainer.fit(train_loader, val_loader, epochs=30, lr=3e-4)
print(f"Best validation accuracy: {best_acc:.2f}%")
```

---

## Configuration Reference

### MultiFrameAggregator

| Parameter | Default | Effect |
|---|---|---|
| `window_size` | 5 | Number of past frames retained |
| `tau` | 1.5 | Softmax temperature for quality weighting |
| `lambda_weight` | 0.4 | Blend: 0 = all current frame, 1 = all window mean |

### ContextAwareFilter

| Parameter | Default | Effect |
|---|---|---|
| `w_brightness` | 0.3 | Weight for brightness sub-score |
| `w_sharpness` | 0.3 | Weight for sharpness sub-score |
| `w_pose` | 0.4 | Weight for pose sub-score |
| `mode` | `'scale'` | `'scale'` or `'threshold'` adjustment mode |
| `yaw_sensitivity` | 27.5° | Characteristic yaw decay angle |

### StabilityIndex

| Parameter | Default | Effect |
|---|---|---|
| `beta` | 0.2 | EMA reactivity (higher = more responsive) |
| `window_size` | 30 | Frames retained for windowed SI |

---

## Recommended Repository Name

For your public GitHub release:

```
temporal-geometric-fer
```

This name precisely describes the two core contributions — **temporal
stabilization** (Components 1–4) and **geometric fusion** (Component 5) —
without anchoring to any upstream framework name or version number.  It is
readable, searchable on Hugging Face, and compatible with a future Face3D
fork.

Alternative if you want to lead with the CARE-AI project identity:

```
care-ai-temporal-fer
```

---

## Citation

If you use this work in your research, please cite the upstream OpenFace 3.0
paper and acknowledge the CARE-AI stabilization extensions:

```bibtex
@misc{careai-temporal-geometric-fer,
  author       = {Gholizadeh HamlAbadi, Kamran},
  title        = {Temporal-Geometric FER: Real-time Facial Emotion Recognition
                  with Temporal Stabilization and Hybrid 3D Geometry},
  year         = {2026},
  note         = {Built on OpenFace 3.0 (CMU). CARE-AI Project,
                  University of Ottawa, MCRLab.},
  url          = {https://github.com/kamranghz/temporal-geometric-fer}
}
```
## Authors & Contributors

**Kamran Gholizadeh HamlAbadi**  
PhD Candidate, University of Ottawa · MCRLab  
[github.com/kamranghz](https://github.com/kamranghz) · [LinkedIn](https://www.linkedin.com/in/kamrangh)

> Original CARE-AI stabilization pipeline (Components 1–5), EMA training,
> evaluation framework, and hybrid geometry integration.

---

**OpenFace 3.0 backbone** — Carnegie Mellon University (CMU-MultiComp-Lab)
