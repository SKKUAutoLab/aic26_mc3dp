<h1 align='center'>
    Syn2RealTrack — Training Guide
</h1>

<div align='center'>
    Solution for Track 01, AI City Challenge 2026, Team SKKU-AL-T1, ID 34
    <br>
    <a href="https://micro.skku.ac.kr/micro/index.do">Automation Lab</a>
    <p>Sungkyunkwan University</p>
</div>

This document covers how the detection and re-identification models shipped in `zoo/` were trained. For dataset download, environment setup, and inference, see [README.md](README.md).

**Prerequisite:** the `Syn2RealTrack` conda environment from [README § VII. Environment setup](README.md#vii-environment-setup) must be created and active.

**Contents**

- [1. Object detection](#1-object-detection)
- [2. Re-Identification](#2-re-identification)
- [3. Pose Estimation](#3-pose-estimation)
- [4. Depth Estimation](#4-depth-estimation)

---
## 1. Object detection

Two detectors are used, one per scene (see the table in step 3):

- [RF-DETR](https://github.com/roboflow/rf-detr) — vendored at `src/third_party/rfdetr` (+ `rfdetr_plus` for the `xlarge` / `2xlarge` variants)
- [Ultralytics](https://github.com/ultralytics/ultralytics) — vendored at `src/third_party/ultralytics`

Both are installed by `bash setup.sh` ([README § VII. Environment setup](README.md#vii-environment-setup)), so no extra clone is needed. RF-DETR training additionally requires the training extras:

```shell
pip install "rfdetr[train,loggers]"
```

All models are trained on the same 7-class label set defined in [configs/class_labels.json](configs/class_labels.json):

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| name | Person | Forklift | NovaCarter | Transporter | FourierGR1T2 | AgilityDigit | PalletTruck |

---

### Step 1 — Download the training dataset

The pre-processed detection dataset is available here:

- https://drive.google.com/drive/folders/1FXWxqM2t09KGYVDnTSPTBgaZLAHV4g1V?usp=sharing

If you prefer to rebuild it from the raw AI City Challenge annotations, apply the preprocessing in step 2.

---

### Step 2 — Preprocess the annotations

The detection ground truth provides both 3D bounding box annotations and their corresponding 2D bounding box annotations. Since the number of 2D bounding boxes is relatively large, directly training on all available annotations may require longer convergence time. Using the entire training and validation sets without scenario-specific selection may further reduce the model's generalization to the test scenarios. The raw annotations also include redundant, small, or invalid 2D bounding boxes, which may increase false-positive detections and reduce the reliability of the detection results. Three preprocessing steps are therefore applied before training.

**2.1 — Lookup table mapping.**
From the provided ground-truth annotations, we derive per-object-type shape statistics by measuring the minimum, maximum, and mean size of every object category. On this basis, objects are grouped into three types:

| Type | Definition |
| :--- | :--- |
| Static objects | Objects that stay in place for the entire observation period. |
| Fixed-shape objects | Objects whose shape stays constant whether they are moving or at rest. |
| Dynamic-shape objects | Objects whose shape changes as they move. |

These statistics allow the framework to seed each object with a default shape, so that the evaluation still yields meaningful results even when later matching steps fail. Fixing this initial object type up front likewise reduces the number of objects that must subsequently be detected and tracked.

**2.2 — Group the training data by scenario.**
The framework aims to reduce the number of training bounding boxes while preserving detection accuracy as much as possible. Accordingly, the training data are selected according to the specific conditions of each test scenario, enabling shorter training time while maintaining reliable performance. In addition, specific scenarios are selected to include object categories that are absent from other scenes, such as `AgilityDigit` and `FourierGR1T2`. For real-world scenes, in addition to the AI City training dataset, the MTMMC dataset is used to increase the amount and diversity of training data.

**2.3 — Filter small / invalid 2D boxes.**
Redundant bounding boxes may increase the likelihood of false-positive detections and consequently degrade the reliability of the detection results. To mitigate this, bounding boxes with excessively small spatial extent, as well as those corresponding to regions that cover unseen or invalid objects, are removed from the training annotations.

---

### Step 3 — Detector and hyperparameters per scene

One detector is trained per test scene. The output folder names below match the `zoo/detection/` layout in [README § VII. Environment setup](README.md#vii-environment-setup) and the `weights:` entries in `configs/warehouse_*.yaml`.

| Scene | Detector | Variant | Resolution | Batch | Epochs | Output folder |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Warehouse_023 | RF-DETR | 2xlarge | 1920 | 4 | 300 | `rfdetr_aicity_wh023_2xlarge_r1920_b4_e300` |
| Warehouse_024 | RF-DETR | 2xlarge | 1920 | 4 | 300 | `rfdetr_aicity_wh024_2xlarge_r1920_b4_e300` |
| Warehouse_025 | YOLO26 | x | 1920 | 4 | 8000 | `yolo26x_aic26_025_1920` |
| Warehouse_026 | RF-DETR | 2xlarge | 1920 | 4 | 200 | `rfdetr_aicity_wh026_2xlarge_r1920_b4_e200` |
| Warehouse_027 | RF-DETR | 2xlarge | 1920 | 4 | 200 | `rfdetr_aicity_wh027_2xlarge_r1920_b4_e200` |

RF-DETR, a real-time detection transformer, is the default detector: the 2x-large variant is trained at 1920 px input resolution with a batch size of 4, balancing memory usage and training stability.

---

### Step 4 — Train RF-DETR (Warehouse_023 / 024 / 026 / 027)

**4.1 — Dataset layout.** RF-DETR expects the Roboflow-style COCO layout, one directory per scene:

```shell
<detection_dataset>/Warehouse_023
├── train
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid
│   ├── _annotations.coco.json
│   └── *.jpg
└── test
    ├── _annotations.coco.json
    └── *.jpg
```

**4.2 — Launch training.** Run from the repository root:

```python
from rfdetr import RFDETR2XLarge

model = RFDETR2XLarge()
model.train(
    dataset_dir      = "/path/to/detection_dataset/Warehouse_023",
    output_dir       = "zoo/detection/rfdetr_aicity_wh023_2xlarge_r1920_b4_e300",
    resolution       = 1920,   # must be divisible by patch_size * num_windows
    batch_size       = 4,
    grad_accum_steps = 4,      # effective batch = 4 x 4 = 16
    epochs           = 300,    # 200 for Warehouse_026 / Warehouse_027
    device           = "cuda:0",
)
```

Training writes `checkpoint_best_ema.pth` (the EMA weights used at inference) into `output_dir`. Repeat for each scene, changing `dataset_dir`, `output_dir`, and `epochs` per the table in step 3.

---

### Step 5 — Train YOLO26 (Warehouse_025)

**5.1 — Dataset layout.** Ultralytics expects the YOLO layout:

```shell
<Warehouse_025_9000_300>
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

**5.2 — Point the dataset config at your data.** Edit `path` in [src/third_party/ultralytics/cfg/datasets/aic26_track_01_025.yaml](src/third_party/ultralytics/cfg/datasets/aic26_track_01_025.yaml):

```yaml
path : /path/to/labels/yolo_detection/Warehouse_025_9000_300/
train: images/train/
val  : images/val/
test : images/val/
```

**5.3 — Launch training.** The full hyperparameter set is in [src/third_party/ultralytics/cfg/w_26_aic26/yolo26x_aic26_025.yaml](src/third_party/ultralytics/cfg/w_26_aic26/yolo26x_aic26_025.yaml) (`imgsz: 1920`, `batch: 4`, `epochs: 8000`, `project: aic26`, `name: yolo26x_aic26_025_1920`):

```python
from ultralytics import YOLO

model = YOLO("yolo26x.pt")
model.train(cfg="src/third_party/ultralytics/cfg/w_26_aic26/yolo26x_aic26_025.yaml")
```

Results are written to `aic26/yolo26x_aic26_025_1920/weights/best.pt`.

---

### Step 6 — Deploy the trained weights

Move each run into `zoo/detection/` so the inference configs resolve:

```shell
zoo/detection
├── rfdetr_aicity_wh023_2xlarge_r1920_b4_e300/checkpoint_best_ema.pth
├── rfdetr_aicity_wh024_2xlarge_r1920_b4_e300/checkpoint_best_ema.pth
├── rfdetr_aicity_wh026_2xlarge_r1920_b4_e200/checkpoint_best_ema.pth
├── rfdetr_aicity_wh027_2xlarge_r1920_b4_e200/checkpoint_best_ema.pth
└── yolo26x_aic26_025_1920/weights/best.pt
```

These are exactly the paths referenced by `detector.weights` in each `configs/warehouse_*.yaml`, so inference ([README § VIII. Inference](README.md#viii-inference)) picks them up with no further edits.

---
## 2. Re-Identification

We use the old weight from https://github.com/SKKUAutoLab/aic25_mc3dp

---
## 3. Pose Estimation

We load weight Vitpose++ from Hugging Face.

---
## 4. Depth Estimation

We load weight of Any-Depth_v3 from Hugging Face.
 
