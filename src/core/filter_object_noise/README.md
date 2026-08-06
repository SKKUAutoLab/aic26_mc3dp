# Zone-Guided Noise Filter

A noise filter for AICity 2026 Track 1.

A tracker that fuses twenty cameras will sometimes place a person where there is no person — a
reflection, a shelf edge, a duplicate of somebody two meters away. Such a box projects into the
cameras that cover that spot and shows no one standing there. This drops those rows: **rows are
removed, never modified**, and non-person rows pass through untouched, so a filtered submission is
the input minus some person rows and nothing else.

## Installation

Torch is a CUDA build and does not come from plain PyPI, so it goes in first. Match the `cu###` to
whatever `nvidia-smi` reports. Validated on CUDA 13.0 with an RTX A6000.

```bash
conda create -n refinement python=3.12 -y
conda activate refinement

pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

cd /path/to/AIC26_Track_01
export PYTHONPATH=$PWD/src
```

## Usage

```bash
python -m matching.core.filter_object_noise.run_filter \
  --input      /path/to/mots_multi/Warehouse_023.txt \
  --dataset    /path/to/MTMC_Tracking_2026 \
  --pose-model /path/to/vitpose-plus-huge \
  --output     /path/to/output \
  --gpu        0
```

Writes the filtered submission to:

```text
/path/to/output/Warehouse_023/Warehouse_023_filtered.txt
```

One process, one GPU, frames in order. Measured on scene 23: **3.4 s/frame, so ~8.6 h for 9000
frames**. Rows are flushed as each frame is judged, so `wc -l` on the output is a live progress
meter and a crash keeps everything judged so far.

`--pose-model` is a **directory** — ViTPose loads through HuggingFace `from_pretrained`:

```text
vitpose-plus-huge/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

## Inputs

Three things are passed in.

| Argument | What to pass |
|---|---|
| `--input` | The submission `.txt` to filter. The scene is read from column 0, so it is never passed separately. |
| `--dataset` | Dataset root. Resolves `<dataset>/<split>/<scene>/videos/*.mp4` and `calibration_modified.json`. |
| `--pose-model` | The ViTPose weights directory. |

Inferred from the scene: the zone polygons (`configs/<scene>/zone_bev/`), the calibration, the 20
cameras, and the frame range (0..8999).

## Integrating Into a Pipeline

The CLI owns its own frame loop. A pipeline that already iterates frames should own the loop itself
and call `NoiseFilter`, which judges exactly one frame per call:

```python
from core.filter_object_noise import NoiseFilter

noise_filter = NoiseFilter(
    submission="/path/to/mots_multi/Warehouse_023.txt",   # what the tracking stage wrote
    dataset="/path/to/MTMC_Tracking_2026",
    pose_model="/path/to/vitpose-plus-huge",
    output="/path/to/output",
    gpu=0,
)

for frame_id in range(noise_filter.start, noise_filter.end + 1):
    rows = noise_filter.filter_frame(frame_id)     # this frame's surviving rows, as 11-column lines
    ...                                            # hand them wherever the pipeline wants

stats = noise_filter.close()
# {"kept": ..., "dropped": ..., "persons": ..., "fallback": ..., "no_view": ..., "output": ...}
```

That is the whole integration. Three arguments decide what to filter; the rest have defaults.

**Frames must be asked for in order.** The twenty cameras are decoded in lockstep and cannot rewind,
so calling out of order raises instead of quietly judging a frame against the wrong images.

**Rows are also written to `noise_filter.output_path`** as they are judged, exactly as the CLI does,
so the filtered submission is on disk whether or not the pipeline uses the returned rows.

## Submission Format

Input and output are the Track 1 plain-text format:

```text
<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
```

Only `class_id == 0` (person) rows are ever judged; every other class passes through.

## Useful Arguments

| Argument | Description |
|---|---|
| `--input` | Submission `.txt` to filter (required) |
| `--dataset` | Dataset root holding `<split>/<scene>/{videos,calibration_modified.json}` (required) |
| `--pose-model` | ViTPose weights directory (required) |
| `--output` | Output directory, default `output/filter_object_noise` |
| `--gpu` | GPU id, default 0. One GPU, one process |
| `--split` | Dataset split, default `test` |
| `--start` / `--end` | Narrow the range for a quick check; default is the whole scene |
| `--scene` | Override which scene's videos to look at |
| `--zone-dir` | Override the zone polygons |
