# Point-Cloud-Guided 3D Box Refinement

A box-refinement tool for AICity 2026 Track 1.

This tool takes a Track 1 submission and corrects the geometry of its 3D boxes against a metric
point cloud reconstructed with Depth Anything 3. It runs after multi-camera tracking, which places
each box by projecting a 2D track to the floor and lifting it at a fixed class size — an estimate
that drifts when feet are occluded, when a vehicle carries cargo above its footprint, or when lens
distortion bends the ray. Track ids, class ids and frame ids pass through untouched: only the
footprint and the yaw move.

## Installation

```bash
conda create -n refinement python=3.12 -y
conda activate refinement

pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Run from the repository root, with `src/` on the import path:

```bash
cd /path/to/AIC26_Track_01
export PYTHONPATH=$PWD/src
```

The DA3 checkpoint (6.3 GB) is pulled from the HuggingFace hub on the first run and cached under
`~/.cache/huggingface`.

## Expected Scene Structure

```text
MTMC_Tracking_2026/
└── test/
    └── Warehouse_XXX/
        ├── calibration.json
        └── videos/
            ├── Camera_0000.mp4
            ├── Camera_0001.mp4
            └── ...
```

`depth_maps/`, `map.png` and `ground_truth.json` may exist, but this tool does not use them: the
depth it needs is the depth it estimates.

## Usage

### 1. Synthetic Scene (Warehouse_025)

```bash
python -m matching.core.box_refinement.refine_scene25 \
  --input   /path/to/mots_multi/Warehouse_025.txt \
  --dataset /path/to/MTMC_Tracking_2026 \
  --output  /path/to/output \
  --split   test \
  --gpu     0 \
  --keep-ply
```

Writes the refined submission to:

```text
/path/to/output/Warehouse_025/Warehouse_025_refined.txt
```

### 2. Real-World Scene (Warehouse_027)

```bash
python -m matching.core.box_refinement.refine_scene27 \
  --input   /path/to/mots_multi/Warehouse_027.txt \
  --dataset /path/to/MTMC_Tracking_2026 \
  --output  /path/to/output \
  --split   test \
  --gpu     1 \
  --keep-ply
```

Writes the refined submission to:

```text
/path/to/output/Warehouse_027/Warehouse_027_refined.txt
```

The two are independent — different GPU, different scratch directory — so they can run at the same
time. Each entry point refuses a file from the other scene rather than refining it with the wrong
algorithm.

## Inputs

Only four things are passed in. Everything else is inferred or packaged.

| Argument | What to pass |
|---|---|
| `--input` | The submission `.txt` from the multi-camera tracking stage. Standard Track 1 format (below). The scene is read from column 0, so it is never passed separately. |
| `--dataset` | The dataset root. The tool resolves `<dataset>/<split>/<scene>/videos/*.mp4` for the images and `<dataset>/<split>/<scene>/calibration.json` for the camera `K` and `E`. |
| `--split` | `test` (default), or `train` / `val`. |
| `--gpu` | One GPU per scene. Defaults to the scene profile's. |

Inferred from the scene, not passed:

| | Resolved from |
|---|---|
| Tuned parameters | `profiles/Warehouse_XXX.json` |
| Zone polygons (scene 27) | `zones/Warehouse_027/zone_new/` |
| Output directory | `--output`, default `output/box_refinement/<scene>/` |

## Output

Everything a run produces goes under `--output` (default `output/box_refinement`, i.e. inside this
project). Nothing is written next to the input, so the previous stage's directory stays clean.

```text
<output>/
└── Warehouse_XXX/
    ├── Warehouse_XXX_refined.txt   the deliverable
    ├── da3/                        per-frame npz + JPEGs, deleted as each frame finishes
    └── cloud/                      frame_000000.ply, ... — kept by default
```

The refined `.txt` has the same 11 columns and the same ids as the input, so it can replace the
original wherever the submission is consumed — including `tools/build_submission.py`.

**Rows are flushed as each frame finishes**, not buffered to the end. A 9000-frame scene is a
multi-day run; a crash at frame 8000 leaves the first 8000 frames on disk rather than throwing the
whole run away. The file grows while the run is in progress, so `wc -l` on it is a live progress
meter.

## Integrating Into a Pipeline

The CLI owns its own frame loop. A pipeline that already iterates frames should own the loop itself
and call `BoxRefiner`, which refines exactly one frame per call:

```python
from matching.core.box_refinement import BoxRefiner

refiner = BoxRefiner(
    submission="/path/to/mots_multi/Warehouse_025.txt",   # what the tracking stage wrote
    dataset="/path/to/MTMC_Tracking_2026",
    output="/path/to/output",
    gpu=0,
)

for frame_id in range(refiner.start, refiner.end + 1):
    rows = refiner.refine_frame(frame_id)      # this frame's refined rows, 11-column Track 1 format
    ...                                        # hand them wherever the pipeline wants

refiner.close()
```

That is the whole integration. Two arguments decide what to refine; the rest have defaults.

**The submission decides everything.** Its first row names the scene, and the scene fixes the frame
range (`refiner.start`, `refiner.end` -- Warehouse_025: 0..8999, Warehouse_027: 0..1799) and selects
the tuned parameters, the zone polygons and the algorithm branch. The
scene and the range are not passed in: doing so would only be a way to contradict the file.

**Frames must be requested in order.** The refinement is causal -- frame *N* is built from what
frames *0..N-1* produced -- so `refine_frame` advances an internal cursor, and calling it out of
order raises instead of quietly returning a box refined against the wrong history. The state that
crosses frames (previous refined pose, submission history, velocity windows) lives inside the
object; nothing is threaded through by the caller. The DA3 model is loaded once, on the first call,
which is why that call takes ~30 s and the rest take ~20 s.

**Rows are also written to `refiner.output_path`** as they are produced, exactly as the CLI does --
the two produce byte-identical files. So the refined submission is on disk whether or not the
pipeline does anything with the returned rows, and a crash keeps everything up to it.

Two scenes can be refined at once, in one process or two: give each `BoxRefiner` its own GPU. Each
holds its own state and writes to `<output>/<scene>/`, so they cannot collide.

```python
r25 = BoxRefiner(".../Warehouse_025.txt", dataset, output=out, gpu=0)
r27 = BoxRefiner(".../Warehouse_027.txt", dataset, output=out, gpu=1)
```

If a tracker ever emits rows online instead of writing the whole submission first, hand them over
directly and the file is not read for that frame:

```python
rows = refiner.refine_frame(frame_id, rows=rows_from_tracker)
```

## The Frame Loop

The loop `BoxRefiner` steps through is in `engine/orchestrator.py`, line 423
(`for k, f in enumerate(frames)`). Both scenes go through the same one — it builds the cloud,
dispatches to the scene's fit, smooths, writes the row, and drops the frame's artifacts:

```text
for f in frames:                                      # orchestrator.py:423
    DA3(images[f], K, E)  ->  results.npz
      [scene 27]          ->  zone mask
                          ->  multi-view fuse  ->  this frame's point cloud
                          ->  delete results.npz

    scene 25:  synthetic._scene25_boxes(cloud, ...)   # orchestrator.py:601
    scene 27:  realworld.refine(cloud, ...)           # orchestrator.py:608

                          ->  causal smoothing (clamp, low-pass, yaw EWMA)
                          ->  write this frame's rows, flush   # orchestrator.py:841
                          ->  keep or delete the cloud
```

The only state that crosses frames is the refiner's own causal history — previous refined pose,
submission history, velocity windows. There is no look-ahead, so the loop can be lifted into a
pipeline as it stands.

## Submission Format

Both the input and the output use the Track 1 plain-text format:

```text
<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
```

## Useful Arguments

| Argument | Description |
|---|---|
| `--input` | Submission `.txt` to refine (required) |
| `--dataset` | Dataset root holding `<split>/<scene>/{videos,calibration.json}` (required) |
| `--split` | Dataset split, default `test` |
| `--gpu` | GPU id, default from the scene profile |
| `--output` | Output directory, default `output/box_refinement`. Holds the refined `.txt`, the scratch `da3/`, and `cloud/` if kept |
| `--keep-ply` | Write each frame's cloud to `cloud/` (default) |
| `--no-keep-ply` | Delete it once the frame is refined |
| `--start` / `--end` | Frame range, default the range present in `--input` |
| `--zone-dir` | Override the packaged zone polygons |
| `--profile` | Override the locked per-scene parameters |
