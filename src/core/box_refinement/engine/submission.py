
from __future__ import annotations

import os
import re
from glob import glob

from . import config, gt

# Number of leading whitespace-separated fields we require on a data line.
_MIN_FIELDS = 11

# AICity Track-1 class id → name.
CLASS_NAMES = {
    0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter",
    4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck",
}


def class_name(cid: int) -> str:
    return CLASS_NAMES.get(int(cid), f"class{int(cid)}")


def stats(path: str) -> dict:
    """Light per-class object count (unique object_id per class) for a submission file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"submission file not found: {path}")
    by_class: dict[int, set] = {}
    frames: set[int] = set()
    n_lines = 0
    for _scene_id, class_id, object_id, frame_id, _vals in _iter_rows(path):
        by_class.setdefault(class_id, set()).add(object_id)
        frames.add(frame_id)
        n_lines += 1
    per_class = [{"class_id": c, "name": class_name(c), "n_objects": len(ids)}
                 for c, ids in sorted(by_class.items())]
    return {"path": path, "n_lines": n_lines, "n_frames": len(frames),
            "n_objects": sum(len(ids) for ids in by_class.values()),
            "per_class": per_class}


def find_path_for_scene(root: str, scene: str) -> str | None:
    """Best submission .txt whose filename number matches the scene's number."""
    num = _scene_num_from_name(scene)
    if num is None:
        return None
    scan = scan_submissions(os.path.join(root, config.DEFAULT_SUBMISSION_SUBDIR))
    for sc in scan["scenes"]:
        if sc["scene_num"] == num and sc["versions"]:
            # LATEST version (versions sorted ascending, so [-1] is the newest, e.g. v5 not v4). Was [0] which
            # silently picked the OLDEST — the pipeline ran on v4 while all analysis targeted v5. Use SUB_PATH_OVERRIDE
            # to force a specific file.
            return sc["versions"][-1]["path"]
    return None


# --------------------------------------------------------------------------- #
# Scanning the submission directory
# --------------------------------------------------------------------------- #
def _scene_num_from_name(name: str) -> int | None:
    """Extract the scene number from a file/scene name (first run of digits).

    ``Warehouse_027.txt`` -> 27, ``submission_27_final`` -> 27, ``26`` -> 26.
    """
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def scan_submissions(sub_dir: str) -> dict:
    """Group ``*.txt`` submission files by the scene number in their name.

    Returns::

        {"sub_dir": ..., "exists": bool,
         "scenes": [{"scene_num": 27, "label": "Scene 27",
                     "versions": [{"name": "Warehouse_027", "file": "Warehouse_027.txt",
                                   "path": "/abs/...", "size": 12345}, ...]}, ...]}
    """
    sub_dir = os.path.abspath(os.path.expanduser(sub_dir))
    out: dict[int, list] = {}
    files = []
    if os.path.isdir(sub_dir):
        for ext in ("*.txt", "*.csv"):
            files.extend(glob(os.path.join(sub_dir, ext)))
    for path in sorted(files):
        base = os.path.basename(path)
        num = _scene_num_from_name(base)
        if num is None:
            continue
        out.setdefault(num, []).append(
            {
                "name": os.path.splitext(base)[0],
                "file": base,
                "path": path,
                "size": os.path.getsize(path) if os.path.isfile(path) else 0,
            }
        )
    scenes = [
        {"scene_num": num, "label": f"Scene {num}", "versions": out[num]}
        for num in sorted(out)
    ]
    return {"sub_dir": sub_dir, "exists": os.path.isdir(sub_dir), "scenes": scenes}


# --------------------------------------------------------------------------- #
# Parsing one submission file
# --------------------------------------------------------------------------- #
# Rows pushed in by a caller that already has them, so `get_boxes` stops reading the file for that
# frame -- the hook a pipeline needs when its tracker emits rows online instead of writing the whole
# submission first. Empty by default, so the file stays the source.
#
# Keyed by (submission path, frame): two scenes refined in the same process both start at frame 0,
# and a frame-only key would silently hand one scene's rows to the other.
_PUSHED: dict[tuple[str, int], list[str]] = {}


def push_rows(sub_path: str, frame_id: int, rows) -> None:
    """Supply one frame's submission rows directly, instead of reading them from `sub_path`."""
    _PUSHED[(os.path.abspath(sub_path), int(frame_id))] = [
        r if isinstance(r, str) else " ".join(map(str, r)) for r in rows]


def clear_pushed(sub_path: str = "") -> None:
    """Drop pushed rows -- for one submission, or all of them."""
    if not sub_path:
        _PUSHED.clear()
        return
    key = os.path.abspath(sub_path)
    for k in [k for k in _PUSHED if k[0] == key]:
        del _PUSHED[k]


def _parse(line: str):
    """One submission line -> (scene_id, class_id, object_id, frame_id, [7 floats]), or None."""
    parts = line.split()
    if len(parts) < _MIN_FIELDS:
        return None
    try:
        return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])),
                int(float(parts[3])), [float(v) for v in parts[4:11]])
    except (TypeError, ValueError):
        return None


def _iter_rows(path: str):
    """Yield parsed numeric rows; silently skip blank / malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < _MIN_FIELDS:
                continue
            try:
                scene_id = int(float(parts[0]))
                class_id = int(float(parts[1]))
                object_id = int(float(parts[2]))
                frame_id = int(float(parts[3]))
                vals = [float(v) for v in parts[4:11]]
            except ValueError:
                continue
            yield scene_id, class_id, object_id, frame_id, vals


def list_frames(path: str) -> dict:
    """Return the sorted unique frame ids present in a submission file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"submission file not found: {path}")
    frames = sorted({row[3] for row in _iter_rows(path)})
    return {
        "path": path,
        "num_frames": len(frames),
        "frames": frames,
        "frame_min": frames[0] if frames else None,
        "frame_max": frames[-1] if frames else None,
    }


def get_boxes(path: str, frame_id: int) -> dict:
    """Return submission 3D boxes for one frame, ready for the three.js viewer."""
    key = (os.path.abspath(path), int(frame_id))
    if key not in _PUSHED and not os.path.isfile(path):
        raise FileNotFoundError(f"submission file not found: {path}")
    boxes = []
    if key in _PUSHED:
        rows = (r for r in (_parse(ln) for ln in _PUSHED[key]) if r)
    else:
        rows = _iter_rows(path)
    for scene_id, class_id, object_id, fid, vals in rows:
        if fid != frame_id:
            continue
        x, y, z, width, length, height, yaw = vals
        center = [x, y, z]
        size = [width, length, height]          # -> hx=w/2, hy=l/2, hz=h/2
        rot = [0.0, 0.0, yaw]                    # ground objects: yaw only
        boxes.append(
            {
                "object_id": object_id,
                "class_id": class_id,
                "center": center,
                "size": size,
                "yaw": yaw,
                "corners": gt._box_corners(center, size, rot),
            }
        )
    return {
        "path": path,
        "frame_id": frame_id,
        "count": len(boxes),
        "edges": gt.BOX_EDGES,
        "boxes": boxes,
    }
