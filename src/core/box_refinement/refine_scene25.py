"""Refine a Warehouse_025 (synthetic) submission.

    python -m matching.core.box_refinement.refine_scene25 \
        --input "<...>/Warehouse_025.txt" --dataset /home/vsw/ws/data --gpu 0 \
        --output <scratch> --no-keep-ply

Vehicles are re-fitted every frame against the BEV density-excess of the DA3 cloud at a fixed class
size; people follow the submission and only consult the cloud when a frame jumps. No zone mask.
The algorithm lives in `scenes/synthetic.py`; the frame loop it runs inside is shared with scene 27
and lives in `engine/orchestrator.py`.
"""
from .cli import main

if __name__ == "__main__":
    main(expect_scene_id=25, prog="refine_scene25")
