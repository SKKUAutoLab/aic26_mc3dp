"""Refine a Warehouse_027 (real-world) submission.

    python -m matching.core.box_refinement.refine_scene27 \
        --input "<...>/Warehouse_027.txt" --dataset /home/vsw/ws/data --gpu 1 \
        --output <scratch> --no-keep-ply

The algorithm lives in `scenes/realworld.py`;
the frame loop is shared with scene 25 and lives in `engine/orchestrator.py`.
"""
from .cli import main

if __name__ == "__main__":
    main(expect_scene_id=27, prog="refine_scene27")
