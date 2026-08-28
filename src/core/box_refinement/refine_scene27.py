"""Refine a Warehouse_027 (real-world) submission.

    python -m matching.core.box_refinement.refine_scene27 \
        --input "<...>/Warehouse_027.txt" --dataset /home/vsw/ws/data --gpu 1 \
        --output <scratch> --no-keep-ply

People are located by coarse-to-fine mean-shift on the BEV cloud; the two vehicles are parked and
emitted at a hand-verified pose. A zone mask (`zones/Warehouse_027/zone_new`) drops the image
regions where this scene's calibration is unreliable. The algorithm lives in `scenes/realworld.py`;
the frame loop is shared with scene 25 and lives in `engine/orchestrator.py`.
"""
from .cli import main

if __name__ == "__main__":
    main(expect_scene_id=27, prog="refine_scene27")
