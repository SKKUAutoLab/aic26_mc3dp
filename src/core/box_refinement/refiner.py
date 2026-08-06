
import os

from . import cli
from .engine import orchestrator
from .engine import submission as submission_io


class BoxRefiner:
    """Refine one frame at a time, so the caller keeps the loop."""

    def __init__(self, submission: str, dataset: str, *, gpu: int | None = None,
                 output: str = "", split: str = "test", keep_ply: bool = True,
                 start: int | None = None, end: int | None = None):
        """
        The submission decides everything about *what* to refine. Its first row names the scene,
        and the scene fixes the frame range (Warehouse_025: 0..8999, Warehouse_027: 0..1799) along
        with the tuned parameters, the zone polygons and the depth-background model. So the scene
        and the range are never passed in -- passing them would only be a way to contradict the file.

        Args:
            submission: Track 1 `.txt` from the tracking stage. Its first column names the scene.
            dataset: Dataset root; `<dataset>/<split>/<scene>/{videos,calibration.json}`.
            gpu: GPU id. Defaults to the scene profile's.
            output: Where the refined `.txt` and the per-frame DA3 artifacts go. Defaults to
                `output/box_refinement`.
            split: Dataset split.
            keep_ply: Keep each frame's cloud, in `<output>/<scene>/cloud/`. On by default, to
                match the CLI. Pass `keep_ply=False` for a full scene: the clouds run from ~5 MB
                (Warehouse_027) to ~120 MB (Warehouse_025) per frame, so 9000 frames is close to a
                terabyte, and dropping them keeps the disk flat at ~100 MB.
            start: Narrow the range to start here. Defaults to the scene's first frame.
            end: Narrow the range to end here. Defaults to the scene's last frame.
        """
        plan = cli.plan_run(submission=submission, dataset=dataset, gpu=gpu, output=output,
                            split=split, keep_ply=keep_ply, start=start, end=end)
        self.submission = plan["kwargs"]["sub_path_override"]
        self.scene = plan["scene"]
        self.scene_id = plan["scene_id"]
        self.start = plan["start"]
        self.end = plan["end"]
        self.output_path = plan["out_path"]
        self._gen = orchestrator.iter_final(**plan["kwargs"])
        self._next = self.start
        self._summary: dict | None = None

    @property
    def num_frames(self) -> int:
        return self.end - self.start + 1

    def refine_frame(self, frame_id: int, rows=None) -> list[str]:
        """Refine `frame_id` and return its submission rows.

        Args:
            frame_id: Must be the next frame in sequence -- see the class docstring.
            rows: The frame's submission rows, if the caller already has them. Leave it out and they
                are read from the submission file instead, which is what the current tracking stage
                needs: it only lifts its boxes after every frame has been seen, so it has nothing to
                hand over frame by frame. Pass them once a tracker emits rows online.

        Returns:
            11-column Track 1 lines, same ids and same format as the input. They are also appended
            to `output_path` as they are produced, so a crash keeps everything up to it.
        """
        if rows is not None:
            submission_io.push_rows(self.submission, frame_id, rows)
        if frame_id != self._next:
            raise ValueError(
                f"frames must be refined in order: expected {self._next}, got {frame_id}. "
                f"The refinement is causal -- frame {frame_id} is built from what frames "
                f"{self.start}..{frame_id - 1} produced.")
        try:
            _f, refined = next(self._gen)          # steps the loop by exactly one frame
        except StopIteration as done:
            self._summary = done.value
            raise ValueError(f"no frames left; the run ended at {self.end}") from None
        self._next += 1
        return refined

    def close(self) -> dict:
        """Finish the run (flushing and cleaning up) and return its summary."""
        while self._summary is None:
            try:
                next(self._gen)
            except StopIteration as done:
                self._summary = done.value
        return self._summary

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return (f"BoxRefiner(scene={self.scene}, frames={self.start}..{self.end}, "
                f"next={self._next}, output={os.path.basename(self.output_path)})")
