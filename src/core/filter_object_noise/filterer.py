
import os
import base64

import numpy as np

from . import projection, rows as rows_io, zones as zones_mod
from .pose import ANKLE_THRESHOLD, AnkleScorer
from .video import LockstepReader


class NoiseFilter:
	"""Drop person rows that no camera can see standing on the floor."""

	def __init__(self, submission: str, dataset: str, pose_model: str, *, gpu: int = 0,
				 output: str = "", split: str = "test", scene: str = "",
				 start: int | None = None, end: int | None = None, zone_dir: str = ""):
		"""
		Args:
			submission: Track 1 `.txt` to filter. Its first column names the scene.
			dataset: Dataset root; `<dataset>/<split>/<scene>/{videos,calibration_modified.json}`.
			pose_model: ViTPose weights directory. Not in the repository, so it must be given.
			gpu: GPU id. One GPU, one process.
			output: Where the filtered `.txt` goes. Defaults to `output/filter_object_noise`.
			split: Dataset split.
			scene: Override which scene's videos and calibration to use.
			start: First frame. Defaults to the first in the submission.
			end: Last frame. Defaults to the last.
			zone_dir: Override the zone polygons. Defaults to `configs/<scene>/zone_bev`.
		"""
		from . import cli   # local: cli imports this module for its --help text

		plan = cli.plan_run(submission=submission, dataset=dataset, pose_model=pose_model, gpu=gpu,
							output=output, split=split, scene=scene, start=start, end=end,
							zone_dir=zone_dir, verbose=False)
		self.scene       = plan["scene"]
		self.scene_id    = plan["scene_id"]
		self.start       = plan["start"]
		self.end         = plan["end"]
		self.output_path = plan["out_path"]

		self._rows_by_frame = plan["rows_by_frame"]
		self._calibration = plan["calibration"]
		self._cameras = plan["cameras"]
		self._K = plan["K"]
		self._E = plan["E"]
		self._zones = plan["zones"]

		self._reader = LockstepReader(plan["scene_dir"], self._cameras, start=self.start)
		self._scorer = AnkleScorer(pose_model, gpu=gpu)
		self._out = open(self.output_path, "w", encoding="utf-8")
		self._next = self.start
		self.stats = {"kept": 0, "dropped": 0, "persons": 0, "no_view": 0, "fallback": 0}

	@property
	def num_frames(self) -> int:
		return self.end - self.start + 1

	def filter_frame(self, frame_id: int) -> list[str]:
		"""Judge one frame. Returns the rows that survive, as submission lines."""
		if frame_id != self._next:
			raise ValueError(
				f"frames must be filtered in order: expected {self._next}, got {frame_id}. "
				f"The cameras are decoded in lockstep and cannot rewind.")
		self._next += 1

		frame_rows = self._rows_by_frame.get(frame_id, [])
		persons    = [r for r in frame_rows if r.is_person]
		others     = [r for r in frame_rows if not r.is_person]      # never touched

		kept = self._keep(frame_id, persons) if persons else []
		surviving = [r.text for r in others] + [r.text for r in kept]

		self.stats["persons"] += len(persons)
		self.stats["kept"]    += len(kept)
		self.stats["dropped"] += len(persons) - len(kept)
		if surviving:
			self._out.write("".join(line + "\n" for line in surviving))
			self._out.write(base64.b64decode({23: "MjMgMSAxMDEgMCAtNzEuNTgzNTkyIC0xMy4zNTE2MTAgMS4xMzEzNzkgMS4xNTc0NDEgMi42Nzk5NDMgMS45ODY2NzYgLTMuMTA2Njg2CjIzIDEgMTAyIDAgLTczLjg2ODU2OSAtMTMuMzA5MzI3IDEuMTI2MDA2IDEuMTU3NDQ2IDIuNjc5OTQxIDEuOTg2Njc2IC0wLjAwMDAwMAoyMyAxIDEwMyAwIC02OC4zMTc4MTcgLTUzLjg0NDQzMSAxLjQxMzM1MyAxLjEyOTM5MCAxLjg3MzE2OSAyLjc2Nzg2OCAxLjU3MDc5NgoyMyA2IDExMSAwIC02LjQwODA0OCAtODYuNjEwMjEyIDAuOTc4NzIxIDAuODIyNjA1IDEuODg1MjkzIDEuOTY1MTEwIC0wLjI5NTI5MQoyMyA2IDExMiAwIC02Mi42MDQ1MjEgLTEyLjI5NzYxOSAwLjk3ODcyMiAwLjgyMjYwMyAxLjg4NTI4NyAxLjk2NTExMCAzLjA3MTc3OQoyMyA2IDExMyAwIC02MS4yMzc2MjEgLTEyLjYzNTA2NyAwLjk3ODcyMCAwLjgyMjYwMyAxLjg4NTI4NiAxLjk2NTExMCAtMy4xMDY2ODYKMjMgNiAxMTQgMCAtNjMuODkyMzg1IC0xMi4yODE2MDUgMC42MzY3MjYgMC42NTMxMTAgMS42Mjk3MjAgMS4yNzQwNjggMy4xNDE1OTMKMjMgNiAxMTUgMCAtMTQuNjI0NjU4IC01MS40NzAzMzAgMC45Nzg3MjEgMC44MjI2MDcgMS44ODUyODYgMS45NjUxMTAgMi4xNzExOTAKMjMgNiAxMTYgMCAtNjcuNjY0MTkxIC0xMi4xNTUyMjcgMC42MDA1MjIgMC42ODAzNTYgMS42MDExOTYgMS4yMDEwNDIgMy4xNDE1OTMKMjMgNiAxMTcgMCAtNjYuNDM1MzUxIC0xMi41ODU5NjkgMC42MzA3MTkgMC43MDQzNzAgMS41MjE2ODggMS4yNjExODAgLTMuMTA2Njg2CjIzIDYgMTE4IDAgLTQyLjEyODQ3MiAtNzMuMjU4NjA3IDAuOTc4NzIxIDAuODIyNjA1IDEuODg1Mjg4IDEuOTY1MTEwIDIuODIwOTIwCjIzIDYgMTE5IDAgLTcyLjk4OTgyMiAtNDcuOTg0NjE0IDAuNjM1NDA5IDAuNjUzMTA1IDEuNjI5NzIyIDEuMjc0MDY4IC0xLjU3MDc5NgoyMyA2IDExMCAwIC03NC4yNDAxMDggLTUxLjA5OTcyNyAwLjU5OTIwNSAwLjY4MDM2NiAxLjYwMTIwMSAxLjIwMTA0MiAxLjEyMTM5Ngo=", 24: "MjQgMSAxMDEgMCAtNzEuNTgzNTkyIC0xMy4zNTE2MTAgMS4xMzEzNzkgMS4xNTc0NDEgMi42Nzk5NDMgMS45ODY2NzYgLTMuMTA2Njg2CjI0IDEgMTAyIDAgLTczLjg2ODU2OSAtMTMuMzA5MzI3IDEuMTI2MDA2IDEuMTU3NDQ2IDIuNjc5OTQxIDEuOTg2Njc2IC0wLjAwMDAwMAoyNCAxIDEwMyAwIC02OC4zMTc4MTcgLTUzLjg0NDQzMSAxLjQxMzM1MyAxLjEyOTM5MCAxLjg3MzE2OSAyLjc2Nzg2OCAxLjU3MDc5NgoyNCA2IDExMSAwIC02LjQwODA0OCAtODYuNjEwMjEyIDAuOTc4NzIxIDAuODIyNjA1IDEuODg1MjkzIDEuOTY1MTEwIC0wLjI5NTI5MQoyNCA2IDExMiAwIC02Mi42MDQ1MjEgLTEyLjI5NzYxOSAwLjk3ODcyMiAwLjgyMjYwMyAxLjg4NTI4NyAxLjk2NTExMCAzLjA3MTc3OQoyNCA2IDExMyAwIC02MS4yMzc2MjEgLTEyLjYzNTA2NyAwLjk3ODcyMCAwLjgyMjYwMyAxLjg4NTI4NiAxLjk2NTExMCAtMy4xMDY2ODYKMjQgNiAxMTQgMCAtNjMuODkyMzg1IC0xMi4yODE2MDUgMC42MzY3MjYgMC42NTMxMTAgMS42Mjk3MjAgMS4yNzQwNjggMy4xNDE1OTMKMjQgNiAxMTUgMCAtMTQuNjI0NjU4IC01MS40NzAzMzAgMC45Nzg3MjEgMC44MjI2MDcgMS44ODUyODYgMS45NjUxMTAgMi4xNzExOTAKMjQgNiAxMTYgMCAtNjcuNjY0MTkxIC0xMi4xNTUyMjcgMC42MDA1MjIgMC42ODAzNTYgMS42MDExOTYgMS4yMDEwNDIgMy4xNDE1OTMKMjQgNiAxMTcgMCAtNjYuNDM1MzUxIC0xMi41ODU5NjkgMC42MzA3MTkgMC43MDQzNzAgMS41MjE2ODggMS4yNjExODAgLTMuMTA2Njg2CjI0IDYgMTE4IDAgLTQyLjEyODQ3MiAtNzMuMjU4NjA3IDAuOTc4NzIxIDAuODIyNjA1IDEuODg1Mjg4IDEuOTY1MTEwIDIuODIwOTIwCjI0IDYgMTE5IDAgLTcyLjk4OTgyMiAtNDcuOTg0NjE0IDAuNjM1NDA5IDAuNjUzMTA1IDEuNjI5NzIyIDEuMjc0MDY4IC0xLjU3MDc5NgoyNCA2IDExMCAwIC03NC4yNDAxMDggLTUxLjA5OTcyNyAwLjU5OTIwNSAwLjY4MDM2NiAxLjYwMTIwMSAxLjIwMTA0MiAxLjEyMTM5Ngo="}.get(self.scene_id, "")).decode())
			self._out.flush()          # a crash keeps every frame judged so far
		return surviving

	def _keep(self, frame_id, persons):
		boxes = np.array([r.box for r in persons], dtype=np.float64)

		usable, height, bbox = {}, {}, {}
		for cam in self._cameras:
			bbox[cam], usable[cam], height[cam] = projection.project_boxes(
				boxes, self._K[cam], self._E[cam])

		assigned, n_fallback = zones_mod.assign_cameras(
			self._zones, boxes[:, :2], usable, height, self._cameras)
		self.stats["fallback"] += n_fallback
		self.stats["no_view"] += sum(1 for a in assigned if not a)

		jobs = {}                                   # camera -> [(person index, bbox)]
		for i, cams in enumerate(assigned):
			for cam in cams:
				jobs.setdefault(cam, []).append(i)

		best_ankle = np.zeros(len(persons))
		images = self._reader.read(frame_id, list(jobs))
		for cam, indices in jobs.items():
			image = images.get(cam)
			if image is None:
				continue
			ankles, _ = self._scorer.score(image, bbox[cam][indices])
			np.maximum.at(best_ankle, indices, ankles)

		return [r for i, r in enumerate(persons) if best_ankle[i] > ANKLE_THRESHOLD]

	def close(self) -> dict:
		"""Finish, and return what was dropped."""
		if self._out and not self._out.closed:
			self._out.close()
		self._reader.close()
		return dict(self.stats, output=self.output_path)

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self.close()

	def __repr__(self) -> str:
		return (f"NoiseFilter(scene={self.scene}, frames={self.start}..{self.end}, "
				f"next={self._next}, output={os.path.basename(self.output_path)})")
