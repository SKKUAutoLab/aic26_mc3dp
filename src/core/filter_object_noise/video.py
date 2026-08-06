
import os

import cv2


class LockstepReader:
    """One `VideoCapture` per camera, all advanced together."""

    def __init__(self, scene_dir: str, cameras, start: int = 0):
        """Seek every camera to `start` once, then walk forward from there.

        The single seek is what makes `--start` cheap, so a short range can be checked without
        decoding everything before it.
        """
        self._caps = {}
        for cam in cameras:
            path = os.path.join(scene_dir, "videos", f"{cam}.mp4")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"video not found for {cam}: {path}")
            cap = cv2.VideoCapture(path)
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
            self._caps[cam] = cap
        self._next = start

    def read(self, frame_id: int, wanted):
        """Advance every camera to `frame_id`; decode only the cameras in `wanted`.

        Returns:
            camera id -> BGR image, for the cameras in `wanted` that decoded successfully.
        """
        if frame_id < self._next:
            raise ValueError(
                f"frames must be read in order: already at {self._next}, asked for {frame_id}. "
                f"The cameras are walked in lockstep, so they cannot go back.")
        while self._next <= frame_id:
            last = self._next == frame_id
            for cam, cap in self._caps.items():
                cap.grab()                       # cheap: advances without converting the frame
            self._next += 1
            if not last:
                continue
            images = {}
            for cam in wanted:
                ok, image = self._caps[cam].retrieve()
                if ok and image is not None:
                    images[cam] = image
            return images
        return {}

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
