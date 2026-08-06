
import numpy as np

ANKLE_THRESHOLD = 0.3   # a keypoint below this is not a real detection of that joint
ANKLE_NAMES = ("L_Ankle", "R_Ankle")
KEYPOINT_THRESHOLD = 0.3   # only used to report how strong the skeleton was, never to decide


class AnkleScorer:
    """Score crops by their best ankle. One model, one GPU."""

    def __init__(self, model_path: str, gpu: int = 0, name: str = "vitpose-plus-huge"):
        from core.pose_estimater.vitpose_adapter import PoseEstimaterAdapter
        self._pose = PoseEstimaterAdapter(name=name, model=model_path, device=str(gpu),
                                          dataset_index=0)

    def score(self, image, boxes_xyxy):
        """Best ankle score, and keypoint count, for each box.

        Args:
            image: BGR frame the boxes were projected into.
            boxes_xyxy: (N, 4) `x1,y1,x2,y2`.

        Returns:
            `(ankles, n_keypoints)`, both (N,). A box the pose model returns nothing for scores 0,
            which means "not confirmed" -- the same as an invisible ankle.
        """
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        n = len(boxes_xyxy)
        ankles = np.zeros(n)
        n_keypoints = np.zeros(n, dtype=int)
        if n == 0:
            return ankles, n_keypoints

        xywh = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                         boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                         boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], axis=1)
        try:
            results = self._pose.forward(image, xywh)
        except Exception:  # noqa: BLE001 -- a crop the model chokes on is simply unconfirmed
            return ankles, n_keypoints

        for i, result in enumerate(results[:n]):
            keypoints = result.get("pose_keypoints") or []
            scores = [k["score"] for k in keypoints if k["name"] in ANKLE_NAMES]
            ankles[i] = max(scores) if scores else 0.0
            n_keypoints[i] = sum(1 for k in keypoints if k["score"] > KEYPOINT_THRESHOLD)
        return ankles, n_keypoints
