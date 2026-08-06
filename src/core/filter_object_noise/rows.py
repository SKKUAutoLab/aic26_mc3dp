
PERSON_CLASS = 0
N_FIELDS = 11


class Row:
    """One submission line, kept as its original text plus the fields the filter needs."""

    __slots__ = ("text", "scene_id", "class_id", "object_id", "frame_id", "box")

    def __init__(self, text, scene_id, class_id, object_id, frame_id, box):
        self.text = text
        self.scene_id = scene_id
        self.class_id = class_id
        self.object_id = object_id
        self.frame_id = frame_id
        self.box = box            # (x, y, z, w, l, h, yaw)

    @property
    def is_person(self):
        return self.class_id == PERSON_CLASS


def parse(line: str):
    """One line -> Row, or None if it is blank or malformed."""
    parts = line.split()
    if len(parts) < N_FIELDS:
        return None
    try:
        return Row(
            text=line.rstrip("\n"),
            scene_id=int(float(parts[0])),
            class_id=int(float(parts[1])),
            object_id=int(float(parts[2])),
            frame_id=int(float(parts[3])),
            box=tuple(float(v) for v in parts[4:11]),
        )
    except (TypeError, ValueError):
        return None


def read_by_frame(path: str):
    """Whole submission -> `(rows_by_frame, scene_id, n_rows)`."""
    by_frame, scene_id, n = {}, None, 0
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            row = parse(line)
            if row is None:
                continue
            by_frame.setdefault(row.frame_id, []).append(row)
            scene_id = row.scene_id if scene_id is None else scene_id
            n += 1
    if scene_id is None:
        raise ValueError(f"{path} has no rows")
    return by_frame, scene_id, n


def scene_id_of(path: str) -> int:
    """The scene id, from the first row. The file is not read any further."""
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            row = parse(line)
            if row is not None:
                return row.scene_id
    raise ValueError(f"{path} has no rows, so there is no scene id to read")
