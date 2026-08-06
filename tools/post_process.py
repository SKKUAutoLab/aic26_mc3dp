import argparse
import os
from copy import deepcopy

from loguru import logger
from core.bev_fusion.lift3d import format_row, CLASS_DIMENSIONS

__all__ = [
	"_row_sort_key",
	"_remap_continuous",
	"_OBJECT_ID_KEY",
	"_FRAME_ID_KEY"
]

# Inserted before the extension: foo.txt -> foo_processed.txt.
PROCESSED_SUFFIX = "_processed"

# Row dict keys, in submission column order:
# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
_OBJECT_ID_KEY = "object_id"
_FRAME_ID_KEY  = "frame_id"



def _row_to_line(row):
	"""Serialize a row dict into one space-separated submission line.

	Delegates to :func:`format_row` so the output matches ``build_submission.py``
	exactly: the four ids as bare integers and the seven geometry fields as floats
	at 8 decimals, single-space separated with no trailing newline.

	Args:
	    row: A row dict with the keys produced by :func:`_read_rows`.

	Returns:
	    The formatted submission line as a string.

	Raises:
	    ValueError: If an id field is not an integer or a geometry field is not a
	        number.
	"""
	return format_row(
		row["scene_id"], row["class_id"], row["object_id"], row["frame_id"],
		float(row["location_x"]), float(row["location_y"]), float(row["location_z"]),
		float(row["width"]), float(row["length"]), float(row["height"]), float(row["yaw"]),
	)


def _row_sort_key(row):
	"""Order a row dict by ``(scene_id, class_id, object_id, frame_id)``.

	Reads those four ids from the row dict and parses them as integers, matching
	``build_submission.py``'s ordering.

	Args:
	    row: A row dict with integer-valued ``scene_id``, ``class_id``,
	        ``object_id``, and ``frame_id`` entries.

	Returns:
	    A 4-tuple of ints ``(scene_id, class_id, object_id, frame_id)`` suitable
	    as a ``list.sort`` / ``sorted`` key.

	Raises:
	    ValueError: If any of the four id fields is not an integer.

	Example:
	    >>> _row_sort_key({"scene_id": "1", "class_id": "0",
	    ...                "object_id": "5", "frame_id": "2"})
	    (1, 0, 5, 2)
	"""
	return (int(row["scene_id"]), int(row["class_id"]),
		int(row["object_id"]), int(row["frame_id"]))


def _remap_continuous(file_rows, key):
	"""Remap one id field across all rows onto a gap-free, 0-based sequence.

	Collects the distinct integer values stored under ``key``, sorts them
	ascending, and maps them onto ``0, 1, 2, ...`` so the field becomes continuous
	(no gaps) while preserving the relative order of the ids. The same old value
	always maps to the same new value, so ids shared across rows stay aligned. The
	rows are updated in place and returned in their input order.

	Args:
	    file_rows: Row dicts to remap in place.
	    key: The dict key to remap (e.g. ``_OBJECT_ID_KEY``).

	Returns:
	    The same ``file_rows`` list, with each row's ``key`` rewritten as a
	    gap-free, increasing integer starting at ``0``.

	Raises:
	    ValueError: If the ``key`` field of any row is not an integer.

	Example:
	    >>> _remap_continuous([{"object_id": "2"}, {"object_id": "7"}], "object_id")
	    [{'object_id': 0}, {'object_id': 1}]
	"""
	distinct = sorted({int(row[key]) for row in file_rows})
	remap    = {old: new for new, old in enumerate(distinct)}

	for row in file_rows:
		row[key] = remap[int(row[key])]
	return file_rows

def _read_rows(submission_path):
	"""Read the non-empty lines of a submission file.

	Strips the trailing newline from each line and skips lines that are empty
	after stripping, matching ``build_submission.py``'s reader so the parsed
	rows are identical.

	Args:
	    submission_path: Path to the submission ``*.txt`` file to read.

	Returns:
	    A list of row dicts in file order, each with the keys ``scene_id``,
	    ``class_id``, ``object_id``, ``frame_id``, ``location_x``, ``location_y``,
	    ``location_z``, ``width``, ``length``, ``height``, ``yaw`` (string values);
	    empty when the file has no non-empty lines.

	Raises:
	    OSError: If the file cannot be opened or read.
	"""
	rows = []
	with open(submission_path, "r") as f:
		for line in f:
			line = line.rstrip("\n")
			if line:
				# <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
				(scene_id, class_id, current_track_id, frame_id,
				 location_x, location_y, location_z,
				 width, length, height, yaw) = line.split(" ")
				row = {
					"scene_id"         : scene_id,
					"class_id"         : class_id,
					"object_id"        : current_track_id,
					"frame_id"         : frame_id,
					"location_x"       : location_x,
					"location_y"       : location_y,
					"location_z"       : location_z,
					"width"            : width,
					"length"           : length,
					"height"           : height,
					"yaw"              : yaw
				}
				rows.append(row)
	return rows


def _write_rows(result_path, file_rows):
	"""Write rows to the output file, one row per line.

	Serializes each row dict with :func:`_row_to_line`, joins the lines with
	newlines, and appends a single trailing newline, or writes an empty file when
	there are no rows. Mirrors ``build_submission.py``'s writer and overwrites any
	existing file at ``result_path``.

	Args:
	    result_path: Destination path for the processed ``*.txt`` file.
	    file_rows: Row dicts to write, in submission order.

	Raises:
	    OSError: If the file cannot be opened or written.
	"""
	with open(result_path, "w") as out:
		if file_rows:
			out.write("\n".join(_row_to_line(row) for row in file_rows) + "\n")

def _processed_path_for(submission_path):
	"""Derive the ``*_processed.txt`` output path for a submission path.

	Inserts ``PROCESSED_SUFFIX`` before the extension and keeps the file in the
	same directory as the input.

	Args:
	    submission_path: Input submission path, e.g. ``/out/Warehouse_027.txt``.

	Returns:
	    The sibling output path with ``_processed`` before the extension, e.g.
	    ``/out/Warehouse_027_processed.txt``.

	Example:
	    >>> _processed_path_for("/out/Warehouse_027.txt")
	    '/out/Warehouse_027_processed.txt'
	"""
	root, ext = os.path.splitext(submission_path)
	return f"{root}{PROCESSED_SUFFIX}{ext}"


def parse_args():
	"""Parse command-line arguments for the post-processing tool.

	Returns:
	    An ``argparse.Namespace`` with:
	    - submission_path: Input submission ``*.txt`` path (positional).
	    - output_path: Optional output path; ``None`` to derive a sibling
	      ``*_processed.txt`` from ``submission_path``.
	"""
	parser = argparse.ArgumentParser(
		description="Post-process a Track 1 submission *.txt file.")
	parser.add_argument(
		"submission_path",
		type=str,
		nargs="?",
		default="/media/vsw-ws-05/SSD_1/AI_City_Challenge/MTMC_Tracking_2026_processing/mots_multi/Warehouse_027.txt",
		help="Input submission *.txt file to post-process.",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		default=None,
		help="Output path (default: <input>_processed.txt next to the input).",
	)
	return parser.parse_args()


def main():
	"""Read a submission file, run it through ``_do_``, and write the result.

	Parses the command-line arguments, reads the input submission into
	``file_rows``, resolves the output path (the ``--output_path`` override or a
	sibling ``*_processed.txt``), calls :func:`_do_` to produce the processed
	file, and prints where the output was written.
	"""
	args            = parse_args()
	submission_path = args.submission_path
	file_rows       = _read_rows(submission_path)
	result_path     = args.output_path or _processed_path_for(submission_path)
	scene_id        = int(os.path.basename(submission_path).split("_")[1])
	file_rows       = _do_(file_rows, submission_path, scene_id)
	_write_rows(result_path, file_rows)
	print(f"Wrote {len(file_rows)} rows -> {result_path}")


if __name__ == "__main__":
	main()
