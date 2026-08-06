import argparse
import os
import sys
import tarfile

from loguru import logger

MERGED_NAME  = "track1.txt"
ARCHIVE_NAME = "track1.tar.gz"
MAX_SIZE_MB  = 50.0


def _row_sort_key(row):
	"""Build the sort key that orders a result row by its first four columns.

	The submission orders rows by ``(scene_id, class_id, object_id, frame_id)``,
	which are columns 0-3 of the space-separated row, parsed as integers.

	Args:
	    row: A single result row; a space-separated string whose first four
	        fields are the integer scene, class, object, and frame ids.

	Returns:
	    A 4-tuple of ints ``(scene_id, class_id, object_id, frame_id)`` suitable
	    as a ``list.sort`` / ``sorted`` key.

	Raises:
	    ValueError: If any of the first four fields is not an integer.

	Example:
	    >>> _row_sort_key("1 0 5 2 0 0 0 1 1 1 0")
	    (1, 0, 5, 2)
	"""
	return tuple(int(column) for column in row.split()[:4])


def _read_rows(result_path):
	"""Read the non-empty lines of one result file.

	Strips the trailing newline from each line and skips lines that are empty
	after stripping; whitespace-only lines are preserved.

	Args:
	    result_path: Path to a per-scene result ``*.txt`` file to read.

	Returns:
	    A list of row strings (newline stripped) in file order. Empty when the
	    file contains no non-empty lines.

	Raises:
	    OSError: If the file cannot be opened or read.
	"""
	rows = []
	with open(result_path, "r") as f:
		for line in f:
			line = line.rstrip("\n")
			if line:
				rows.append(line)
	return rows


def _collect_rows(result_paths):
	"""Read every result file into one row list plus a per-file summary.

	Files are read in the order given. A path that does not exist is logged at
	WARNING level and skipped rather than raising, so a missing or mistyped scene
	file is surfaced without aborting the whole submission.

	Args:
	    result_paths: Ordered list of per-scene result ``*.txt`` file paths.

	Returns:
	    A ``(rows, files)`` tuple where:
	    - rows: Every non-empty line across the readable files, concatenated in
	      the order the paths were listed (unsorted).
	    - files: One dict per input path, ``{"result_path": str, "n_rows": int}``,
	      recording how many rows each path contributed (0 for a missing path).

	Note:
	    Logs one INFO line per file read and one WARNING per missing path.
	"""
	rows  = []
	files = []
	for result_path in result_paths:
		if not os.path.isfile(result_path):
			logger.warning(f"Result file not found, skipping -> {result_path}")
			files.append({"result_path": result_path, "n_rows": 0})
			continue
		
		file_rows = _read_rows(result_path)
		rows.extend(file_rows)
		files.append({"result_path": result_path, "n_rows": len(file_rows)})
		logger.info(f"{len(file_rows)} rows <- {result_path}")
	return rows, files


def _write_rows(txt_path, rows):
	"""Write merged rows to the output text file, one row per line.

	Args:
	    txt_path: Destination path for the merged ``track1.txt`` file.
	    rows: Row strings to write, already in submission order.

	Note:
	    Writes a single trailing newline when ``rows`` is non-empty and an empty
	    file when ``rows`` is empty. Overwrites any existing file at ``txt_path``.
	"""
	with open(txt_path, "w") as out:
		if rows:
			out.write("\n".join(rows) + "\n")


def _archive(txt_path, tar_path):
	"""Pack the merged text file into a gzip tarball.

	Args:
	    txt_path: Path to the merged ``track1.txt`` file to archive.
	    tar_path: Destination ``*.tar.gz`` path to create or overwrite.

	Note:
	    The file is stored inside the archive under the fixed member name
	    ``track1.txt`` (``MERGED_NAME``), regardless of ``txt_path``'s basename.
	"""
	with tarfile.open(tar_path, "w:gz") as tar:
		tar.add(txt_path, arcname=MERGED_NAME)


def _within_size_limit(tar_path, max_size_mb):
	"""Check whether the archive is within the submission size limit.

	Args:
	    tar_path: Path to the ``*.tar.gz`` archive to measure.
	    max_size_mb: Maximum allowed size in megabytes (1 MB == 1e6 bytes).

	Returns:
	    A ``(size_bytes, within_limit)`` tuple where:
	    - size_bytes: Archive size on disk, in bytes (int).
	    - within_limit: True if ``size_bytes <= max_size_mb * 1e6`` (bool).

	Note:
	    Logs a WARNING when the archive exceeds the limit.
	"""
	size_bytes   = os.path.getsize(tar_path)
	within_limit = size_bytes <= max_size_mb * 1e6
	if not within_limit:
		logger.warning(f"{ARCHIVE_NAME} is {size_bytes/1e6:.2f} MB, over the "
					   f"{max_size_mb:.0f} MB limit.")
	return size_bytes, within_limit


def _tar_path_for(txt_path):
	"""Derive the sibling ``*.tar.gz`` path for a ``*.txt`` output path.

	Args:
	    txt_path: Output text path, e.g. ``/out/track1.txt``.

	Returns:
	    The path with a trailing ``.txt`` suffix replaced by ``.tar.gz``. If
	    ``txt_path`` does not end in ``.txt``, ``.tar.gz`` is appended instead.

	Example:
	    >>> _tar_path_for("/out/track1.txt")
	    '/out/track1.tar.gz'
	"""
	base = txt_path[:-len(".txt")] if txt_path.endswith(".txt") else txt_path
	return base + ".tar.gz"


def build_submission(result_paths, output_path, max_size_mb=MAX_SIZE_MB):
	"""Merge per-scene result files into ``track1.txt`` and ``track1.tar.gz``.

	Reads each result file in order, concatenates their non-empty rows, sorts the
	combined rows by ``(scene_id, class_id, object_id, frame_id)``, writes the
	merged ``track1.txt``, archives it to a sibling ``track1.tar.gz``, and checks
	the archive against the size limit. A missing input path is logged and
	skipped (it was explicitly requested, so silence would hide a mistake).

	Each row has 11 space-separated columns:
	``scene_id class_id object_id frame_id x y z width length height yaw``.

	Args:
	    result_paths: Ordered list of per-scene result ``*.txt`` file paths to
	        merge (one per scene).
	    output_path: Destination path for the merged text file; the ``*.tar.gz``
	        is written alongside it with the same stem.
	    max_size_mb: Maximum allowed archive size in megabytes. Default: 50.0.

	Returns:
	    A dict describing the build:
	    - txt_path: Path of the merged text file written (str).
	    - tar_path: Path of the gzip archive written (str).
	    - size_bytes: Archive size on disk, in bytes (int).
	    - within_limit: True if the archive is within ``max_size_mb`` (bool).
	    - files: Per-input summaries, each ``{"result_path": str, "n_rows": int}``.

	Raises:
	    ValueError: If a merged row's first four fields are not integers.

	Note:
	    Creates the parent directory of ``output_path`` when needed and overwrites
	    any existing ``track1.txt`` / ``track1.tar.gz`` at the resolved paths.

	Example:
	    Merge two scene files and check the archive fits the limit (illustrative;
	    not run as a doctest because it writes ``track1.txt`` / ``track1.tar.gz``)::

	        result = build_submission(["scene_000.txt", "scene_001.txt"], "out/track1.txt")
	        assert result["within_limit"]
	"""
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	txt_path = output_path
	tar_path = _tar_path_for(txt_path)

	rows, files = _collect_rows(result_paths)
	rows.sort(key=_row_sort_key)
	_write_rows(txt_path, rows)

	_archive(txt_path, tar_path)
	size_bytes, within_limit = _within_size_limit(tar_path, max_size_mb)

	logger.info(f"Merged {len(result_paths)} files ({len(rows)} rows) -> {txt_path}; "
				f"archive {size_bytes/1e6:.2f} MB -> {tar_path}")
	return {
		"txt_path"    : txt_path,
		"tar_path"    : tar_path,
		"size_bytes"  : size_bytes,
		"within_limit": within_limit,
		"files"       : files,
	}


def main(result_paths, output_path=None):
	"""Build the submission for the given result files (programmatic entry point).

	Args:
	    result_paths: Ordered list of per-scene result ``*.txt`` file paths.
	    output_path: Destination path for ``track1.txt``. When omitted, defaults
	        to ``track1.txt`` next to the first result file, since per-scene
	        results usually share a directory. Default: None.
	"""
	# All per-scene result files usually live in the same dir; default the output
	# there as track1.txt (next to the first result file).
	output_path = output_path or os.path.join(os.path.dirname(result_paths[0]), MERGED_NAME)
	build_submission(result_paths, output_path)


def parse_args():
	"""Parse command-line arguments for the submission builder.

	Returns:
	    An ``argparse.Namespace`` with:
	    - results: List of per-scene result ``*.txt`` paths (``--results``).
	    - output_path: Output path for ``track1.txt`` / ``track1.tar.gz``
	      (``--output_path``, default ``"track1.txt"``).
	"""
	parser = argparse.ArgumentParser(description="Build the track1.txt submission archive.")
	parser.add_argument("--results", type=str, nargs="+", 
						default=[
							"/media/vsw-ws-05/SSD_1/AI_City_Challenge/MTMC_Tracking_2026_processing/mots_multi/Warehouse_027.txt"
						],
						help="Per-scene result *.txt files to merge (one per scene).")
	parser.add_argument("--output_path", type=str, default="track1.txt",
						help="Output path for track1.txt / track1.tar.gz "
							 "(default: the directory of the first result file).")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	main(args.results, args.output_path)
