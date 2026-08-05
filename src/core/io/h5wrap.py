#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Schema-agnostic HDF5 (de)serialization for arbitrarily nested structures.

`save_dict_to_h5` and `load_dict_from_h5` round-trip any nesting of dicts,
lists, tuples, ``None``, strings, bytes, Python/NumPy scalars and NumPy arrays
without the structure being known in advance. Every node is tagged with a
``type`` attribute so the loader can rebuild the exact same object. NumPy
arrays are stored compressed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
	"CompressionConfig",
	"H5SerializationError",
	"load_dict_from_h5",
	"save_dict_to_h5",
]


# MARK: - Constants

_TYPE_ATTR     = "type"
_KEY_TYPE_ATTR = "key_type"
_LENGTH_ATTR   = "length"
_ROOT_KEY      = "root"
_STRING_DTYPE  = h5py.string_dtype(encoding="utf-8")
_BYTES_DTYPE   = "uint8"

DEFAULT_COMPRESSION       = "gzip"
DEFAULT_COMPRESSION_LEVEL = 4


# MARK: - Errors


class H5SerializationError(Exception):
	"""Raised when a value cannot be serialized to or restored from HDF5."""


# MARK: - NodeType


class NodeType(StrEnum):
	"""Tag stored on each HDF5 node describing the Python type it encodes."""

	DICT    = "dict"
	LIST    = "list"
	TUPLE   = "tuple"
	NONE    = "none"
	STR     = "str"
	BYTES   = "bytes"
	BOOL    = "bool"
	INT     = "int"
	FLOAT   = "float"
	NDARRAY = "ndarray"


# MARK: - CompressionConfig


@dataclass(frozen=True)
class CompressionConfig:
	"""Compression settings applied to NumPy-array datasets.

	Attributes:
		algorithm: HDF5 compression filter name, e.g. ``"gzip"``.
		level: Compression level passed to HDF5 as ``compression_opts``. Use
			``None`` for filters that take no options, such as ``"lzf"``.
	"""

	algorithm: str = DEFAULT_COMPRESSION
	level: int | None = DEFAULT_COMPRESSION_LEVEL


# MARK: - Writer


class _H5Writer:
	"""Recursively writes an arbitrary structure into an open HDF5 group.

	The compression settings are held as state so they need not be threaded
	through every recursive call.
	"""

	def __init__(self, compression: CompressionConfig) -> None:
		self._compression = compression

	def write(self, parent: h5py.Group, key: str, value: Any) -> None:
		"""Write `value` under ``parent[key]``, tagged with its node type."""
		if isinstance(value, dict):
			self._write_dict(parent, key, value)
		elif isinstance(value, (list, tuple)):
			self._write_sequence(parent, key, value)
		elif value is None:
			self._write_none(parent, key)
		else:
			self._write_leaf(parent, key, value)

	def _write_dict(self, parent: h5py.Group, key: str, value: dict) -> None:
		group = parent.create_group(key)
		group.attrs[_TYPE_ATTR] = NodeType.DICT.value
		for child_key, child_value in value.items():
			name = str(child_key)
			self.write(group, name, child_value)
			# Remember the original key type so non-str keys round-trip.
			group[name].attrs[_KEY_TYPE_ATTR] = type(child_key).__name__

	def _write_sequence(
		self, parent: h5py.Group, key: str, value: list | tuple
	) -> None:
		group     = parent.create_group(key)
		node_type = NodeType.TUPLE if isinstance(value, tuple) else NodeType.LIST
		group.attrs[_TYPE_ATTR]   = node_type.value
		group.attrs[_LENGTH_ATTR] = len(value)
		for index, item in enumerate(value):
			self.write(group, str(index), item)

	def _write_none(self, parent: h5py.Group, key: str) -> None:
		group = parent.create_group(key)
		group.attrs[_TYPE_ATTR] = NodeType.NONE.value

	def _write_leaf(self, parent: h5py.Group, key: str, value: Any) -> None:
		if isinstance(value, str):
			dataset = parent.create_dataset(key, data=value, dtype=_STRING_DTYPE)
			self._tag(dataset, NodeType.STR)
		elif isinstance(value, (bytes, bytearray)):
			raw = np.frombuffer(bytes(value), dtype=_BYTES_DTYPE)
			self._tag(parent.create_dataset(key, data=raw), NodeType.BYTES)
		elif isinstance(value, np.ndarray):
			self._write_array(parent, key, value)
		else:
			self._write_scalar(parent, key, value)

	def _write_array(self, parent: h5py.Group, key: str, value: np.ndarray) -> None:
		options: dict[str, Any] = {}
		# Compression needs a chunkable (non-empty, non-scalar) dataset.
		if value.size > 0 and value.ndim > 0:
			options["compression"] = self._compression.algorithm
			# Some filters (e.g. "lzf") reject compression_opts entirely.
			if self._compression.level is not None:
				options["compression_opts"] = self._compression.level
		self._tag(parent.create_dataset(key, data=value, **options), NodeType.NDARRAY)

	def _write_scalar(self, parent: h5py.Group, key: str, value: Any) -> None:
		# `bool` must be checked before `int`, since `bool` subclasses `int`.
		if isinstance(value, (bool, np.bool_)):
			self._tag(parent.create_dataset(key, data=bool(value)), NodeType.BOOL)
		elif isinstance(value, (int, np.integer)):
			self._tag(parent.create_dataset(key, data=np.int64(value)), NodeType.INT)
		elif isinstance(value, (float, np.floating)):
			self._tag(parent.create_dataset(key, data=np.float64(value)), NodeType.FLOAT)
		else:
			raise H5SerializationError(
				f"Cannot serialize value of type {type(value).__name__!r} "
				f"at key {key!r}."
			)

	@staticmethod
	def _tag(dataset: h5py.Dataset, node_type: NodeType) -> None:
		dataset.attrs[_TYPE_ATTR] = node_type.value


# MARK: - Reader


class _H5Reader:
	"""Recursively reconstructs a structure from a tagged HDF5 node.

	Dispatch is table-driven on the stored ``type`` tag rather than a chain of
	conditionals, so each node type maps to exactly one focused handler.
	"""

	def __init__(self) -> None:
		self._handlers: dict[NodeType, Callable[[Any], Any]] = {
			NodeType.DICT   : self._read_dict,
			NodeType.LIST   : self._read_list,
			NodeType.TUPLE  : self._read_tuple,
			NodeType.NONE   : self._read_none,
			NodeType.STR    : self._read_str,
			NodeType.BYTES  : self._read_bytes,
			NodeType.BOOL   : self._read_bool,
			NodeType.INT    : self._read_int,
			NodeType.FLOAT  : self._read_float,
			NodeType.NDARRAY: self._read_array,
		}

	def read(self, node: h5py.Group | h5py.Dataset) -> Any:
		"""Reconstruct the Python object encoded by `node`."""
		if _TYPE_ATTR not in node.attrs:
			raise H5SerializationError(f"Node {node.name!r} is missing its type tag.")
		try:
			node_type = NodeType(_as_str(node.attrs[_TYPE_ATTR]))
		except ValueError as exc:
			raise H5SerializationError(
				f"Unknown node type tag on {node.name!r}: {node.attrs[_TYPE_ATTR]!r}."
			) from exc
		return self._handlers[node_type](node)

	def _read_dict(self, node: h5py.Group) -> dict:
		result = {}
		for name in node:
			child    = node[name]
			key_type = _as_str(child.attrs.get(_KEY_TYPE_ATTR, NodeType.STR.value))
			result[_cast_key(name, key_type)] = self.read(child)
		return result

	def _read_list(self, node: h5py.Group) -> list:
		return self._read_items(node)

	def _read_tuple(self, node: h5py.Group) -> tuple:
		return tuple(self._read_items(node))

	def _read_items(self, node: h5py.Group) -> list:
		length = int(node.attrs[_LENGTH_ATTR])
		return [self.read(node[str(index)]) for index in range(length)]

	@staticmethod
	def _read_none(node: h5py.Group) -> None:
		return None

	@staticmethod
	def _read_str(node: h5py.Dataset) -> str:
		value = node[()]
		return value.decode("utf-8") if isinstance(value, bytes) else str(value)

	@staticmethod
	def _read_bytes(node: h5py.Dataset) -> bytes:
		return np.asarray(node[()], dtype=_BYTES_DTYPE).tobytes()

	@staticmethod
	def _read_bool(node: h5py.Dataset) -> bool:
		return bool(node[()])

	@staticmethod
	def _read_int(node: h5py.Dataset) -> int:
		return int(node[()])

	@staticmethod
	def _read_float(node: h5py.Dataset) -> float:
		return float(node[()])

	@staticmethod
	def _read_array(node: h5py.Dataset) -> np.ndarray:
		return np.asarray(node[()])


# MARK: - Helpers


def _as_str(value: str | bytes) -> str:
	"""Normalize an HDF5 attribute value to a `str`."""
	return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _cast_key(name: str, key_type: str) -> Any:
	"""Cast a stringified dict key back to its original scalar type."""
	if key_type == NodeType.INT:
		return int(name)
	if key_type == NodeType.FLOAT:
		return float(name)
	if key_type == NodeType.BOOL:
		return name == "True"
	return name


# MARK: - Public API


def save_dict_to_h5(
	path: str | Path,
	data: Any,
	*,
	compression: CompressionConfig | None = None,
) -> None:
	"""Serialize an arbitrarily nested structure to an HDF5 file.

	The structure does not need to be known in advance. Supported nodes are
	`dict`, `list`, `tuple`, `None`, `str`, `bytes`, `bool`, `int`, `float`,
	NumPy scalars and NumPy arrays, nested to any depth.

	Args:
		path: Destination ``.h5`` file path.
		data: Any supported (possibly nested) value to serialize.
		compression: Array-compression settings. Defaults to gzip level 4.

	Raises:
		H5SerializationError: If `data` contains an unsupported type.
	"""
	writer = _H5Writer(compression or CompressionConfig())
	with h5py.File(Path(path), "w") as h5_file:
		writer.write(h5_file, _ROOT_KEY, data)


def load_dict_from_h5(path: str | Path) -> Any:
	"""Load and reconstruct a structure written by `save_dict_to_h5`.

	Args:
		path: Path to the ``.h5`` file.

	Returns:
		The reconstructed (possibly nested) structure.

	Raises:
		H5SerializationError: If the file holds an unknown or untagged node.
	"""
	reader = _H5Reader()
	with h5py.File(Path(path), "r") as h5_file:
		return reader.read(h5_file[_ROOT_KEY])
