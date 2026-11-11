"""Read contour files produced by ``contour_io.write``."""

import os
import struct

import numpy as np


def read_contours(folder: str) -> tuple[float, np.ndarray, list[int]]:
    """Read one contour level and return its value, vertices, and boundaries.

    Boundaries are returned as point offsets suitable for slicing ``vertices``.
    """
    folder_name = os.path.basename(os.path.normpath(folder))
    if not folder_name.isdigit():
        raise ValueError(f"Contour level folder must have a numeric name: {folder}")

    metadata_path = os.path.join(folder, f"{folder_name}_metadata.bin")
    vertices_path = os.path.join(folder, f"{folder_name}_data.bin")

    with open(metadata_path, "rb") as metadata_file:
        metadata = metadata_file.read()
    if len(metadata) < 12:
        raise ValueError(f"Contour metadata is too short: {metadata_path}")

    contour_level, point_count, start_count = struct.unpack_from("<fII", metadata)
    expected_size = 12 + start_count * 4
    if len(metadata) < expected_size:
        raise ValueError(
            f"Contour metadata is truncated: expected {expected_size} bytes, "
            f"got {len(metadata)}"
        )

    start_offsets = (
        list(struct.unpack_from(f"<{start_count}I", metadata, 12))
        if start_count
        else []
    )

    with open(vertices_path, "rb") as vertices_file:
        vertex_data = vertices_file.read()
    if len(vertex_data) % 8 != 0:
        raise ValueError(f"Contour vertex data is not x/y-pair aligned: {vertices_path}")

    vertices = np.frombuffer(vertex_data, dtype="<f4").reshape(-1, 2)
    float_count = vertices.size
    if len(vertices) != point_count:
        raise ValueError(
            f"Contour metadata declares {point_count} points, "
            f"but vertex data contains {len(vertices)}: {vertices_path}"
        )
    if any(
        offset > float_count or offset % 2 != 0
        for offset in start_offsets
    ) or any(
        start > end for start, end in zip(start_offsets, start_offsets[1:])
    ):
        raise ValueError(f"Invalid contour start offsets in {metadata_path}")

    boundaries = [offset // 2 for offset in start_offsets]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != len(vertices):
        boundaries.append(len(vertices))

    return contour_level, vertices, boundaries
