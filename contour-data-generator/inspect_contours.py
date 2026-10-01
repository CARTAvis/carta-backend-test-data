#!/usr/bin/env python3
"""Print contour binary files in a human-readable form."""

import argparse
import os
import struct


def read_top_metadata(folder):
    folder_name = os.path.basename(os.path.normpath(folder))
    candidates = [
        os.path.join(folder, f"{folder_name}_metadata.bin"),
        f"{folder}_metadata.bin",
    ]
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        return None

    with open(path, "rb") as file:
        data = file.read()

    if len(data) < 4:
        raise ValueError(f"Top-level metadata is too short: {path}")

    level_count = struct.unpack_from("<i", data)[0]
    if level_count < 0:
        raise ValueError(f"Invalid level count in {path}: {level_count}")

    expected_size = 4 + (level_count * 8) + 8
    if len(data) < expected_size:
        raise ValueError(
            f"Top-level metadata is truncated: expected at least {expected_size} bytes, got {len(data)}"
        )

    offset = 4
    indices = list(struct.unpack_from(f"<{level_count}i", data, offset)) if level_count else []
    offset += level_count * 4
    levels = list(struct.unpack_from(f"<{level_count}i", data, offset)) if level_count else []
    offset += level_count * 4
    decimation_factor, uncompressed_size = struct.unpack_from("<ii", data, offset)

    return path, indices, levels, decimation_factor, uncompressed_size


def read_level(folder, level_index):
    level_folder = os.path.join(folder, str(level_index))
    metadata_path = os.path.join(level_folder, f"{level_index}_metadata.bin")
    data_path = os.path.join(level_folder, f"{level_index}_data.bin")

    with open(metadata_path, "rb") as file:
        metadata = file.read()
    if len(metadata) < 8:
        raise ValueError(f"Level metadata is too short: {metadata_path}")

    level, contour_count = struct.unpack_from("<iI", metadata)
    expected_size = 8 + contour_count * 8
    if len(metadata) < expected_size:
        raise ValueError(
            f"Level metadata is truncated: expected {expected_size} bytes, got {len(metadata)}"
        )

    end_indices = list(struct.unpack_from(f"<{contour_count}Q", metadata, 8)) if contour_count else []
    with open(data_path, "rb") as file:
        vertex_data = file.read()
    if len(vertex_data) % 4 != 0:
        raise ValueError(f"Vertex data is not float32-aligned: {data_path}")

    float_count = len(vertex_data) // 4
    vertices = list(struct.unpack(f"<{float_count}f", vertex_data)) if float_count else []
    return level, end_indices, vertices, metadata_path, data_path


def format_points(segment, max_points):
    point_count = len(segment) // 2
    shown_count = point_count if max_points is None else min(point_count, max_points)
    points = [
        f"({segment[offset]:.4f}, {segment[offset + 1]:.4f})"
        for offset in range(0, shown_count * 2, 2)
    ]
    suffix = "" if shown_count == point_count else f", ... {point_count - shown_count} more"
    return f"[{', '.join(points)}{suffix}]", point_count


def inspect_folder(folder, max_points):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError(f"Not a contour folder: {folder}")

    print(f"Contour folder: {folder}")
    top_metadata = read_top_metadata(folder)
    if top_metadata:
        path, indices, levels, decimation_factor, uncompressed_size = top_metadata
        print(f"Top metadata: {path}")
        print(f"  level indices: {indices}")
        print(f"  levels: {levels}")
        print(f"  decimation factor: {decimation_factor}")
        print(f"  uncompressed coordinates size: {uncompressed_size} bytes")
    else:
        print("Top-level metadata file not found.")

    level_indices = sorted(
        int(name)
        for name in os.listdir(folder)
        if name.isdigit() and os.path.isdir(os.path.join(folder, name))
    )
    if not level_indices:
        print("No per-level contour folders found.")
        return

    for level_index in level_indices:
        level, end_indices, vertices, metadata_path, data_path = read_level(folder, level_index)
        float_count = len(vertices)
        boundaries = [0, *end_indices]
        if boundaries[-1] != float_count:
            boundaries.append(float_count)

        print(f"\nIndex {level_index}, level {level}")
        print(f"  metadata: {metadata_path}")
        print(f"  data: {data_path}")
        print(f"  floats: {float_count}; points: {float_count // 2}; contours: {len(boundaries) - 1}")
        print(f"  stored end offsets: {end_indices}")

        if any(start > end for start, end in zip(boundaries, boundaries[1:])):
            print("  ERROR: contour boundaries are not ordered.")
        if any(boundary < 0 or boundary > float_count for boundary in boundaries):
            print("  ERROR: a contour boundary is outside the vertex data.")

        for contour_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start < 0 or end > float_count or start > end:
                print(f"  contour {contour_index}: invalid float range [{start}, {end})")
                continue

            segment = vertices[start:end]
            if len(segment) % 2 != 0:
                print(f"  contour {contour_index}: ERROR, {len(segment)} floats is not an x/y pair count")
            formatted, point_count = format_points(segment, max_points)
            print(f"  contour {contour_index}: floats [{start}, {end}), {point_count} points: {formatted}")


def main():
    parser = argparse.ArgumentParser(description="Inspect generated contour binary files.")
    parser.add_argument("folder", help="Contour output folder containing numeric level subfolders")
    parser.add_argument(
        "--max-points",
        type=int,
        default=5,
        help="Maximum coordinate pairs to print per contour; use 0 to print all (default: 5)",
    )
    args = parser.parse_args()
    if args.max_points < 0:
        parser.error("--max-points must be zero or greater")

    inspect_folder(args.folder, None if args.max_points == 0 else args.max_points)


if __name__ == "__main__":
    main()
