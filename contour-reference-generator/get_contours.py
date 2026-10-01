#!/usr/bin/env python3

# Run your desired backend executable like this: CARTA_AUTH_TOKEN=TEST_TOKEN carta --no_browser

import os
import argparse
import sys
import subprocess
import signal
import struct
import time

from cartaproto.client import Client
from cartaproto.messages import OpenFile, SetContourParameters, ImageBounds, IntBounds
from cartaproto.enums import SmoothingMode
import zstandard


def write_contour_top(index: list[int], levels: list[int], decimation_factor: int, uncompressed_coordinates_size: int, base: str):
    """Write the top-level contour metadata file."""
    folder_name = base
    # folder the same name as the image file
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' ready.")

    top_meta_file_name = f"{folder_name}_metadata.bin"
    top_meta_file_path = os.path.join(folder_name, top_meta_file_name)

    with open(top_meta_file_path, "wb") as f_meta:
        # Write number of levels
        levels = [int(level) for level in levels]
        f_meta.write(struct.pack('<i', len(levels)))
        # Write array of index values
        index = [int(v) for v in index]
        f_meta.write(struct.pack(f'<{len(index)}i', *index))
        # Write array of level values
        if levels:
            f_meta.write(struct.pack(f'<{len(levels)}i', *levels))
        # Write decimation factor
        f_meta.write(struct.pack('<i', int(decimation_factor)))
        # Write uncompressed coordinates size
        f_meta.write(struct.pack('<i', int(uncompressed_coordinates_size)))

    print(f"Metadata saved to: {top_meta_file_path}")
    return top_meta_file_path


def write_contour(index: int, level: int, base: str, vertices: list, indices: list):
    """Write one level of contour data."""
    folder_name = base
    # folder the same name as the image file
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' ready.")

    # folder for level
    level_folder = os.path.join(folder_name, str(index))
    os.makedirs(level_folder, exist_ok=True)
    print(f"Folder '{level_folder}' ready.")

    indices = list(indices)
    if not indices:
        indices = [0]
    elif indices[0] != 0:
        indices.insert(0, 0)

    if len(vertices) % 2 != 0:
        print(f"Warning: odd vertex count for level {level}; dropping trailing value.")
        vertices = vertices[:-1]

    indices = [int(idx) for idx in indices]
    if any(idx < 0 or idx > len(vertices) or idx % 2 != 0 for idx in indices):
        raise ValueError(
            f"Contour boundaries for level {level} must be even float offsets "
            f"between 0 and {len(vertices)}; got {indices}."
        )
    if indices != sorted(indices):
        raise ValueError(f"Contour boundaries for level {level} must be ordered; got {indices}.")

    num_contours = len(indices) - 1 if indices else 0

    # File paths for metadata and vertices
    meta_file_name = f"{index}_metadata.bin"
    verts_file_name = f"{index}_data.bin"

    meta_file_path = os.path.join(level_folder, meta_file_name)
    verts_file_path = os.path.join(level_folder, verts_file_name)

    if not vertices:
        print(f"Warning: no vertex data for level {level}; writing empty contour file.")

    with open(meta_file_path, "wb") as f_meta:
        f_meta.write(struct.pack('<i', int(level)))
        f_meta.write(struct.pack('<I', num_contours))
        for idx in indices[1:]:
            f_meta.write(struct.pack('<Q', idx))

    with open(verts_file_path, "wb") as f_verts:
        for i in range(0, len(vertices), 2):
            if i + 1 >= len(vertices):
                break
            x, y = vertices[i:i+2]
            f_verts.write(struct.pack("<ff", float(x), float(y)))

    print(f"Metadata saved to: {meta_file_path}")
    print(f"Vertices saved to: {verts_file_path}")
    return meta_file_path, verts_file_path


def decode_contour_payload(payload):
    raw_coordinates = bytes(payload.raw_coordinates)
    expected_size = int(payload.uncompressed_coordinates_size)
    decimation_factor = int(payload.decimation_factor)
    if decimation_factor > 0 and raw_coordinates:
        encoded_coordinates = zstandard.ZstdDecompressor().decompress(
            raw_coordinates,
            max_output_size=expected_size,
        )
    else:
        encoded_coordinates = raw_coordinates

    if len(encoded_coordinates) != expected_size:
        raise ValueError(
            f"Contour level {payload.level} decoded to {len(encoded_coordinates)} bytes; "
            f"expected {expected_size}."
        )
    if len(encoded_coordinates) % 4 != 0:
        raise ValueError(f"Contour coordinates for level {payload.level} are not float32-aligned.")

    if decimation_factor == 0:
        vertices = list(struct.unpack(f"<{len(encoded_coordinates) // 4}f", encoded_coordinates))
    else:
        if decimation_factor < 0:
            raise ValueError(f"Invalid contour decimation factor: {decimation_factor}.")
        if len(encoded_coordinates) % 8 != 0:
            raise ValueError(f"Contour coordinates for level {payload.level} do not contain complete x/y pairs.")

        shuffled = bytearray(encoded_coordinates)
        byte_order = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
        full_blocks = (len(shuffled) // 16) * 16
        unshuffled = bytearray(len(shuffled))
        for block_start in range(0, full_blocks, 16):
            block = shuffled[block_start:block_start + 16]
            for shuffled_index, original_index in enumerate(byte_order):
                unshuffled[block_start + original_index] = block[shuffled_index]
        unshuffled[full_blocks:] = shuffled[full_blocks:]

        deltas = struct.unpack(f"<{len(unshuffled) // 4}i", unshuffled)
        vertices = []
        x = 0
        y = 0
        for offset in range(0, len(deltas), 2):
            x += deltas[offset]
            y += deltas[offset + 1]
            vertices.extend((x / decimation_factor, y / decimation_factor))

    raw_start_indices = bytes(payload.raw_start_indices)
    if len(raw_start_indices) % 4 != 0:
        raise ValueError(f"Contour start indices for level {payload.level} are not uint32-aligned.")

    indices = list(struct.unpack(f"<{len(raw_start_indices) // 4}I", raw_start_indices))
    return vertices, indices


def save_contour_message(messages, base_dir):
    generator_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "contour-data-generator"))
    if generator_root not in sys.path:
        sys.path.insert(0, generator_root)

    from image_io import write_contour

    if not messages:
        return []

    def contour_payload(message):
        if hasattr(message, "contour_set"):
            return message.contour_set
        if hasattr(message, "contour_sets"):
            contour_sets = list(message.contour_sets)
            if contour_sets:
                return contour_sets[0]
        return message

    level_index = list(range(len(messages)))
    payloads = [contour_payload(message) for message in messages]
    levels = [int(p.level) for p in payloads]
    decimation_factor = next(
        (int(payload.decimation_factor) for payload in payloads if int(payload.decimation_factor) > 0),
        0,
    )
    uncompressed_coordinates_size = sum(int(p.uncompressed_coordinates_size) for p in payloads)

    write_contour.write_contour_top(level_index, levels, decimation_factor, uncompressed_coordinates_size, base_dir)

    written = []
    for idx, payload in enumerate(payloads):
        vertices, indices = decode_contour_payload(payload)
        if not indices:
            indices = [0]
        elif indices[0] != 0:
            indices.insert(0, 0)

        file_paths = write_contour.write_contour(idx, int(payload.level), base_dir, vertices, indices)
        written.append((int(payload.level), file_paths))

    return written

parser = argparse.ArgumentParser(description='Generate contour reference values.')
parser.add_argument('image', help='path to image file')
parser.add_argument('xmin', type=int, help='Minimum X value')
parser.add_argument('xmax', type=int, help='Maximum X value')
parser.add_argument('ymin', type=int, help='Minimum Y value')
parser.add_argument('ymax', type=int, help='Maximum Y value')
parser.add_argument('--levels', type=float, default=[0, 50, 100, 150, 200])
parser.add_argument('--smoothing', type=str, default="none")
parser.add_argument('--smoothing-factor', type=int, default=4)
parser.add_argument('--decimation-factor', type=int, default=4)
parser.add_argument('--compression-level', type=int, default=8)
parser.add_argument('--chunk-size', type=int, default=100000)

args = parser.parse_args()

cmd = ["/home/michaela/Desktop/carta/carta-backend/build/carta_backend", "--no_browser", "--top_level_folder", os.getcwd(), "--frontend_folder", "/home/michaela/Desktop/carta/carta-frontend/build"]
env = os.environ.copy()
env["CARTA_AUTH_TOKEN"] = "TEST_TOKEN"

proc = subprocess.Popen(cmd, preexec_fn=os.setpgrp, env=env)

time.sleep(1)

try:
    client = Client.from_parts("localhost", 3002, "TEST_TOKEN")

    ack = client.received_history[-1]
    if "Invalid ICD version number" in ack.message:
        sys.exit(ack.message)

    file_path = args.image
    file_dir, file_name = os.path.split(file_path)

    client.send(OpenFile(
        file=file_name, 
        directory=file_dir, 
        file_id=0
    ))

    levels = args.levels
    smoothing_mode = SmoothingMode.NoSmoothing
    if(args.smoothing == "gaussian"):
        print("gaussian smoothing")
        smoothing_mode = SmoothingMode.GaussianBlur
    if(args.smoothing == "block"):
        print("block smoothing")
        smoothing_mode = SmoothingMode.BlockAverage

    decimation_factor = args.decimation_factor
    compression_level = args.compression_level
    chunk_size = args.chunk_size
    bounds_min = 0
    bounds_max = 0

    message = SetContourParameters(
        file_id=0, 
        reference_file_id=0,
        image_bounds=(
            ImageBounds(x_min=args.xmin, x_max=args.xmax, y_min=args.ymin, y_max=args.ymax)
        ),
        levels=levels,
        smoothing_mode=smoothing_mode,
        smoothing_factor=decimation_factor,
        decimation_factor=decimation_factor,
        compression_level=compression_level,
        contour_chunk_size=chunk_size,
        channel_range=IntBounds(min=bounds_min, max=bounds_max)
    )

    client.send(message)

    client.receive()

    # last = client.received_history[-1]

    received = client.received_history
    contour_msgs = [
        msg for msg in received
        if type(msg).__name__ == "ContourImageData"
    ]

    base_name, _ = os.path.splitext(os.path.basename(args.image))
    base_name = base_name + "-" + args.smoothing
    saved = save_contour_message(contour_msgs, base_name)
    print(f"Saved {len(saved)} contour levels to '{base_name}'")
    for level, paths in saved:
        print(level, paths)

    print("EOT")
finally:
    if proc is not None:
        try:
            pgrp = os.getpgid(proc.pid)
            os.killpg(pgrp, signal.SIGINT)
            proc.wait()
        except ProcessLookupError:
            print("Could not shut down backend because it was no longer running.")