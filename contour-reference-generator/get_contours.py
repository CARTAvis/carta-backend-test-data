#!/usr/bin/env python3

# Run your desired backend executable like this: CARTA_AUTH_TOKEN=TEST_TOKEN carta --no_browser

import os
import argparse
import sys
import subprocess
import signal
import struct
import time
from array import array
from cartaproto.client import Client
from cartaproto.messages import OpenFile, SetContourParameters, ImageBounds, IntBounds, ContourImageData, ContourSet
from cartaproto.enums import SmoothingMode


def save_contours(folder_name: str, output_dir: str, level_index: int, num_messages: int, message: ContourImageData):
    file_dest = os.path.join(output_dir, folder_name)
    os.makedirs(file_dest, exist_ok=True)
    print(f"Folder '{file_dest}' ready.")

    top_meta_file_name = f"{folder_name}_metadata.bin"
    top_meta_file_path = os.path.join(file_dest, top_meta_file_name)

    with open(top_meta_file_path, "wb") as f_meta:
        # Write number of levels
        f_meta.write(struct.pack('<I', num_messages))

    for contour_set in message.contour_sets:
        save_level(file_dest, level_index, contour_set)

    print(f"Metadata saved to: {top_meta_file_path}")
    return top_meta_file_path


def save_level(parent_folder: str, level_index: int, contour_set: ContourSet):
    """Write one level of contour data."""

    # folder for level
    level_folder = os.path.join(parent_folder, str(level_index))
    os.makedirs(level_folder, exist_ok=True)
    print(f"Folder '{level_folder}' ready.")

    # if not indices:
    #     indices = [0, 0]
    #     if indices[0] != 0:
    #         raise ValueError(f"Invalid start index: {indices[0]}")

    

    # if len(coords) % 2:
    #     raise ValueError(f"Odd vertex count for level: {level}")

    # if any(idx < 0 or idx > len(coords) or idx % 2 for idx in indices):
    #     raise ValueError(
    #         f"Contour boundaries for level {level} must be even float offsets "
    #         f"between 0 and {len(coords)}; got {indices}."
    #     )
    # if indices != sorted(indices):
    #     raise ValueError(f"Contour boundaries for level {level} must be ordered; got {indices}.")

    num_points = contour_set.uncompressed_coordinates_size / 2

    meta_file_name = f"{level_index}_metadata.bin"
    verts_file_name = f"{level_index}_data.bin"

    meta_file_path = os.path.join(level_folder, meta_file_name)
    verts_file_path = os.path.join(level_folder, verts_file_name)

    indices = array("I")
    indices.frombytes(contour_set.raw_start_indices)

    with open(meta_file_path, "wb") as f_meta:
        f_meta.write(struct.pack('<f', contour_set.level)) # level
        f_meta.write(struct.pack('<I', int(num_points))) # number of points
        # f_meta.write(indices.tobytes())
        f_meta.write(struct.pack('<I', len(indices))) # number of indices
        f_meta.write(contour_set.raw_start_indices) # indices

    # float_coords = array("f")
    # float_coords.frombytes(contour_set.raw_coordinates)
    with open(verts_file_path, "wb") as f_verts:
        # f_verts.write(float_coords.tobytes())
        f_verts.write(contour_set.raw_coordinates)

    print(f"Metadata saved to: {meta_file_path}")
    print(f"Vertices saved to: {verts_file_path}")

    level_index += 1
    return meta_file_path, verts_file_path

parser = argparse.ArgumentParser(description="Script to request and save contour data from the backend")
parser.add_argument("image", help="Path to input image")
parser.add_argument("levels", type=float, nargs="+", help="Contour levels (a sequence of numbers; can be floating point)")
parser.add_argument("--output-dir", help="Path to output parent directory", default=".")
parser.add_argument("--smoothing", type=SmoothingMode.Value, default=SmoothingMode.NoSmoothing, help=f"Smoothing type. One of: {SmoothingMode.keys()}")
parser.add_argument("--smoothing-factor", type=int, default=0)

args = parser.parse_args()

if args.smoothing and not args.smoothing_factor:
    args.smoothing_factor = 4

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

    message = SetContourParameters(
        file_id=0, 
        reference_file_id=0,
        image_bounds=(
            ImageBounds(x_min=0, x_max=500, y_min=0, y_max=500)
        ),
        levels=args.levels,
        smoothing_mode=args.smoothing,
        smoothing_factor=args.smoothing_factor,
        decimation_factor=0,
        compression_level=0,
    )

    print(message)

    client.send(message)

    client.receive()

    received = client.received_history
    contour_msgs = [
        msg for msg in received
        if type(msg).__name__ == "ContourImageData"
    ]

    base_name, _ = os.path.splitext(os.path.basename(args.image))
    base_name = base_name + "-" + SmoothingMode.Name(args.smoothing)
    # output_dir = os.path.join(args.output_dir, base_name)

    for index, message in enumerate(contour_msgs):
        save_contours(base_name, args.output_dir, index, len(contour_msgs), message)

    print("EOT")
finally:
    if proc is not None:
        try:
            pgrp = os.getpgid(proc.pid)
            os.killpg(pgrp, signal.SIGINT)
            proc.wait()
        except ProcessLookupError:
            print("Could not shut down backend because it was no longer running.")