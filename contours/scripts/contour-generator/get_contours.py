#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import time

from cartaproto.client import Client
from cartaproto.messages import ImageBounds, OpenFile, SetContourParameters
from cartaproto.enums import SmoothingMode

from contour_io.write import save_contours


def parse_args():
    parser = argparse.ArgumentParser(
        description="Request and save contour data from the backend"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--backend-executable",
        help="Path to the CARTA backend executable",
    )
    parser.add_argument(
        "--frontend-folder",
        help="Path to the CARTA frontend build directory",
    )
    parser.add_argument(
        "levels",
        type=float,
        nargs="+",
        help="Contour levels (a sequence of numbers; can be floating point)",
    )
    parser.add_argument("--output-dir", help="Path to output parent directory")
    parser.add_argument(
        "--smoothing",
        type=SmoothingMode.Value,
        default=SmoothingMode.NoSmoothing,
        help=f"Smoothing type. One of: {SmoothingMode.keys()}",
    )
    parser.add_argument("--smoothing-factor", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoothing and not args.smoothing_factor:
        args.smoothing_factor = 4

    command = [
        args.backend_executable,
        "--no_browser",
        "--top_level_folder",
        os.getcwd(),
        "--frontend_folder",
        args.frontend_folder,
    ]
    env = os.environ.copy()
    env["CARTA_AUTH_TOKEN"] = "TEST_TOKEN"
    proc = subprocess.Popen(command, preexec_fn=os.setpgrp, env=env)

    time.sleep(1)
    try:
        client = Client.from_parts("localhost", 3002, "TEST_TOKEN")
        ack = client.received_history[-1]
        if "Invalid ICD version number" in ack.message:
            sys.exit(ack.message)

        file_dir, file_name = os.path.split(args.image)
        client.send(OpenFile(file=file_name, directory=file_dir, file_id=0))

        message = SetContourParameters(
            file_id=0,
            reference_file_id=0,
            image_bounds=ImageBounds(x_min=0, x_max=500, y_min=0, y_max=500),
            levels=args.levels,
            smoothing_mode=args.smoothing,
            smoothing_factor=args.smoothing_factor,
            decimation_factor=0,
            compression_level=0,
        )
        print(message)
        client.send(message)
        client.receive()

        contour_messages = [
            received_message
            for received_message in client.received_history
            if type(received_message).__name__ == "ContourImageData"
        ]
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        base_name += f"_{SmoothingMode.Name(args.smoothing)}"

        for index, contour_message in enumerate(contour_messages):
            save_contours(
                base_name,
                args.output_dir,
                index,
                len(contour_messages),
                contour_message,
            )

        print("EOT")
    finally:
        try:
            process_group = os.getpgid(proc.pid)
            os.killpg(process_group, signal.SIGINT)
            proc.wait()
        except ProcessLookupError:
            print("Could not shut down backend because it was no longer running.")

main()
