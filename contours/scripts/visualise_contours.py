import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from contour_io import read_contours
from image_io import load_fits_image


def add_contours_to_image(
    image: np.ndarray | None,
    levels: list[int],
    output_image: str = "contour_multi",
    folder_name: str = "",
):
    """Plot each requested contour level, optionally over an image, and save a JPEG.

    Contour data is read from numbered subfolders under ``folder_name``. The
    output is saved there when provided, or in the current directory otherwise.
    """
    fig, ax = plt.subplots()
    if image is not None:
        image_cmap = plt.get_cmap("gray").copy()
        image_cmap.set_bad("yellow")
        ax.imshow(image, cmap=image_cmap)

    cmap = plt.get_cmap("tab10")

    for i, level in enumerate(levels):
        contour_folder = os.path.join(folder_name, str(i))
        _, vertices, boundaries = read_contours(contour_folder)
        color = cmap(i % 10)

        for contour_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            segment = vertices[start:end]
            if len(segment) > 0:
                label_name = f"Level {level}" if contour_index == 0 else ""
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    linestyle="-",
                    marker=None,
                    linewidth=0.5,
                    color=color,
                    label=label_name,
                )

    ax.axis("off")
    ax.legend(loc="upper left")

    levels_str = "_".join(map(str, levels))
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, f"{output_image}_levels_{levels_str}.jpg")
    else:
        save_path = f"{output_image}_levels_{levels_str}.jpg"

    plt.savefig(save_path, format="jpg", dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path}")

    plt.close()


def main():
    """Parse command-line arguments and render contours with an optional image."""
    parser = argparse.ArgumentParser(
        description="Visualise contours on original image."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Input image file (.fits or .h5); omit with --no-background",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help=(
            "Folder where the saved data is (or the only positional argument "
            "with --no-background)"
        ),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 50, 100, 150, 200],
        help="The number of levels",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Save the contours without loading or displaying the input image",
    )

    args = parser.parse_args()

    if args.no_background:
        folder = args.folder or args.filename
        if folder is None:
            parser.error("a contour folder is required")
    else:
        if args.filename is None or args.folder is None:
            parser.error(
                "filename and folder are required unless --no-background is set"
            )
        folder = args.folder

    image = None
    if not args.no_background:
        print(f"Reading image: {args.filename}")
        image = load_fits_image(args.filename)

    if args.no_background:
        print("Saving contours without a background image")
    else:
        print(f"Saving image contours onto image: {args.filename}")
    add_contours_to_image(image, args.levels, "with_contours", folder)

main()
