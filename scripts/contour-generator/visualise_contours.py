import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from contour_io import read_contours
from image_io import load_fits_image


def add_contours_to_image(
    image: np.ndarray,
    levels: list[int],
    output_image: str = "contour_multi",
    folder_name: str = "",
):
    """Plot each requested contour level over an image and save a JPEG.

    Contour data is read from numbered subfolders under ``folder_name``. The
    output is saved there when provided, or in the current directory otherwise.
    """
    fig, ax = plt.subplots()
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
    """Parse command-line arguments and render contours over the input image."""
    parser = argparse.ArgumentParser(
        description="Visualise contours on original image."
    )
    parser.add_argument("filename", help="Input image file (.fits or .h5)")
    parser.add_argument("folder", help="Folder where the saved data is")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 50, 100, 150, 200],
        help="The number of levels",
    )

    args = parser.parse_args()

    print(f"Reading image: {args.filename}")
    image = load_fits_image(args.filename)

    print(f"Saving image contours onto image: {args.filename}")
    add_contours_to_image(image, args.levels, "with_contours", args.folder)


if __name__ == "__main__":
    main()