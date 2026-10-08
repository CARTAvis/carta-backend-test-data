import argparse
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

from image_io import load_fits_image


def read_contour(folder: str):
    folder_name = os.path.basename(os.path.normpath(folder))
    if not folder_name.isdigit():
        raise ValueError(f"Contour level folder must have a numeric name: {folder}")
    level_index = int(folder_name)
    metadata_path = os.path.join(folder, f"{level_index}_metadata.bin")
    vertices_path = os.path.join(folder, f"{level_index}_data.bin")

    with open(metadata_path, "rb") as metadata_file:
        metadata = metadata_file.read()
    if len(metadata) < 8:
        raise ValueError(f"Contour metadata is too short: {metadata_path}")

    contour_level, contour_count = struct.unpack_from("<iI", metadata)
    expected_size = 8 + contour_count * 8
    if len(metadata) < expected_size:
        raise ValueError(f"Contour metadata is truncated: {metadata_path}")
    end_offsets = list(
        struct.unpack_from(f"<{contour_count}Q", metadata, 8)
    ) if contour_count else []

    with open(vertices_path, "rb") as vertices_file:
        vertex_data = vertices_file.read()
    if len(vertex_data) % 8 != 0:
        raise ValueError(f"Contour vertex data is not x/y-pair aligned: {vertices_path}")

    vertices = np.frombuffer(vertex_data, dtype="<f4").reshape(-1, 2)
    float_count = vertices.size
    boundaries = [0, *end_offsets]
    if boundaries[-1] != float_count:
        boundaries.append(float_count)
    if any(
        boundary < 0
        or boundary > float_count
        or boundary % 2 != 0
        for boundary in boundaries
    ) or any(start > end for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError(f"Invalid contour boundaries in {metadata_path}")

    point_boundaries = [boundary // 2 for boundary in boundaries]
    return contour_level, vertices, point_boundaries


def add_contours_to_image(
    image: np.ndarray,
    image_name: str,
    contour_folders: tuple[str, str],
    output_folder: str,
):
    figure, axis = plt.subplots()
    image_cmap = plt.get_cmap("gray").copy()
    image_cmap.set_bad("yellow")
    axis.imshow(image, cmap=image_cmap)

    contour_levels = []
    plotted_folders = []
    empty_folders = []
    colors = ("tab:blue", "tab:orange")
    for folder, color in zip(contour_folders, colors):
        contour_level, vertices, boundaries = read_contour(folder)
        contour_levels.append(contour_level)
        if len(vertices) == 0:
            empty_folders.append(folder)
            continue

        plotted_folders.append(folder)
        label = os.path.basename(os.path.dirname(os.path.normpath(folder)))

        for start, end in zip(boundaries, boundaries[1:]):
            segment = vertices[start:end]
            if len(segment):
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    linewidth=0.7,
                    color=color,
                    label=label,
                )
                label = ""

    if len(set(contour_levels)) != 1:
        raise ValueError(
            "The selected folders contain different contour levels: "
            f"{contour_levels}"
        )
    if empty_folders:
        print(f"Warning: no contour vertices found in: {', '.join(empty_folders)}")
    if not plotted_folders:
        raise ValueError(
            "Neither contour folder contains vertices. Choose folders for a level "
            "that has contours."
        )

    axis.axis("off")
    axis.legend(loc="lower left")

    output_path = os.path.join(
        output_folder, f"{image_name}_contour_comparison_level_{contour_levels[0]}.jpg"
    )
    figure.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot two contour levels on a FITS image."
    )
    parser.add_argument("filename", help="Input FITS image file")
    parser.add_argument(
        "contour_folder_1",
        help="First numeric contour-level folder, e.g. contours/example-none/0",
    )
    parser.add_argument(
        "contour_folder_2",
        help="Second numeric contour-level folder, e.g. contours/example-gaussian/0",
    )
    args = parser.parse_args()

    image_name = os.path.splitext(os.path.basename(args.filename))[0]
    image = load_fits_image.load_fits_image(args.filename)
    output_folder = os.path.dirname(os.path.abspath(args.filename))
    add_contours_to_image(
        image,
        image_name,
        (args.contour_folder_1, args.contour_folder_2),
        output_folder,
    )


if __name__ == "__main__":
    main()