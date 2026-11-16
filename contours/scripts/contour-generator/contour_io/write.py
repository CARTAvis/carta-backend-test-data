"""Write contour data received from the CARTA backend."""

import os
import struct
from array import array

from cartaproto.messages import ContourImageData, ContourSet


def save_contours(
    folder_name: str,
    output_dir: str,
    level_index: int,
    num_messages: int,
    message: ContourImageData,
) -> str:
    """Write the message's contour sets and top-level metadata.

    Args:
        folder_name: Name of the output folder for this image and smoothing mode.
        output_dir: Parent directory in which to create the output folder.
        level_index: Numeric subfolder name for the contour sets.
        num_messages: Number of contour image messages in the response.
        message: Backend response containing the contour sets to write.

    Returns:
        The path to the top-level metadata file.
    """
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Folder '{output_folder}' ready.")

    metadata_path = os.path.join(output_folder, f"{folder_name}_metadata.bin")
    with open(metadata_path, "wb") as metadata_file:
        metadata_file.write(struct.pack("<I", num_messages))

    for contour_set in message.contour_sets:
        save_level(output_folder, level_index, contour_set)

    print(f"Metadata saved to: {metadata_path}")
    return metadata_path


def save_level(
    parent_folder: str, level_index: int, contour_set: ContourSet
) -> tuple[str, str]:
    """Write a contour set's metadata and coordinate data.

    Args:
        parent_folder: Directory under which the numeric level folder is created.
        level_index: Numeric name for the contour level folder and its files.
        contour_set: Backend contour set whose binary data is written.

    Returns:
        Paths to the level metadata file and coordinate data file, respectively.
    """
    level_folder = os.path.join(parent_folder, str(level_index))
    os.makedirs(level_folder, exist_ok=True)
    print(f"Folder '{level_folder}' ready.")

    num_points = contour_set.uncompressed_coordinates_size / 2
    metadata_path = os.path.join(level_folder, f"{level_index}_metadata.bin")
    vertices_path = os.path.join(level_folder, f"{level_index}_data.bin")

    indices = array("I")
    indices.frombytes(contour_set.raw_start_indices)

    with open(metadata_path, "wb") as metadata_file:
        metadata_file.write(struct.pack("<f", contour_set.level))
        metadata_file.write(struct.pack("<I", int(num_points)))
        metadata_file.write(struct.pack("<I", len(indices)))
        metadata_file.write(contour_set.raw_start_indices)

    with open(vertices_path, "wb") as vertices_file:
        vertices_file.write(contour_set.raw_coordinates)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Vertices saved to: {vertices_path}")
    return metadata_path, vertices_path
