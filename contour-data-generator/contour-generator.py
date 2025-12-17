#!/usr/bin/env python3
"""
Generate contour lines from FITS or HDF5 images.
Outputs one .txt file per contour level and optionally a JPG visualization.
"""

import argparse
import math
import numpy as np
import h5py
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import os

# -------------------------------
# Image I/O
# -------------------------------
def load_image(filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".fits", ".fit"]:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
    elif ext in [".h5", ".hdf5"]:
        with h5py.File(filename, "r") as f:
            # Try to find the first dataset in the file
            def first_dataset(g):
                for key in g.keys():
                    if isinstance(g[key], h5py.Dataset):
                        return g[key][()]
                    elif isinstance(g[key], h5py.Group):
                        result = first_dataset(g[key])
                        if result is not None:
                            return result
                return None
            data = first_dataset(f)
            if data is None:
                raise ValueError("No dataset found in HDF5 file.")
            data = np.array(data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure 2D
    if data.ndim > 2:
        data = data[0]
    return np.nan_to_num(data)


# -------------------------------
# Smoothing
# -------------------------------
def apply_smoothing(image: np.ndarray, mode: str, factor: float) -> np.ndarray:
    if mode == "none":
        return image
    elif mode == "gaussian":
        return gaussian_filter(image, sigma=factor)
    elif mode == "block":
        return block_reduce(image, block_size=(int(factor), int(factor)), func=np.mean)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")


# -------------------------------
# Contour Extraction
# -------------------------------

def trace_segment(image, visited, width, height, scale, offset, level, x_cell, y_cell, side, vertices):
    i = x_cell
    j = y_cell
    orig_side = side
    first_iteration = True
    done = (i < 0 or i >= width - 1 or j < 0 or j >= height - 1)

    while not done:
        flag = False
        a = image[j * width + i]
        b = image[j * width + i + 1]
        c = image[(j + 1) * width + i + 1]
        d = image[(j + 1) * width + i]

        # Replace NaNs with negative infinity
        a = a if not math.isnan(a) else -float('inf')
        b = b if not math.isnan(b) else -float('inf')
        c = c if not math.isnan(c) else -float('inf')
        d = d if not math.isnan(d) else -float('inf')

        x = y = 0.0

        if first_iteration:
            first_iteration = False
            if side == 0:  # TopEdge
                x = (level - a) / (b - a) + i
                y = j
            elif side == 1:  # RightEdge
                x = i + 1
                y = (level - b) / (c - b) + j
            elif side == 2:  # BottomEdge
                x = (level - c) / (d - c) + i
                y = j + 1
            elif side == 3:  # LeftEdge
                x = i
                y = (level - a) / (d - a) + j
        else:
            if side == 0:  # Mark visited on top edge
                visited[j * width + i] = True

            while not flag:
                side = (side + 1) % 4

                if side == 0:  # TopEdge
                    if a >= level and level > b:
                        flag = True
                        x = (level - a) / (b - a) + i
                        y = j
                        j -= 1
                elif side == 1:  # RightEdge
                    if b >= level and level > c:
                        flag = True
                        x = i + 1
                        y = (level - b) / (c - b) + j
                        i += 1
                elif side == 2:  # BottomEdge
                    if c >= level and level > d:
                        flag = True
                        x = (level - d) / (c - d) + i
                        y = j + 1
                        j += 1
                elif side == 3:  # LeftEdge
                    if d >= level and level > a:
                        flag = True
                        x = i
                        y = (level - a) / (d - a) + j
                        i -= 1

            side = (side + 2) % 4

            if (i == x_cell and j == y_cell and side == orig_side) or \
                (i < 0 or i >= width - 1 or j < 0 or j >= height - 1):
                done = True

        # Shift to pixel center
        x_val = x + 0.5
        y_val = y + 0.5
        vertices.append(scale * x_val + offset)
        vertices.append(scale * y_val + offset)

def trace_level(image, width, height, scale, offset, level):
    num_pixels = width * height
    checked_pixels = 0

    visited = [False] * num_pixels
    vertices = []
    indices = []

    # ---- Top Edge ----
    for j in range(0, 1):
        for i in range(width - 1):
            pt_a = image[j * width + i]
            pt_b = image[j * width + i + 1]

            if (math.isnan(pt_a) or pt_a < level) and level <= pt_b:
                indices.append(len(vertices))
                trace_segment(image, visited, width, height, scale, offset, level, i, j, 0, vertices)
            checked_pixels += 1

    # ---- Right Edge ----
    i = width - 1
    for j in range(height - 1):
        pt_a = image[j * width + i]
        pt_b = image[(j + 1) * width + i]

        if (math.isnan(pt_a) or pt_a < level) and level <= pt_b:
            indices.append(len(vertices))
            trace_segment(image, visited, width, height, scale, offset, level, i - 1, j, 1, vertices)
        checked_pixels += 1

    # ---- Bottom Edge ----
    j = height - 1
    for i in range(width - 2, -1, -1):
        pt_a = image[j * width + i + 1]
        pt_b = image[j * width + i]

        if (math.isnan(pt_a) or pt_a < level) and level <= pt_b:
            indices.append(len(vertices))
            trace_segment(image, visited, width, height, scale, offset, level, i, j - 1, 2, vertices)
        checked_pixels += 1

    # ---- Left Edge ----
    i = 0
    for j in range(height - 2, -1, -1):
        pt_a = image[(j + 1) * width + i]
        pt_b = image[j * width + i]

        if (math.isnan(pt_a) or pt_a < level) and level <= pt_b:
            indices.append(len(vertices))
            trace_segment(image, visited, width, height, scale, offset, level, i, j, 3, vertices)
        checked_pixels += 1

    # ---- Interior ----
    for j in range(1, height - 1):
        for i in range(width - 1):
            pt_a = image[j * width + i]
            pt_b = image[j * width + i + 1]

            if (not visited[j * width + i]) and (math.isnan(pt_a) or pt_a < level) and level <= pt_b:
                indices.append(len(vertices))
                trace_segment(image, visited, width, height, scale, offset, level, i, j, 0, vertices)
            checked_pixels += 1

    return vertices, indices

def generate_contours(image: np.ndarray, smoothing_mode: str, level: float):
    vertex_map = []
    index_map = []

    # contour level
    vertex_map, index_map = trace_level(
        image.flatten(),
        image.shape[1],
        image.shape[0],
        scale=1.0,
        offset=0.0,
        level=level
    )

    return vertex_map, index_map


# -------------------------------
# Output Writing
# -------------------------------
def write_contour_files(level: float, base: str, vertices: list, indices: list, formatted: bool):
    folder_name = base
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    
    # If no indices, treat entire list as a single contour
    if not indices:
        indices = [0]

    # append end marker
    indices_sorted = sorted(indices)
    indices_sorted.append(len(vertices))

    for idx_num in range(len(indices_sorted)-1):
        start = indices_sorted[idx_num]
        end = indices_sorted[idx_num+1]
        contour_vertices = vertices[start:end]
        file_name = f"{base}_level_{level}.txt"
        file_path = os.path.join(folder_name, file_name)

        with open(file_path, "a") as f:
            if formatted:
                f.write(f"# Contour Level: {level}\n")
                f.write(f"# Part: {idx_num+1}\n")
                f.write(f"# Number of vertices: {len(contour_vertices) // 2}\n")
                f.write("# X, Y coordinates\n\n")
                for i in range(0, len(contour_vertices), 2):
                    x, y = contour_vertices[i], contour_vertices[i + 1]
                    f.write(f"{x:.6f}, {y:.6f}\n")
            else:
                for i in range(0, len(contour_vertices), 2):
                    x, y = contour_vertices[i], contour_vertices[i + 1]
                    f.write(f"{x:.6f} {y:.6f}\n")
            f.write(f"\n")
        print(f"✅ Saved contour to: {file_name}")

# def save_contours_to_file(file_name: str, levels: list, vertex_data: list, index_data: list):
#     with open(file_name, 'w') as file:
#         for l in range(len(levels)):
#             file.write(f"LEVEL: {levels[l]}\n")
#             file.write(f"VERTICES: {len(vertex_data[l]) // 2}\n")
            
#             # Write vertices as (x, y) pairs
#             for v in range(0, len(vertex_data[l]), 2):
#                 file.write(f"{vertex_data[l][v]} {vertex_data[l][v + 1]}\n")
            
#             file.write(f"INDICES: {len(index_data[l])}\n")
#             for idx in index_data[l]:
#                 file.write(f"{idx}\n")
            
#             file.write("---\n")  # Separator between levels
#     print(f"✅ Saved: {file_name}")


# -------------------------------
# Visualisation
# -------------------------------
def show_contours(image: np.ndarray, vertices: np.ndarray, indices: list, level: float, output_image: str | None = None, folder_name: str = ""):
    # If no indices, plot the whole array as a single contour
    if not indices:
        indices = [0]
    indices_sorted = sorted(indices)
    indices_sorted.append(len(vertices))

    fig, ax = plt.subplots()
    ax.imshow(image)
    # plot each contour segment separately
    for idx_num in range(len(indices_sorted)-1):
        start = indices_sorted[idx_num]
        end = indices_sorted[idx_num+1]
        seg = np.array(vertices[start:end]).reshape(-1, 2)
        ax.plot(seg[:, 0], seg[:, 1], linewidth=1.5)
    ax.axis('off')

    plt.savefig(f"{folder_name}/{output_image}_level_{level}.jpg", format='jpg', dpi=300)
    print(f"Contour image saved to: {output_image}_level_{level}.jpg")

    plt.close()


# -------------------------------
# Main CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate contour data from FITS or HDF5 images.")
    parser.add_argument("filename", help="Input image file (.fits or .h5)")
    parser.add_argument("--levels", nargs="+", type=float, default=[-1, 0, 1],
                        help="Contour levels (default: -1 0 1)")
    parser.add_argument("--smoothing", nargs="+", default=["none"],
                        help="Smoothing mode: none | gaussian <sigma> | block <factor>")
    parser.add_argument("--formatted", action="store_true", help="Save human-readable contour files.")
    parser.add_argument("--show", action="store_true", help="Save contours overlay on images as JPG.")

    args = parser.parse_args()

    # Parse smoothing arguments
    smoothing_mode = args.smoothing[0]
    smoothing_value = float(args.smoothing[1]) if len(args.smoothing) > 1 else 1.0

    print(f"Reading image: {args.filename}")
    image = load_image(args.filename)

    print(f"Image shape: {image.shape}")

    # print(f"Applying smoothing: {smoothing_mode} ({smoothing_value})")
    # smoothed = apply_smoothing(image, smoothing_mode, smoothing_value)

    base = os.path.splitext(os.path.basename(args.filename))[0]

    for level in args.levels:
        print(f"Generating contours for levels: {level}")
        vertices, indices = generate_contours(image, "none", level)

        write_contour_files(level, base + "_contours", vertices, indices, args.formatted)

    for level in args.levels:
        # write_contour_files(level, base, vertices, indices, args.formatted)
        if args.show:
            show_contours(image, vertices, indices, level, base, f"{base}_contours")

if __name__ == "__main__":
    main()
