#!/usr/bin/env python3
"""
Generate contour lines from FITS or HDF5 images.
Outputs .txt or .bin file per contour level and optionally a JPG visualisation.
"""

import argparse
import math
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import os

# -------------------------------
# Image I/O
# -------------------------------
HEADER_KEYS = (
    'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
    'CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'CTYPE1', 'CTYPE2', 'EQUINOX', 'SIMPLE', 'BZERO', 'BSCALE', 'NAXIS',
    'NAXIS1', 'NAXIS2'
)


def normalize_header_value(value):
    if isinstance(value, bytes):
        try:
            return value.decode('ascii')
        except UnicodeDecodeError:
            return value.decode('utf-8', errors='ignore')
    if hasattr(value, 'item'):
        return value.item()
    return value


def extract_metadata_from_header(header):
    metadata = {}
    for key in HEADER_KEYS:
        if key in header:
            metadata[key] = normalize_header_value(header[key])
    return metadata


def extract_metadata_from_hdf5(file_obj):
    metadata = {}

    def collect_attrs(name, obj):
        if hasattr(obj, 'attrs'):
            for key, value in obj.attrs.items():
                key_up = key.upper()
                if key_up in HEADER_KEYS and key_up not in metadata:
                    metadata[key_up] = normalize_header_value(value)

    file_obj.visititems(collect_attrs)
    return metadata


def find_first_dataset(group):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            return item
        if isinstance(item, h5py.Group):
            result = find_first_dataset(item)
            if result is not None:
                return result
    return None


def load_image(filename: str) -> tuple[np.ndarray, dict]:
    ext = os.path.splitext(filename)[1].lower()
    metadata = {}

    if ext in [".fits", ".fit"]:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            metadata = extract_metadata_from_header(hdul[0].header)
            x_offset = metadata.get('CRPIX1', 0)
            y_offset = metadata.get('CRPIX2', 0)
            print(f"FITS Header - CRPIX1 (X): {x_offset}, CRPIX2 (Y): {y_offset}")
    elif ext in [".h5", ".hdf5"]:
        with h5py.File(filename, "r") as f:
            dataset = find_first_dataset(f)
            if dataset is None:
                raise ValueError("No dataset found in HDF5 file.")
            data = np.array(dataset[()], dtype=np.float32)
            metadata = extract_metadata_from_hdf5(f)
            if metadata:
                print(f"HDF5 metadata keys: {', '.join(sorted(metadata.keys()))}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure 2D
    if data.ndim > 2:
        data = data[0]

    return np.nan_to_num(data), metadata


def metadata_to_wcs(metadata):
    if not metadata:
        return None

    header = fits.Header()
    for key, value in metadata.items():
        header[key] = value

    try:
        return WCS(header)
    except Exception:
        return None

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
def write_contour_binary(folder_name: str, level: float, base: str, vertices: list, indices: list):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    
    # If no indices, treat entire list as a single contour
    if not indices:
        indices = [0]

    # append end marker
    indices_sorted = sorted(indices)
    indices_sorted.append(len(vertices))

    file_name = f"{level}.bin"
    file_path = os.path.join(folder_name, file_name)
    
    with open(file_path, 'wb') as f:
        segment_count = len(indices_sorted) - 1
        for idx_num in range(segment_count):
            start = indices_sorted[idx_num]
            end = indices_sorted[idx_num+1]
            contour_vertices = vertices[start:end]
            
            # Convert to numpy array and cast to float32 (little-endian)
            data_to_save = np.array(contour_vertices, dtype=np.float32)
            
            # Write header for this segment
            f.write(b'CTRN')  # Magic number (4 bytes)
            f.write(np.uint32(1).tobytes())  # Version 1 (4 bytes, little-endian)
            f.write(np.uint32(len(contour_vertices) // 2).tobytes())  # Num coordinate pairs (4 bytes)
            f.write(b'\x00\x00\x00\x00')  # Reserved (4 bytes)
            
            # Write coordinate data (float32, little-endian)
            f.write(data_to_save.astype(np.float32).tobytes())


def read_contour_binary(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        # Read and validate header
        magic = f.read(4)
        if magic != b'CTRN':
            raise ValueError(f"Invalid binary file: magic number mismatch (expected 'CTRN', got {magic})")
        
        version = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
        if version != 1:
            raise ValueError(f"Unsupported binary format version: {version}")
        
        num_coords = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
        reserved = f.read(4)  # Skip reserved bytes
        
        # Read coordinate data
        data = np.frombuffer(f.read(), dtype=np.float32)
        
        if len(data) != num_coords * 2:
            raise ValueError(f"Data size mismatch: expected {num_coords * 2} floats, got {len(data)}")
    
    # Reshape to coordinate pairs (N, 2) if needed
    return data.reshape(-1, 2) if len(data) > 0 else data


def write_contour_files(level: float, base: str, vertices: list, indices: list, formatted: bool, wcs=None, output_format: str = "text"):
    """Write contour files in specified format (text or binary)."""
    folder_name = base

    print(f"DEBUG: Level {level} | Total vertices in list: {len(vertices)}")
    
    if output_format == "binary":
        write_contour_binary(folder_name, level, os.path.basename(base), vertices, indices)
    elif output_format == "both":
        write_contour_binary(folder_name, level, os.path.basename(base), vertices, indices)
        write_contour_text(folder_name, level, base, vertices, indices, formatted, wcs)
    else:  # text or default
        write_contour_text(folder_name, level, base, vertices, indices, formatted, wcs)


def write_contour_text(folder_name: str, level: float, base: str, vertices: list, indices: list, formatted: bool, wcs=None):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    
    # If no indices, treat entire list as a single contour
    if not indices:
        indices = [0]

    indices_sorted = sorted(indices)
    indices_sorted.append(len(vertices))

    file_name = f"level_{level}.txt"
    file_path = os.path.join(folder_name, file_name)

    with open(file_path, "w") as f:
        segment_count = len(indices_sorted) - 1
        for idx_num in range(segment_count):
            start = indices_sorted[idx_num]
            end = indices_sorted[idx_num+1]
            contour_vertices = vertices[start:end]
            
            if idx_num > 0:
                f.write("---\n")  # Segment separator
            
            if formatted:
                f.write(f"# Contour Level: {level}\n")
                f.write(f"# Segment: {idx_num+1} of {segment_count}\n")
                f.write(f"# Number of vertices: {len(contour_vertices) // 2}\n")
                if wcs is not None:
                    f.write("# Columns: X_pixel, Y_pixel, RA, DEC\n\n")
                    coords = np.array(contour_vertices).reshape(-1, 2)
                    xs = coords[:, 0]
                    ys = coords[:, 1]
                    try:
                        world = wcs.all_pix2world(xs, ys, 0)
                        # world may be (N,2) or tuple; normalize
                        if isinstance(world, tuple) or (isinstance(world, np.ndarray) and world.ndim == 2 and world.shape[1] == 2):
                            if isinstance(world, tuple):
                                lon, lat = world
                            else:
                                lon = world[:, 0]
                                lat = world[:, 1]
                        else:
                            lon = [None] * len(xs)
                            lat = [None] * len(xs)
                    except Exception:
                        lon = [None] * len(xs)
                        lat = [None] * len(xs)

                    for i in range(len(xs)):
                        x, y = xs[i], ys[i]
                        ra = lon[i]
                        dec = lat[i]
                        if ra is None or dec is None:
                            f.write(f"{x:.6f}, {y:.6f}\n")
                        else:
                            f.write(f"{x:.6f}, {y:.6f}, {ra:.6f}, {dec:.6f}\n")
                else:
                    f.write("# X, Y coordinates\n\n")
                    for i in range(0, len(contour_vertices), 2):
                        x, y = contour_vertices[i], contour_vertices[i + 1]
                        f.write(f"{x:.6f}, {y:.6f}\n")
            else:
                for i in range(0, len(contour_vertices), 2):
                    x, y = contour_vertices[i], contour_vertices[i + 1]
                    f.write(f"{x:.6f} {y:.6f}\n")
            f.write(f"\n")

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
    parser.add_argument("--emit-world", action="store_true", help="Include world coordinates (WCS) in formatted output when available.")
    parser.add_argument("--format", choices=["text", "binary", "both"], default="text",
                        help="Output format: text (default), binary (compact .bin files), or both")

    args = parser.parse_args()

    # Parse smoothing arguments
    smoothing_mode = args.smoothing[0]
    smoothing_value = float(args.smoothing[1]) if len(args.smoothing) > 1 else 1.0

    print(f"Reading image: {args.filename}")
    image, metadata = load_image(args.filename)
    wcs = metadata_to_wcs(metadata)

    print(f"Image shape: {image.shape}")
    if wcs is not None and wcs.has_celestial:
        print("WCS detected: using coordinate metadata for consistency.")
    elif metadata:
        print("Header metadata found, but WCS could not be constructed.")

    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    file_ext = os.path.splitext(args.filename)[1].lower().lstrip('.')
    
    # Construct output directory name with smoothing mode
    if file_ext == "fits":
        if smoothing_mode == "none":
            output_dir = f"{base_name}_contours"
        else:
            output_dir = f"{base_name}_{smoothing_mode}_contours"
    else:  # hdf5
        if smoothing_mode == "none":
            output_dir = f"{base_name}_{file_ext}_contours"
        else:
            output_dir = f"{base_name}_{file_ext}_{smoothing_mode}_contours"

    for level in args.levels:
        print(f"Generating contours for levels: {level}")
        vertices, indices = generate_contours(image, smoothing_mode, level)

        write_contour_files(level, output_dir, vertices, indices, args.formatted, wcs if args.emit_world else None, args.format)

    for level in args.levels:
        # write_contour_files(level, base, vertices, indices, args.formatted)
        if args.show:
            show_contours(image, vertices, indices, level, base_name, f"{base_name}_contours")

if __name__ == "__main__":
    main()
