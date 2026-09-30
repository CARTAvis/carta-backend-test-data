#!/usr/bin/env python3
"""
Generate contour lines from FITS or HDF5 images.
Outputs .txt or .bin file per contour level and optionally a JPG visualisation.
"""

import argparse
import math
import struct
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
    if ext in [".fits", ".fit"]:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            metadata = extract_metadata_from_header(hdul[0].header)
            header = hdul[0].header
        
            x_offset = header.get('CRPIX1', 0)
            y_offset = header.get('CRPIX2', 0)
            
            print(f"FITS Header - CRPIX1 (X): {x_offset}, CRPIX2 (Y): {y_offset}")
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

    return data, metadata


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
        # Replicating carta::GaussianSmooth and separable 1D RunKernel
        src_height, src_width = image.shape
        smoothing_factor = int(factor)
        apron_height = smoothing_factor - 1
        dest_width = src_width - 2 * apron_height
        dest_height = src_height - 2 * apron_height
        
        if dest_width <= 0 or dest_height <= 0:
            return np.array([[]], dtype=np.float32)

        sigma = (smoothing_factor - 1) / 2.0
        mask_size = (smoothing_factor - 1) * 2 + 1
        kernel_radius = apron_height
        
        # MakeKernel (NormPdf)
        kernel = np.zeros(mask_size, dtype=np.float32)
        for j in range(kernel_radius + 1):
            val = math.exp(-0.5 * (j * j) / (sigma * sigma)) / sigma
            kernel[kernel_radius + j] = val
            kernel[kernel_radius - j] = val

        # Horizontal 1D pass
        temp_buffer = np.zeros((src_height, dest_width), dtype=np.float32)
        for y in range(src_height):
            for x in range(dest_width):
                src_x = x + kernel_radius
                s_sum = 0.0
                w_sum = 0.0
                for i in range(-kernel_radius, kernel_radius + 1):
                    val = image[y, src_x + i]
                    if not math.isnan(val) and not math.isinf(val):
                        w = kernel[i + kernel_radius]
                        s_sum += val * w
                        w_sum += w
                temp_buffer[y, x] = s_sum / w_sum if w_sum > 0.0 else float('nan')

        # Vertical 1D pass + NaN re-injection
        dest_data = np.zeros((dest_height, dest_width), dtype=np.float32)
        for y in range(dest_height):
            src_y = y + kernel_radius
            for x in range(dest_width):
                orig_val = image[src_y, x + kernel_radius]
                if math.isnan(orig_val) or math.isinf(orig_val):
                    dest_data[y, x] = float('nan')
                    continue
                
                s_sum = 0.0
                w_sum = 0.0
                for i in range(-kernel_radius, kernel_radius + 1):
                    val = temp_buffer[src_y + i, x]
                    if not math.isnan(val) and not math.isinf(val):
                        w = kernel[i + kernel_radius]
                        s_sum += val * w
                        w_sum += w
                dest_data[y, x] = s_sum / w_sum if w_sum > 0.0 else float('nan')

        return dest_data

    elif mode == "block":
        # Replicating carta::BlockSmoothScalar
        src_height, src_width = image.shape
        factor_int = int(factor)
        dest_width = math.ceil(src_width / factor_int)
        dest_height = math.ceil(src_height / factor_int)
        
        dest_data = np.zeros((dest_height, dest_width), dtype=np.float32)
        for j in range(dest_height):
            for i in range(dest_width):
                image_row = j * factor_int
                image_col = i * factor_int
                rows_left = min(factor_int, src_height - image_row)
                cols_left = min(factor_int, src_width - image_col)
                
                pixel_sum = 0.0
                pixel_count = 0
                for py in range(rows_left):
                    for px in range(cols_left):
                        pix_val = image[image_row + py, image_col + px]
                        if not math.isnan(pix_val) and not math.isinf(pix_val):
                            pixel_count += 1
                            pixel_sum += pix_val
                            
                dest_data[j, i] = pixel_sum / pixel_count if pixel_count > 0 else float('nan')
        return dest_data
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")

# -------------------------------
# Contour Extraction
# -------------------------------

def is_below(val, level):
    return not math.isnan(val) and val < level

def is_ge(val, level):
    return not math.isnan(val) and val >= level

def trace_segment(image, visited, width, height, scale, offset, level, x_cell, y_cell, side, vertices):
    i = x_cell
    j = y_cell
    orig_side = side
    first_iteration = True
    done = (i < 0 or i >= width - 1 or (j < 0 and j >= height - 1))

    while not done:
        flag = False
        a = image[j * width + i]
        b = image[j * width + i + 1]
        c = image[(j + 1) * width + i + 1]
        d = image[(j + 1) * width + i]

        max_float = 3.4028234663852886e+38 # C++ float max limit
        a = -max_float if math.isnan(a) else a
        b = -max_float if math.isnan(b) else b
        c = -max_float if math.isnan(c) else c
        d = -max_float if math.isnan(d) else d

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
                        x = (level - d) / (d - c) + i
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

            if is_below(pt_a, level) and is_ge(pt_b, level):
                indices.append(len(vertices))
                trace_segment(image, visited, width, height, scale, offset, level, i, j, 0, vertices)
            checked_pixels += 1

    # ---- Right Edge ----
    i = width - 1
    for j in range(height - 1):
        pt_a = image[j * width + i]
        pt_b = image[(j + 1) * width + i]

        if is_below(pt_a, level) and is_ge(pt_b, level):
            indices.append(len(vertices))
            trace_segment(image, visited, width, height, scale, offset, level, i - 1, j, 1, vertices)
        checked_pixels += 1

    # ---- Bottom Edge ----
    j = height - 1
    for i in range(width - 2, -1, -1):
        pt_a = image[j * width + i + 1]
        pt_b = image[j * width + i]

        if is_below(pt_a, level) and is_ge(pt_b, level):
            indices.append(len(vertices))
            trace_segment(image, visited, width, height, scale, offset, level, i, j - 1, 2, vertices)
        checked_pixels += 1

    # ---- Left Edge ----
    i = 0
    for j in range(height - 2, -1, -1):
        pt_a = image[(j + 1) * width + i]
        pt_b = image[j * width + i]

        if is_below(pt_a, level) and is_ge(pt_b, level):
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

def generate_contours(image: np.ndarray, smoothing_mode: str, smoothing_factor: float, level: float):
    scale = smoothing_factor if smoothing_mode == "block" else 1.0
    
    vertex_map, index_map = trace_level(
        image.flatten(),
        image.shape[1],
        image.shape[0],
        scale=scale,
        offset=0.0,
        level=level
    )
    if len(vertex_map) == 0:
        arr = np.array([], dtype=np.float32)
    else:
        arr = np.array(vertex_map, dtype=np.float32).reshape(-1, 2)

    return arr, index_map


# -------------------------------
# Output Writing
# -------------------------------
def _get_binary_dtype(precision: int):
    if precision == 64:
        return np.dtype(np.float64)
    if precision == 32:
        return np.dtype(np.float32)
    if precision == 16:
        return np.dtype(np.float16)
    if precision == 8:
        return np.dtype(np.uint8)
    raise ValueError(f"Unsupported precision {precision}; expected one of: 8, 16, 32, 64")


def write_contour_binary(folder_name: str, level: float, base: str, vertices: list, precision: int):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    file_name = f"level_{str(int(level))}.bin"
    file_path = os.path.join(folder_name, file_name)

    dtype = _get_binary_dtype(precision)
    data_to_save = np.array(vertices, dtype=dtype)
    count = data_to_save.size

    with open(file_path, 'wb') as f:
        f.write(struct.pack('<Q', count))         # number of floats
        # f.write(struct.pack('<I', precision))     # saved precision value
        f.write(data_to_save.tobytes())


def read_contour_binary(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        count_data = f.read(8)
        if len(count_data) < 8:
            raise ValueError("Binary file is too small to contain a vertex count")

        count = struct.unpack('<Q', count_data)[0]

        data = np.frombuffer(f.read(), dtype=_get_binary_dtype(32))

        if len(data) != count:
            raise ValueError(f"Data size mismatch: expected {count} floats, got {len(data)}")

    return data.reshape(-1, 2) if len(data) > 0 else data

# def read_contour_binary(file_path: str) -> np.ndarray:
#     with open(file_path, 'rb') as f:
#         count_data = f.read(8)
#         if len(count_data) < 8:
#             raise ValueError("Binary file is too small to contain a vertex count")

#         count = struct.unpack('<Q', count_data)[0]

#         precision_data = f.read(4)
#         if len(precision_data) < 4:
#             raise ValueError("Binary file is too small to contain a precision value")

#         precision = struct.unpack('<I', precision_data)[0]
#         dtype = _get_binary_dtype(precision)
#         data = np.frombuffer(f.read(), dtype=dtype)

#         if len(data) != count:
#             raise ValueError(f"Data size mismatch: expected {count} floats, got {len(data)}")

#     return data.reshape(-1, 2) if len(data) > 0 else data


def write_contour_files(level: float, base: str, vertices: list, indices: list, formatted: bool, wcs=None, output_format: str = "text"):
    """Write contour files in specified format (text or binary)."""
    folder_name = base

    print(f"DEBUG: Level {level} | Total vertices in list: {len(vertices)}")
    
    if output_format == "binary":
        write_contour_binary(folder_name, level, os.path.basename(base), vertices, 32)
    elif output_format == "both":
        write_contour_binary(folder_name, level, os.path.basename(base), vertices, 32)
        write_contour_text(folder_name, level, os.path.basename(base), vertices)
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

    file_name = f"level_{str(int(level))}.txt"
    file_path = os.path.join(folder_name, file_name)

    with open(file_path, "w") as f:
        segment_count = len(indices_sorted) - 1
        for idx_num in range(segment_count):
            start = indices_sorted[idx_num]
            end = indices_sorted[idx_num+1]
            contour_vertices = np.array(vertices[start:end], dtype=np.float32).reshape(-1, 2)
            
            if idx_num > 0:
                f.write("---\n")  # Segment separator
            
            if formatted:
                f.write(f"# Contour Level: {level}\n")
                f.write(f"# Segment: {idx_num+1} of {segment_count}\n")
                f.write(f"# Number of vertices: {len(contour_vertices)}\n")
                if wcs is not None:
                    f.write("# Columns: X_pixel, Y_pixel, RA, DEC\n\n")
                    xs = contour_vertices[:, 0]
                    ys = contour_vertices[:, 1]
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

                    for x, y, ra, dec in zip(xs, ys, lon, lat):
                        if ra is None or dec is None:
                            f.write(f"{x:.6f}, {y:.6f}\n")
                        else:
                            f.write(f"{x:.6f}, {y:.6f}, {ra:.6f}, {dec:.6f}\n")
                else:
                    f.write("# X, Y coordinates\n\n")
                    for x, y in contour_vertices:
                        f.write(f"{x:.6f}, {y:.6f}\n")
            else:
                for x, y in contour_vertices:
                    f.write(f"{x:.6f} {y:.6f}\n")
            f.write("\n")

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
    parser.add_argument("--percision", default=32,
                            help="percision: <percision> (default: 32)")
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
        vertices, indices = generate_contours(image, smoothing_mode, smoothing_value, level)

        # write_contour_files(level, output_dir, vertices, indices, args.formatted, wcs if args.emit_world else None, args.format)

        write_contour_binary(output_dir, level, os.path.basename(output_dir), vertices, args.percision)

        write_contour_text(output_dir, level, os.path.basename(output_dir), vertices, indices, True, wcs if args.emit_world else None)

    for level in args.levels:
        # write_contour_files(level, base, vertices, indices, args.formatted)
        if args.show:
            show_contours(image, vertices, indices, level, base_name, f"{base_name}_contours")

if __name__ == "__main__":
    main()
