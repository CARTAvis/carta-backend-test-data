import argparse

from image_io import load_fits_image

import os
import numpy as np
import matplotlib.pyplot as plt
import struct

def read_contour_binary_split(index: int, base: str):
    folder_name = base
    meta_file_path = os.path.join(folder_name, f"{index}/{index}_metadata.bin")
    verts_file_path = os.path.join(folder_name, f"{index}/{index}_data.bin")

    # Read metadata file
    with open(meta_file_path, "rb") as f_meta:
        read_level = struct.unpack('<i', f_meta.read(4))[0]
        num_contours = struct.unpack('<I', f_meta.read(4))[0]
        
        indices = [0]
        for _ in range(num_contours):
            idx_bytes = f_meta.read(8)
            if not idx_bytes:
                break
            idx = struct.unpack('<Q', idx_bytes)[0]
            indices.append(idx)

    # Read vertices file
    vertices = []
    with open(verts_file_path, "rb") as f_verts:
        while True:
            chunk = f_verts.read(8)  # 4 bytes for x (float) + 4 bytes for y (float)
            if not chunk or len(chunk) < 8:
                break
            x, y = struct.unpack('<ff', chunk)
            vertices.extend([x, y])

    return read_level, vertices, indices

def add_contours_to_image(image: np.ndarray, levels: list[int], output_image: str = "contour_multi", folder_name: str = ""):
    fig, ax = plt.subplots()
    image_cmap = plt.get_cmap("gray").copy()
    image_cmap.set_bad("yellow")
    ax.imshow(image, cmap=image_cmap)
    
    cmap = plt.get_cmap('tab10')
    
    for i, level in enumerate(levels):
        _, vertices, indices = read_contour_binary_split(index=i, base=folder_name)

        if not indices or len(indices) == 1:
            indices = [0]
            
        indices_sorted = sorted(indices)
        if indices_sorted[-1] != len(vertices):
            indices_sorted.append(len(vertices))

        invalid_indices = [
            idx for idx in indices_sorted
            if idx < 0 or idx > len(vertices) or idx % 2 != 0
        ]
        if invalid_indices:
            invalid_sample = invalid_indices[:10]
            remaining_count = len(invalid_indices) - len(invalid_sample)
            remaining = f" and {remaining_count} more" if remaining_count else ""
            raise ValueError(
                f"Invalid contour boundaries for level {level}: "
                f"{invalid_sample}{remaining}. "
                f"Boundaries must be even float offsets between 0 and {len(vertices)}. "
                "Regenerate the contour files."
            )

        color = cmap(i % 10)
        
        for idx_num in range(len(indices_sorted) - 1):
            start = indices_sorted[idx_num]
            end = indices_sorted[idx_num + 1]
            seg = np.array(vertices[start:end]).reshape(-1, 2)
            if len(seg) > 0:
                label_name = f'Level {level}' if idx_num == 0 else ""
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    linestyle="-",
                    marker=None,
                    linewidth=0.5,
                    color=color,
                    label=label_name,
                )
                
    ax.axis('off')
    ax.legend(loc='upper left')

    levels_str = "_".join(map(str, levels))
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, f"{output_image}_levels_{levels_str}.jpg")
    else:
        save_path = f"{output_image}_levels_{levels_str}.jpg"

    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualise contours on original image.")
    parser.add_argument("filename", help="Input image file (.fits or .h5)")
    parser.add_argument("folder", help="Folder where the saved data is")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 50, 100, 150, 200], help="The number of levels")

    args = parser.parse_args()

    print(f"Reading image: {args.filename}")
    image = load_fits_image.load_fits_image(args.filename)
    
    print(f"Saving image contours onto image: {args.filename}")
    add_contours_to_image(image, args.levels, "with_contours", args.folder)

if __name__ == "__main__":
    main()