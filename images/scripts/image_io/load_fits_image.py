import os

import h5py
import numpy as np
from astropy.io import fits


def _first_dataset(group):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            return item[()]
        if isinstance(item, h5py.Group):
            result = _first_dataset(item)
            if result is not None:
                return result
    return None


def load_fits_image(filename: str) -> np.ndarray:
    """Load a FITS or HDF5 image as float32 data."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".fits", ".fit"):
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

            x_offset = header.get("CRPIX1", 0)
            y_offset = header.get("CRPIX2", 0)

            print(f"FITS Header - CRPIX1 (X): {x_offset}, CRPIX2 (Y): {y_offset}")
    elif ext in (".h5", ".hdf5"):
        with h5py.File(filename, "r") as hdf_file:
            data = _first_dataset(hdf_file)
            if data is None:
                raise ValueError("No dataset found in HDF5 file.")
            data = np.array(data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure 2D
    if data.ndim > 2:
        data = data[0]

    return data
