import numpy as np
import h5py
from astropy.io import fits
import os

def load_fits_image(filename: str) -> np.ndarray:
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".fits", ".fit"]:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
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
    # return np.nan_to_num(data)
    return data
