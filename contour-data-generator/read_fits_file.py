import os
import numpy as np
from astropy.io import fits

def print_fits_contents(filename: str):
    """Inspects and prints the structure, header, and data summary of a FITS file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".fits", ".fit"]:
        raise ValueError(f"Unsupported file format for FITS inspection: {ext}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with fits.open(filename) as hdul:
        print("=== FITS HDU Structure ===")
        hdul.info()
        
        print("\n=== Primary HDU Header ===")
        header = hdul[0].header
        for card in header.cards:
            print(f"{card.keyword:8} = {str(card.value):<20} / {card.comment}")
            
        print("\n=== Data Summary ===")
        data = hdul[0].data
        if data is not None:
            print(f"Shape : {data.shape}")
            print(f"Type  : {data.dtype}")
            print(f"Min   : {np.nanmin(data)}")
            print(f"Max   : {np.nanmax(data)}")
            print(f"Mean  : {np.nanmean(data)}")
            
            # Show a slice of the data array
            if data.ndim >= 2:
                print("\nPreview (Top-left 5x5 pixels):")
                print(data[:5, :5])
            else:
                print("\nPreview (First 10 elements):")
                print(data[:10])
        else:
            print("No data array found in the primary HDU.")

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    print_fits_contents(file_path)