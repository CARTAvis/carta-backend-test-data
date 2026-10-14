import argparse

from image_io import load_fits_image
import numpy as np

import numpy as np
from astropy.io import fits

def save_new_fits_image(output_filename: str, image: np.ndarray, header: fits.Header = None):
    """
    Writes a completely new FITS file from a numpy array and an optional header.
    """
    
    image = image.astype(np.float32)
    
    if header is not None:
        hdu = fits.PrimaryHDU(data=image, header=header)
    else:
        hdu = fits.PrimaryHDU(data=image)
        
    hdu.writeto(output_filename, overwrite=True)
    print(f"Successfully wrote new FITS file to {output_filename}")

def replace_pixels_with_nans(image: np.ndarray, value: float = 0.0):
    image[image == value] = np.nan
    return image

def main():
    parser = argparse.ArgumentParser(description="Replace pixels with nans.")
    parser.add_argument("filename", help="Input image file (.fits or .h5)")
    parser.add_argument("--value", type=float, default=255.0,
                        help="Value to be replaced (default: 0)")

    args = parser.parse_args()

    value = args.value

    print(f"Reading image: {args.filename}")
    image = load_fits_image.load_fits_image(args.filename)

    print("Adding nans...")
    image = replace_pixels_with_nans.replace_pixels_with_nans(image, value)
    
    print(f"Saving image: {args.filename}")
    save_new_fits_image.save_new_fits_image(args.filename+"_new", image)

if __name__ == "__main__":
    main()