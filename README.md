# carta-backend-test-data

## Overview

This repository contains generated test files used for validating and benchmarking the [carta-backend](https://github.com/CARTAvis/carta-backend). 
The files are stored using Git Large File Storage (Git LFS) to efficiently manage large files.

An optional FITS → HDF5 conversion step is also supported.

##Regenerating Test Data

`bash scripts/remake_files.sh`

## Generating Test Files with `make_image.py` in `image-generator`
This script generates synthetic FITS files with optional NaNs, Infs, noise, patterns, and Gaussian models.

### Basic Usage
`./make_image.py 256 256 -o myimage.fits`
Creates a 256x256 image with default values.

### Positional Arguments
| Argument | Description |
|---|---|
| dimensions | Required. 2–4 integers: width height [depth] [stokes] |

### Optional Flags and Parameters
#### General Options
| Flag | Type | Description |
|---|---|---|
| -o, --output | string | Output FITS file name. Default: image-[dims].fits |
| -m, --max-bytes | int | Max size (bytes) to hold in memory at once. |
| -s, --seed | int | Seed for random number generator. |
| -H, --header | string | Additional FITS header entries (newline-separated string). |

### NaN Insertion
| Flag | Type | Description |
|---|---|---|
| -n, --nans | list of strings | Where to insert NaNs. Options: pixel, row, column, channel, stokes, image |
| -d, --nan-density | float (%) | Percentage of values to replace with NaNs (default: 25.0) |

> 💡 "image" overrides all other --nans and fills the image entirely with NaNs.

### Infinity Insertion
| Flag | Type | Description |
|---|---|---|
| -i, --infs | list of strings | Insert infinities. Options: positive, negative |
| --inf-density | float (%) | Percentage of values to replace with +inf or -inf (default: 1.0) |

### Image Patterns
| Flag | Type | Description |
|---|---|---|
| -c, --checkerboard | int | Create a checkerboard pattern. Value = size of each square (px). |
| --gaussian-model | float list | Define Gaussian blobs. Format: n x y amp fwhm_x fwhm_y pa_deg [repeat] |

### Examples
Basic Gaussian noise (2D), makes a futs and hdf5 file:

`./make_image.py 256 256 --hdf5`

NaNs randomly in pixels:

`./make_image.py 256 256 -n pixel -d 10`

Checkerboard pattern with NaNs:

`./make_image.py 256 256 -c 8 -n pixel`

Gaussian model added:

`./make_image.py 256 256 --gaussian-model 1 128 128 1.0 30 30 0`

Full 4D cube with Infs:

`./make_image.py 64 64 8 4 -i positive -d 5`

A synthetic FITS image with 10% of rows and 10% of columns randomly set to NaN values, using a fixed random seed (0) to ensure the output is reproducible:

`./make_image.py 256 256 -s 0 -n row column -d 10`