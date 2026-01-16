# Gaussian noise FITS image generator

This is a script for generating FITS and HDF5 images filled with Gaussian noise, suitable for testing. The method for generating images larger than the available memory is taken from [the AstroPy documentation](https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html).

Various additional options are provided, for example for adding different distributions of NAN and INF pixels. Not all of these options are fully compatible with the option to restrict memory usage.


## Requirements

Python 3, numpy, astropy.

`fits2idia` for HDF5 conversion support. See instructions below.

## Installation

This is a single-file script. Copy or symlink it into your path.

### HDF5 Conversion (Optional)

The script can generate HDF5 versions of FITS files using the `--hdf5` flag. This requires the `fits2idia` tool:

- **Ubuntu/Debian**: `sudo apt install fits2idia`
- **macOS (Homebrew)**: `brew install cartavis/tap/fits2idia`
- **Build from source**: https://github.com/CARTAvis/fits2idia
- **AppImage**: Download from https://github.com/CARTAvis/fits2idia/releases

The `fits2idia` tool is a battle-tested C++ converter from the CARTAvis project that produces HDF5 files compatible with CARTA's test suite.

## Usage

Type `make_image.py -h` for a list of options.

To generate both FITS and HDF5 versions of an image:

```bash
make_image.py 500 500 --hdf5
```

This will create `500x500.fits` and `500x500.hdf5` using the official `fits2idia` converter.
