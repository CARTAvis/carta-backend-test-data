# Contour generator

These scripts request contour data from the CARTA backend, save it in binary
files, and provide utilities to inspect or visualize the results.

## Requirements

- A built CARTA backend and frontend.
- Python dependencies used by `get_contours.py`, including `cartaproto`.
- A Python environment containing those dependencies.

## Configure machine-specific paths

The Makefile loads `config.mk` when it is present. To configure paths for your
system, copy `config.mk.example` to `config.mk` and edit the Python interpreter,
backend executable, and frontend build directory:

```bash
cp config.mk.example config.mk
```

`config.mk` is ignored by Git so local paths are not committed. Values can also
be overridden on the `make` command line, for example
`make BACKEND_EXECUTABLE=/path/to/carta_backend`.

## Generate contours

Run one request with `get_contours.py`:

```bash
python3 get_contours.py /path/to/image.fits 0 50 100 \
  --backend-executable /path/to/carta_backend \
  --frontend-folder /path/to/frontend/build \
  --output-dir /path/to/output \
  --smoothing GaussianBlur \
  --smoothing-factor 4
```

For the configured batch of two sample images, five contour levels, and three
smoothing modes, run either:

```bash
make BACKEND_EXECUTABLE=/path/to/carta_backend \
     FRONTEND_FOLDER=/path/to/frontend/build
```

or:

```bash
./generate_all_contours.sh \
  --backend-executable /path/to/carta_backend \
  --frontend-folder /path/to/frontend/build
```

The batch inputs, levels, and smoothing modes are defined in the Makefile or
shell script. The Makefile also accepts overrides such as `PYTHON`, `OUTPUT_DIR`,
`LEVELS`, and `SMOOTHING_FACTOR`.

## Output layout

Each result is placed in a directory named from the image stem and smoothing
mode, for example `500x500_contours_GaussianBlur/`. It contains top-level
metadata and numbered subdirectories with per-message contour metadata and
coordinate data. The contour values are stored in the metadata.

## Inspect and visualize

- `inspect_contours.py` prints the contents of a generated contour directory.
- `visualise_contours.py` overlays several contour levels on an image.
- `visualise_contour_smoothing.py` compares two contour outputs for the same
  contour level.
- `read_fits_file.py` summarizes a FITS file, and `add_nans_to_image.py` can
  replace a selected pixel value with NaNs.
