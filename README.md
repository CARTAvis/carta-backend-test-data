# carta-backend-test-data

This repository contains sample data and utility scripts for testing CARTA
backend functionality. It includes both ready-to-use test files and tools for
generating or inspecting test data.

## Repository contents

- `images/` — image test data in formats such as FITS, HDF5, CASA, and Miriad,
  including examples with NaN values and different image dimensions.
- `contours/` — generated contour output for sample images and smoothing modes.
- `tables/` — table test data, including FITS and XML examples.
- `tiles/` — tile-related test data and a tile data generator.
- `settings/` — JSON examples for testing settings handling.
- `scripts/` — utilities for generating images and contours, inspecting contour
  output, and visualising contours.

The scripts are optional tools; the repository also contains test data that can
be used directly by backend tests.
