#!/bin/bash

set -e

# Path to the generator script
GEN=./image-generator/make_image.py
OUTDIR=images/fits

mkdir -p "$OUTDIR"

# 128x128 Gaussian model, one component
$GEN 128 128 --gaussian-model 1 64 64 1.0 30 30 0 -o $OUTDIR/128_128_gaussian_model_one_component.fits

# 128x128 Gaussian model, three components
$GEN 128 128 --gaussian-model 3 32 32 1.0 20 20 0 96 32 0.8 20 20 0 64 96 0.5 30 30 0 -o $OUTDIR/128_128_gaussian_model_three_components.fits

# 500x500 with default options (Gaussian noise)
$GEN 500 500 -o $OUTDIR/500_500_image_opts.fits

# 500x500 with NaNs (default density)
$GEN 500 500 -n pixel -o $OUTDIR/500_500_image_opts_nan.fits

# 64x64x8x4 with positive infinities (example from README, adjust density as needed)
$GEN 64 64 8 4 -i positive -o $OUTDIR/noise_4d.fits

# 64x64x8x4 with positive infinities, CASA variant (if needed, otherwise same as above)
$GEN 64 64 8 4 -i positive -o $OUTDIR/noise_4d_casa.fits

# 64x64x8 with default options (Gaussian noise)
$GEN 64 64 8 -o $OUTDIR/noise_3d.fits

# 64x64x8 with default options, CASA variant
$GEN 64 64 8 -o $OUTDIR/noise_3d_degen_casa.fits

# 64x64x8 with default options, non-CASA variant
$GEN 64 64 8 -o $OUTDIR/noise_3d_degen.fits

# 10x10 checkerboard pattern
$GEN 10 10 -c 10 -o $OUTDIR/noise_10px_10px.fits

# Small per-plane beam (example, adjust as needed)
$GEN 10 10 2 1 -o $OUTDIR/small_perplanebeam.fits

# M17_SWex_unittest.fits is likely real data and not reproducible with make_image.py

echo "Synthetic FITS files regenerated in $OUTDIR"