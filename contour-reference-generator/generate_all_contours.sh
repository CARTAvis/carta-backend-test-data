#!/bin/bash

# make executable 
# chmod +x generate_all_contours.sh

# run script
# ./generate_all_contours.sh

set -e

PYTHON_CMD="${HOME}/.venvs/carta-contours/bin/python3"
PYTHON_SCRIPT="get_contours.py"

DATA_DIR="../images/fits"

declare -a cases=(
    "$DATA_DIR/sensible-picture-noisey.fits 0 50 100 150 200  --output-dir ../contours --smoothing NoSmoothing --smoothing-factor 4"
    "$DATA_DIR/sensible-picture-noisey.fits 0 50 100 150 200  --output-dir ../contours --smoothing BlockAverage --smoothing-factor 4"
    "$DATA_DIR/sensible-picture-noisey.fits 0 50 100 150 200  --output-dir ../contours --smoothing GaussianBlur --smoothing-factor 4"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 50 100 150 200  --output-dir ../contours --smoothing NoSmoothing --smoothing-factor 4"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 50 100 150 200  --output-dir ../contours --smoothing BlockAverage --smoothing-factor 4"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 50 100 150 200  --output-dir ../contours --smoothing GaussianBlur --smoothing-factor 4"
)

for case in "${cases[@]}"; do
    echo "Executing: $PYTHON_CMD $PYTHON_SCRIPT $case"
    "$PYTHON_CMD" "$PYTHON_SCRIPT" $case
done

echo "All contour cases completed!"