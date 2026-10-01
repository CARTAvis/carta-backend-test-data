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
    # "$DATA_DIR/sensible-picture-noisey.fits 0 500 0 500 --smoothing none"
    # "$DATA_DIR/sensible-picture-noisey.fits 0 500 0 500 --smoothing block"
    # "$DATA_DIR/sensible-picture-noisey.fits 0 500 0 500 --smoothing gaussian"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 500 0 500 --smoothing none"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 500 0 500 --smoothing block"
    "$DATA_DIR/sensible-picture-noisey-nans.fits 0 500 0 500 --smoothing gaussian"
)

for case in "${cases[@]}"; do
    echo "Executing: $PYTHON_CMD $PYTHON_SCRIPT $case"
    "$PYTHON_CMD" "$PYTHON_SCRIPT" $case
done

echo "All contour cases completed!"