#!/bin/bash

# make executable 
# chmod +x generate_all_contours.sh

# run script
# ./generate_all_contours.sh

PYTHON_CMD="./.venv/bin/python3"
PYTHON_SCRIPT="contour_generator.py"

DATA_DIR="./contour-images"

declare -a cases=(
    "$DATA_DIR/sensible-picture-noisey.fits --levels 50 100 150 200 250 300 350 400 450 --smoothing none"
    # "$DATA_DIR/10x10.fits --levels -1 0 1 --smoothing none"
    # "$DATA_DIR/10x10.fits --levels -1 0 1 --smoothing block 4.0"
    # "$DATA_DIR/10x10.fits --levels -1 0 1 --smoothing gaussian 1.5"
    # "$DATA_DIR/500x500.fits --levels -1 0 1 --smoothing none"
    # "$DATA_DIR/500x500.fits --levels -1 0 1 --smoothing block 4.0"
    # "$DATA_DIR/500x500.fits --levels -1 0 1 --smoothing gaussian 1.5"
    # "$DATA_DIR/500x500_nans.fits --levels -1 0 1 --smoothing none"
    # "$DATA_DIR/500x500_nans.fits --levels -1 0 1 --smoothing block 4.0"
    # "$DATA_DIR/500x500_nans.fits --levels -1 0 1 --smoothing gaussian 1.5"
)

for case in "${cases[@]}"; do
    echo "Executing: $PYTHON_CMD $PYTHON_SCRIPT $case"
    "$PYTHON_CMD" "$PYTHON_SCRIPT" $case
done

echo "All contour cases completed!"