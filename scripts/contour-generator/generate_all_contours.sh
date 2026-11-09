#!/usr/bin/env bash
#
# Generate contour files for each configured FITS image and smoothing mode.
# Edit the images, levels, or smoothing_modes arrays below to change the cases.
# Arguments supplied to this script are passed through to get_contours.py.
#
# Example:
#   ./generate_all_contours.sh \
#     --backend-executable /path/to/carta_backend \
#     --frontend-folder /path/to/frontend/build

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python3}"
GET_CONTOURS="$SCRIPT_DIR/get_contours.py"
OUTPUT_DIR="$REPO_ROOT/contours"

cd "$REPO_ROOT"

images=(
    "$REPO_ROOT/images/fits/500x500_contours.fits"
    "$REPO_ROOT/images/fits/500x500_contours_nans.fits"
)
levels=(0 50 100 150 200)
smoothing_modes=(NoSmoothing BlockAverage GaussianBlur)

for image in "${images[@]}"; do
    if [[ ! -f "$image" ]]; then
        printf 'Input FITS image not found: %s\n' "$image" >&2
        exit 1
    fi
done

for image in "${images[@]}"; do
    for smoothing in "${smoothing_modes[@]}"; do
        command=(
            "$PYTHON_CMD"
            "$GET_CONTOURS"
            "$image"
            "${levels[@]}"
            --output-dir "$OUTPUT_DIR"
            --smoothing "$smoothing"
            --smoothing-factor 4
            "$@"
        )
        printf 'Running:'
        printf ' %q' "${command[@]}"
        printf '\n'
        "${command[@]}"
    done
done

printf 'All contour cases completed.\n'
