# Batch-generate contours for the configured FITS images and smoothing modes.
# Override values on the command line, for example:
#   make PYTHON=/path/to/python BACKEND_EXECUTABLE=/path/to/carta_backend \
#        FRONTEND_FOLDER=/path/to/frontend/build

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MAKEFILE_DIR)/../..)
-include $(MAKEFILE_DIR)config.mk

GET_CONTOURS := $(MAKEFILE_DIR)get_contours.py
OUTPUT_DIR ?= $(REPO_ROOT)/contours

PYTHON ?= $(HOME)/.venvs/carta-contours/bin/python3
# BACKEND_EXECUTABLE ?= /home/michaela/Desktop/carta/carta-backend/build/carta_backend
# FRONTEND_FOLDER ?= /home/michaela/Desktop/carta/carta-frontend/build

IMAGES := \
	$(REPO_ROOT)/images/fits/500x500_contours.fits \
	$(REPO_ROOT)/images/fits/500x500_contours_nans.fits
LEVELS ?= 0 50 100 150 200
SMOOTHING_MODES ?= NoSmoothing BlockAverage GaussianBlur
SMOOTHING_FACTOR ?= 4

.PHONY: all check-inputs

all: check-inputs
	@set -e; \
	for image in $(foreach image,$(IMAGES),"$(image)"); do \
		for smoothing in $(SMOOTHING_MODES); do \
			printf 'Running contours for %s with %s\n' "$$image" "$$smoothing"; \
			"$(PYTHON)" "$(GET_CONTOURS)" "$$image" $(LEVELS) \
				--output-dir "$(OUTPUT_DIR)" \
				--smoothing "$$smoothing" \
				--smoothing-factor "$(SMOOTHING_FACTOR)" \
				--backend-executable "$(BACKEND_EXECUTABLE)" \
				--frontend-folder "$(FRONTEND_FOLDER)"; \
		done; \
	done
	@printf 'All contour cases completed.\n'

check-inputs:
	@for image in $(foreach image,$(IMAGES),"$(image)"); do \
		if [ ! -f "$$image" ]; then \
			printf 'Input FITS image not found: %s\n' "$$image" >&2; \
			exit 1; \
		fi; \
	done
