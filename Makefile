# Batch-generate contours for the configured FITS images and smoothing modes.
# Override values on the command line, for example:
#   make PYTHON=/path/to/python BACKEND_EXECUTABLE=/path/to/carta_backend \
#        FRONTEND_FOLDER=/path/to/frontend/build
# Or use a config.mk to specify command line values

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MAKEFILE_DIR))
-include $(MAKEFILE_DIR)config.mk

GET_CONTOURS := $(MAKEFILE_DIR)contours/scripts/contour-generator/get_contours.py
OUTPUT_DIR ?= $(REPO_ROOT)/contours
INPUT_DIR ?= $(REPO_ROOT)/images/fits

PYTHON ?= $(HOME)/.venvs/carta-contours/bin/python3

IMAGES := \
	500x500_contours \
	500x500_contours_nans
LEVELS ?= 0 50 100 150 200
SMOOTHING_MODES ?= NoSmoothing BlockAverage GaussianBlur
SMOOTHING_FACTOR ?= 4

define rule_with_smoothing
$$(OUTPUT_DIR)/%_$(1): $$(INPUT_DIR)/%.fits
	"$$(PYTHON)" "$$(GET_CONTOURS)" "$$<" $$(LEVELS) \
		--output-dir "$$(OUTPUT_DIR)" \
		--smoothing "$(1)" \
		--smoothing-factor "$$(SMOOTHING_FACTOR)" \
		--backend-executable "$$(BACKEND_EXECUTABLE)" \
		--frontend-folder "$$(FRONTEND_FOLDER)"
endef
$(foreach sm,$(SMOOTHING_MODES),$(eval $(call rule_with_smoothing,$(sm))))

.PHONY: all-contours

all-contours: $(foreach image,$(IMAGES),$(foreach sm,$(SMOOTHING_MODES),$(OUTPUT_DIR)/$(image)_$(sm)))
