# SPDX-License-Identifier: Apache-2.0

.PHONY: clean clean_all

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXTENSION_NAME=bolt

# Mark the extension as C_STRUCT_UNSTABLE: DuckDB 2.x requires the _init_c_api_v2 entrypoint.
# Binaries load only in the exact TARGET_DUCKDB_VERSION, not in newer DuckDB releases.
USE_UNSTABLE_C_API=1
CAPI_VERSION=2

# Target DuckDB version
TARGET_DUCKDB_VERSION=v2.0.0

all: configure debug

# Include makefiles from DuckDB
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/rust.Makefile

# Auto-enable cuda-stub when neither CUDA_PATH nor /usr/local/cuda is present,
# unless BOLT_CUDA_STUB=0 is explicitly passed. (craton-bolt's build.rs also only
# consults CUDA_PATH and the standard install locations, not CUDA_HOME.)
ifeq ($(BOLT_CUDA_STUB),)
ifeq ($(wildcard /usr/local/cuda)$(CUDA_PATH),)
BOLT_CUDA_STUB=1
$(warning No CUDA toolkit found: CUDA_PATH is unset and /usr/local/cuda does not exist. Building with the cuda-stub feature - the resulting bolt.duckdb_extension is CPU-only and always uses the host-planner-fallback backend. Set CUDA_PATH or pass BOLT_CUDA_STUB=0 to force a GPU build.)
endif
endif

# Build workspace libraries rather than the CI tools' Wasm example target.
IS_EXAMPLE=
ifneq ($(DUCKDB_WASM_PLATFORM),)
	TARGET_INFO=--target $(TARGET) --package duckdb-bolt-wasm
else
	TARGET_INFO+=--package duckdb-bolt-extension
ifneq ($(filter-out 0,$(BOLT_CUDA_STUB)),)
	TARGET_INFO+=--features cuda-stub
endif
endif

# Wasm link step (W1-E-002). The pinned extension-ci-tools base.Makefile links the side module with
# -sEXPORTED_FUNCTIONS="_$(EXTENSION_NAME)_init_c_api" (the C API v1 entrypoint), but
# #[duckdb_entrypoint_c_api_v2] emits `<name>_init_c_api_v2`, which is what DuckDB 2.x looks up.
# Override both recipes (GNU make keeps the last recipe and prints an "overriding recipe" warning)
# and make the link depend on the library build so `make -j` cannot link a missing or stale
# libbolt.a. Remove this block once extension-ci-tools exports the v2 entrypoint itself.
ifneq ($(DUCKDB_WASM_PLATFORM),)
WASM_EXPORTED_FUNCTIONS=-sEXPORTED_FUNCTIONS="_$(EXTENSION_NAME)_init_c_api_v2"

link_wasm_debug: build_extension_library_debug
	emcc $(EXTENSION_BUILD_PATH)/debug/$(EXTENSION_LIB_FILENAME) -o $(EXTENSION_BUILD_PATH)/debug/$(EXTENSION_FILENAME_NO_METADATA) -O3 -g -sSIDE_MODULE=2 $(WASM_EXPORTED_FUNCTIONS)

link_wasm_release: build_extension_library_release
	emcc $(EXTENSION_BUILD_PATH)/release/$(EXTENSION_LIB_FILENAME) -o $(EXTENSION_BUILD_PATH)/release/$(EXTENSION_FILENAME_NO_METADATA) -O3 -sSIDE_MODULE=2 $(WASM_EXPORTED_FUNCTIONS)
endif

configure: venv platform extension_version

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_debug
test_debug: test_extension_debug
test_release: test_extension_release

clean: clean_build clean_rust
clean_all: clean_configure clean
