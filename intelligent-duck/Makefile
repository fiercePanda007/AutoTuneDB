PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=autoDB
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# DuckDB's linux_amd64_gcc4 platform expects the old libstdc++ ABI.
# Force the extension build to match, otherwise LOAD can fail at runtime.
ifeq ($(DUCKDB_PLATFORM),linux_amd64_gcc4)
EXT_FLAGS += -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
