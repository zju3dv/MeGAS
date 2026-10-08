#!/bin/bash
# Tested on Ubuntu with torch 2.0 + CUDA 12.4, NVIDIA RTX 4070 Ti (12 GB).
# Install PyTorch matching your CUDA version before running this script.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt

# CUDA extensions
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn

# OptiX Gaussian ray tracer: the PTX files are built with CMake before the Python package
cd submodules/gstracer
rm -rf build && mkdir build && cd build && cmake .. && make && cd ..
pip install --no-build-isolation .
