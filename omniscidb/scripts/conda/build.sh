#!/bin/sh

set -vxe
cd $(dirname "$0")/../../..

test -z "$CC" || export CUDAFLAGS="--compiler-bindir $CC"

nvcc=$(ls -1 /usr/local/cuda-*/bin/nvcc | head -1)
export PATH=$(dirname "$nvcc"):$PATH

cmake -B build -S . $@
cmake --build build --parallel $(nproc)

