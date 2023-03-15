#!/bin/sh

set -vxe
cd $(dirname "$0")/../../..

test -z "$CC" || export CUDAFLAGS="--compiler-bindir $CC"
cmake -B build -S . $@
cmake --build build --parallel $(nproc)

