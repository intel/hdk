#!/usr/bin/env bash

set -ex

test -z "$CC" || export CUDAFLAGS="--compiler-bindir $CC"

cmake -Wno-dev \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_PREFIX=$PREFIX/$INSTALL_BASE \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_AWS_S3=off -B build -S .

cmake --build build --parallel $(nproc)
cmake --install build --prefix $PREFIX/$INSTALL_BASE

