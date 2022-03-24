# Heterogeneous Data Kernels

A low-level execution library for analytic data processing.

## Clone

Cloning a project with submodules. Either use `git clone --recruse-submodules` to clone the repo, or clone as normal and
then run:

```
git submodule init
git submodule update
```

## Build

```
mkdir build
cd build
cmake ..
make -j
```

## MLIR

Install notes:

* Set `LLVM_ENABLE_RTTI=ON`
