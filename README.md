# Heterogeneous Data Kernels

A low-level execution library for analytic data processing. 

## Clone

Cloning a project with submodules. Either use `git clone --recurse-submodules` to clone the repo, or clone as normal and then run:

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

## Windows Build

This guide assumes Windows C++ build tools have been installed. Ensure the MSVC spectre-mitigated libraries are installed as well. See https://code.visualstudio.com/docs/cpp/config-msvc for more information. 

1. Install dependencies using conda:
```
conda env create -f omniscidb/scripts/mapd-deps-conda-windows-env.yml
```
2. Enter build dir.
```
cd build
```
3. Initialize cmake. 
```
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_CONDA=on -DBUILD_SHARED_LIBS=off -G "Visual Studio 17 2022"  -A x64 ..
```
4. Build.
```
cmake --build . --config Debug
```