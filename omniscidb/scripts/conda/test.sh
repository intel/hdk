#!/bin/bash

usage() { echo "Usage: $0 [-s (skip large buffers)]" 1>&2; exit 1; }

TEST_FLAGS=''

while getopts "s" o; do
    case "${o}" in
        s)
            TEST_FLAGS='SKIP_LARGE_BUFFERS=true'
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

set -ex

export ${TEST_FLAGS}

cd $(dirname "$0")

# Omnisci UDF support uses CLangTool for parsing Load-time UDF C++
# code to AST. If the C++ code uses C++ std headers, we need to
# specify the locations of include directories
export CXX=g++
. ./get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=$(get_cxx_include_path)

cd ../../../build

make sanity_tests

