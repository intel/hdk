#!/bin/sh

find . \
    -name python -prune -or \
    -name build -prune -or \
    -name ThirdParty -prune -or \
    -name third_party -prune -or \
    \( -name '*.cpp' -or -name '*.h' \) -print | \
    xargs clang-format --dry-run --Werror -style=file

