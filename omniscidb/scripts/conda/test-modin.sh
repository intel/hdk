#!/bin/sh

set -evx
cmake --install build

pytest -s python/tests/ --ignore=python/tests/modin

(
    cd modin && pip install -e .
)

python -c 'import pyhdk'
python python/tests/modin/modin_smoke_test.py
