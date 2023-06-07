#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0


from pytest import fixture
from dataclasses import dataclass


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="CPU", help="Choose device type (CPU/GPU)."
    )


@dataclass
class ExecutionConfig:
    enable_heterogeneous: bool
    device_type: str


cpu_cfg = ExecutionConfig(False, "CPU")
gpu_cfg = ExecutionConfig(False, "GPU")
het_cfg = ExecutionConfig(True, "AUTO")


def get_execution_config(device_type: str):
    if device_type.lower() == "cpu":
        return cpu_cfg
    if device_type.lower() == "gpu":
        return gpu_cfg
    raise ValueError("Unsupported exeuction config: " + str(device_type))


@fixture
def exe_cfg(request):
    return get_execution_config(request.config.getoption("--device"))
