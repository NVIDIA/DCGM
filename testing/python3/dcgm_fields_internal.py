# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##
# Python bindings for the internal API of DCGM library (dcgm_fields_internal.hpp)
##

from ctypes import *
from ctypes.util import find_library
import dcgm_structs
import dcgm_fields

# Provides access to functions
dcgmFP = dcgm_structs._dcgmGetFunctionPointer


# internal-only fields
DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES = 210  # Memory utilization samples
DCGM_FI_DEV_GPU_UTIL_SAMPLES = 211  # SM utilization samples
DCGM_FI_DEV_GRAPHICS_PIDS = 220  # Graphics processes running on the GPU.
DCGM_FI_DEV_COMPUTE_PIDS = 221  # Compute processes running on the GPU.

DCGM_FI_SYSMON_FIRST_ID = dcgm_fields.DCGM_FI_DEV_CPU_UTIL_TOTAL
DCGM_FI_SYSMON_LAST_ID = dcgm_fields.DCGM_FI_DEV_CPU_MODEL

DCGM_FI_PROF_FIRST_ID = dcgm_fields.DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO
DCGM_FI_PROF_LAST_ID = dcgm_fields.DCGM_FI_PROF_FP16_CYCLES_ACTIVE_TOTAL

# PCIe throughput field range (returns MiB/s from NVML GPM).
DCGM_FI_PROF_PCIE_THROUGHPUT_FIRST = dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES
DCGM_FI_PROF_PCIE_THROUGHPUT_LAST = dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES

# NVLink aggregate throughput field range (returns MiB/s from NVML GPM).
DCGM_FI_PROF_NVLINK_AGG_THROUGHPUT_FIRST = dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES
DCGM_FI_PROF_NVLINK_AGG_THROUGHPUT_LAST = dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES

# Legacy per-link NVLink throughput field range L0-L17 (returns MiB/s from NVML GPM).
DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_FIRST = dcgm_fields.DCGM_FI_PROF_NVLINK_L0_TX_BYTES
DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_LAST = dcgm_fields.DCGM_FI_PROF_NVLINK_L17_RX_BYTES
# Per-link NVLink throughput fields keyed by dcgm_link_t (returns MiB/s from NVML GPM).
DCGM_FI_PROF_NVLINK_THROUGHPUT_PER_LINK_FIRST = dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK
DCGM_FI_PROF_NVLINK_THROUGHPUT_PER_LINK_LAST = dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK

# C2C throughput field range (returns MiB/s from NVML GPM).
DCGM_FI_PROF_C2C_THROUGHPUT_FIRST = dcgm_fields.DCGM_FI_PROF_C2C_TX_ALL_BYTES
DCGM_FI_PROF_C2C_THROUGHPUT_LAST = dcgm_fields.DCGM_FI_PROF_C2C_RX_DATA_BYTES
