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
# Python bindings for the internal API of DCGM library (dcgm_fields.h)
##

from ctypes import *
from ctypes.util import find_library
import dcgm_structs

# Provides access to functions
dcgmFP = dcgm_structs._dcgmGetFunctionPointer

# Field Types are a single byte. List these in ASCII order
DCGM_FT_BINARY = 'b'  # Blob of binary data representing a structure
DCGM_FT_DOUBLE = 'd'  # 8-byte double precision
DCGM_FT_INT64 = 'i'  # 8-byte signed integer
DCGM_FT_STRING = 's'  # Null-terminated ASCII Character string
DCGM_FT_TIMESTAMP = 't'  # 8-byte signed integer usec since 1970

# Field scope. What are these fields associated with
DCGM_FS_GLOBAL = 0              # Field is global (ex: driver version)
# Field is associated with an entity (GPU, VGPU, ..etc)
DCGM_FS_ENTITY = 1
# Field is associated with a device. Deprecated. Use DCGM_FS_ENTITY
DCGM_FS_DEVICE = DCGM_FS_ENTITY

# DCGM_FI_DEV_CLOCKS_EVENT_REASONS is a bitmap of clock events
# These macros are masks for relevant clock events, and are a 1:1 map to the NVML
# reasons documented in nvml.h. The notes for the header are copied blow:

# Nothing is running on the GPU and the clocks are dropping to Idle state
DCGM_CLOCKS_EVENT_REASON_GPU_IDLE = 0x0000000000000001

# GPU clocks are limited by current setting of applications clocks
DCGM_CLOCKS_EVENT_REASON_CLOCKS_SETTING = 0x0000000000000002

# SW Power Scaling algorithm is reducing the clocks below requested clocks
DCGM_CLOCKS_EVENT_REASON_SW_POWER_CAP = 0x0000000000000004

# HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - temperature being too high
#  - External Power Brake Assertion is triggered (e.g. by the system power supply)
#  - Power draw is too high and Fast Trigger protection is reducing the clocks
#  - May be also reported during PState or clock change
#  - This behavior may be removed in a later release.

DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN = 0x0000000000000008

# Sync Boost
#
# This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
# order to maximize performance per watt. All GPUs in the sync boost group
# will boost to the minimum possible clocks across the entire group. Look at
# the throttle reasons for other GPUs in the system to see why those GPUs are
# holding this one at lower clocks.
DCGM_CLOCKS_EVENT_REASON_SYNC_BOOST = 0x0000000000000010

# SW Thermal Slowdown
#
# This is an indicator of one or more of the following:
#  - Current GPU temperature above the GPU Max Operating Temperature
#  - Current memory temperature above the Memory Max Operating Temperature
DCGM_CLOCKS_EVENT_REASON_SW_THERMAL = 0x0000000000000020

# HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - temperature being too high
DCGM_CLOCKS_EVENT_REASON_HW_THERMAL = 0x0000000000000040

# HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
#
# This is an indicator of:
#  - External Power Brake Assertion being triggered (e.g. by the system power supply)
DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE = 0x0000000000000080

# GPU clocks are limited by current setting of Display clocks
DCGM_CLOCKS_EVENT_REASON_DISPLAY_CLOCKS = 0x0000000000000100

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_GPU_IDLE instead
DCGM_CLOCKS_THROTTLE_REASON_GPU_IDLE = DCGM_CLOCKS_EVENT_REASON_GPU_IDLE

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_CLOCKS_SETTING instead
DCGM_CLOCKS_THROTTLE_REASON_CLOCKS_SETTING = DCGM_CLOCKS_EVENT_REASON_CLOCKS_SETTING

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_SW_POWER_CAP instead
DCGM_CLOCKS_THROTTLE_REASON_SW_POWER_CAP = DCGM_CLOCKS_EVENT_REASON_SW_POWER_CAP

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN instead
DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_SYNC_BOOST instead
DCGM_CLOCKS_THROTTLE_REASON_SYNC_BOOST = DCGM_CLOCKS_EVENT_REASON_SYNC_BOOST

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_SW_THERMAL instead
DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL = DCGM_CLOCKS_EVENT_REASON_SW_THERMAL

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_HW_THERMAL instead
DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL = DCGM_CLOCKS_EVENT_REASON_HW_THERMAL

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE instead
DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE = DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE

# Deprecated: Use DCGM_CLOCKS_EVENT_REASON_DISPLAY_CLOCKS instead
DCGM_CLOCKS_THROTTLE_REASON_DISPLAY_CLOCKS = DCGM_CLOCKS_EVENT_REASON_DISPLAY_CLOCKS

# Field entity groups. Which type of entity is this field or field value associated with
# Field is not associated with an entity. Field scope should be DCGM_FS_GLOBAL
DCGM_FE_NONE = 0
DCGM_FE_GPU = 1  # Field is associated with a GPU entity
DCGM_FE_VGPU = 2  # Field is associated with a VGPU entity
DCGM_FE_SWITCH = 3  # Field is associated with a Switch entity
DCGM_FE_GPU_I = 4  # Field is associated with a GPU Instance entity
DCGM_FE_GPU_CI = 5  # Field is associated with a GPU Compute Instance entity
DCGM_FE_LINK = 6  # Field is associated with an NVLINK
DCGM_FE_CPU = 7  # Field is associated with a CPU
DCGM_FE_CPU_CORE = 8  # Field is associated with a CPU core
DCGM_FE_CONNECTX = 9  # Field is associated with a ConnectX card

# Represents an identifier for an entity within a field entity. For
# instance, this is the gpuId for DCGM_FE_GPU.
c_dcgm_field_eid_t = c_uint32

# System attributes
DCGM_FI_SYSTEM_FIELD_UNKNOWN = 0
DCGM_FI_SYSTEM_DRIVER_VERSION = 1  # Driver Version
DCGM_FI_SYSTEM_NVML_VERSION = 2  # Underlying NVML version
# Process Name. Will be nv-hostengine or your process's name in embedded mode
DCGM_FI_SYSTEM_PROCESS_NAME = 3
DCGM_FI_SYSTEM_GPU_QUANTITY = 4  # Number of Devices on the node
# Cuda Driver Version as an integer. CUDA 11.1 = 11100
DCGM_FI_CUDA_DRIVER_VERSION = 5
# GPU bind/unbind event notification. Values: SystemReinitializing=1,
# SystemReinitializationCompleted=2, recommended watch frequency: 1 second
DCGM_FI_SYSTEM_GPU_BIND_EVENT = 6

# Device attributes
DCGM_FI_DEV_GPU_NAME = 50  # Name of the GPU device
DCGM_FI_DEV_GPU_BRAND = 51  # Device Brand
DCGM_FI_DEV_NVML_INDEX = 52  # NVML index of this GPU
DCGM_FI_DEV_BOARD_SERIAL = 53  # Device Serial Number
DCGM_FI_DEV_GPU_UUID = 54  # UUID corresponding to the device
DCGM_FI_DEV_GPU_MINOR_NUMBER = 55  # Device node minor number /dev/nvidia#
DCGM_FI_DEV_INFOROM_OEM_VERSION = 56  # OEM inforom version
DCGM_FI_DEV_PCI_BUS_ID = 57  # PCI attributes for the device
# The combined 16-bit device id and 16-bit vendor id
DCGM_FI_DEV_PCI_COMBINED_ID = 58
DCGM_FI_DEV_PCI_SUBSYS_ID = 59  # The 32-bit Sub System Device ID
# Topology of all GPUs on the system via PCI (static)
DCGM_FI_SYSTEM_PCI_TOPOLOGY = 60
# Topology of all GPUs on the system via NVLINK (static)
DCGM_FI_SYSTEM_NVLINK_TOPOLOGY = 61
# Affinity of all GPUs on the system (static)
DCGM_FI_SYSTEM_GPU_AFFINITY = 62
# Cuda compute capability for the device
DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY = 63
# A bitmap of the P2P NVLINK status from this GPU to others on this host.
DCGM_FI_DEV_NVLINK_P2P_STATUS = 64
DCGM_FI_DEV_GPU_COMPUTE_MODE = 65  # Compute mode for the device
DCGM_FI_DEV_GPU_PERSISTENCE_MODE = 66  # Persistence mode for the device
DCGM_FI_DEV_MIG_MODE = 67  # MIG mode for the device
# String value for CUDA_VISIBLE_DEVICES for the device
DCGM_FI_CUDA_GPU_VISIBLE_DEVICES = 68
DCGM_FI_DEV_MIG_MAX_SLICES = 69  # The maximum number of slices this GPU supports
DCGM_FI_DEV_CPU_AFFINITY_0 = 70  # Device CPU affinity. part 1/8 = cpus 0 - 63
DCGM_FI_DEV_CPU_AFFINITY_1 = 71  # Device CPU affinity. part 1/8 = cpus 64 - 127
DCGM_FI_DEV_CPU_AFFINITY_2 = 72  # Device CPU affinity. part 2/8 = cpus 128 - 191
DCGM_FI_DEV_CPU_AFFINITY_3 = 73  # Device CPU affinity. part 3/8 = cpus 192 - 255
DCGM_FI_DEV_CC_MODE = 74  # Device CC/APM mode
DCGM_FI_DEV_MIG_ATTRIBUTES = 75  # MIG device attributes
DCGM_FI_DEV_MIG_GI_INFO = 76  # GPU instance profile information
DCGM_FI_DEV_MIG_CI_INFO = 77  # Compute instance profile information
DCGM_FI_DEV_INFOROM_ECC_VERSION = 80  # ECC inforom version
# Power management object inforom version
DCGM_FI_DEV_INFOROM_POWER_VERSION = 81
DCGM_FI_DEV_INFOROM_IMAGE_VERSION = 82  # Inforom image version
DCGM_FI_DEV_INFOROM_CHECKSUM = 83  # Inforom configuration checksum
# Reads the infoROM from the flash and verifies the checksums
DCGM_FI_DEV_INFOROM_VALID = 84
DCGM_FI_DEV_VBIOS_VERSION = 85  # VBIOS version of the device
# Device MEM affinity. part 1/4 = nodes 0 - 63
DCGM_FI_DEV_MEMORY_AFFINITY_0 = 86
# Device MEM affinity. part 1/4 = nodes 64 - 127
DCGM_FI_DEV_MEMORY_AFFINITY_1 = 87
# Device MEM affinity. part 1/4 = nodes 128 - 191
DCGM_FI_DEV_MEMORY_AFFINITY_2 = 88
# Device MEM affinity. part 1/4 = nodes 192 - 255
DCGM_FI_DEV_MEMORY_AFFINITY_3 = 89
DCGM_FI_DEV_BAR1_TOTAL = 90  # Total BAR1 of the GPU
DCGM_FI_SYSTEM_GPU_SYNC_BOOST = 91  # Deprecated - Sync boost settings on the node
DCGM_FI_DEV_BAR1_USED = 92  # Used BAR1 of the GPU in MB
DCGM_FI_DEV_BAR1_FREE = 93  # Free BAR1 of the GPU in MB
DCGM_FI_DEV_GPM_SUPPORT = 94  # GPM support for the device
# Clocks and power
DCGM_FI_DEV_SM_CLOCK = 100  # SM clock for the device
DCGM_FI_DEV_MEM_CLOCK = 101  # Memory clock for the device
DCGM_FI_DEV_VIDEO_CLOCK = 102  # Video encoder/decoder clock for the device
DCGM_FI_DEV_APP_SM_CLOCK = 110  # SM Application clocks
DCGM_FI_DEV_APP_MEM_CLOCK = 111  # Memory Application clocks
# Current clocks event reasons (bitmask of DCGM_CLOCKS_EVENT_REASON_*)
DCGM_FI_DEV_CLOCKS_EVENT_REASONS = 112
# Deprecated: Use DCGM_FI_DEV_CLOCKS_EVENT_REASONS instead
DCGM_FI_DEV_CLOCK_THROTTLE_REASONS = DCGM_FI_DEV_CLOCKS_EVENT_REASONS
DCGM_FI_DEV_MAX_SM_CLOCK = 113  # Maximum supported SM clock for the device
DCGM_FI_DEV_MAX_MEM_CLOCK = 114  # Maximum supported Memory clock for the device
# Maximum supported Video encoder/decoder clock for the device
DCGM_FI_DEV_MAX_VIDEO_CLOCK = 115
# Auto-boost for the device (1 = enabled. 0 = disabled)
DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE = 120
DCGM_FI_DEV_CLOCKS_SUPPORTED = 130  # Supported clocks for the device
DCGM_FI_DEV_MEMORY_TEMP_CELSIUS = 140  # Memory temperature for the device
# Current temperature readings for the device, in degrees C
DCGM_FI_DEV_GPU_TEMP_CELSIUS = 150
# Maximum operating temperature for the memory of this GPU
DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS = 151
# Maximum operating temperature for this GPU
DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS = 152
# Thermal margin temperature (distance to nearest slowdown threshold)
DCGM_FI_DEV_GPU_TEMP_MARGIN_CELSIUS = 153
DCGM_FI_DEV_BOARD_POWER_WATTS = 155  # Power usage for the device in Watts
# Total energy consumption for the GPU in mJ since the driver was last reloaded
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 156
# Current instantaneous power usage of the device in Watts
DCGM_FI_DEV_BOARD_POWER_RAW_WATTS = 157
# Slowdown temperature for the device
DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS = 158
# Shutdown temperature for the device
DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS = 159
# Current Power limit for the device
DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS = 160
# Minimum power management limit for the device
DCGM_FI_DEV_BOARD_POWER_LIMIT_MIN_WATTS = 161
# Maximum power management limit for the device
DCGM_FI_DEV_BOARD_POWER_LIMIT_MAX_WATTS = 162
# Default power management limit for the device
DCGM_FI_DEV_BOARD_POWER_LIMIT_DEFAULT_WATTS = 163
# Effective power limit that the driver enforces after taking into account
# all limiters
DCGM_FI_DEV_BOARD_POWER_LIMIT_ENFORCED_WATTS = 164
# Requested workload power profile mask (Blackwell and newer)
DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK = 165
# Enforced workload power profile mask (Blackwell and newer)
DCGM_FI_DEV_BOARD_POWER_PROFILE_ENFORCED_MASK = 166
# Supported workload power profile mask (Blackwell and newer)
DCGM_FI_DEV_BOARD_POWER_PROFILE_SUPPORTED_MASK = 167
# The status of the fabric manager - a value from dcgmFabricManagerStatus_t
DCGM_FI_DEV_FABRIC_MANAGER_STATUS = 170
# The failure that happened while starting the Fabric Manager, if any
DCGM_FI_DEV_FABRIC_MANAGER_ERROR = 171
# The uuid of the cluster to which this GPU belongs
DCGM_FI_DEV_FABRIC_CLUSTER_UUID = 172
# The ID of the fabric clique to which this GPU belongs
DCGM_FI_DEV_FABRIC_CLIQUE_ID = 173
DCGM_FI_DEV_FABRIC_HEALTH_MASK = 174  # GPU Fabric health Status Mask
DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY = 175  # GPU Fabric Health Summary
DCGM_FI_DEV_GPU_PSTATE = 190  # Performance state (P-State) 0-15. 0=highest
DCGM_FI_DEV_FAN_SPEED = 191  # Fan speed for the device in percent 0-100
# Device utilization and telemetry
# Deprecated - PCIe Tx utilization information
DCGM_FI_DEV_PCIE_TX_THROUGHPUT = 200
# Deprecated - PCIe Rx utilization information
DCGM_FI_DEV_PCIE_RX_THROUGHPUT = 201
DCGM_FI_DEV_PCIE_REPLAY_TOTAL = 202  # PCIe replay counter
DCGM_FI_DEV_GPU_UTIL_RATIO = 203  # GPU Utilization
DCGM_FI_DEV_MEM_COPY_UTIL = 204  # Memory Utilization
DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS = 205  # Process accounting stats
DCGM_FI_DEV_ENC_UTIL = 206  # Encoder utilization
DCGM_FI_DEV_DEC_UTIL = 207  # Decoder utilization
# Fields 210, 211, 220, and 221 are internal-only. see dcgm_fields_internal.py
DCGM_FI_DEV_XID_ERROR = 230  # XID errors. The value is the specific XID error
DCGM_FI_DEV_PCIE_MAX_LINK_GEN = 235  # PCIe Max Link Generation
DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH = 236  # PCIe Max Link Width
DCGM_FI_DEV_PCIE_LINK_GEN = 237  # PCIe Current Link Generation
DCGM_FI_DEV_PCIE_LINK_WIDTH = 238  # PCIe Current Link Width
# Violation counters
DCGM_FI_DEV_POWER_VIOLATION = 240  # Power Violation time in usec
DCGM_FI_DEV_THERMAL_VIOLATION = 241  # Thermal Violation time in usec
DCGM_FI_DEV_SYNC_BOOST_VIOLATION = 242  # Sync Boost Violation time in usec
DCGM_FI_DEV_BOARD_LIMIT_VIOLATION = 243  # Board Limit Violation time in usec.
DCGM_FI_DEV_LOW_UTIL_VIOLATION = 244  # Low Utilization Violation time in usec.
DCGM_FI_DEV_RELIABILITY_VIOLATION = 245  # Reliability Violation time in usec.
# App Clocks Violation time in usec.
DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION = 246
# Base Clocks Violation time in usec.
DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION = 247
# Framebuffer usage
DCGM_FI_DEV_FB_TOTAL = 250  # Total framebuffer memory in MB
DCGM_FI_DEV_FB_FREE = 251  # Total framebuffer used in MB
DCGM_FI_DEV_FB_USED = 252  # Total framebuffer free in MB
DCGM_FI_DEV_FB_RESERVED = 253  # Total framebuffer reserved in MB
DCGM_FI_DEV_FB_USED_RATIO = 254  # Frame buffer used ratio
# C2C Link Information (Grace-Hopper)
DCGM_FI_DEV_C2C_LINK_QUANTITY = 285  # C2C Link Quantity
DCGM_FI_DEV_C2C_LINK_STATUS = 286  # C2C Link Status
DCGM_FI_DEV_C2C_MAX_BANDWIDTH = 287  # C2C Link Max Bandwidth
# Device ECC Counters
DCGM_FI_DEV_ECC_MODE = 300  # Current ECC mode for the device
DCGM_FI_DEV_ECC_PENDING = 301  # Pending ECC mode for the device
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL = 310  # Total single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL = 311  # Total double bit volatile ecc errors
# Total single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_TOTAL = 312
# Total double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_TOTAL = 313

# Start of ECC Counters

DCGM_FI_DEV_ECC_SBE_VOL_L1 = 314  # L1 cache single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_L1 = 315  # L1 cache double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_L2 = 316  # L2 cache single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_L2 = 317  # L2 cache double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_DEV = 318  # Device memory single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_DEV = 319  # Device memory double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_REG = 320  # Register file single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_REG = 321  # Register file double bit volatile ecc errors
DCGM_FI_DEV_ECC_SBE_VOL_TEX = 322  # Texture memory single bit volatile ecc errors
DCGM_FI_DEV_ECC_DBE_VOL_TEX = 323  # Texture memory double bit volatile ecc errors
# L1 cache single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_L1 = 324
# L1 cache double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_L1 = 325
# L2 cache single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_L2 = 326
# L2 cache double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_L2 = 327
# Device memory single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_DEV = 328
# Device memory double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_DEV = 329
# Register File single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_REG = 330
# Register File double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_REG = 331
# Texture memory single bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_SBE_AGG_TEX = 332
# Texture memory double bit aggregate (persistent) ecc errors
DCGM_FI_DEV_ECC_DBE_AGG_TEX = 333

DCGM_FI_DEV_ECC_SBE_VOL_SHM = 334  # Texture SHM single bit volatile ECC erro
DCGM_FI_DEV_ECC_DBE_VOL_SHM = 335  # Texture SHM double bit volatile ECC errors
DCGM_FI_DEV_ECC_SBE_VOL_CBU = 336  # CBU single bit ECC volatile errors
DCGM_FI_DEV_ECC_DBE_VOL_CBU = 337  # CBU double bit ECC volatile errors
DCGM_FI_DEV_ECC_SBE_AGG_SHM = 338  # Texture SHM single bit aggregate ECC errors
DCGM_FI_DEV_ECC_DBE_AGG_SHM = 339  # Texture SHM double bit aggregate ECC errors
DCGM_FI_DEV_ECC_SBE_AGG_CBU = 340  # CBU single bit ECC aggregate errors
DCGM_FI_DEV_ECC_DBE_AGG_CBU = 341  # CBU double bit ECC aggregate errors

# Turing and later fields

DCGM_FI_DEV_ECC_SBE_VOL_SRM = 342  # SRAM single bit ECC volatile errors
DCGM_FI_DEV_ECC_DBE_VOL_SRM = 343  # SRAM double bit ECC volatile errors
DCGM_FI_DEV_ECC_SBE_AGG_SRM = 344  # SRAM single bit ECC aggregate errors
DCGM_FI_DEV_ECC_DBE_AGG_SRM = 345  # SRAM double bit ECC aggregate errors

DCGM_FI_DEV_SRAM_EXCEEDED = 346  # SRAM Error Status

# End of ECC Counters

DCGM_FI_DEV_DIAG_MEMORY_RESULT = 350  # Result of the GPU Memory test
DCGM_FI_DEV_DIAG_DIAGNOSTIC_RESULT = 351  # Result of the Diagnostics test
DCGM_FI_DEV_DIAG_PCIE_RESULT = 352  # Result of the PCIe + NVLink test
# Result of the Targeted Stress test
DCGM_FI_DEV_DIAG_TARGETED_STRESS_RESULT = 353
DCGM_FI_DEV_DIAG_TARGETED_POWER_RESULT = 354  # Result of the Targeted Power test
# Result of the Memory Bandwidth test
DCGM_FI_DEV_DIAG_MEMORY_BANDWIDTH_RESULT = 355
DCGM_FI_DEV_DIAG_MEMTEST_RESULT = 356  # Result of the Memory Stress test
# Result of the Input Energy Delayed Product power (EDPp) test (a.k.a. the
# pulse test)
DCGM_FI_DEV_DIAG_PULSE_TEST_RESULT = 357
# Result of the Extended Utility Diagnostics (EUD) test
DCGM_FI_DEV_DIAG_EUD_RESULT = 358
# Result of the CPU Extended Utility Diagnostics (EUD) test
DCGM_FI_DEV_DIAG_CPU_EUD_RESULT = 359
DCGM_FI_DEV_DIAG_SOFTWARE_RESULT = 360  # Result of the Software test
DCGM_FI_DEV_DIAG_NVBANDWIDTH_RESULT = 361  # Result of the NVBandwidth test
DCGM_FI_DEV_DIAG_STATUS = 362  # Status of the current diag run
DCGM_FI_DEV_DIAG_NCCL_TESTS_RESULT = 363  # Result of the nccl-tests test

# Remap availability histogram for each memory bank on the GPU.
DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX = 385
DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH = 386
DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL = 387
DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW = 388
DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE = 389
# Number of retired pages because of single bit errors
DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL = 390
# Number of retired pages because of double bit errors
DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL = 391
DCGM_FI_DEV_PAGE_RETIRED_PENDING = 392  # Number of pages pending retirement
# Row remapper fields (Ampere and newer)
# Number of remapped rows for uncorrectable errors
DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL = 393
# Number of remapped rows for correctable errors
DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL = 394
DCGM_FI_DEV_ROW_REMAP_FAILED = 395  # Whether remapping of rows has failed
DCGM_FI_DEV_ROW_REMAP_PENDING = 396  # Whether remapping of rows is pending

# Device NvLink Throughput and Error Counters
# NV Link flow control CRC  Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL = 400
# NV Link flow control CRC  Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL = 401
# NV Link flow control CRC  Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL = 402
# NV Link flow control CRC  Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL = 403
# NV Link flow control CRC  Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL = 404
# NV Link flow control CRC  Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL = 405
# NV Link flow control CRC  Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL = 409
# NV Link data CRC Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL = 410
# NV Link data CRC Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL = 411
# NV Link data CRC Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL = 412
# NV Link data CRC Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL = 413
# NV Link data CRC Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL = 414
# NV Link data CRC Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL = 415
# NV Link data CRC Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL = 419
# NV Link Replay Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL = 420
# NV Link Replay Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL = 421
# NV Link Replay Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL = 422
# NV Link Replay Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL = 423
# NV Link Replay Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL = 424
# NV Link Replay Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL = 425
# NV Link Replay Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL = 429

# NV Link Recovery Error Counter for Lane 0
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL = 430
# NV Link Recovery Error Counter for Lane 1
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL = 431
# NV Link Recovery Error Counter for Lane 2
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL = 432
# NV Link Recovery Error Counter for Lane 3
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL = 433
# NV Link Recovery Error Counter for Lane 4
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL = 434
# NV Link Recovery Error Counter for Lane 5
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL = 435
# NV Link Recovery Error Counter total for all Lanes
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL = 439
DCGM_FI_DEV_NVLINK_THROUGHPUT_L0 = 440  # NV Link Throughput for Lane 0
DCGM_FI_DEV_NVLINK_THROUGHPUT_L1 = 441  # NV Link Throughput for Lane 1
DCGM_FI_DEV_NVLINK_THROUGHPUT_L2 = 442  # NV Link Throughput for Lane 2
DCGM_FI_DEV_NVLINK_THROUGHPUT_L3 = 443  # NV Link Throughput for Lane 3
DCGM_FI_DEV_NVLINK_THROUGHPUT_L4 = 444  # NV Link Throughput for Lane 4
DCGM_FI_DEV_NVLINK_THROUGHPUT_L5 = 445  # NV Link Throughput for Lane 5
# NV Link Throughput total for all Lanes
DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL = 449
DCGM_FI_DEV_NVLINK_ERROR = 450  # GPU NVLink error information
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL = 451
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL = 452
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL = 453
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL = 454
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL = 455
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL = 456
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL = 406
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL = 407
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL = 408
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL = 481
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL = 482
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL = 483
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL = 457
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL = 458
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL = 459
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL = 460
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL = 461
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL = 462
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL = 416
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL = 417
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL = 418
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL = 484
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL = 485
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL = 486
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL = 463
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL = 464
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL = 465
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL = 466
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL = 467
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL = 468
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL = 426
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL = 427
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL = 428
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL = 487
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL = 488
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL = 489
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL = 469
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL = 470
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL = 471
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL = 472
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL = 473
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL = 474
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL = 436
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL = 437
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL = 438
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL = 491
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL = 492
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL = 493
DCGM_FI_DEV_NVLINK_THROUGHPUT_L6 = 475
DCGM_FI_DEV_NVLINK_THROUGHPUT_L7 = 476
DCGM_FI_DEV_NVLINK_THROUGHPUT_L8 = 477
DCGM_FI_DEV_NVLINK_THROUGHPUT_L9 = 478
DCGM_FI_DEV_NVLINK_THROUGHPUT_L10 = 479
DCGM_FI_DEV_NVLINK_THROUGHPUT_L11 = 480
DCGM_FI_DEV_NVLINK_THROUGHPUT_L12 = 446
DCGM_FI_DEV_NVLINK_THROUGHPUT_L13 = 447
DCGM_FI_DEV_NVLINK_THROUGHPUT_L14 = 448
DCGM_FI_DEV_NVLINK_THROUGHPUT_L15 = 494
DCGM_FI_DEV_NVLINK_THROUGHPUT_L16 = 495
DCGM_FI_DEV_NVLINK_THROUGHPUT_L17 = 496
DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL = 497
DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL = 498
DCGM_FI_DEV_NVLINK_REPLAY_TOTAL = 499

# Device Attributes associated with virtualization
DCGM_FI_DEV_GPU_VIRTUAL_MODE = 500  # Operating mode of the GPU
# Includes Count and Supported vGPU type information
DCGM_FI_DEV_VGPU_SUPPORTED_INFO = 501
# Includes Count and List of Creatable vGPU type IDs
DCGM_FI_DEV_VGPU_CREATABLE_IDS = 502
# Includes Count and List of vGPU instance IDs
DCGM_FI_DEV_VGPU_INSTANCE_INFO = 503
# Utilization values for vGPUs running on the device
DCGM_FI_DEV_VGPU_UTIL_INFO = 504
# Utilization values for processes running within vGPU VMs using the device
DCGM_FI_DEV_VGPU_PROCESS_UTIL_INFO = 505
DCGM_FI_DEV_ENC_STATS = 506  # Current encoder statistics for a given device
# Statistics of current active frame buffer capture sessions on a given device
DCGM_FI_DEV_FBC_STATS = 507
# Information about active frame buffer capture sessions on a target device
DCGM_FI_DEV_FBC_SESSIONS_INFO = 508
# Includes Count and currently Supported vGPU types on a device
DCGM_FI_DEV_VGPU_SUPPORTED_IDS = 509
# Includes Static info of vGPU types supported on a device
DCGM_FI_DEV_VGPU_TYPE_INFO = 510
# Includes the name of a vGPU type supported on a device
DCGM_FI_DEV_VGPU_TYPE_NAME = 511
# Includes the class of a vGPU type supported on a device
DCGM_FI_DEV_VGPU_TYPE_CLASS = 512
# Includes the license info for a vGPU type supported on a device
DCGM_FI_DEV_VGPU_TYPE_LICENSE = 513
# Related to vGPU Instance IDs
DCGM_FI_DEV_VGPU_VM_ID = 520  # vGPU VM ID
DCGM_FI_DEV_VGPU_VM_NAME = 521  # vGPU VM name
DCGM_FI_DEV_VGPU_TYPE = 522  # vGPU type of the vGPU instance
DCGM_FI_DEV_VGPU_UUID = 523  # UUID of the vGPU instance
DCGM_FI_DEV_VGPU_DRIVER_VERSION = 524  # Driver version of the vGPU instance
DCGM_FI_DEV_VGPU_MEMORY_USAGE = 525  # Memory usage of the vGPU instance
DCGM_FI_DEV_VGPU_LICENSE_STATUS = 526  # License status of the vGPU
# Frame rate limit of the vGPU instance
DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT = 527
DCGM_FI_DEV_VGPU_ENC_STATS = 528  # Current encoder statistics of the vGPU instance
# Information about all active encoder sessions on the vGPU instance
DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO = 529
# Statistics of current active frame buffer capture sessions on the vGPU
# instance
DCGM_FI_DEV_VGPU_FBC_STATS = 530
# Information about active frame buffer capture sessions on the vGPU instance
DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO = 531
# License state information of the vGPU instance
DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS = 532
DCGM_FI_DEV_VGPU_PCI_ID = 533  # PCI Id of the vGPU instance
# GPU Instance Id of the vGPU instance
DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID = 534

# Infiniband GUID string (e.g. xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
DCGM_FI_DEV_PLATFORM_INFINIBAND_GUID = 571
# GUID string of the rack containing the GPU (e.g.
# xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
DCGM_FI_DEV_PLATFORM_CHASSIS_SERIAL_NUMBER = 572
# Slot number in the rack containing the GPU (includes switches)
DCGM_FI_DEV_PLATFORM_CHASSIS_SLOT_NUMBER = 573
# Index within the compute slots in the rack containing the GPU (does not
# include switches)
DCGM_FI_DEV_PLATFORM_TRAY_INDEX = 574
# Index of the node within the slot containing the GPU
DCGM_FI_DEV_PLATFORM_HOST_ID = 575
# Platform indicated NVLink-peer type (e.g. switch present or not)
DCGM_FI_DEV_PLATFORM_PEER_TYPE = 576
DCGM_FI_DEV_PLATFORM_MODULE_ID = 577  # ID of the GPU within the node

# Link-based PRM metrics for NvLink (use dcgm_link_t to specify GPU ID +
# port number)
DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY = 580  # PPRM recovery operation status
# Time in seconds since last PRM recovery
DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST = 581
# Time in milliseconds between last two recoveries
DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO = 582
# Total successful recovery events counter
DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SUCCESSFUL_TOTAL = 583
# Physical layer successful recovery events
DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_RECOVERY_SUCCESSFUL_TOTAL = 584
# Physical layer link down counter
DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_TOTAL = 585
DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL = 586  # PLR received codewords counter
# PLR received code error counter
DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL = 587
# PLR received uncorrectable codes counter
DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_UNCORRECTABLE_TOTAL = 588
# PLR transmitted codewords counter
DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL = 589
# PLR transmitted retry codes counter
DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_CODE_TOTAL = 590
# PLR transmitted retry events counter
DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL = 591
DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENT_TOTAL = 592  # PLR sync events counter

# Internal fields reserve the range 600..699
# below fields related to NVSwitch
# Starting field ID of the NVSwitch instance
DCGM_FI_FIRST_NVSWITCH_FIELD_ID = 700
DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT = 701
DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ = 702
DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV = 703
DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD = 704
DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS = 705
DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS = 706
DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS = 707
DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX = 780
DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX = 781
DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS = 782
DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS = 783
DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL = 784
DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERROR_TOTAL = 785
DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERROR_TOTAL = 786
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_TOTAL = 787
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_TOTAL = 788
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0 = 789
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1 = 790
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2 = 791
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3 = 792
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0 = 793
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1 = 794
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2 = 795
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3 = 796
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0 = 797
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1 = 798
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2 = 799
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3 = 800
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0 = 801
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1 = 802
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2 = 803
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3 = 804
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL = 805
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL = 806
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL = 807
DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL = 808
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L0_TOTAL = 809
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L1_TOTAL = 810
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L2_TOTAL = 811
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L3_TOTAL = 812
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L0_TOTAL = 813
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L1_TOTAL = 814
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L2_TOTAL = 815
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L3_TOTAL = 816
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L4_TOTAL = 817
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L5_TOTAL = 818
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L6_TOTAL = 819
DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L7_TOTAL = 820
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L4_TOTAL = 821
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L5_TOTAL = 822
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L6_TOTAL = 823
DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L7_TOTAL = 824
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0 = 825
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L1 = 826
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L2 = 827
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L3 = 828
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L4 = 829
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L5 = 830
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L6 = 831
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L7 = 832
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L8 = 833
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L9 = 834
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L10 = 835
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L11 = 836
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L12 = 837
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L13 = 838
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L14 = 839
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L15 = 840
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L16 = 841
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L17 = 842
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL = 843
DCGM_FI_DEV_SXID_FATAL_ERROR = 856
DCGM_FI_DEV_SXID_NON_FATAL_ERROR = 857
DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS = 858
DCGM_FI_DEV_NVSWITCH_TEMP_SLOWDOWN_CELSIUS = 859
DCGM_FI_DEV_NVSWITCH_TEMP_SHUTDOWN_CELSIUS = 860
DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX = 861
DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX = 862
DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID = 863
DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED = 864
DCGM_FI_DEV_NVSWITCH_LINK_ID = 865
DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN = 866
DCGM_FI_DEV_NVSWITCH_PCIE_BUS = 867
DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE = 868
DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION = 869
DCGM_FI_DEV_NVSWITCH_LINK_STATUS = 870
DCGM_FI_DEV_NVSWITCH_LINK_TYPE = 871
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN = 872
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS = 873
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE = 874
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION = 875
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID = 876
DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID = 877

DCGM_FI_DEV_NVSWITCH_UUID = 878

DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0 = 879
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L1 = 880
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L2 = 881
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L3 = 882
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L4 = 883
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L5 = 884
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L6 = 885
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L7 = 886
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L8 = 887
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L9 = 888
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L10 = 889
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L11 = 890
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L12 = 891
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L13 = 892
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L14 = 893
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L15 = 894
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L16 = 895
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L17 = 896
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL = 897

DCGM_FI_LAST_NVSWITCH_FIELD_ID = 899  # Last field ID of the NVSwitch instance

'''
Profiling Fields
'''
DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO = 1001  # Ratio of time the graphics engine is active. The graphics engine is
# active if a graphics/compute context is bound and the graphics pipe or
# compute pipe is busy.

# The ratio of cycles an SM has at least 1 warp assigned
DCGM_FI_PROF_SM_UTIL_RATIO = 1002
# (computed from the number of cycles and elapsed cycles)

# The ratio of number of warps resident on an SM.
DCGM_FI_PROF_SM_OCCUPANCY_RATIO = 1003
# (number of resident as a ratio of the theoretical
# maximum number of warps per elapsed cycle)

# The ratio of cycles the any tensor pipe is active
DCGM_FI_PROF_TENSOR_UTIL_RATIO = 1004
# (off the peak sustained elapsed cycles)

# The ratio of cycles the device memory interface is active sending or
# receiving data.
DCGM_FI_PROF_DRAM_UTIL_RATIO = 1005
# Ratio of cycles the fp64 pipe is active.
DCGM_FI_PROF_FP64_UTIL_RATIO = 1006
# Ratio of cycles the fp32 pipe is active.
DCGM_FI_PROF_FP32_UTIL_RATIO = 1007
# Ratio of cycles the fp16 pipe is active. This does not include HMMA.
DCGM_FI_PROF_FP16_UTIL_RATIO = 1008
# The number of bytes of active PCIe tx (transmit) data including both
# header and payload.
DCGM_FI_PROF_PCIE_TX_BYTES = 1009
# The number of bytes of active PCIe rx (read) data including both header
# and payload.
DCGM_FI_PROF_PCIE_RX_BYTES = 1010
# The number of bytes of active NvLink tx (transmit) data including both
# header and payload.
DCGM_FI_PROF_NVLINK_TX_BYTES = 1011
# The number of bytes of active NvLink rx (receive) data including both
# header and payload.
DCGM_FI_PROF_NVLINK_RX_BYTES = 1012
# The ratio of cycles the IMMA tensor pipe is active (off the peak
# sustained elapsed cycles)
DCGM_FI_PROF_IMMA_UTIL_RATIO = 1013
# The ratio of cycles the HMMA tensor pipe is active (off the peak
# sustained elapsed cycles)
DCGM_FI_PROF_HMMA_UTIL_RATIO = 1014
# The ratio of cycles the tensor (DFMA) pipe is active (off the peak
# sustained elapsed cycles)
DCGM_FI_PROF_DFMA_UTIL_RATIO = 1015
# Ratio of cycles the integer pipe is active.
DCGM_FI_PROF_INT_UTIL_RATIO = 1016

# Ratio of cycles each of the NVDEC engines are active.
DCGM_FI_PROF_NVDEC_UTIL_0_RATIO = 1017
DCGM_FI_PROF_NVDEC_UTIL_1_RATIO = 1018
DCGM_FI_PROF_NVDEC_UTIL_2_RATIO = 1019
DCGM_FI_PROF_NVDEC_UTIL_3_RATIO = 1020
DCGM_FI_PROF_NVDEC_UTIL_4_RATIO = 1021
DCGM_FI_PROF_NVDEC_UTIL_5_RATIO = 1022
DCGM_FI_PROF_NVDEC_UTIL_6_RATIO = 1023
DCGM_FI_PROF_NVDEC_UTIL_7_RATIO = 1024

# Ratio of cycles each of the NVJPG engines are active.
DCGM_FI_PROF_NVJPG_UTIL_0_RATIO = 1025
DCGM_FI_PROF_NVJPG_UTIL_1_RATIO = 1026
DCGM_FI_PROF_NVJPG_UTIL_2_RATIO = 1027
DCGM_FI_PROF_NVJPG_UTIL_3_RATIO = 1028
DCGM_FI_PROF_NVJPG_UTIL_4_RATIO = 1029
DCGM_FI_PROF_NVJPG_UTIL_5_RATIO = 1030
DCGM_FI_PROF_NVJPG_UTIL_6_RATIO = 1031
DCGM_FI_PROF_NVJPG_UTIL_7_RATIO = 1032

# Ratio of cycles each of the NVOFA engines are active.
DCGM_FI_PROF_NVOFA_UTIL_0_RATIO = 1033
DCGM_FI_PROF_NVOFA_UTIL_1_RATIO = 1034

'''
The per-link number of bytes of active NvLink TX (transmit) or RX (transmit) data including both header and payload.
For example: DCGM_FI_PROF_NVLINK_L0_TX_BYTES -> L0 TX
To get the bandwidth for a link, add the RX and TX value together like
total = DCGM_FI_PROF_NVLINK_L0_TX_BYTES + DCGM_FI_PROF_NVLINK_L0_RX_BYTES
'''
DCGM_FI_PROF_NVLINK_L0_TX_BYTES = 1040
DCGM_FI_PROF_NVLINK_L0_RX_BYTES = 1041
DCGM_FI_PROF_NVLINK_L1_TX_BYTES = 1042
DCGM_FI_PROF_NVLINK_L1_RX_BYTES = 1043
DCGM_FI_PROF_NVLINK_L2_TX_BYTES = 1044
DCGM_FI_PROF_NVLINK_L2_RX_BYTES = 1045
DCGM_FI_PROF_NVLINK_L3_TX_BYTES = 1046
DCGM_FI_PROF_NVLINK_L3_RX_BYTES = 1047
DCGM_FI_PROF_NVLINK_L4_TX_BYTES = 1048
DCGM_FI_PROF_NVLINK_L4_RX_BYTES = 1049
DCGM_FI_PROF_NVLINK_L5_TX_BYTES = 1050
DCGM_FI_PROF_NVLINK_L5_RX_BYTES = 1051
DCGM_FI_PROF_NVLINK_L6_TX_BYTES = 1052
DCGM_FI_PROF_NVLINK_L6_RX_BYTES = 1053
DCGM_FI_PROF_NVLINK_L7_TX_BYTES = 1054
DCGM_FI_PROF_NVLINK_L7_RX_BYTES = 1055
DCGM_FI_PROF_NVLINK_L8_TX_BYTES = 1056
DCGM_FI_PROF_NVLINK_L8_RX_BYTES = 1057
DCGM_FI_PROF_NVLINK_L9_TX_BYTES = 1058
DCGM_FI_PROF_NVLINK_L9_RX_BYTES = 1059
DCGM_FI_PROF_NVLINK_L10_TX_BYTES = 1060
DCGM_FI_PROF_NVLINK_L10_RX_BYTES = 1061
DCGM_FI_PROF_NVLINK_L11_TX_BYTES = 1062
DCGM_FI_PROF_NVLINK_L11_RX_BYTES = 1063
DCGM_FI_PROF_NVLINK_L12_TX_BYTES = 1064
DCGM_FI_PROF_NVLINK_L12_RX_BYTES = 1065
DCGM_FI_PROF_NVLINK_L13_TX_BYTES = 1066
DCGM_FI_PROF_NVLINK_L13_RX_BYTES = 1067
DCGM_FI_PROF_NVLINK_L14_TX_BYTES = 1068
DCGM_FI_PROF_NVLINK_L14_RX_BYTES = 1069
DCGM_FI_PROF_NVLINK_L15_TX_BYTES = 1070
DCGM_FI_PROF_NVLINK_L15_RX_BYTES = 1071
DCGM_FI_PROF_NVLINK_L16_TX_BYTES = 1072
DCGM_FI_PROF_NVLINK_L16_RX_BYTES = 1073
DCGM_FI_PROF_NVLINK_L17_TX_BYTES = 1074
DCGM_FI_PROF_NVLINK_L17_RX_BYTES = 1075

# C2C (Chip-to-Chip) interface metrics.
DCGM_FI_PROF_C2C_TX_ALL_BYTES = 1076
DCGM_FI_PROF_C2C_TX_DATA_BYTES = 1077
DCGM_FI_PROF_C2C_RX_ALL_BYTES = 1078
DCGM_FI_PROF_C2C_RX_DATA_BYTES = 1079

# GPU Host Memory Utilization
DCGM_FI_PROF_HOSTMEM_CACHE_HIT = 1080
DCGM_FI_PROF_HOSTMEM_CACHE_MISS = 1081

# GPU Peer Memory Utilization
DCGM_FI_PROF_PEERMEM_CACHE_HIT = 1082
DCGM_FI_PROF_PEERMEM_CACHE_MISS = 1083

# Cumulative GPM counter metrics - core
DCGM_FI_PROF_SM_CYCLES_ELAPSED_TOTAL = 1084
DCGM_FI_PROF_SM_CYCLES_ACTIVE_TOTAL = 1085
DCGM_FI_PROF_MMA_CYCLES_ACTIVE_TOTAL = 1086
DCGM_FI_PROF_DMMA_CYCLES_ACTIVE_TOTAL = 1087
DCGM_FI_PROF_HMMA_CYCLES_ACTIVE_TOTAL = 1088
DCGM_FI_PROF_IMMA_CYCLES_ACTIVE_TOTAL = 1089
DCGM_FI_PROF_DFMA_CYCLES_ACTIVE_TOTAL = 1090
DCGM_FI_PROF_PCIE_TX_BYTES_TOTAL = 1091
DCGM_FI_PROF_PCIE_RX_BYTES_TOTAL = 1092
DCGM_FI_PROF_INT_CYCLES_ACTIVE_TOTAL = 1093
DCGM_FI_PROF_FP64_CYCLES_ACTIVE_TOTAL = 1094
DCGM_FI_PROF_FP32_CYCLES_ACTIVE_TOTAL = 1095
DCGM_FI_PROF_FP16_CYCLES_ACTIVE_TOTAL = 1096

DCGM_FI_DEV_CPU_UTIL_TOTAL = 1100  # CPU Utilization, total
DCGM_FI_DEV_CPU_UTIL_USER = 1101  # CPU Utilization, user
DCGM_FI_DEV_CPU_UTIL_NICE = 1102  # CPU Utilization, nice
DCGM_FI_DEV_CPU_UTIL_SYS = 1103  # CPU Utilization, system time
DCGM_FI_DEV_CPU_UTIL_IRQ = 1104  # CPU Utilization, interrupt servicing
DCGM_FI_DEV_CPU_TEMP_CELSIUS = 1110  # CPU temperature
DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS = 1111  # CPU warning temparature
DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS = 1112  # CPU critical temparature
DCGM_FI_DEV_CPU_CLOCK_CURRENT = 1120  # CPU instantaneous clock speed
DCGM_FI_DEV_CPU_POWER_WATTS = 1130  # CPU power utilization
DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS = 1131  # CPU power limit
DCGM_FI_DEV_SYSIO_POWER_UTIL_CURRENT = 1132  # SysIO power limit
DCGM_FI_DEV_MODULE_POWER_UTIL_CURRENT = 1133  # Module power limit
DCGM_FI_DEV_CPU_VENDOR = 1140  # CPU vendor name
DCGM_FI_DEV_CPU_MODEL = 1141  # CPU model name

# Total Tx packets on the link in NVLink5
DCGM_FI_DEV_NVLINK_TX_PACKET_TOTAL = 1200

# Total Tx bytes on the link in NVLink5
DCGM_FI_DEV_NVLINK_TX_BYTES_TOTAL = 1201

# Total Rx packets on the link in NVLink5
DCGM_FI_DEV_NVLINK_RX_PACKET_TOTAL = 1202

# Total Rx bytes on the link in NVLink5
DCGM_FI_DEV_NVLINK_RX_BYTES_TOTAL = 1203

# Number of packets Rx on a link where packets are malformed
DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL = 1204

# Number of packets that were discarded on Rx due to buffer overrun
DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL = 1205

# Total number of packets with errors Rx on a link
DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL = 1206

# Total number of packets Rx - stomp/EBP marker
DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL = 1207

# Total number of packets Rx with header mismatch
DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL = 1208

# Total number of times that the count of local errors exceeded a threshold
DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL = 1209

# Total number of tx error packets that were discarded
DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS = 1210

# Number of times link went from Up to recovery, succeeded and link came
# back up
DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL = 1211

# Number of times link went from Up to recovery, failed and link was
# declared down
DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL = 1212

# Number of times link went from Up to recovery, irrespective of the result
DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL = 1213

# Number of errors in rx symbols
DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL = 1214

# BER for symbol errors - raw value
DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW = 1215

# BER for symbol errors - decoded float
DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO = 1216

# Effective BER for effective errors - raw value
DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW = 1217

# Effective BER for effective errors - decoded float
DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO = 1218

# Sum of the number of errors in each Nvlink packet
DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL = 1219

# NVLink ECC Data Error Counter total for all Links
DCGM_FI_DEV_NVLINK_ECC_ERROR_TOTAL = 1220

# First field id of ConnectX
DCGM_FI_DEV_FIRST_CONNECTX_FIELD_ID = 1300

# Health state of ConnectX
DCGM_FI_DEV_CONNECTX_HEALTH = 1300

# Active PCIe link width
DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_WIDTH = 1301

# Active PCIe link speed
DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_SPEED = 1302

# Expect PCIe link width
DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_WIDTH = 1303

# Expect PCIe link speed
DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_SPEED = 1304

# Correctable error status
DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_STATUS = 1305

# Correctable error mask
DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_MASK = 1306

# Uncorrectable error status
DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS = 1307

# Uncorrectable error mask
DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK = 1308

# Uncorrectable error severity
DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY = 1309

# Device temperature
DCGM_FI_DEV_CONNECTX_DEVICE_TEMPERATURE = 1310

# The last field id of ConnectX
DCGM_FI_DEV_LAST_CONNECTX_FIELD_ID = 1399

DCGM_FI_DEV_C2C_LINK_ERROR_INTR = 1400
DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY = 1401
DCGM_FI_DEV_C2C_LINK_ERROR_REPLAY_B2B = 1402
DCGM_FI_DEV_C2C_LINK_POWER_STATE = 1403
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0 = 1404
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_1 = 1405
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_2 = 1406
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_3 = 1407
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_4 = 1408
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_5 = 1409
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_6 = 1410
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_7 = 1411
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_8 = 1412
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_9 = 1413
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_10 = 1414
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_11 = 1415
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_12 = 1416
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_13 = 1417
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_14 = 1418
DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15 = 1419
DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP_NS = 1420
DCGM_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST_COUNT_NS = 1421
DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN_NS = 1422
DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN_NS = 1423
DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN_NS = 1424

DCGM_FI_DEV_PWR_SMOOTHING_ENABLED = 1425
DCGM_FI_DEV_PWR_SMOOTHING_PRIV_LVL = 1426
DCGM_FI_DEV_PWR_SMOOTHING_IMM_RAMP_DOWN_ENABLED = 1427
DCGM_FI_DEV_PWR_SMOOTHING_APPLIED_TMP_CEIL = 1428
DCGM_FI_DEV_PWR_SMOOTHING_APPLIED_TMP_FLOOR = 1429
DCGM_FI_DEV_PWR_SMOOTHING_MAX_PERCENT_TMP_FLOOR_SETTING = 1430
DCGM_FI_DEV_PWR_SMOOTHING_MIN_PERCENT_TMP_FLOOR_SETTING = 1431
DCGM_FI_DEV_PWR_SMOOTHING_HW_CIRCUITRY_PERCENT_LIFETIME_REMAINING = 1432
DCGM_FI_DEV_PWR_SMOOTHING_MAX_NUM_PRESET_PROFILES = 1433
DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_PERCENT_TMP_FLOOR = 1434
DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_RAMP_UP_RATE = 1435
DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_RAMP_DOWN_RATE = 1436
DCGM_FI_DEV_PWR_SMOOTHING_PROFILE_RAMP_DOWN_HYST_VAL = 1437
DCGM_FI_DEV_PWR_SMOOTHING_ACTIVE_PRESET_PROFILE = 1438
DCGM_FI_DEV_PWR_SMOOTHING_ADMIN_OVERRIDE_PERCENT_TMP_FLOOR = 1439
DCGM_FI_DEV_PWR_SMOOTHING_ADMIN_OVERRIDE_RAMP_UP_RATE = 1440
DCGM_FI_DEV_PWR_SMOOTHING_ADMIN_OVERRIDE_RAMP_DOWN_RATE = 1441
DCGM_FI_DEV_PWR_SMOOTHING_ADMIN_OVERRIDE_RAMP_DOWN_HYST_VAL = 1442
# 1443 to 1500 entries reserved for power smoothing fields

# PCIe Correctable Errors Counter
DCGM_FI_DEV_PCIE_CORRECTABLE_ERROR_TOTAL = 1501

# IMEX (Infiniband Management EXtension) fields
# IMEX domain status (UP, DOWN, DEGRADED, NOT_INSTALLED, NOT_CONFIGURED,
# UNAVAILABLE)
DCGM_FI_IMEX_DOMAIN_STATUS = 1502
# IMEX daemon status (0-7: INITIALIZING through UNAVAILABLE, -1:
# NOT_INSTALLED, -2: NOT_CONFIGURED, -3: COMMAND_ERROR)
DCGM_FI_IMEX_DAEMON_STATUS = 1503

# 1504 to 1507 reserved for power IMEX fields

DCGM_FI_DEV_MEMORY_UNREPAIRABLE = 1507

# NVLink State (see NVML_FI_DEV_NVLINK_GET_STATE for return values)
# This field expects a dcgm_link_t entity to specify the GPU and link index.
# Use DCGM_FE_LINK entity group when accessing this field.
DCGM_FI_DEV_NVLINK_GET_STATE = 1508

# InfiniBand Port Counter: Port Transmit Wait
# (see NVML_PRM_COUNTER_ID_PPCNT_PORTCOUNTERS_PORT_XMIT_WAIT for details)
# This field expects a dcgm_link_t entity to specify the GPU and link index.
# Use DCGM_FE_LINK entity group when accessing this field.
DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT = 1509

# 1510 to 1522 reserved for future use

# GPU Recovery Action (see nvmlDeviceGpuRecoveryAction_t for return values)
DCGM_FI_DEV_GPU_RECOVERY_ACTION = 1523

# NVSwitch firmware version string
DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION = 1524

# Per-link NVLink metrics keyed by a dcgm_link_t entity (DCGM_FE_LINK). A single field id covers
# every link; the link index is carried in the entity. Legacy L0-L17 fields remain as aliases.
DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_PER_LINK_TOTAL = 1525
DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_PER_LINK_TOTAL = 1526
DCGM_FI_DEV_NVLINK_REPLAY_ERROR_PER_LINK_TOTAL = 1527
DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_PER_LINK_TOTAL = 1528
DCGM_FI_DEV_NVLINK_THROUGHPUT_PER_LINK = 1529
DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_PER_LINK = 1530
DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_PER_LINK = 1531
# Per-link NVLink GPM throughput counters (bytes), keyed by a dcgm_link_t entity (DCGM_FE_LINK).
DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK = 1532
DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK = 1533

# 1 greater than maximum fields above. This value can increase in the future
DCGM_FI_MAX_FIELDS = DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK + 1

DCGM_DEPRECATED = 1

"""
   If DCGM_DEPRECATED is not zero the following deprecated symbols will be
   resolved for backward compatibility. Setting it to zero lets us test that
   Nvidia code does not depend on deprecated symbols.
"""

# autopep8: off
_DEPRECATED_ALIASES = {
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L0":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L1":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L1",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L2":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L2",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L3":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L3",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L4":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L4",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L5":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L5",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L6":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L6",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L7":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L7",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L8":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L8",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L9":  "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L9",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L10": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L10",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L11": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L11",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L12": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L12",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L13": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L13",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L14": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L14",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L15": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L15",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L16": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L16",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L17": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L17",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L0":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L0",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L1":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L1",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L2":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L2",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L3":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L3",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L4":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L4",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L5":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L5",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_TOTAL": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL",
    "DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_TOTAL": "DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL":  "DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L6":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L6",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L7":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L7",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L8":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L8",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L9":     "DCGM_FI_DEV_NVLINK_THROUGHPUT_L9",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L10":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L10",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L11":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L11",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L12":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L12",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L13":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L13",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L14":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L14",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L15":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L15",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L16":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L16",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_L17":    "DCGM_FI_DEV_NVLINK_THROUGHPUT_L17",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L0":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L1":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L1",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L2":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L2",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L3":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L3",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L4":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L4",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L5":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L5",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L6":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L6",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L7":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L7",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L8":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L8",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L9":  "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L9",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L10": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L10",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L11": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L11",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L12": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L12",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L13": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L13",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L14": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L14",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L15": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L15",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L16": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L16",
    "DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L17": "DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L17",

    # Deprecated: renamed per DCGM Field Naming Standard (see dcgmlib/dcgm_field_naming.py)
    "DCGM_FI_UNKNOWN":                                              "DCGM_FI_SYSTEM_FIELD_UNKNOWN",                                #    0
    "DCGM_FI_DRIVER_VERSION":                                       "DCGM_FI_SYSTEM_DRIVER_VERSION",                               #    1
    "DCGM_FI_NVML_VERSION":                                         "DCGM_FI_SYSTEM_NVML_VERSION",                                 #    2
    "DCGM_FI_PROCESS_NAME":                                         "DCGM_FI_SYSTEM_PROCESS_NAME",                                 #    3
    "DCGM_FI_DEV_COUNT":                                            "DCGM_FI_SYSTEM_GPU_QUANTITY",                                 #    4
    "DCGM_FI_BIND_UNBIND_EVENT":                                    "DCGM_FI_SYSTEM_GPU_BIND_EVENT",                               #    6
    "DCGM_FI_DEV_NAME":                                             "DCGM_FI_DEV_GPU_NAME",                                        #   50
    "DCGM_FI_DEV_BRAND":                                            "DCGM_FI_DEV_GPU_BRAND",                                       #   51
    "DCGM_FI_DEV_SERIAL":                                           "DCGM_FI_DEV_BOARD_SERIAL",                                    #   53
    "DCGM_FI_DEV_UUID":                                             "DCGM_FI_DEV_GPU_UUID",                                        #   54
    "DCGM_FI_DEV_MINOR_NUMBER":                                     "DCGM_FI_DEV_GPU_MINOR_NUMBER",                                #   55
    "DCGM_FI_DEV_OEM_INFOROM_VER":                                  "DCGM_FI_DEV_INFOROM_OEM_VERSION",                             #   56
    "DCGM_FI_DEV_PCI_BUSID":                                        "DCGM_FI_DEV_PCI_BUS_ID",                                      #   57
    "DCGM_FI_GPU_TOPOLOGY_PCI":                                     "DCGM_FI_SYSTEM_PCI_TOPOLOGY",                                 #   60
    "DCGM_FI_GPU_TOPOLOGY_NVLINK":                                  "DCGM_FI_SYSTEM_NVLINK_TOPOLOGY",                              #   61
    "DCGM_FI_GPU_TOPOLOGY_AFFINITY":                                "DCGM_FI_SYSTEM_GPU_AFFINITY",                                 #   62
    "DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY":                          "DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY",                         #   63
    "DCGM_FI_DEV_P2P_NVLINK_STATUS":                                "DCGM_FI_DEV_NVLINK_P2P_STATUS",                               #   64
    "DCGM_FI_DEV_COMPUTE_MODE":                                     "DCGM_FI_DEV_GPU_COMPUTE_MODE",                                #   65
    "DCGM_FI_DEV_PERSISTENCE_MODE":                                 "DCGM_FI_DEV_GPU_PERSISTENCE_MODE",                            #   66
    "DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR":                         "DCGM_FI_CUDA_GPU_VISIBLE_DEVICES",                            #   68
    "DCGM_FI_DEV_ECC_INFOROM_VER":                                  "DCGM_FI_DEV_INFOROM_ECC_VERSION",                             #   80
    "DCGM_FI_DEV_POWER_INFOROM_VER":                                "DCGM_FI_DEV_INFOROM_POWER_VERSION",                           #   81
    "DCGM_FI_DEV_INFOROM_IMAGE_VER":                                "DCGM_FI_DEV_INFOROM_IMAGE_VERSION",                           #   82
    "DCGM_FI_DEV_INFOROM_CONFIG_CHECK":                             "DCGM_FI_DEV_INFOROM_CHECKSUM",                                #   83
    "DCGM_FI_DEV_INFOROM_CONFIG_VALID":                             "DCGM_FI_DEV_INFOROM_VALID",                                   #   84
    "DCGM_FI_DEV_MEM_AFFINITY_0":                                   "DCGM_FI_DEV_MEMORY_AFFINITY_0",                               #   86
    "DCGM_FI_DEV_MEM_AFFINITY_1":                                   "DCGM_FI_DEV_MEMORY_AFFINITY_1",                               #   87
    "DCGM_FI_DEV_MEM_AFFINITY_2":                                   "DCGM_FI_DEV_MEMORY_AFFINITY_2",                               #   88
    "DCGM_FI_DEV_MEM_AFFINITY_3":                                   "DCGM_FI_DEV_MEMORY_AFFINITY_3",                               #   89
    "DCGM_FI_SYNC_BOOST":                                           "DCGM_FI_SYSTEM_GPU_SYNC_BOOST",                               #   91
    "DCGM_FI_DEV_AUTOBOOST":                                        "DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE",                           #  120
    "DCGM_FI_DEV_SUPPORTED_CLOCKS":                                 "DCGM_FI_DEV_CLOCKS_SUPPORTED",                                #  130
    "DCGM_FI_DEV_MEMORY_TEMP":                                      "DCGM_FI_DEV_MEMORY_TEMP_CELSIUS",                             #  140
    "DCGM_FI_DEV_GPU_TEMP":                                         "DCGM_FI_DEV_GPU_TEMP_CELSIUS",                                #  150
    "DCGM_FI_DEV_MEM_MAX_OP_TEMP":                                  "DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS",                      #  151
    "DCGM_FI_DEV_GPU_MAX_OP_TEMP":                                  "DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS",                         #  152
    "DCGM_FI_DEV_GPU_TEMP_LIMIT":                                   "DCGM_FI_DEV_GPU_TEMP_MARGIN_CELSIUS",                         #  153
    "DCGM_FI_DEV_POWER_USAGE":                                      "DCGM_FI_DEV_BOARD_POWER_WATTS",                               #  155
    "DCGM_FI_DEV_POWER_USAGE_INSTANT":                              "DCGM_FI_DEV_BOARD_POWER_RAW_WATTS",                           #  157
    "DCGM_FI_DEV_SLOWDOWN_TEMP":                                    "DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS",                       #  158
    "DCGM_FI_DEV_SHUTDOWN_TEMP":                                    "DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS",                       #  159
    "DCGM_FI_DEV_POWER_MGMT_LIMIT":                                 "DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS",               #  160
    "DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN":                             "DCGM_FI_DEV_BOARD_POWER_LIMIT_MIN_WATTS",                     #  161
    "DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX":                             "DCGM_FI_DEV_BOARD_POWER_LIMIT_MAX_WATTS",                     #  162
    "DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF":                             "DCGM_FI_DEV_BOARD_POWER_LIMIT_DEFAULT_WATTS",                 #  163
    "DCGM_FI_DEV_ENFORCED_POWER_LIMIT":                             "DCGM_FI_DEV_BOARD_POWER_LIMIT_ENFORCED_WATTS",                #  164
    "DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK":                     "DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK",              #  165
    "DCGM_FI_DEV_ENFORCED_POWER_PROFILE_MASK":                      "DCGM_FI_DEV_BOARD_POWER_PROFILE_ENFORCED_MASK",               #  166
    "DCGM_FI_DEV_VALID_POWER_PROFILE_MASK":                         "DCGM_FI_DEV_BOARD_POWER_PROFILE_SUPPORTED_MASK",              #  167
    "DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE":                        "DCGM_FI_DEV_FABRIC_MANAGER_ERROR",                            #  171
    "DCGM_FI_DEV_PSTATE":                                           "DCGM_FI_DEV_GPU_PSTATE",                                      #  190
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER":                              "DCGM_FI_DEV_PCIE_REPLAY_TOTAL",                               #  202
    "DCGM_FI_DEV_GPU_UTIL":                                         "DCGM_FI_DEV_GPU_UTIL_RATIO",                                  #  203
    "DCGM_FI_DEV_ACCOUNTING_DATA":                                  "DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS",                        #  205
    "DCGM_FI_DEV_XID_ERRORS":                                       "DCGM_FI_DEV_XID_ERROR",                                       #  230
    "DCGM_FI_DEV_FB_USED_PERCENT":                                  "DCGM_FI_DEV_FB_USED_RATIO",                                   #  254
    "DCGM_FI_DEV_C2C_LINK_COUNT":                                   "DCGM_FI_DEV_C2C_LINK_QUANTITY",                               #  285
    "DCGM_FI_DEV_ECC_CURRENT":                                      "DCGM_FI_DEV_ECC_MODE",                                        #  300
    "DCGM_FI_DEV_THRESHOLD_SRM":                                    "DCGM_FI_DEV_SRAM_EXCEEDED",                                   #  346
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_MAX":                       "DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX",                            #  385
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_HIGH":                      "DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH",                           #  386
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_PARTIAL":                   "DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL",                        #  387
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW":                       "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW",                            #  388
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE":                      "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE",                           #  389
    "DCGM_FI_DEV_RETIRED_SBE":                                      "DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL",                          #  390
    "DCGM_FI_DEV_RETIRED_DBE":                                      "DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL",                          #  391
    "DCGM_FI_DEV_RETIRED_PENDING":                                  "DCGM_FI_DEV_PAGE_RETIRED_PENDING",                            #  392
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS":                      "DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL",                   #  393
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS":                        "DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL",                     #  394
    "DCGM_FI_DEV_ROW_REMAP_FAILURE":                                "DCGM_FI_DEV_ROW_REMAP_FAILED",                                #  395

    # NVLink indexed fields grouped together by error type
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL",                  #  400
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL",                  #  401
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL",                  #  402
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL",                  #  403
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL",                  #  404
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL",                  #  405
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL",                  #  451
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL",                  #  452
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL",                  #  453
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9":                   "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL",                  #  454
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL",                 #  455
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL",                 #  456
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL",                 #  406
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL",                 #  407
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL",                 #  408
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL",                 #  481
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL",                 #  482
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17":                  "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL",                 #  483

    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL",                  #  410
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL",                  #  411
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL",                  #  412
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL",                  #  413
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL",                  #  414
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL",                  #  415
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL",                  #  457
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL",                  #  458
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL",                  #  459
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9":                   "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL",                  #  460
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL",                 #  461
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL",                 #  462
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL",                 #  416
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL",                 #  417
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL",                 #  418
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL",                 #  484
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL",                 #  485
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17":                  "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL",                 #  486

    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL",                    #  420
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL",                    #  421
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL",                    #  422
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL",                    #  423
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL",                    #  424
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL",                    #  425
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL",                    #  463
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL",                    #  464
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL",                    #  465
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9":                     "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL",                    #  466
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL",                   #  467
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL",                   #  468
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL",                   #  426
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL",                   #  427
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL",                   #  428
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL",                   #  487
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL",                   #  488
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17":                    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL",                   #  489

    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL",                  #  430
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL",                  #  431
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL",                  #  432
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL",                  #  433
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL",                  #  434
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL",                  #  435
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL",                  #  469
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL",                  #  470
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL",                  #  471
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9":                   "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL",                  #  472
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL",                 #  473
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL",                 #  474
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL",                 #  436
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL",                 #  437
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL",                 #  438
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL",                 #  491
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL",                 #  492
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17":                  "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL",                 #  493

    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL":                "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL",                     #  409
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL":                "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL",                     #  419
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL":                  "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL",                       #  429
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL":                "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL",                     #  439
    "DCGM_FI_DEV_GPU_NVLINK_ERRORS":                                "DCGM_FI_DEV_NVLINK_ERROR",                                    #  450
    "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC":                              "DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL",                          #  497
    "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY":                         "DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL",                           #  498
    "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY":                           "DCGM_FI_DEV_NVLINK_REPLAY_TOTAL",                             #  499

    "DCGM_FI_DEV_VIRTUAL_MODE":                                     "DCGM_FI_DEV_GPU_VIRTUAL_MODE",                                #  500
    "DCGM_FI_DEV_SUPPORTED_TYPE_INFO":                              "DCGM_FI_DEV_VGPU_SUPPORTED_INFO",                             #  501
    "DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS":                          "DCGM_FI_DEV_VGPU_CREATABLE_IDS",                              #  502
    "DCGM_FI_DEV_VGPU_INSTANCE_IDS":                                "DCGM_FI_DEV_VGPU_INSTANCE_INFO",                              #  503
    "DCGM_FI_DEV_VGPU_UTILIZATIONS":                                "DCGM_FI_DEV_VGPU_UTIL_INFO",                                  #  504
    "DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION":                     "DCGM_FI_DEV_VGPU_PROCESS_UTIL_INFO",                          #  505
    "DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS":                          "DCGM_FI_DEV_VGPU_SUPPORTED_IDS",                              #  509
    "DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE":                      "DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS",                    #  532
    "DCGM_FI_DEV_VGPU_VM_GPU_INSTANCE_ID":                          "DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID",                            #  534
    "DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TOTAL_SUCCESSFUL_EVENTS":    "DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SUCCESSFUL_TOTAL",          #  583
    "DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_SUCCESSFUL_RECOVERY_EVENTS": "DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_RECOVERY_SUCCESSFUL_TOTAL", #  584
    "DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_COUNTER":          "DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_TOTAL",           #  585
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RCV_CODES":                       "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL",                  #  586
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RCV_CODE_ERR":                    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL",            #  587
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RCV_UNCORRECTABLE_CODE":          "DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_UNCORRECTABLE_TOTAL",    #  588
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_XMIT_CODES":                      "DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL",                  #  589
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_XMIT_RETRY_CODES":                "DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_CODE_TOTAL",            #  590
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_XMIT_RETRY_EVENTS":               "DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL",           #  591
    "DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENTS":                     "DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENT_TOTAL",               #  592
    "DCGM_FI_DEV_NVSWITCH_POWER_VDD":                               "DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS",                        #  705
    "DCGM_FI_DEV_NVSWITCH_POWER_DVDD":                              "DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS",                       #  706
    "DCGM_FI_DEV_NVSWITCH_POWER_HVDD":                              "DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS",                       #  707
    "DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS":                      "DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL",                #  784
    "DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS":                    "DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERROR_TOTAL",              #  785
    "DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS":                        "DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERROR_TOTAL",                  #  786
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS":                         "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_TOTAL",                   #  787
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS":                         "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_TOTAL",                   #  788

    "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0":                  "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL",          #  805
    "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1":                  "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL",          #  806
    "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2":                  "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL",          #  807
    "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3":                  "DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL",          #  808

    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L0_TOTAL",                #  809
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L1_TOTAL",                #  810
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L2_TOTAL",                #  811
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L3_TOTAL",                #  812
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE4":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L4_TOTAL",                #  817
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE5":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L5_TOTAL",                #  818
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE6":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L6_TOTAL",                #  819
    "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE7":                   "DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L7_TOTAL",                #  820

    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L0_TOTAL",                #  813
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L1_TOTAL",                #  814
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L2_TOTAL",                #  815
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L3_TOTAL",                #  816
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE4":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L4_TOTAL",                #  821
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE5":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L5_TOTAL",                #  822
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE6":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L6_TOTAL",                #  823
    "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE7":                   "DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L7_TOTAL",                #  824

    "DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT":                     "DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS",                           #  858
    "DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN":              "DCGM_FI_DEV_NVSWITCH_TEMP_SLOWDOWN_CELSIUS",                  #  859
    "DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN":              "DCGM_FI_DEV_NVSWITCH_TEMP_SHUTDOWN_CELSIUS",                  #  860
    "DCGM_FI_DEV_NVSWITCH_PHYS_ID":                                 "DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID",                            #  863
    "DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS":                            "DCGM_FI_DEV_SXID_FATAL_ERROR",                                #  856
    "DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS":                        "DCGM_FI_DEV_SXID_NON_FATAL_ERROR",                            #  857
    "DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_ID":                     "DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID",                    #  876
    "DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_SID":                    "DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID",                   #  877
    "DCGM_FI_DEV_NVSWITCH_DEVICE_UUID":                             "DCGM_FI_DEV_NVSWITCH_UUID",                                   #  878
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE":                                "DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO",                           # 1001
    "DCGM_FI_PROF_SM_ACTIVE":                                       "DCGM_FI_PROF_SM_UTIL_RATIO",                                  # 1002
    "DCGM_FI_PROF_SM_OCCUPANCY":                                    "DCGM_FI_PROF_SM_OCCUPANCY_RATIO",                             # 1003
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE":                              "DCGM_FI_PROF_TENSOR_UTIL_RATIO",                              # 1004
    "DCGM_FI_PROF_DRAM_ACTIVE":                                     "DCGM_FI_PROF_DRAM_UTIL_RATIO",                                # 1005
    "DCGM_FI_PROF_PIPE_FP64_ACTIVE":                                "DCGM_FI_PROF_FP64_UTIL_RATIO",                                # 1006
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE":                                "DCGM_FI_PROF_FP32_UTIL_RATIO",                                # 1007
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE":                                "DCGM_FI_PROF_FP16_UTIL_RATIO",                                # 1008
    "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE":                         "DCGM_FI_PROF_IMMA_UTIL_RATIO",                                # 1013
    "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE":                         "DCGM_FI_PROF_HMMA_UTIL_RATIO",                                # 1014
    "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE":                         "DCGM_FI_PROF_DFMA_UTIL_RATIO",                                # 1015
    "DCGM_FI_PROF_PIPE_INT_ACTIVE":                                 "DCGM_FI_PROF_INT_UTIL_RATIO",                                 # 1016
    "DCGM_FI_PROF_NVDEC0_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_0_RATIO",                             # 1017
    "DCGM_FI_PROF_NVDEC1_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_1_RATIO",                             # 1018
    "DCGM_FI_PROF_NVDEC2_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_2_RATIO",                             # 1019
    "DCGM_FI_PROF_NVDEC3_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_3_RATIO",                             # 1020
    "DCGM_FI_PROF_NVDEC4_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_4_RATIO",                             # 1021
    "DCGM_FI_PROF_NVDEC5_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_5_RATIO",                             # 1022
    "DCGM_FI_PROF_NVDEC6_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_6_RATIO",                             # 1023
    "DCGM_FI_PROF_NVDEC7_ACTIVE":                                   "DCGM_FI_PROF_NVDEC_UTIL_7_RATIO",                             # 1024
    "DCGM_FI_PROF_NVJPG0_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_0_RATIO",                             # 1025
    "DCGM_FI_PROF_NVJPG1_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_1_RATIO",                             # 1026
    "DCGM_FI_PROF_NVJPG2_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_2_RATIO",                             # 1027
    "DCGM_FI_PROF_NVJPG3_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_3_RATIO",                             # 1028
    "DCGM_FI_PROF_NVJPG4_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_4_RATIO",                             # 1029
    "DCGM_FI_PROF_NVJPG5_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_5_RATIO",                             # 1030
    "DCGM_FI_PROF_NVJPG6_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_6_RATIO",                             # 1031
    "DCGM_FI_PROF_NVJPG7_ACTIVE":                                   "DCGM_FI_PROF_NVJPG_UTIL_7_RATIO",                             # 1032
    "DCGM_FI_PROF_NVOFA0_ACTIVE":                                   "DCGM_FI_PROF_NVOFA_UTIL_0_RATIO",                             # 1033
    "DCGM_FI_PROF_NVOFA1_ACTIVE":                                   "DCGM_FI_PROF_NVOFA_UTIL_1_RATIO",                             # 1034
    "DCGM_FI_DEV_CPU_TEMP_CURRENT":                                 "DCGM_FI_DEV_CPU_TEMP_CELSIUS",                                # 1110
    "DCGM_FI_DEV_CPU_TEMP_WARNING":                                 "DCGM_FI_DEV_CPU_TEMP_WARNING_CELSIUS",                        # 1111
    "DCGM_FI_DEV_CPU_TEMP_CRITICAL":                                "DCGM_FI_DEV_CPU_TEMP_CRITICAL_CELSIUS",                       # 1112
    "DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT":                           "DCGM_FI_DEV_CPU_POWER_WATTS",                                 # 1130
    "DCGM_FI_DEV_CPU_POWER_LIMIT":                                  "DCGM_FI_DEV_CPU_POWER_LIMIT_WATTS",                           # 1131
    "DCGM_FI_DEV_NVLINK_COUNT_TX_PACKETS":                          "DCGM_FI_DEV_NVLINK_TX_PACKET_TOTAL",                          # 1200
    "DCGM_FI_DEV_NVLINK_COUNT_TX_BYTES":                            "DCGM_FI_DEV_NVLINK_TX_BYTES_TOTAL",                           # 1201
    "DCGM_FI_DEV_NVLINK_COUNT_RX_PACKETS":                          "DCGM_FI_DEV_NVLINK_RX_PACKET_TOTAL",                          # 1202
    "DCGM_FI_DEV_NVLINK_COUNT_RX_BYTES":                            "DCGM_FI_DEV_NVLINK_RX_BYTES_TOTAL",                           # 1203
    "DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS":          "DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL",                # 1204
    "DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS":            "DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL",                  # 1205
    "DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS":                           "DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL",                           # 1206
    "DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS":                    "DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL",                    # 1207
    "DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS":                   "DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL",                   # 1208
    "DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS":         "DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL",                    # 1209
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS":     "DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL",                # 1211
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS":         "DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL",                    # 1212
    "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS":                "DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL",                     # 1213
    "DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS":                    "DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL",                    # 1214
    "DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER":                          "DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW",                           # 1215
    "DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT":                    "DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO",                         # 1216
    "DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER":                       "DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW",                        # 1217
    "DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT":                 "DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO",                      # 1218
    "DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS":                    "DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL",                    # 1219
    "DCGM_FI_DEV_NVLINK_ECC_DATA_ERROR_COUNT_TOTAL":                "DCGM_FI_DEV_NVLINK_ECC_ERROR_TOTAL",                          # 1220
    "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_STATUS":                "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS",             # 1307
    "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_MASK":                  "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK",               # 1308
    "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERR_SEVERITY":              "DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY",           # 1309
    "DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS":                    "DCGM_FI_DEV_PCIE_CORRECTABLE_ERROR_TOTAL",                    # 1501
    "DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG":                         "DCGM_FI_DEV_MEMORY_UNREPAIRABLE",                             # 1507
    "DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION":                          "DCGM_FI_DEV_GPU_RECOVERY_ACTION",                             # 1523
}
# autopep8: on


def _install_deprecated_alias_resolver():
    """Resolve deprecated field names with a DeprecationWarning, or raise AttributeError."""
    import sys
    import types
    import warnings

    class _DcgmFieldsModule(types.ModuleType):
        def __getattr__(self, name):
            if name in _DEPRECATED_ALIASES:
                if DCGM_DEPRECATED == 0:
                    raise AttributeError(
                        f"{name} is deprecated and DCGM_DEPRECATED=0")
                new_name = _DEPRECATED_ALIASES[name]
                warnings.warn(
                    f"{name} is deprecated, use {new_name} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(self, new_name)
            raise AttributeError(
                f"module {self.__name__!r} has no attribute {name!r}")

    sys.modules[__name__].__class__ = _DcgmFieldsModule


_install_deprecated_alias_resolver()


class struct_c_dcgm_field_meta_t(dcgm_structs._DcgmStructure):
    # struct_c_dcgm_field_meta_t structure
    pass  # opaque handle


dcgm_field_meta_t = POINTER(struct_c_dcgm_field_meta_t)


class _PrintableStructure(dcgm_structs._DcgmStructure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)

    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"

    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}

    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" + ', '.join(result) + ")"


# Provides access to functions from dcgm_agent_internal
dcgmFP = dcgm_structs._dcgmGetFunctionPointer

SHORTNAME_LENGTH = 10
UNIT_LENGTH = 4

# Structure to hold formatting information for values


class c_dcgm_field_output_format_t(_PrintableStructure):
    _fields_ = [
        ('shortName', c_char * SHORTNAME_LENGTH),
        ('unit', c_char * UNIT_LENGTH),
        ('width', c_short)
    ]


dcgm_field_output_format_t = POINTER(c_dcgm_field_output_format_t)

TAG_LENGTH = 48

# Structure to represent device information


class c_dcgm_field_meta_t(_PrintableStructure):
    _fields_ = [
        # version must always be first
        ('fieldId', c_short),
        ('fieldType', c_char),
        ('size', c_ubyte),
        ('tag', c_char * TAG_LENGTH),
        ('scope', c_int),
        ('nvmlFieldId', c_int),
        ('entityLevel', c_uint),
        ('valueFormat', dcgm_field_output_format_t)
    ]


# Class for maintaining properties for each sampling type like Power,
# Utilization and Clock.
class pySamplingProperties:
    '''
    The instance of this class is used to hold information related to each sampling event type.
    '''

    def __init__(
            self,
            name,
            sampling_type,
            sample_val_type,
            timeIntervalIdle,
            timeIntervalBoost,
            min_value,
            max_value):
        self.name = name
        self.sampling_type = sampling_type
        self.timeIntervalIdle = timeIntervalIdle
        self.timeIntervalBoost = timeIntervalBoost
        self.min_value = min_value
        self.max_value = max_value
        self.sample_val_type = sample_val_type


def DcgmFieldsInit():
    fn = dcgmFP("DcgmFieldsInit")
    ret = fn()
    assert ret == 0, "Got return %d from DcgmFieldsInit" % ret


def DcgmFieldGetById(fieldId):
    '''
    Get metadata for a field, given its fieldId

    :param fieldId: Field ID to get metadata for
    :return: c_dcgm_field_meta_t struct on success. None on error.
    '''
    DcgmFieldsInit()

    retVal = c_dcgm_field_meta_t()
    fn = dcgmFP("DcgmFieldGetById")
    fn.restype = POINTER(c_dcgm_field_meta_t)
    c_field_meta_ptr = fn(fieldId)
    if not c_field_meta_ptr:
        return None

    retVal = c_dcgm_field_meta_t()
    memmove(addressof(retVal), c_field_meta_ptr, sizeof(retVal))
    return retVal


def DcgmFieldGetByTag(tag):
    '''
    Get metadata for a field, given its string tag

    :param tag: Field tag to get metadata for. Example 'brand'
    :return: c_dcgm_field_meta_t struct on success. None on error.
    '''
    DcgmFieldsInit()

    retVal = c_dcgm_field_meta_t()
    fn = dcgmFP("DcgmFieldGetByTag")
    fn.restype = POINTER(c_dcgm_field_meta_t)
    c_field_meta_ptr = fn(c_char_p(tag.encode('utf-8')))
    if not c_field_meta_ptr:
        return None

    retVal = c_dcgm_field_meta_t()
    memmove(addressof(retVal), c_field_meta_ptr, sizeof(retVal))
    return retVal


def DcgmFieldGetTagById(fieldId):
    field = DcgmFieldGetById(fieldId)
    if field:
        return field.tag
    else:
        return None


def DcgmFieldGetIdByTag(fieldTag):
    field = DcgmFieldGetByTag(fieldTag)
    if field:
        return field.fieldId
    else:
        return None
