# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dcgm_fields import *
from dcgm_fields_internal import *
import sys

class CollectdMetadata:
    '''
    Constructor
    @params:
    name: string identifying the dcgm field. The field_name as opposed to
          field_id.Address:port of the host to connect. Defaults to localhost
    kind: collectd type string.
    used: a bool indicating whether or not the field is to be defined in
          a collectd types.db file when GenerateCollectdTypesDB() is called
          (generally if this file is run as a python3 mainline). We enumerate
          all the dcgm fields, but only generate types.db records for those
          supported at the current time. Others may or may not have correct
          collectd type definitions (generally  one might be a guage where it
          is more correctly a counter). The idea is that an intrepid user may
          enable generation of additional dcgm fields that they wish to collect
          but are not officially supported yet.
     '''
    def __init__(self, name, kind, used = False):
        self.name = name
        self.kind = kind
        self.used = used

# collectd metadata definition table.    
    
CollectdMetadataDict = { DCGM_FI_DRIVER_VERSION: None,
                         DCGM_FI_NVML_VERSION: None,
                         DCGM_FI_PROCESS_NAME: None,
                         DCGM_FI_CUDA_DRIVER_VERSION: CollectdMetadata("cuda_driver_version", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_COUNT: CollectdMetadata("device_count", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NAME: None,
                         DCGM_FI_DEV_BRAND: None,
                         DCGM_FI_DEV_NVML_INDEX: CollectdMetadata("nvml_index", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_SERIAL: None,
                         DCGM_FI_DEV_CPU_AFFINITY_0: CollectdMetadata("cpu_affinity_0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CPU_AFFINITY_1: CollectdMetadata("cpu_affinity_1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CPU_AFFINITY_2: CollectdMetadata("cpu_affinity_2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CPU_AFFINITY_3: CollectdMetadata("cpu_affinity_3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_UUID: None,
                         DCGM_FI_DEV_MINOR_NUMBER: CollectdMetadata("minor_number", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_OEM_INFOROM_VER: None,
                         DCGM_FI_DEV_ECC_INFOROM_VER: None,
                         DCGM_FI_DEV_POWER_INFOROM_VER: None,
                         DCGM_FI_DEV_INFOROM_IMAGE_VER: None,
                         DCGM_FI_DEV_INFOROM_CONFIG_CHECK: CollectdMetadata("inforom_config_checksum", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCI_BUSID: None,
                         DCGM_FI_DEV_PCI_COMBINED_ID: CollectdMetadata("pci_combined_id", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCI_SUBSYS_ID: CollectdMetadata("pci_subsys_id", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCIE_TX_THROUGHPUT: CollectdMetadata("pcie_tx_throughput", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_PCIE_RX_THROUGHPUT: CollectdMetadata("pcie_rx_throughput", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_PCIE_REPLAY_COUNTER: CollectdMetadata("pcie_replay_counter", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_SM_CLOCK: CollectdMetadata("sm_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_MEM_CLOCK: CollectdMetadata("memory_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_VIDEO_CLOCK: CollectdMetadata("video_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_APP_SM_CLOCK: CollectdMetadata("sm_app_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_APP_MEM_CLOCK: CollectdMetadata("mem_app_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_CLOCKS_EVENT_REASONS: CollectdMetadata("current_clocks_event_reasons", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_MAX_SM_CLOCK: CollectdMetadata("sm_max_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_MAX_MEM_CLOCK: CollectdMetadata("memory_max_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_MAX_VIDEO_CLOCK: CollectdMetadata("video_max_clock", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_AUTOBOOST: CollectdMetadata("autoboost", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_GPU_TEMP: CollectdMetadata("gpu_temp", "value:GAUGE:U:U", True),
                         DCGM_FI_DEV_MEM_MAX_OP_TEMP: CollectdMetadata("gpu_mem_max_op_temp", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_GPU_MAX_OP_TEMP: CollectdMetadata("gpu_max_op_temp", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_SLOWDOWN_TEMP: CollectdMetadata("slowdown_temp", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_SHUTDOWN_TEMP: CollectdMetadata("shutdown_temp", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_MGMT_LIMIT: CollectdMetadata("power_management_limit", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN: CollectdMetadata("power_management_limit_min", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX: CollectdMetadata("power_management_limit_max", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF: CollectdMetadata("power_management_limit_default", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_USAGE: CollectdMetadata("power_usage", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION: CollectdMetadata("total_energy_consumption", "value:GAUGE:0:U", True), # left as guage since zeroed at driver reload
                         DCGM_FI_DEV_ENFORCED_POWER_LIMIT: CollectdMetadata("enforced_power_limit", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PSTATE: CollectdMetadata("pstate", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_FAN_SPEED: CollectdMetadata("fan_speed", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_COMPUTE_MODE: CollectdMetadata("compute_mode", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PERSISTENCE_MODE: CollectdMetadata("persistance_mode", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_MIG_MODE: CollectdMetadata("mig_mode", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_GPM_SUPPORT: CollectdMetadata("gpm_support", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR: None,
                         DCGM_FI_DEV_MIG_MAX_SLICES: CollectdMetadata("mig_max_slices", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_CURRENT: CollectdMetadata("ecc", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_PENDING: CollectdMetadata("ecc_pending", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_VOL_TOTAL: CollectdMetadata("ecc_sbe_volatile_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_ECC_DBE_VOL_TOTAL: CollectdMetadata("ecc_dbe_volatile_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_ECC_SBE_AGG_TOTAL: CollectdMetadata("ecc_sbe_aggregate_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_ECC_DBE_AGG_TOTAL: CollectdMetadata("ecc_dbe_aggregate_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_ECC_SBE_VOL_L1: CollectdMetadata("ecc_sbe_volatile_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_VOL_L1: CollectdMetadata("ecc_dbe_volatile_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_VOL_L2: CollectdMetadata("ecc_sbe_volatile_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_VOL_L2: CollectdMetadata("ecc_dbe_volatile_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_VOL_DEV: CollectdMetadata("ecc_sbe_volatile_device", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_VOL_DEV: CollectdMetadata("ecc_dbe_volatile_device", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_VOL_REG: CollectdMetadata("ecc_sbe_volatile_register", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_VOL_REG: CollectdMetadata("ecc_dbe_volatile_register", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_VOL_TEX: CollectdMetadata("ecc_sbe_volatile_texture", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_VOL_TEX: CollectdMetadata("ecc_dbe_volatile_texture", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_AGG_L1: CollectdMetadata("ecc_sbe_aggregate_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_AGG_L1: CollectdMetadata("ecc_dbe_aggregate_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_AGG_L2: CollectdMetadata("ecc_sbe_aggregate_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_AGG_L2: CollectdMetadata("ecc_dbe_aggregate_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_AGG_DEV: CollectdMetadata("ecc_sbe_aggregate_device", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_AGG_DEV: CollectdMetadata("ecc_dbe_aggregate_device", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_AGG_REG: CollectdMetadata("ecc_sbe_aggregate_register", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_AGG_REG: CollectdMetadata("ecc_dbe_aggregate_register", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_SBE_AGG_TEX: CollectdMetadata("ecc_sbe_aggregate_texture", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ECC_DBE_AGG_TEX: CollectdMetadata("ecc_dbe_aggregate_texture", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_GPU_UTIL: CollectdMetadata("gpu_utilization", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_DEV_MEM_COPY_UTIL: CollectdMetadata("mem_copy_utilization", "value:GAUGE:0:100", True),
                         DCGM_FI_DEV_ENC_UTIL: CollectdMetadata("enc_utilization", "value:GAUGE:0:100"),
                         DCGM_FI_DEV_DEC_UTIL: CollectdMetadata("dec_utilization", "value:GAUGE:0:100"),
                         DCGM_FI_DEV_VBIOS_VERSION: None,
                         DCGM_FI_DEV_BAR1_TOTAL: CollectdMetadata("bar1_total", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_BAR1_USED: CollectdMetadata("bar1_used", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_BAR1_FREE: CollectdMetadata("bar1_free", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_FB_TOTAL: CollectdMetadata("fb_total", "value:GAUGE:0.0:U", True),
                         DCGM_FI_DEV_FB_FREE: CollectdMetadata("fb_free", "value:GAUGE:0.0:U", True),
                         DCGM_FI_DEV_FB_USED: CollectdMetadata("fb_used", "value:GAUGE:0.0:U", True),
                         DCGM_FI_DEV_FB_RESERVED: CollectdMetadata("fb_resv", "value:GAUGE:0.0:U", True),
                         DCGM_FI_DEV_VIRTUAL_MODE: CollectdMetadata("virtualization_mode", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_INSTANCE_IDS: None,
                         DCGM_FI_DEV_VGPU_UTILIZATIONS: None,
                         DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION: None,
                         DCGM_FI_DEV_VGPU_VM_ID: None,
                         DCGM_FI_DEV_VGPU_VM_NAME: None,
                         DCGM_FI_DEV_VGPU_TYPE: CollectdMetadata("vgpu_instance_type", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_UUID: None,
                         DCGM_FI_DEV_VGPU_DRIVER_VERSION: None,
                         DCGM_FI_DEV_VGPU_MEMORY_USAGE: CollectdMetadata("vgpu_instance_memory_usage", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE: CollectdMetadata("vgpu_instance_license_state", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_LICENSE_STATUS: CollectdMetadata("vgpu_instance_license_status", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT: CollectdMetadata("vgpu_instance_frame_rate_limit", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_PCI_ID: CollectdMetadata("vgpu_instance_pci_id", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_VGPU_ENC_STATS: None,
                         DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO: None,
                         DCGM_FI_DEV_VGPU_FBC_STATS: None,
                         DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO: None,
                         DCGM_FI_DEV_VGPU_VM_GPU_INSTANCE_ID: None,
                         DCGM_FI_DEV_SUPPORTED_TYPE_INFO: None,
                         DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS: None,
                         DCGM_FI_DEV_VGPU_TYPE_INFO: None,
                         DCGM_FI_DEV_VGPU_TYPE_NAME: None,
                         DCGM_FI_DEV_VGPU_TYPE_CLASS: None,
                         DCGM_FI_DEV_VGPU_TYPE_LICENSE: None,
                         DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS: None,
                         DCGM_FI_DEV_ENC_STATS: None,
                         DCGM_FI_DEV_FBC_STATS: None,
                         DCGM_FI_DEV_FBC_SESSIONS_INFO: None,
                         DCGM_FI_DEV_ACCOUNTING_DATA: None,
                         DCGM_FI_DEV_RETIRED_SBE: CollectdMetadata("retired_pages_sbe", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_RETIRED_DBE: CollectdMetadata("retired_pages_dbe", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_GRAPHICS_PIDS: None,
                         DCGM_FI_DEV_COMPUTE_PIDS: None,
                         DCGM_FI_DEV_SUPPORTED_CLOCKS: None,
                         DCGM_FI_SYNC_BOOST: None,
                         DCGM_FI_DEV_RETIRED_PENDING: CollectdMetadata("retired_pages_pending", "value:GAUGE:0:1", True), # boolean 1 = yes, 0 = no
                         DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS: CollectdMetadata("uncorrectable_remapped_rows", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS: CollectdMetadata("correctable_remapped_rows", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ROW_REMAP_FAILURE: CollectdMetadata("row_remap_failure", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_ROW_REMAP_PENDING: CollectdMetadata("row_remap_pending", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_INFOROM_CONFIG_VALID: CollectdMetadata("inforom_config_valid", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_XID_ERRORS: CollectdMetadata("xid_errors", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_PCIE_MAX_LINK_GEN: CollectdMetadata("pcie_max_link_gen", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH: CollectdMetadata("pcie_max_link_width", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCIE_LINK_GEN: CollectdMetadata("pcie_link_gen", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_PCIE_LINK_WIDTH: CollectdMetadata("pcie_link_width", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_POWER_VIOLATION: CollectdMetadata("power_violation", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_THERMAL_VIOLATION: CollectdMetadata("thermal_violation", "value:COUNTER:0:U", True),
                         DCGM_FI_GPU_TOPOLOGY_PCI: None,
                         DCGM_FI_GPU_TOPOLOGY_NVLINK: None,
                         DCGM_FI_GPU_TOPOLOGY_AFFINITY: None,
                         DCGM_FI_DEV_SYNC_BOOST_VIOLATION: CollectdMetadata("sync_boost_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_BOARD_LIMIT_VIOLATION: CollectdMetadata("board_limit_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_LOW_UTIL_VIOLATION: CollectdMetadata("low_util_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_RELIABILITY_VIOLATION: CollectdMetadata("reliability_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION: CollectdMetadata("app_clock_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION: CollectdMetadata("base_clock_violation", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES: CollectdMetadata("mem_util_samples", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_GPU_UTIL_SAMPLES: CollectdMetadata("gpu_util_samples", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0: CollectdMetadata("nvlink_flit_crc_error_count_l0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1: CollectdMetadata("nvlink_flit_crc_error_count_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2: CollectdMetadata("nvlink_flit_crc_error_count_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3: CollectdMetadata("nvlink_flit_crc_error_count_l3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4: CollectdMetadata("nvlink_flit_crc_error_count_l4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5: CollectdMetadata("nvlink_flit_crc_error_count_l5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL: CollectdMetadata("nvlink_flit_crc_error_count_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0: CollectdMetadata("nvlink_data_crc_error_count_l0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1: CollectdMetadata("nvlink_data_crc_error_count_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2: CollectdMetadata("nvlink_data_crc_error_count_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3: CollectdMetadata("nvlink_data_crc_error_count_l3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4: CollectdMetadata("nvlink_data_crc_error_count_l4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5: CollectdMetadata("nvlink_data_crc_error_count_l5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL: CollectdMetadata("nvlink_data_crc_error_count_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0: CollectdMetadata("nvlink_replay_error_count_l0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1: CollectdMetadata("nvlink_replay_error_count_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2: CollectdMetadata("nvlink_replay_error_count_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3: CollectdMetadata("nvlink_replay_error_count_l3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4: CollectdMetadata("nvlink_replay_error_count_l4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5: CollectdMetadata("nvlink_replay_error_count_l5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL: CollectdMetadata("nvlink_replay_error_count_total", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0: CollectdMetadata("nvlink_recovery_error_count_l0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1: CollectdMetadata("nvlink_recovery_error_count_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2: CollectdMetadata("nvlink_recovery_error_count_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3: CollectdMetadata("nvlink_recovery_error_count_l3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4: CollectdMetadata("nvlink_recovery_error_count_l4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5: CollectdMetadata("nvlink_recovery_error_count_l5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL: CollectdMetadata("nvlink_recovery_error_count_total", "value:COUNTER:0:U", True),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L0: CollectdMetadata("nvlink_bandwidth_l0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L1: CollectdMetadata("nvlink_bandwidth_l1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L2: CollectdMetadata("nvlink_bandwidth_l2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L3: CollectdMetadata("nvlink_bandwidth_l3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L4: CollectdMetadata("nvlink_bandwidth_l4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L5: CollectdMetadata("nvlink_bandwidth_l5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL: CollectdMetadata("nvlink_bandwidth_total", "value:GAUGE:0:U", True),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6: CollectdMetadata("nvlink_flit_crc_error_count_l6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7: CollectdMetadata("nvlink_flit_crc_error_count_l7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8: CollectdMetadata("nvlink_flit_crc_error_count_l8", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9: CollectdMetadata("nvlink_flit_crc_error_count_l9", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10: CollectdMetadata("nvlink_flit_crc_error_count_l10", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11: CollectdMetadata("nvlink_flit_crc_error_count_l11", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6: CollectdMetadata("nvlink_data_crc_error_count_l6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7: CollectdMetadata("nvlink_data_crc_error_count_l7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8: CollectdMetadata("nvlink_data_crc_error_count_l8", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9: CollectdMetadata("nvlink_data_crc_error_count_l9", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10: CollectdMetadata("nvlink_data_crc_error_count_l10", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11: CollectdMetadata("nvlink_data_crc_error_count_l11", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6: CollectdMetadata("nvlink_replay_error_count_l6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7: CollectdMetadata("nvlink_replay_error_count_l7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8: CollectdMetadata("nvlink_replay_error_count_l8", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9: CollectdMetadata("nvlink_replay_error_count_l9", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10: CollectdMetadata("nvlink_replay_error_count_l10", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11: CollectdMetadata("nvlink_replay_error_count_l11", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6: CollectdMetadata("nvlink_recovery_error_count_l6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7: CollectdMetadata("nvlink_recovery_error_count_l7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8: CollectdMetadata("nvlink_recovery_error_count_l8", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9: CollectdMetadata("nvlink_recovery_error_count_l9", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10: CollectdMetadata("nvlink_recovery_error_count_l10", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11: CollectdMetadata("nvlink_recovery_error_count_l11", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L6: CollectdMetadata("nvlink_bandwidth_l6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L7: CollectdMetadata("nvlink_bandwidth_l7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L8: CollectdMetadata("nvlink_bandwidth_l8", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L9: CollectdMetadata("nvlink_bandwidth_l9", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L10: CollectdMetadata("nvlink_bandwidth_l10", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVLINK_BANDWIDTH_L11: CollectdMetadata("nvlink_bandwidth_l11", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_MEMORY_TEMP: CollectdMetadata("memory_temp", "value:GAUGE:U:U", True),
                         DCGM_FI_DEV_GPU_NVLINK_ERRORS: CollectdMetadata("gpu_nvlink_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX: CollectdMetadata("nvswitch_link_bandwidth_tx", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX: CollectdMetadata("nvswitch_link_bandwidth_rx", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS: CollectdMetadata("nvswitch_link_fatal_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS: CollectdMetadata("nvswitch_link_non_fatal_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS: CollectdMetadata("nvswitch_link_recovery_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS: CollectdMetadata("nvswitch_link_flit_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS:CollectdMetadata("nvswitch_link_crc_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS:CollectdMetadata("nvswitch_link_ecc_errors", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0:CollectdMetadata("nvswitch_link_latency_low_vc0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1:CollectdMetadata("nvswitch_link_latency_low_vc1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2:CollectdMetadata("nvswitch_link_latency_low_vc2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3:CollectdMetadata("nvswitch_link_latency_low_vc3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0:CollectdMetadata("nvswitch_link_latency_medium_vc0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1:CollectdMetadata("nvswitch_link_latency_medium_vc1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2:CollectdMetadata("nvswitch_link_latency_medium_vc2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3:CollectdMetadata("nvswitch_link_latency_medium_vc3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0:CollectdMetadata("nvswitch_link_latency_high_vc0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1:CollectdMetadata("nvswitch_link_latency_high_vc1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2:CollectdMetadata("nvswitch_link_latency_high_vc2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3:CollectdMetadata("nvswitch_link_latency_high_vc3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0:CollectdMetadata("nvswitch_link_latency_panic_vc0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1:CollectdMetadata("nvswitch_link_latency_panic_vc1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2:CollectdMetadata("nvswitch_link_latency_panic_vc2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3:CollectdMetadata("nvswitch_link_latency_panic_vc3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0:CollectdMetadata("nvswitch_link_latency_count_vc0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1:CollectdMetadata("nvswitch_link_latency_count_vc1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2:CollectdMetadata("nvswitch_link_latency_count_vc2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3:CollectdMetadata("nvswitch_link_latency_count_vc3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0:CollectdMetadata("nvswitch_link_crc_errors_lane0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1:CollectdMetadata("nvswitch_link_crc_errors_lane1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2:CollectdMetadata("nvswitch_link_crc_errors_lane2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3:CollectdMetadata("nvswitch_link_crc_errors_lane3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE4:CollectdMetadata("nvswitch_link_crc_errors_lane4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE5:CollectdMetadata("nvswitch_link_crc_errors_lane5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE6:CollectdMetadata("nvswitch_link_crc_errors_lane6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE7:CollectdMetadata("nvswitch_link_crc_errors_lane7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0:CollectdMetadata("nvswitch_link_ecc_errors_lane0", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1:CollectdMetadata("nvswitch_link_ecc_errors_lane1", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2:CollectdMetadata("nvswitch_link_ecc_errors_lane2", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3:CollectdMetadata("nvswitch_link_ecc_errors_lane3", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE4:CollectdMetadata("nvswitch_link_ecc_errors_lane4", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE5:CollectdMetadata("nvswitch_link_ecc_errors_lane5", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE6:CollectdMetadata("nvswitch_link_ecc_errors_lane6", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE7:CollectdMetadata("nvswitch_link_ecc_errors_lane7", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS: CollectdMetadata("nvswitch_fatal_error", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS: CollectdMetadata("nvswitch_non_fatal_error", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT: CollectdMetadata("nvswitch_temperature_current", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN: CollectdMetadata("nvswitch_temperature_limit_slowdown", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN: CollectdMetadata("nvswitch_temperature_limit_shutdown", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX: CollectdMetadata("nvswitch_throughput_tx", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX: CollectdMetadata("nvswitch_throughput_rx", "value:GAUGE:U:U"),
                         DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY: CollectdMetadata("cuda_compute_capability", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_GR_ENGINE_ACTIVE: CollectdMetadata("gr_engine_active", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_PROF_SM_ACTIVE: CollectdMetadata("sm_active", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_PROF_SM_OCCUPANCY: CollectdMetadata("sm_occupancy", "value:GAUGE:0:U", True),
                         DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: CollectdMetadata("tensor_active", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_PROF_DRAM_ACTIVE: CollectdMetadata("dram_active", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_PROF_PIPE_FP64_ACTIVE: CollectdMetadata("fp64_active", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_PIPE_FP32_ACTIVE: CollectdMetadata("fp32_active", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_PIPE_FP16_ACTIVE: CollectdMetadata("fp16_active", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_PCIE_TX_BYTES: CollectdMetadata("pcie_tx_bytes", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_PCIE_RX_BYTES: CollectdMetadata("pcie_rx_bytes", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_NVLINK_TX_BYTES: CollectdMetadata("nvlink_tx_bytes", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_NVLINK_RX_BYTES: CollectdMetadata("nvlink_rx_bytes", "value:GAUGE:U:U"),
                         DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE: CollectdMetadata("tensor_imma_active", "value:GAUGE:0.0:1.0", True),
                         DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE: CollectdMetadata("tensor_hmma_active", "value:GAUGE:0.0:1.0", True),
                         }

__fieldDict = None

def GenerateCollectdTypesDB():
    length = max(map(lambda x: len(x.name) if x else 0, CollectdMetadataDict.values()))

    fmt = "{0:<" + str(length) + "}"
    fail = False

    for item in filter(None, CollectdMetadataDict.values()):
        item_list = item.kind.split(':')

        # Some rudimentary syntax checking.
        
        if len(item_list) != 4:
            sys.stderr.write('Item ' + item.name + ' has wrong number of collectd type fields - four required.\n')
            fail = True

        if item_list[1] not in ['GAUGE', 'COUNTER', 'DERIVE', 'ABSOLUTE']:
            sys.stderr.write('Item ' + item.name + ' should be one of GAUGE, COUNTER, DERIVE, ABSOLUTE.\n')
            fail = True

        # We check this so we can enumerate all dcgm fields for possible
        # inclusion, even if some are not (yet) formally supported.
        
        if item.used:
            print(fmt.format(item.name), item.kind)

    if fail:
        exit("Failed on db.types table syntax errors.\n")

def GetFieldByName(name):
    global __fieldDict

    if name.isnumeric():
        item = CollectdMetadataDict[int(name)]

        if item == None:
            return -1

        if item.used:
            return int(name)

        return -1

    if __fieldDict == None:
        __fieldDict = {}

        for key in CollectdMetadataDict:
            item = CollectdMetadataDict[key]

            if item != None:
                __fieldDict[item.name] = key;

    if name not in __fieldDict.keys():
        return -1

    return __fieldDict[name]

if __name__ == '__main__':
    GenerateCollectdTypesDB()
