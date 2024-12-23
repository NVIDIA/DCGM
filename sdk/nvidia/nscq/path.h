//
// Copyright 2020-2023 NVIDIA Corporation.  All rights reserved.
//
// NOTICE TO USER:
//
// This source code is subject to NVIDIA ownership rights under U.S. and
// international Copyright laws.  Users and possessors of this source code
// are hereby granted a nonexclusive, royalty-free license to use this code
// in individual and commercial software.
//
// NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
// CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
// IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
// REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
// OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
// OR PERFORMANCE OF THIS SOURCE CODE.
//
// U.S. Government End Users.   This source code is a "commercial item" as
// that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
// "commercial computer  software"  and "commercial computer software
// documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
// and is provided to the U.S. Government only as a commercial end item.
// Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
// 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
// source code with only those rights set forth herein.
//
// Any use of this source code in individual and commercial software must
// include, in the user documentation and internal comments to the code,
// the above Disclaimer and U.S. Government End Users Notice.
//

#ifndef _NSCQ_PATH_H_
#define _NSCQ_PATH_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define NSCQ_PATH(p) static_cast<const char*>(nscq_##p)
#else
#define NSCQ_PATH(p) nscq_##p
#endif

#define _NSCQ_DEF_PATH(name, path) static const char name[] = path

_NSCQ_DEF_PATH(nscq_nvswitch_drv_version, "/drv/nvswitch/version");
_NSCQ_DEF_PATH(nscq_nvswitch_device_uuid_path, "/drv/nvswitch/{device}/uuid");
_NSCQ_DEF_PATH(nscq_nvswitch_device_blacklisted, "/drv/nvswitch/{device}/blacklisted");
_NSCQ_DEF_PATH(nscq_nvswitch_device_pcie_location, "/drv/nvswitch/{device}/pcie/location");

_NSCQ_DEF_PATH(nscq_nvswitch_phys_id, "/{nvswitch}/id/phys_id");
_NSCQ_DEF_PATH(nscq_nvswitch_uuid, "/{nvswitch}/id/uuid");
_NSCQ_DEF_PATH(nscq_nvswitch_arch, "/{nvswitch}/id/arch");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_id, "/{nvswitch}/id/nvlink");
_NSCQ_DEF_PATH(nscq_nvswitch_firmware_version, "/{nvswitch}/version/firmware");
_NSCQ_DEF_PATH(nscq_nvswitch_inforom_version, "/{nvswitch}/version/inforom");
_NSCQ_DEF_PATH(nscq_nvswitch_pcie_location, "/{nvswitch}/pcie/location");
_NSCQ_DEF_PATH(nscq_nvswitch_fabric_status, "/{nvswitch}/status/fabric");
_NSCQ_DEF_PATH(nscq_nvswitch_reset_required, "/{nvswitch}/status/reset_required");
_NSCQ_DEF_PATH(nscq_nvswitch_power, "/{nvswitch}/status/power");
_NSCQ_DEF_PATH(nscq_nvswitch_voltage, "/{nvswitch}/status/voltage");
_NSCQ_DEF_PATH(nscq_nvswitch_temperature_current, "/{nvswitch}/status/temperature/current");
_NSCQ_DEF_PATH(nscq_nvswitch_temperature_sensors, "/{nvswitch}/status/temperature/sensors");
_NSCQ_DEF_PATH(nscq_nvswitch_temperature_limit_slowdown, "/{nvswitch}/status/temperature/limit_slowdown");
_NSCQ_DEF_PATH(nscq_nvswitch_temperature_limit_shutdown, "/{nvswitch}/status/temperature/limit_shutdown");
_NSCQ_DEF_PATH(nscq_nvswitch_error_fatal, "/{nvswitch}/status/error/fatal");
_NSCQ_DEF_PATH(nscq_nvswitch_error_nonfatal, "/{nvswitch}/status/error/nonfatal");
_NSCQ_DEF_PATH(nscq_nvswitch_error_history_nvlink, "/{nvswitch}/status/error/history/nvlink");
_NSCQ_DEF_PATH(nscq_nvswitch_error_history_sxid, "/{nvswitch}/status/error/history/sxid");
_NSCQ_DEF_PATH(nscq_nvswitch_error_history_ecc_count, "/{nvswitch}/status/error/history/ecc/count");
_NSCQ_DEF_PATH(nscq_nvswitch_error_history_ecc_log, "/{nvswitch}/status/error/history/ecc/entry");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_throughput_counters, "/{nvswitch}/nvlink/throughput_counters");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_raw_throughput_counters, "/{nvswitch}/nvlink/raw_throughput_counters");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_ports_num, "/{nvswitch}/nvlink/ports_num");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_ports_mask, "/{nvswitch}/nvlink/ports_mask");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_vcs_num, "/{nvswitch}/nvlink/vcs_num");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_clock_info, "/{nvswitch}/nvlink/clock_info");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_voltage_info, "/{nvswitch}/nvlink/voltage_info");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_current_info, "/{nvswitch}/nvlink/current_info");
_NSCQ_DEF_PATH(nscq_nvswitch_port_link_version, "/{nvswitch}/nvlink/{port}/link_version");
_NSCQ_DEF_PATH(nscq_nvswitch_port_sublink_width, "/{nvswitch}/nvlink/{port}/sublink_width");
_NSCQ_DEF_PATH(nscq_nvswitch_port_link_bandwidth, "/{nvswitch}/nvlink/{port}/link_bandwidth");
_NSCQ_DEF_PATH(nscq_nvswitch_port_link_data_rate, "/{nvswitch}/nvlink/{port}/link_data_rate");
_NSCQ_DEF_PATH(nscq_nvswitch_port_remote_device_type, "/{nvswitch}/nvlink/{port}/remote_device/type");
_NSCQ_DEF_PATH(nscq_nvswitch_port_remote_device_pcie_location, "/{nvswitch}/nvlink/{port}/remote_device/pcie/location");
_NSCQ_DEF_PATH(nscq_nvswitch_port_remote_device_link, "/{nvswitch}/nvlink/{port}/remote_device/id/link");
_NSCQ_DEF_PATH(nscq_nvswitch_port_remote_device_nvlink, "/{nvswitch}/nvlink/{port}/remote_device/id/nvlink");
_NSCQ_DEF_PATH(nscq_nvswitch_port_remote_device_uuid, "/{nvswitch}/nvlink/{port}/remote_device/id/uuid");
_NSCQ_DEF_PATH(nscq_nvswitch_port_link_status, "/{nvswitch}/nvlink/{port}/status/link");
_NSCQ_DEF_PATH(nscq_nvswitch_port_rx_sublink_state, "/{nvswitch}/nvlink/{port}/status/rx_sublink_state");
_NSCQ_DEF_PATH(nscq_nvswitch_port_tx_sublink_state, "/{nvswitch}/nvlink/{port}/status/tx_sublink_state");
_NSCQ_DEF_PATH(nscq_nvswitch_port_reset_required,
               "/{nvswitch}/nvlink/{port}/status/reset_required");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_fatal, "/{nvswitch}/nvlink/{port}/status/error/fatal");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_nonfatal, "/{nvswitch}/nvlink/{port}/status/error/nonfatal");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_replay_count, "/{nvswitch}/nvlink/{port}/status/error/replay_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_recovery_count, "/{nvswitch}/nvlink/{port}/status/error/recovery_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_flit_err_count, "/{nvswitch}/nvlink/{port}/status/error/flit_err_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_lane_crc_err_count_aggregate, "/{nvswitch}/nvlink/{port}/status/error/lane_crc_err_count_aggregate");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_lane_ecc_err_count_aggregate, "/{nvswitch}/nvlink/{port}/status/error/lane_ecc_err_count_aggregate");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_max_correctable_flit_crc_error_rate_daily,
               "/{nvswitch}/nvlink/{port}/status/error/max_correctable_flit_crc_error_rate/daily");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_max_correctable_flit_crc_error_rate_monthly,
               "/{nvswitch}/nvlink/{port}/status/error/max_correctable_flit_crc_error_rate/monthly");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_threshold, "/{nvswitch}/nvlink/{port}/error_threshold");
_NSCQ_DEF_PATH(nscq_nvswitch_port_write_error_threshold, "/{nvswitch}/nvlink/{port}/write/error_threshold");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_crc_err_count, "/{nvswitch}/nvlink/{port}/{lane}/status/error/crc_err_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_ecc_err_count, "/{nvswitch}/nvlink/{port}/{lane}/status/error/ecc_err_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_max_correctable_lane_crc_error_rate_daily,
               "/{nvswitch}/nvlink/{port}/{lane}/status/error/max_correctable_lane_crc_error_rate/daily");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_max_correctable_lane_crc_error_rate_monthly,
               "/{nvswitch}/nvlink/{port}/{lane}/status/error/max_correctable_lane_crc_error_rate/monthly");
_NSCQ_DEF_PATH(nscq_nvswitch_port_vc_latency, "/{nvswitch}/nvlink/{port}/{vc}/latency");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_port_throughput_counters, "/{nvswitch}/nvlink/{port}/throughput_counters");
_NSCQ_DEF_PATH(nscq_nvswitch_nvlink_port_raw_throughput_counters, "/{nvswitch}/nvlink/{port}/raw_throughput_counters");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_raw_cmis_presence, "/{nvswitch}/cci/raw/cmis_presence");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_raw_cmis_lane_mapping, "/{nvswitch}/cci/raw/cmis_lane_mapping");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_raw_cmis_read, "/{nvswitch}/cci/raw/cmis_read");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_presence, "/{nvswitch}/cci/{osfp}/presence");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_num_osfp, "/{nvswitch}/cci/num_osfp");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_lane_mapping, "/{nvswitch}/cci/{osfp}/lane_mapping");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_cable_type, "/{nvswitch}/cci/{osfp}/cable_type");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_module_media_type, "/{nvswitch}/cci/{osfp}/module_media_type");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_vendor_name, "/{nvswitch}/cci/{osfp}/vendor_name");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_cable_length, "/{nvswitch}/cci/{osfp}/cable_length");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_part_number, "/{nvswitch}/cci/{osfp}/part_number");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_revision_number, "/{nvswitch}/cci/{osfp}/revision_number");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_serial_number, "/{nvswitch}/cci/{osfp}/serial_number");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_lane_monitor, "/{nvswitch}/cci/{osfp}/lane_monitor");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_module_monitor, "/{nvswitch}/cci/{osfp}/module_monitor");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_module_firmware_version, "/{nvswitch}/cci/{osfp}/module_firmware_version");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_signal_integrity, "/{nvswitch}/cci/{osfp}/signal_integrity");
_NSCQ_DEF_PATH(nscq_nvswitch_cci_osfp_data_path_state, "/{nvswitch}/cci/{osfp}/data_path_state");

#undef _NSCQ_DEF_PATH

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _NSCQ_PATH_H_
