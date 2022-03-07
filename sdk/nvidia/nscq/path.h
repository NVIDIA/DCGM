//
// Copyright 2021 NVIDIA Corporation.  All rights reserved.
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

#define _NSCQ_DEF_PATH(name, path) const char name[] = path

_NSCQ_DEF_PATH(nscq_nvswitch_drv_version, "/drv/nvswitch/version");
_NSCQ_DEF_PATH(nscq_nvswitch_device_uuid_path, "/drv/nvswitch/{device}/uuid");
_NSCQ_DEF_PATH(nscq_nvswitch_device_blacklisted, "/drv/nvswitch/blacklisted");
_NSCQ_DEF_PATH(nscq_nvswitch_device_pcie_location, "/drv/nvswitch/pcie/location");

_NSCQ_DEF_PATH(nscq_nvswitch_phys_id, "/{nvswitch}/id/phys_id");
_NSCQ_DEF_PATH(nscq_nvswitch_uuid, "/{nvswitch}/id/uuid");
_NSCQ_DEF_PATH(nscq_nvswitch_firmware_version, "/{nvswitch}/version/firmware");
_NSCQ_DEF_PATH(nscq_nvswitch_pcie_location, "/{nvswitch}/pcie/location");
_NSCQ_DEF_PATH(nscq_nvswitch_fabric_status, "/{nvswitch}/status/fabric");
_NSCQ_DEF_PATH(nscq_nvswitch_reset_required, "/{nvswitch}/status/reset_required");
_NSCQ_DEF_PATH(nscq_nvswitch_error_fatal, "/{nvswitch}/status/error/fatal");
_NSCQ_DEF_PATH(nscq_nvswitch_port_link_status, "/{nvswitch}/nvlink/{port}/status/link");
_NSCQ_DEF_PATH(nscq_nvswitch_port_reset_required,
               "/{nvswitch}/nvlink/{port}/status/reset_required");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_fatal, "/{nvswitch}/nvlink/{port}/status/error/fatal");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_replay_count, "/{nvswitch}/nvlink/{port}/status/error/replay_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_recovery_count, "/{nvswitch}/nvlink/{port}/status/error/recovery_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_flit_err_count, "/{nvswitch}/nvlink/{port}/status/error/flit_err_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_lane_crc_err_count_aggregate, "/{nvswitch}/nvlink/{port}/status/error/lane_crc_err_count_aggregate");
_NSCQ_DEF_PATH(nscq_nvswitch_port_error_lane_ecc_err_count_aggregate, "/{nvswitch}/nvlink/{port}/status/error/lane_ecc_err_count_aggregate");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_crc_err_count, "/{nvswitch}/nvlink/{port}/{lane}/status/error/crc_err_count");
_NSCQ_DEF_PATH(nscq_nvswitch_port_lane_ecc_err_count, "/{nvswitch}/nvlink/{port}/{lane}/status/error/ecc_err_count");

#undef _NSCQ_DEF_PATH

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _NSCQ_PATH_H_
