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

#ifndef _NSCQ_H_
#define _NSCQ_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

#define NSCQ_API_VERSION(major, minor, patch)                                       \
    (((((uint32_t)major) & 0xFFu) << 24u) | ((((uint32_t)minor) & 0xFFFu) << 12u) | \
     ((((uint32_t)patch) & 0xFFFu) << 0u))
#define NSCQ_API_VERSION_CODE_MAJOR(code) (((code) >> 24u) & 0xFFu)
#define NSCQ_API_VERSION_CODE_MINOR(code) (((code) >> 12u) & 0xFFFu)
#define NSCQ_API_VERSION_CODE_PATCH(code) (((code) >> 0u) & 0xFFFu)

#define NSCQ_API_VERSION_CODE \
    NSCQ_API_VERSION(2, 0, 0)
#define NSCQ_API_VERSION_DEVEL "g7aa2171"

extern const uint32_t nscq_api_version;
extern const char nscq_api_version_devel[];

// nscq_rc_t value ranges:
//  0          : success
//  1 to 127   : warnings (success, but with caveats)
//  -128 to -1 : errors
#define NSCQ_RC_SUCCESS                      (0)
#define NSCQ_RC_WARNING_RDT_INIT_FAILURE     (1)
#define NSCQ_RC_ERROR_NOT_IMPLEMENTED        (-1)
#define NSCQ_RC_ERROR_INVALID_UUID           (-2)
#define NSCQ_RC_ERROR_RESOURCE_NOT_MOUNTABLE (-3)
#define NSCQ_RC_ERROR_OVERFLOW               (-4)
#define NSCQ_RC_ERROR_UNEXPECTED_VALUE       (-5)
#define NSCQ_RC_ERROR_UNSUPPORTED_DRV        (-6)
#define NSCQ_RC_ERROR_DRV                    (-7)
#define NSCQ_RC_ERROR_TIMEOUT                (-8)
#define NSCQ_RC_ERROR_EXT                    (-127)
#define NSCQ_RC_ERROR_UNSPECIFIED            (-128)

// The pointer-cast-dereference is done so that these macros can also be used
// with the nscq_*_result_t types, which embed the result code as the first
// member of the result struct.
#ifdef __cplusplus
#define NSCQ_SUCCESS(result) (*(reinterpret_cast<nscq_rc_t*>(&(result))) == NSCQ_RC_SUCCESS)
#define NSCQ_WARNING(result) (*(reinterpret_cast<nscq_rc_t*>(&(result))) > NSCQ_RC_SUCCESS)
#define NSCQ_ERROR(result)   (*(reinterpret_cast<nscq_rc_t*>(&(result))) < NSCQ_RC_SUCCESS)
#else
#define NSCQ_SUCCESS(result) (*((nscq_rc_t*)&(result)) == NSCQ_RC_SUCCESS)
#define NSCQ_WARNING(result) (*((nscq_rc_t*)&(result)) > NSCQ_RC_SUCCESS)
#define NSCQ_ERROR(result)   (*((nscq_rc_t*)&(result)) < NSCQ_RC_SUCCESS)
#endif

#define _NSCQ_RESULT_TYPE(t, m) \
    typedef struct {            \
        nscq_rc_t rc;           \
        t m;                    \
    } nscq_##m##_result_t

typedef int8_t nscq_rc_t;
typedef struct nscq_session_st* nscq_session_t;
typedef struct nscq_observer_st* nscq_observer_t;
typedef struct nscq_writer_st* nscq_writer_t;

// All function callbacks (e.g., used for path observers) are passed using a single type.
// These are cast internally to the appropriate function types internally before use.
typedef void (*nscq_fn_t)(void);

// Convenience macro for casting a function pointer to the common nscq_fn_t type.
#ifdef __cplusplus
#define NSCQ_FN(fn) reinterpret_cast<nscq_fn_t>(fn)
#else
#define NSCQ_FN(fn) ((nscq_fn_t)&fn)
#endif

typedef struct {
    uint8_t bytes[16];
} nscq_uuid_t;

#define NSCQ_DRIVER_FABRIC_STATE_UNKNOWN         (-1) // unknown state
#define NSCQ_DRIVER_FABRIC_STATE_OFFLINE         (0)  // offline (No driver loaded)
#define NSCQ_DRIVER_FABRIC_STATE_STANDBY         (1)  // driver up, no FM
#define NSCQ_DRIVER_FABRIC_STATE_CONFIGURED      (2)  // driver up, FM up
#define NSCQ_DRIVER_FABRIC_STATE_MANAGER_TIMEOUT (3)  // driver up, FM timed out
#define NSCQ_DRIVER_FABRIC_STATE_MANAGER_ERROR   (4)  // driver up, FM error state

#define NSCQ_DEVICE_FABRIC_STATE_UNKNOWN     (-1) // unknown state
#define NSCQ_DEVICE_FABRIC_STATE_OFFLINE     (0)  // offline: No driver, no FM
#define NSCQ_DEVICE_FABRIC_STATE_STANDBY     (1)  // driver up, no FM, !blacklisted
#define NSCQ_DEVICE_FABRIC_STATE_CONFIGURED  (2)  // driver up, FM up, !blacklisted
#define NSCQ_DEVICE_FABRIC_STATE_BLACKLISTED (3)  // device is blacklisted

#define NSCQ_NVLINK_STATE_UNKNOWN (-1)
#define NSCQ_NVLINK_STATE_OFF     (0)
#define NSCQ_NVLINK_STATE_SAFE    (1)
#define NSCQ_NVLINK_STATE_ACTIVE  (2)
#define NSCQ_NVLINK_STATE_ERROR   (3)
#define NSCQ_NVLINK_STATE_SLEEP   (4)

typedef int8_t nscq_nvlink_state_t;

#define NSCQ_NVLINK_DEVICE_TYPE_GPU        (1)
#define NSCQ_NVLINK_DEVICE_TYPE_SWITCH     (2)

typedef uint8_t nscq_device_type_t;

#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_UNKNOWN      (-1)
#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_HIGH_SPEED_1 (0)
#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_SINGLE_LANE  (1)
#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_TRAINING     (2)
#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_SAFE_MODE    (3)
#define NSCQ_NVLINK_STATUS_SUBLINK_RX_STATE_OFF          (4)

typedef int8_t nscq_nvlink_rx_sublink_state_t;

#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_UNKNOWN (-1)
#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_HIGH_SPEED_1 (0)
#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_SINGLE_LANE  (1)
#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_TRAINING     (2)
#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_SAFE_MODE    (3)
#define NSCQ_NVLINK_STATUS_SUBLINK_TX_STATE_OFF          (4)

typedef int8_t nscq_nvlink_tx_sublink_state_t;

typedef struct {
    int16_t driver;
    int16_t device;
} nscq_fabric_state_t;

#define NSCQ_DEVICE_BLACKLIST_REASON_UNKNOWN                    (-1)
#define NSCQ_DEVICE_BLACKLIST_REASON_NONE                       (0)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND         (1)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_IN_BAND             (2)
#define NSCQ_DEVICE_BLACKLIST_REASON_MANUAL_PEER                (3)
#define NSCQ_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE         (4)
#define NSCQ_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE_PEER    (5)
#define NSCQ_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE        (6)
#define NSCQ_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE_PEER   (7)
#define NSCQ_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE      (8)
#define NSCQ_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE_PEER (9)

typedef int8_t nscq_blacklist_reason_t;

#define NSCQ_ARCH_SV10  (0)
#define NSCQ_ARCH_LR10  (1)
#define NSCQ_ARCH_LS10  (2)

typedef int8_t nscq_arch_t;

typedef struct {
    uint32_t domain;
    uint8_t bus;
    uint8_t device;
    uint8_t function;
} nscq_pcie_location_t;

typedef struct {
    char data[64];
} nscq_label_t;

typedef struct {
    uint8_t protocol_version;
    uint8_t link_width;
    uint32_t bandwidth;
} nscq_link_caps_t;

typedef struct {
    uint64_t rx; // Mibits
    uint64_t tx; // Mibits
} nscq_link_throughput_t;

typedef struct
{
    uint32_t error_value;
    uint64_t time;
} nscq_error_t;

typedef struct
{
    uint32_t last_updated;
    uint32_t errors_per_minute;
} nscq_link_max_correctable_error_rate;

typedef struct
{
    uint8_t  instance;
    uint32_t error;
    uint32_t timestamp;
    uint64_t count;
} nscq_link_error_t;

typedef struct
{
    uint64_t uncorrected_total;
    uint64_t corrected_total;
    uint32_t error_count;
} nscq_ecc_error_count_t;

typedef struct
{
    uint32_t sxid;
    uint8_t  link_id;
    uint32_t timestamp;
    uint8_t  address_valid;
    uint32_t address;
    uint32_t corrected_count;
    uint32_t uncorrected_count;
} nscq_ecc_error_entry_t;

typedef struct
{
    uint32_t sxid;
    uint32_t timestamp;
} nscq_sxid_error_entry_t;

typedef struct
{
    uint64_t low;
    uint64_t medium;
    uint64_t high;
    uint64_t panic;
    uint64_t count;
} nscq_vc_latency_t;

typedef struct {
    uint32_t vdd_mvolt;
    uint32_t dvdd_mvolt;
    uint32_t hvdd_mvolt;
} nscq_nvswitch_voltage_t;

typedef struct {
    uint32_t vdd_w;
    uint32_t dvdd_w;
    uint32_t hvdd_w;
} nscq_nvswitch_power_t;

typedef struct
{
    uint32_t freq_khz;
    uint32_t vcofreq_khz;
} nscq_nvlink_clock_info_t;

typedef struct
{
    uint32_t voltage_mvolt;
} nscq_nvlink_voltage_info_t;

typedef struct
{
    uint32_t iddq;
    uint32_t iddq_rev;
    uint32_t iddq_dvdd;
} nscq_nvlink_current_info_t;

typedef struct {
    uint32_t cages_mask;
    uint32_t modules_mask;
} nscq_cci_raw_cmis_presence_t;

typedef struct {
    uint64_t link_mask;
    uint64_t encoded_value;
} nscq_cci_raw_cmis_lane_mapping_t;

#define NSCQ_OSFP_MAX_LANE 8
typedef struct {
    // port[i] indicates nvswitch_port that occupies osfp lane i
    // Port value starts from 0. 0xFF value indicates no port.
    uint8_t port[NSCQ_OSFP_MAX_LANE];
} nscq_cci_osfp_lane_mapping_t;

#define NSCQ_OSFP_STR_SZ 16
typedef struct {
    uint8_t data[NSCQ_OSFP_STR_SZ];
} nscq_cci_osfp_str_t;

typedef struct {
    uint32_t tx_power[NSCQ_OSFP_MAX_LANE];
    uint32_t rx_power[NSCQ_OSFP_MAX_LANE];
    uint32_t tx_bias[NSCQ_OSFP_MAX_LANE];
} nscq_cci_osfp_lane_monitor_t;

typedef struct {
    int16_t temperature;
    uint32_t supply_volt;
} nscq_cci_osfp_module_monitor_t;

typedef struct {
    uint8_t major_rev;
    uint8_t minor_rev;
} nscq_cci_osfp_module_firmware_version_t;

typedef struct {
   uint8_t tx_input_eq[NSCQ_OSFP_MAX_LANE]; // CMIS Table 6-4 code
   uint8_t rx_output_pre_cursor[NSCQ_OSFP_MAX_LANE]; //CMIS Table 6-5
   uint8_t rx_output_post_cursor[NSCQ_OSFP_MAX_LANE]; //CMIS Table 6-5
} nscq_cci_osfp_signal_integrity_t;

typedef struct {
   uint8_t state[NSCQ_OSFP_MAX_LANE];
} nscq_cci_osfp_data_path_state_t;

#define NSCQ_CMIS_MAX_READ 128
typedef struct {
    uint8_t data[NSCQ_CMIS_MAX_READ];
} nscq_cci_raw_cmis_read_t;

#define NSCQ_TEMP_MAX_SENSOR 16
typedef struct {
   int32_t sensors[NSCQ_TEMP_MAX_SENSOR];
   uint8_t num_sensor;
} nscq_temperature_sensors_t;

_NSCQ_RESULT_TYPE(nscq_session_t, session);
_NSCQ_RESULT_TYPE(nscq_observer_t, observer);
_NSCQ_RESULT_TYPE(nscq_writer_t, writer);

nscq_rc_t nscq_uuid_to_label(const nscq_uuid_t*, nscq_label_t*, uint32_t);

#define NSCQ_SESSION_CREATE_MOUNT_DEVICES (0x1u)

nscq_session_result_t nscq_session_create(uint32_t);
void nscq_session_destroy(nscq_session_t);

nscq_rc_t nscq_session_mount(nscq_session_t, const nscq_uuid_t*, uint32_t);
void nscq_session_unmount(nscq_session_t, const nscq_uuid_t*);

nscq_rc_t nscq_session_path_observe(nscq_session_t, const char*, nscq_fn_t, void*, uint32_t);
nscq_rc_t nscq_session_path_write(nscq_session_t, const char*, void*, uint32_t);

#define NSCQ_SESSION_PATH_REGISTER_OBSERVER_WATCH (0x1u)

enum {
    NSCQ_SESSION_PATH_OBSERVE_FLAG_NONE = 0,
    NSCQ_SESSION_PATH_OBSERVE_FLAG_INPUT,
};

#define NSCQ_SESSION_PATH_OBSERVE_FLAG_MASK (0x3FFFFFFF)
#define NSCQ_SESSION_PATH_OBSERVE_FLAG_INPUT_TYPE_SHIFT 30
#define NSCQ_SESSION_PATH_OBSERVE_FLAG_TYPE(flag) (flag >> NSCQ_SESSION_PATH_OBSERVE_FLAG_INPUT_TYPE_SHIFT)

// Input
// 31-30: input type
//
// CMIS input
// 29-25: osfp (0-31)
// 24-23: bank (0-3)
// 22-15: page (0-255)
// 14-7:  byte offset (0-255)
// 6-0:   byte length (1-128)
#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_PACK(osfp, bank, page, offset, length) \
   ((NSCQ_SESSION_PATH_OBSERVE_FLAG_INPUT << NSCQ_SESSION_PATH_OBSERVE_FLAG_INPUT_TYPE_SHIFT) | \
    ((osfp & 0x1F) << 25) | ((bank & 0x3) << 23) | ((page & 0xFF) << 15) | ((offset & 0xFF) << 7) | ((length - 1) & 0x7F))

#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_GET_OSFP(input)    ((input >> 25) & 0x1F)
#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_GET_BANK(input)    ((input >> 23) & 0x3)
#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_GET_PAGE(input)    ((input >> 15) & 0xFF)
#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_GET_OFFSET(input)  ((input >> 7) & 0xFF)
#define NSCQ_SESSION_PATH_OBSERVE_CMIS_INPUT_GET_LENGTH(input)  ((input & 0x7F) + 1)

#define NSCQ_NVLINK_ERROR_THRESHOLD_CASE_ID_MIN 1
#define NSCQ_NVLINK_ERROR_THRESHOLD_CASE_ID_MAX 5

#define NSCQ_NVLINK_ERROR_THRESHOLD_THD_MAN_IDX 0
#define NSCQ_NVLINK_ERROR_THRESHOLD_THD_EXP_IDX 1
#define NSCQ_NVLINK_ERROR_THRESHOLD_TS_MAN_IDX  2
#define NSCQ_NVLINK_ERROR_THRESHOLD_TS_EXP_IDX  3
#define NSCQ_NVLINK_ERROR_THRESHOLD_MAX_IDX     4

typedef struct
{
    uint32_t errorThresholdId;
    bool bInterruptEn;
    bool bInterruptTrigerred;
    bool bReset;
} nscq_nvlink_error_threshold_t;

nscq_observer_result_t nscq_session_path_register_observer(nscq_session_t, const char*, nscq_fn_t,
                                                           void*, uint32_t);
void nscq_observer_deregister(nscq_observer_t);
nscq_rc_t nscq_observer_observe(nscq_observer_t, uint32_t);
nscq_rc_t nscq_session_observe(nscq_session_t, uint32_t);
nscq_rc_t nscq_session_watch(nscq_session_t, uint32_t, uint32_t);

#ifdef __cplusplus
}
#endif

#endif // _NSCQ_H_
