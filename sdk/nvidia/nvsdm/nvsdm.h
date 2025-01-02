//
// Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#ifndef __NVSDM_H__
#define __NVSDM_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVSDM_CURR_VERSION_MAJOR 1
#define NVSDM_CURR_VERSION_MINOR 1
#define NVSDM_CURR_VERSION_PATCH 0

#define NVSDM_VERSION(major, minor, patch) \
    ( (uint64_t)(major) << 32 | (uint64_t)(minor) << 16 | (uint64_t)patch )

#define NVSDM_CURR_VERSION NVSDM_VERSION(NVSDM_CURR_VERSION_MAJOR, \
                                       NVSDM_CURR_VERSION_MINOR, \
                                       NVSDM_CURR_VERSION_PATCH)

#define NVSDM_GET_MAJOR_NUMBER(version) ((uint64_t)version >> 32 & 0xffff)
#define NVSDM_GET_MINOR_NUMBER(version) ((uint64_t)version >> 16 & 0xffff)
#define NVSDM_GET_PATCH_NUMBER(version) ((uint64_t)version & 0xffff)

#define NVSDM_MAX_DEVICES 128        /* The maximum number of devices discoverable */

#define NVSDM_INFO_ARRAY_SIZE 64
#define NVSDM_DESC_ARRAY_SIZE (NVSDM_INFO_ARRAY_SIZE + 1)
#define NVSDM_DEV_INFO_ARRAY_SIZE (128 + 1)
#define NVSDM_PORT_INFO_ARRAY_SIZE (128 + 1)

#define NVSDM_SMP_DATA_SIZE 64 // IB_SMP_DATA_SIZE


/**
 * Special port number for the management port
 */
#define NVSDM_MANAGEMENT_PORT_NUMBER 0xFFFFFFFF

#define NVSDM_STRUCT_VERSION(pre, ver) ( (ver) << 24 |  (sizeof(pre ## _v ## ver ## _t)) )

/**
 * NVSDM Return Types
*/
typedef enum
{
    NVSDM_SUCCESS                              = 0, //!< The operation was successful
    NVSDM_ERROR_UNINITIALIZED                  = 1, //!< NVML was not first initialized with nvmlInit()
    NVSDM_ERROR_NOT_SUPPORTED                  = 2, //!< The requested operation is not available on target device
    NVSDM_ERROR_INVALID_ARG                    = 3, //!< A supplied argument is invalid
    NVSDM_ERROR_INSUFFICIENT_SIZE              = 4, //!< An input argument is not large enough
    NVSDM_ERROR_VERSION_NOT_SUPPORTED          = 5, //!< API version is not supported
    NVSDM_ERROR_MEMORY                         = 6, //!< Insufficient memory
    NVSDM_ERROR_DEVICE_DISCOVERY_FAILURE       = 7, //!< Device discovery failed
    NVSDM_ERROR_LIBRARY_LOAD                   = 8, //!< Library load failed
    NVSDM_ERROR_FUNCTION_NOT_FOUND             = 9, //!< Function not found
    NVSDM_ERROR_INVALID_CTR                    = 10, //!< Invalid counter
    NVSDM_ERROR_TELEMETRY_READ                 = 11, //!< Telemetry read failed
    NVSDM_ERROR_DEVICE_NOT_FOUND               = 12, //!< Device not found
    NVSDM_ERROR_UMAD_INIT                      = 13, //!< UMAD init failure
    NVSDM_ERROR_UMAD_LIB_CALL                  = 14, //!< UMAD library call failed
    NVSDM_ERROR_MAD_LIB_CALL                   = 15, //!< UMAD library call failed
    NVSDM_ERROR_NLSOCKET_OPEN_FAILED           = 16,
    NVSDM_ERROR_NLSOCKET_BIND_FAILED           = 17,
    NVSDM_ERROR_NLSOCKET_SEND_FAILED           = 18,
    NVSDM_ERROR_NLSOCKET_RECV_FAILED           = 19,
    NVSDM_ERROR_FILE_OPEN_FAILED               = 20,
    NVSDM_ERROR_INSUFFICIENT_PERMISSION        = 21, //!< Insufficient permissions
    NVSDM_ERROR_UNKNOWN                        = 0xFFFFFFFF,
} nvsdmRet_t;

/**
 * NVSDM Log Levels
 */
enum nvsdmLogLevel
{
    NVSDM_LOG_LEVEL_FATAL         = 0, //!< Log fatal errors
    NVSDM_LOG_LEVEL_ERROR         = 1, //!< Log all errors
    NVSDM_LOG_LEVEL_WARN          = 2, //!< Log all warnings
    NVSDM_LOG_LEVEL_DEBUG         = 3, //!< Log all debug messages
    NVSDM_LOG_LEVEL_INFO          = 4, //!< Log all info messages
    NVSDM_LOG_LEVEL_NONE          = 0xFFFFFFFF, //!< Log none
};

/**
 * NVSDM Device Types
 */
enum nvsdmDevType
{
    NVSDM_DEV_TYPE_CA            = 1, //!< Channel adapter device
    NVSDM_DEV_TYPE_SWITCH        = 2, //!< Switch device
    NVSDM_DEV_TYPE_ROUTER        = 3, //!< Routere device
    NVSDM_DEV_TYPE_RNIC          = 4, //< RNIC device
    NVSDM_DEV_TYPE_GPU           = 5, //< GPU device
    NVSDM_DEV_TYPE_MAX           = NVSDM_DEV_TYPE_GPU,
    NVSDM_DEV_TYPE_NONE          = 0xFFFFFFFF,
};

/**
 * Opaque handle for port descriptors
 */
typedef struct nvsdmPort *nvsdmPort_t;

/**
 * Opaque handle for device/node descriptors
 */
typedef struct nvsdmDevice *nvsdmDevice_t;

/**
 * Opaque handle for port iterators
 */
typedef struct nvsdmPortIter *nvsdmPortIter_t;

/**
 * Opaque handle for device iterators
 */
typedef struct nvsdmDeviceIter *nvsdmDeviceIter_t;

/**
 * NVSDM telemetry type
 */
typedef enum
{
    NVSDM_TELEM_TYPE_PORT        = 1,    //!< Basic port telemetry
    NVSDM_TELEM_TYPE_PLATFORM    = 2,    //!< Telemetry from QM3 ASIC sensors
    NVSDM_TELEM_TYPE_CUSTOM      = 3,    //!< Custom counter
    NVSDM_TELEM_TYPE_CONNECTX    = 4,    //!< ConnectX telemetry counter
    NVSDM_TELEM_TYPE_MAX         = NVSDM_TELEM_TYPE_CONNECTX,

    NVSDM_TELEM_TYPE_NONE        = 99,
} nvsdmTelemType_t;

/**
 * NVSDM counter IDs
 */
typedef enum
{
    NVSDM_PORT_TELEM_CTR_RCV_PKTS                                                          = 1,
    NVSDM_PORT_TELEM_CTR_RCV_DATA                                                          = 2,
    NVSDM_PORT_TELEM_CTR_MCAST_RCV_PKTS                                                    = 3,
    NVSDM_PORT_TELEM_CTR_UCAST_RCV_PKTS                                                    = 4,
    NVSDM_PORT_TELEM_CTR_MFRMD_PKTS                                                        = 5,
    NVSDM_PORT_TELEM_CTR_VL15_DROPPED                                                      = 6,
    NVSDM_PORT_TELEM_CTR_RCV_ERR                                                           = 7,
    NVSDM_PORT_TELEM_CTR_XMIT_PKTS                                                         = 8,
    //NVSDM_PORT_TELEM_CTR_XMIT_PKTS_VL15                                                  = 9,
    NVSDM_PORT_TELEM_CTR_XMIT_DATA                                                         = 10,
    //NVSDM_PORT_TELEM_CTR_XMIT_DATA_VL15                                                  = 11,
    NVSDM_PORT_TELEM_CTR_UCAST_XMIT_PKTS                                                   = 12,
    NVSDM_PORT_TELEM_CTR_MCAST_XMIT_PKTS                                                   = 13,
    NVSDM_PORT_TELEM_CTR_XMIT_DISCARD                                                      = 14,
    NVSDM_PORT_TELEM_CTR_NBR_MTU_DISCARDS                                                  = 15,
    NVSDM_PORT_TELEM_CTR_SYM_ERR                                                           = 16,
    NVSDM_PORT_TELEM_CTR_LNK_ERR_REC_CTR                                                   = 17,
    NVSDM_PORT_TELEM_CTR_LNK_DWND_CTR                                                      = 18,
    NVSDM_PORT_TELEM_CTR_RCV_RMT_PHY_ERR                                                   = 19,
    NVSDM_PORT_TELEM_CTR_RCV_SWTCH_REL_ERR                                                 = 20,
    NVSDM_PORT_TELEM_CTR_QP1_DROPPED                                                       = 21,
    NVSDM_PORT_TELEM_CTR_XMIT_WAIT                                                         = 22,
    NVSDM_PORT_TELEM_CTR_PDDR_OP_INFO_LINK_SPEED_ACTIVE                                    = 23,
    NVSDM_PORT_TELEM_CTR_PDDR_OP_INFO_LINK_WIDTH_ACTIVE                                    = 24,

    /* Group 0x12 counters */
    NVSDM_PORT_TELEM_CTR_GRP_X12_SYM_ERR                                                   = 25,
    NVSDM_PORT_TELEM_CTR_GRP_X12_SYNC_HDR_ERR                                              = 26,
    NVSDM_PORT_TELEM_CTR_GRP_X12_EDPL_BIP_ERR_LANE_0                                       = 27,
    NVSDM_PORT_TELEM_CTR_GRP_X12_EDPL_BIP_ERR_LANE_1                                       = 28,
    NVSDM_PORT_TELEM_CTR_GRP_X12_EDPL_BIP_ERR_LANE_2                                       = 29,
    NVSDM_PORT_TELEM_CTR_GRP_X12_EDPL_BIP_ERR_LANE_3                                       = 30,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_BLKS_LANE_0                                    = 31,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_BLKS_LANE_1                                    = 32,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_BLKS_LANE_2                                    = 33,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_BLKS_LANE_3                                    = 34,
    NVSDM_PORT_TELEM_CTR_GRP_X12_RS_FEC_COR_BLKS                                           = 35,
    NVSDM_PORT_TELEM_CTR_GRP_X12_RS_FEC_UNCOR_BLKS                                         = 36,
    NVSDM_PORT_TELEM_CTR_GRP_X12_RS_FEC_NO_ERR_BLKS                                        = 37,
    NVSDM_PORT_TELEM_CTR_GRP_X12_RS_FEC_COR_SYM_TOTAL                                      = 38,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_SYM_LANE_0                                     = 39,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_SYM_LANE_1                                     = 40,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_SYM_LANE_2                                     = 41,
    NVSDM_PORT_TELEM_CTR_GRP_X12_FC_FEC_COR_SYM_LANE_3                                     = 42,
    NVSDM_PORT_TELEM_CTR_GRP_X12_SUCCESSFUL_RECOV_EVENTS                                   = 43,

    /* Group 0x16 counters */
    NVSDM_PORT_TELEM_CTR_GRP_X16_TIME_SINCE_LAST_CLEAR                                     = 44,
    NVSDM_PORT_TELEM_CTR_GRP_X16_PHY_RECV_BITS                                             = 45,
    NVSDM_PORT_TELEM_CTR_GRP_X16_PHY_SYM_ERR                                               = 46,
    NVSDM_PORT_TELEM_CTR_GRP_X16_PHY_CORR_BITS                                             = 47,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_0                                            = 48,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_1                                            = 49,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_2                                            = 50,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_3                                            = 51,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_4                                            = 52,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_5                                            = 53,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_6                                            = 54,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_ERR_LANE_7                                            = 55,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER                                                   = 56,
    NVSDM_PORT_TELEM_CTR_GRP_X16_EFF_BER                                                   = 57,
    NVSDM_PORT_TELEM_CTR_GRP_X16_SYM_BER                                                   = 58,
    NVSDM_PORT_TELEM_CTR_GRP_X16_PHY_EFFECTIVE_ERR                                         = 59,

    /* PPCNT Group 0x20 counters */
    NVSDM_PORT_TELEM_CTR_GRP_X20_SYM_ERR_CTR_TO_BE_REMOVED                                 = 60, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_LINK_ERR_REC_CTR                                          = 61,
    NVSDM_PORT_TELEM_CTR_GRP_X20_LINK_DWNED_CTR                                            = 62,
    NVSDM_PORT_TELEM_CTR_GRP_X20_PORT_RCV_ERR                                              = 63,
    NVSDM_PORT_TELEM_CTR_GRP_X20_PORT_RCV_REM_PHY_ERR                                      = 64,
    NVSDM_PORT_TELEM_CTR_GRP_X20_RCV_SWTCH_REL_ERR_TO_BE_REMOVED                           = 65, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_XMIT_DISCARD_TO_BE_REMOVED                                = 66, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_XMIT_CNSTR_ERR                                            = 67,
    NVSDM_PORT_TELEM_CTR_GRP_X20_RCV_CNSTR_ERR                                             = 68,
    NVSDM_PORT_TELEM_CTR_GRP_X20_LOCAL_LINK_INTEG_ERR                                      = 69,
    NVSDM_PORT_TELEM_CTR_GRP_X20_EXSV_BUFF_OVRUN_ERR                                       = 70,
    NVSDM_PORT_TELEM_CTR_GRP_X20_VL15_DROPPED_TO_BE_REMOVED                                = 71, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_XMIT_DATA_TO_BE_REMOVED                                   = 72, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_RCV_DATA_TO_BE_REMOVED                                    = 73, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_XMIT_PKTS_TO_BE_REMOVED                                   = 74, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_RCV_PKTS_TO_BE_REMOVED                                    = 75, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X20_XMIT_WAIT_TO_BE_REMOVED                                   = 76, /* Already collected via regular MAD */

    /* Group 0x22 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X22_RCV_CODES                                                 = 77,
    NVSDM_PORT_TELEM_CTR_GRP_X22_RCV_CODES_ERR                                             = 78,
    NVSDM_PORT_TELEM_CTR_GRP_X22_RCV_UNCORRECTABLE_CODE                                    = 79,
    NVSDM_PORT_TELEM_CTR_GRP_X22_XMIT_CODES                                                = 80,
    NVSDM_PORT_TELEM_CTR_GRP_X22_XMIT_RETRY_CODES                                          = 81,
    NVSDM_PORT_TELEM_CTR_GRP_X22_XMIT_RETRY_EVENTS                                         = 82,
    NVSDM_PORT_TELEM_CTR_GRP_X22_SYNC_EVENTS                                               = 83,
    NVSDM_PORT_TELEM_CTR_GRP_X22_CODES_LOSS                                                = 84,
    NVSDM_PORT_TELEM_CTR_GRP_X22_XMIT_RETRY_EVTS_WTHN_TSEC_MAX                             = 85,
    NVSDM_PORT_TELEM_CTR_GRP_X22_TIME_SINCE_LAST_CLEAR                                     = 86,

    /* Group 0x23 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_0                                             = 87,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_1                                             = 88,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_2                                             = 89,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_3                                             = 90,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_4                                             = 91,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_5                                             = 92,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_6                                             = 93,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_7                                             = 94,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_8                                             = 95,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_9                                             = 96,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_10                                            = 97,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_11                                            = 98,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_12                                            = 99,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_13                                            = 100,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_14                                            = 101,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_15                                            = 102,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_16                                            = 103,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_17                                            = 104,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_18                                            = 105,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_19                                            = 106,
    NVSDM_PORT_TELEM_CTR_GRP_X23_PHY_RS_HIST_20                                            = 107,

    /* PPCNT Group 0x25 counters */
    NVSDM_PORT_TELEM_CTR_GRP_X25_TIME_SINCE_LAST_CLEAR                                     = 108,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_UCAST_XMIT_PKTS_TO_BE_REMOVED                        = 109, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_UCAST_RCV_PKTS_TO_BE_REMOVED                         = 110, /* Already collected via regular MAD */
    NVSDM_PORT_TELEM_CTR_GRP_X25_SYNC_HDR_ERR_CTR                                          = 111,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_LOC_PHY_ERR                                          = 112,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_MFRMD_PKT_ERR                                        = 113,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_BUFF_OVRUN_ERR                                       = 114,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_DLID_MAP_ERR                                         = 115,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_VL_MAP_ERR                                           = 116,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_LOOP_ERR                                             = 117,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_INACT_DISC                                           = 118,
    NVSDM_PORT_TELEM_CTR_GRP_X25_PORT_NBOR_MTU_DISC                                        = 119,

    /* Group 0x26 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X26_RQ_GENERAL_ERROR                                          = 120,

    /* Group 0x12 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X12_RS_FEC_SINGLE_ERR_BLKS_INTERNAL_ONLY                      = 121,
    NVSDM_PORT_TELEM_CTR_GRP_X12_LINK_DOWN_EVENTS_INTERNAL_ONLY                            = 122,

    /* Group 0x16 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_0_INTERNAL_ONLY                              = 123,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_1_INTERNAL_ONLY                              = 124,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_2_INTERNAL_ONLY                              = 125,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_3_INTERNAL_ONLY                              = 126,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_4_INTERNAL_ONLY                              = 127,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_5_INTERNAL_ONLY                              = 128,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_6_INTERNAL_ONLY                              = 129,
    NVSDM_PORT_TELEM_CTR_GRP_X16_RAW_BER_LANE_7_INTERNAL_ONLY                              = 130,

    /* Group 0x26 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X26_PORT_RCV_DATA_QWORD_INTERNAL_ONLY                         = 131,
    NVSDM_PORT_TELEM_CTR_GRP_X26_PORT_XMIT_DATA_QWORD_INTERNAL_ONLY                        = 132,
    NVSDM_PORT_TELEM_CTR_GRP_X26_PORT_XMIT_EBP_INTERNAL_ONLY                               = 133,
    NVSDM_PORT_TELEM_CTR_GRP_X26_PORT_SW_HOQ_LIFETIME_LIM_DISCARDS_INTERNAL_ONLY           = 134,
    NVSDM_PORT_TELEM_CTR_GRP_X26_DQS2LLU_XMIT_WAIT_ARB_GLOBAL_INTERNAL_ONLY                = 135,
    NVSDM_PORT_TELEM_CTR_GRP_X26_GRXB_FECN_MARK_INTERNAL_ONLY                              = 136,
    NVSDM_PORT_TELEM_CTR_GRP_X26_GENERAL_TRANSMIT_DISC_EXT_CONT_INTERNAL_ONLY              = 137,

    /* Group 0x27 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_ENTRY_QUIET_INTERNAL_ONLY                              = 138,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_FORCE_ENTRY_INTERNAL_ONLY                              = 139,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_LOCAL_DESIRED_INTERNAL_ONLY                       = 140,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_LOCAL_RECAL_INTERNAL_ONLY                         = 141,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_LOCAL_RECAL_WTH_TRAFFIC_INTERNAL_ONLY             = 142,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_LOCAL_RECAL_WTHOUT_TRAFFIC_INTERNAL_ONLY          = 143,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_REMOTE_INTERNAL_ONLY                              = 144,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_EXIT_REMOTE_PROBABLY_RECAL_INTERNAL_ONLY               = 145,
    NVSDM_PORT_TELEM_CTR_GRP_X27_LOCAL_FULL_BW_EXIT_INTERNAL_ONLY                          = 146,
    NVSDM_PORT_TELEM_CTR_GRP_X27_LOCAL_LOW_POWER_ENTER_INTERNAL_ONLY                       = 147,
    NVSDM_PORT_TELEM_CTR_GRP_X27_REMOTE_FULL_BW_EXIT_INTERNAL_ONLY                         = 148,
    NVSDM_PORT_TELEM_CTR_GRP_X27_REMOTE_LOW_POWER_ENTER_INTERNAL_ONLY                      = 149,
    NVSDM_PORT_TELEM_CTR_GRP_X27_LOCAL_LOW_POWER_EXIT_INTERNAL_ONLY                        = 150,
    NVSDM_PORT_TELEM_CTR_GRP_X27_LOCAL_FULL_BW_ENTER_INTERNAL_ONLY                         = 151,
    NVSDM_PORT_TELEM_CTR_GRP_X27_REMOTE_LOW_POWER_EXIT_INTERNAL_ONLY                       = 152,
    NVSDM_PORT_TELEM_CTR_GRP_X27_REMOTE_FULL_BW_ENTER_INTERNAL_ONLY                        = 153,
    NVSDM_PORT_TELEM_CTR_GRP_X27_HIGH_SPEED_STEADY_STATE_INTERNAL_ONLY                     = 154,
    NVSDM_PORT_TELEM_CTR_GRP_X27_L1_STEADY_STATE_INTERNAL_ONLY                             = 155,
    NVSDM_PORT_TELEM_CTR_GRP_X27_OTHER_STATE_INTERNAL_ONLY                                 = 156,
    NVSDM_PORT_TELEM_CTR_GRP_X27_TIME_SINCE_LAST_CLEAR_INTERNAL_ONLY                       = 157,

    /* Group 0x28 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R8_INTERNAL_ONLY                                      = 158,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R16_INTERNAL_ONLY                                     = 159,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_16_24_INTERNAL_ONLY                                 = 160,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_25_32_INTERNAL_ONLY                                 = 161,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_33_64_INTERNAL_ONLY                                 = 162,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_65_128_INTERNAL_ONLY                                = 163,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_129_256_INTERNAL_ONLY                               = 164,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_257_512_INTERNAL_ONLY                               = 165,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_R_513_MTU_INTERNAL_ONLY                               = 166,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T8_INTERNAL_ONLY                                      = 167,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T16_INTERNAL_ONLY                                     = 168,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_16_24_INTERNAL_ONLY                                 = 169,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_25_32_INTERNAL_ONLY                                 = 170,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_33_64_INTERNAL_ONLY                                 = 171,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_65_128_INTERNAL_ONLY                                = 172,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_129_256_INTERNAL_ONLY                               = 173,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_257_512_INTERNAL_ONLY                               = 174,
    NVSDM_PORT_TELEM_CTR_GRP_X28_CNT_T_513_MTU_INTERNAL_ONLY                               = 175,

    /* Group 0x29 Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X29_RCV_IBG1_NVL_PKTS_INTERNAL_ONLY                           = 176,
    NVSDM_PORT_TELEM_CTR_GRP_X29_RCV_IBG1_NON_NVL_PKTS_INTERNAL_ONLY                       = 177,
    NVSDM_PORT_TELEM_CTR_GRP_X29_RCV_IBG2_PKTS_INTERNAL_ONLY                               = 178,
    NVSDM_PORT_TELEM_CTR_GRP_X29_XMIT_IBG1_NVL_PKTS_INTERNAL_ONLY                          = 179,
    NVSDM_PORT_TELEM_CTR_GRP_X29_XMIT_IBG1_NON_NVL_PKTS_INTERNAL_ONLY                      = 180,
    NVSDM_PORT_TELEM_CTR_GRP_X29_XMIT_IBG2_PKTS_INTERNAL_ONLY                              = 181,

    /* Group 0x2A Counters */
    NVSDM_PORT_TELEM_CTR_GRP_X2A_DQS2LLU_XMIT_WAIT_INTERNAL_ONLY                           = 182,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_DQS2LLU_XMIT_WAIT_ARB_INTERNAL_ONLY                       = 183,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_RCV_PKTS_VL_INTERNAL_ONLY                            = 184,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_RCV_PKTS_VL15_INTERNAL_ONLY                          = 185,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_RCV_DATA_VL_INTERNAL_ONLY                            = 186,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_RCV_DATA_VL15_INTERNAL_ONLY                          = 187,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_XMIT_PKTS_VL_INTERNAL_ONLY                           = 188,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_XMIT_PKTS_VL15_INTERNAL_ONLY                         = 189,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_XMIT_DATA_VL_INTERNAL_ONLY                           = 190,
    NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_XMIT_DATA_VL15_INTERNAL_ONLY                         = 191,

    NVSDM_PORT_TELEM_CTR_MAX                                                               = NVSDM_PORT_TELEM_CTR_GRP_X2A_PORT_XMIT_DATA_VL15_INTERNAL_ONLY,

    NVSDM_PORT_TELEM_CTR_NONE                                                              = 9999,
} nvsdmPortTelemCounter_t;

/**
 * NVSDM platform telemetry IDs
 */
typedef enum
{
    NVSDM_PLATFORM_TELEM_CTR_VOLTAGE                   = 1,
    NVSDM_PLATFORM_TELEM_CTR_CURRENT_NOT_SUPPORTED     = 2,
    NVSDM_PLATFORM_TELEM_CTR_POWER                     = 3,
    NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE               = 4,
    NVSDM_PLATFORM_TELEM_CTR_MAX                       = NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE,
    NVSDM_PLATFORM_TELEM_CTR_NONE                      = 9999,
} nvsdmPlatformTelemCounter_t;

/**
 * NVSDM ConnectX telemetry IDs
 */
typedef enum
{
    NVSDM_CONNECTX_TELEM_CTR_ACTIVE_PCIE_LINK_WIDTH     = 1,
    NVSDM_CONNECTX_TELEM_CTR_ACTIVE_PCIE_LINK_SPEED     = 2,
    NVSDM_CONNECTX_TELEM_CTR_EXPECT_PCIE_LINK_WIDTH     = 3,
    NVSDM_CONNECTX_TELEM_CTR_EXPECT_PCIE_LINK_SPEED     = 4,
    NVSDM_CONNECTX_TELEM_CTR_CORRECTABLE_ERR_STATUS     = 5,
    NVSDM_CONNECTX_TELEM_CTR_CORRECTABLE_ERR_MASK       = 6,
    NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_STATUS   = 7,
    NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_MASK     = 8,
    NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_SEVERITY = 9,
    NVSDM_CONNECTX_TELEM_CTR_MAX                        = NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_SEVERITY,
    NVSDM_CONNECTX_TELEM_CTR_NONE                       = 9999,
} nvsdmConnectXTelemCounter_t;

/**
 * NVSDM value types
 */
typedef enum
{
    NVSDM_VAL_TYPE_DOUBLE    = 1,
    NVSDM_VAL_TYPE_UINT64    = 2,
    NVSDM_VAL_TYPE_INT64     = 3,
    NVSDM_VAL_TYPE_FLOAT     = 4,
    NVSDM_VAL_TYPE_UINT32    = 5,
    NVSDM_VAL_TYPE_INT32     = 6,
    NVSDM_VAL_TYPE_UINT16    = 7,
    NVSDM_VAL_TYPE_INT16     = 8,
    NVSDM_VAL_TYPE_UINT8     = 9,
    NVSDM_VAL_TYPE_INT8      = 10,
    /* New types below */
    /* New types above */
    NVSDM_VAL_TYPE_NONE      = 999,
} nvsdmValType_t;

/**
 * NVSDM value union
 */
typedef union
{
    double      dVal; //!< Double
    uint64_t    u64Val; //!< 64 bit unsigned int
    int64_t     s64Val; //!< 64 bit signed int
    float       fVal;   //!< Float
    uint32_t    u32Val; //!< 32 bit unsigned int
    int32_t     s32Val; //!< 32 bit signed int
    uint16_t    u16Val; //!< 16 bit unsigned int
    int16_t     s16Val; //!< 16 bit signed int
    uint8_t     u8Val;  //!< 8 bit unsigned int
    int8_t      s8Val;  //!< 8 bit signed int
    /* New entries below */
} nvsdmVal_t;

/**
 * Currently ignored
 */
typedef enum
{
    NVSDM_OPERATION_GET    = 0x1,
    NVSDM_OPERATION_SET    = 0x2,
    NVSDM_OPERATION_RESET  = 0x4,
    NVSDM_OPERATION_NONE   = 0xff
} nvsdmOperationType_t;

/**
 * NVSDM telemetry structure
 */
typedef struct
{
    uint8_t telemType;      //!< One of @p nvsdmTelemType_t
    uint8_t reserved;       //!< Reserved for future use
    uint16_t telemCtr;      //!< One of @p nvsdmPortTelemCounter_t or @p nvsdmPlatformTelemCounter_t or @p nvsdmConnectXTelemCounter_t a custom telemetry counter, depending on the value of @p telemType
    uint64_t timestamp;     //!< Timestamp when sample was collected; only valid if @ref status is @p NVSDM_SUCCESS
    uint16_t valType;       //!< One of @p nvsdmValType_t
    nvsdmVal_t val;         //!< The retrieved telemtry value; only valid if @ref status is @p NVSDM_SUCCESS
    uint16_t status;        //!< One of @p nvsdmRet_t
} nvsdmTelem_v1_t;

/**
 * NVSDM telemetry structure
 */
typedef struct
{
    uint32_t version;                   //!< Structure version number
    uint32_t numTelemEntries;           //!< Number of entries in the @p telemValsArray
    nvsdmTelem_v1_t *telemValsArray;    //!< Array of tnvsdmTelem_v1_t instances
} nvsdmTelemParam_v1_t;
typedef nvsdmTelemParam_v1_t nvsdmTelemParam_t;
#define nvsdmTelemParam_v1 NVSDM_STRUCT_VERSION(nvsdmTelemParam, 1)

/**
 * NVSDM Port States
 */
typedef enum
{
    NVSDM_PORT_STATE_NO_STATE_CHANGE   = 0, //!< No state change
    NVSDM_PORT_STATE_DOWN              = 1, //!< State down
    NVSDM_PORT_STATE_INITIALIZE        = 2, //!< initialized state
    NVSDM_PORT_STATE_ARMED             = 3, //!< Armed state
    NVSDM_PORT_STATE_ACTIVE            = 4, //!< Active state
    NVSDM_PORT_STATE_MAX               = NVSDM_PORT_STATE_ACTIVE,
    NVSDM_PORT_STATE_NONE              = 0xFF,
} nvsdmPortState_t;

/**
 * NVSDM Port Info
 */
typedef struct
{
    uint32_t version; //!< The version number of this struct
    int32_t portState; //!< Port state. One of @p nvsdmPortState_t
    int32_t portPhysState; //!< Port physical state. One of @p nvsdmPortState_t
    uint32_t linkWidthActive; //!< Link width active
    uint32_t linkWidthEnabled; //!< Link width enabled
    uint32_t linkWidthSupported; //!< Link width supported
    uint32_t linkSpeedSupported; //!< Link speed supported
    uint32_t linkSpeedActive; //!< Link speed active
    uint32_t linkSpeedEnabled; //!< Link speed enabled
    uint32_t linkSpeedExtActive; //!< Link speed ext active
    uint32_t fdr10; //!< FDR10
} nvsdmPortInfo_v1_t;
typedef nvsdmPortInfo_v1_t nvsdmPortInfo_t;
#define nvsdmPortInfo_v1 NVSDM_STRUCT_VERSION(nvsdmPortInfo, 1)

/**
 * NVSDM Switch Info
 */
typedef struct
{
    uint32_t version; //!< The version number of this struct
    uint16_t linearFDBCap; //!< linear FDB cap
    uint16_t randomFDBCap; //!< random FDB cap

    uint16_t multicastFDBCap; //!< multicast FDB cap
    uint16_t linearFDBTop; //!< linear FDB top

    uint8_t defaultPort; //!< Default port
    uint8_t defaultMulticastPrimaryPort; //!< Default multicast primary port
    uint8_t defaultMulticastNotPrimaryPort; //!< Default multicast not primary port
    uint8_t lifeTimeValue; //!< Lifetime value
    uint8_t portStateChange; //!< Port state change
    uint8_t optimizedSLToVLMappingProgramming; //!< Optimized SL To VL Mapping Programming

    uint16_t lidsPerPort; //!< LIDs per port
    uint16_t partitionEnforcementCap; //!< partition encorcement cap

    uint8_t inboundEnforcementCap; //!< inbound Enforcement cap
    uint8_t outboundEnforcementCap; //!< outbound enforcementcap
    uint8_t filterRawInboundCap; //!< filter raw inbound cap
    uint8_t filterRawOutboundCap; //!< filter raw outbound cap
    uint8_t enhancedPort0; //!< enhanced port 0
    uint8_t disableDRSMPClassVersionValidationSupportedNotAvailable; //!< No entry in "enum MAD_FIELDS" in infinband "mad.h" header
    uint8_t disableDRSMPClassVersionValidationEnabledNotAvailable; //!< No entry in "enum MAD_FIELDS" in infinband "mad.h" header
    uint16_t multicastFDBTop; //!< multicast FDB
} nvsdmSwitchInfo_v1_t;
typedef nvsdmSwitchInfo_v1_t nvsdmSwitchInfo_t;
#define nvsdmSwitchInfo_v1 NVSDM_STRUCT_VERSION(nvsdmSwitchInfo, 1)

/**
 * NVSDM Device Health Type
 */
typedef enum
{
    NVSDM_DEVICE_STATE_HEALTHY = 0,
    NVSDM_DEVICE_STATE_ERROR   = 1,
    NVSDM_DEVICE_STATE_UNKNOWN = 0xFF,
} nvsdmDeviceHealthType_t;

/**
 * NVSDM Device Health Status
 */
typedef struct
{
    uint32_t version;
    uint32_t state;
} nvsdmDeviceHealthStatus_v1_t;
typedef nvsdmDeviceHealthStatus_v1_t nvsdmDeviceHealthStatus_t;
#define nvsdmDeviceHealthStatus_v1 NVSDM_STRUCT_VERSION(nvsdmDeviceHealthStatus, 1)

/* *****************************************
 * The exported interface
 * *****************************************
 */

/**
 * Initialize library
 *
 * @returns         @ref NVSDM_SUCCESS on success
 */
nvsdmRet_t nvsdmInitialize();

/**
 * Perform cleanup prior to exit
 *
 * @returns         @ref NVSDM_SUCCESS on success
 */
nvsdmRet_t nvsdmFinalize();

/* *****************************************
 * Discovery and topology functions
 * *****************************************
 */

/**
 * Discover network topology
 * @param[in]       srcCA       CA of the start point i.e. the CA from which to start the topology discovery process.
 *                              Set to NULL to use the first CA with a connected port.
 * @param[in]       srcPort     Port number of the start point. Ignored if @p srcCA is NULL.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 */
nvsdmRet_t nvsdmDiscoverTopology(char *srcCA, int srcPort);

/**
 * Retrieve an iterator to the list of devices of every type.
 * @param[out]      iter        Pointer to an opaque device iterator handle. Use
 *                              the @ref nvsdmGetNextDevice or @ref nvsdmIterateDevices functions
 *                              to iterate over the list.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p iter  is NULL
 */
nvsdmRet_t nvsdmGetAllDevices(nvsdmDeviceIter_t *iter);

/**
 * Retrieve an iterator to the list of devices of type @p type
 * @param[in]       type        Device type, one of @ref nvsdmDevType
 * @param[out]      iter        Pointer to an opaque device iterator handle. Use
 *                              the @ref nvsdmGetNextDevice or @ref nvsdmIterateDevices functions
 *                              to iterate over the list.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if type is not one of @ref nvsdmDevType, or if @p iter  is NULL
 */
nvsdmRet_t nvsdmGetAllDevicesOfType(int type, nvsdmDeviceIter_t *iter);

/**
 * Retrieve the device corresponding to the given GUID.
 * @param[in]       guid        The device GUID
 * @param[out]      dev         The device corresponding to the target @p guid; undefined
 *                              if the return code is @ref NVSDM_ERROR_DEVICE_NOT_FOUND
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 * @returns         @ref NVSDM_ERROR_DEVICE_NOT_FOUND      if no device with GUID equal to @p guid is present
 */
nvsdmRet_t nvsdmGetDeviceGivenGUID(uint64_t guid, nvsdmDevice_t *dev);

/**
 * Retrieve the device corresponding to the given LID.
 * @param[in]       lid        The device LID
 * @param[out]      dev        The device corresponding to the target @p lid; undefined
 *                             if the return code is @ref NVSDM_ERROR_DEVICE_NOT_FOUND
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 * @returns         @ref NVSDM_ERROR_DEVICE_NOT_FOUND      if no device with LID equal to @p lid is present
 */
nvsdmRet_t nvsdmGetDeviceGivenLID(uint16_t lid, nvsdmDevice_t *dev);

/**
 * Retrieve the next valid device handle in the list being iterated over by @p iter.
 * @param[in]       iter        Iterator to the list being iterated over.
 * @param[out]      device      Reference to the next valid device handle, or NULL if no such handle exists.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p iter or @p device are NULL
 */
nvsdmRet_t nvsdmGetNextDevice(nvsdmDeviceIter_t iter, nvsdmDevice_t *device);

/**
 * Iterate over a list of devices, calling @p callback on each
 * @param[in]       iter        Iterator to the list being iterated over.
 * @param[in]       callback    Callback function to invoke for each device instance.
 * @param[in]       cbData      User-supplied data to pass to @p callback
 *
 * @returns         @ref NVSDM_SUCCESS on success
 *                  any error return codes from @p callback
 */
nvsdmRet_t nvsdmIterateDevices(nvsdmDeviceIter_t iter, nvsdmRet_t (*callback)(nvsdmDevice_t const, void *), void *cbData);

/**
 * Reset the iterator to point to the first device in the list
 * @param[in]       iter        Iterator to the list being reset.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 */
nvsdmRet_t nvsdmResetDeviceList(nvsdmDeviceIter_t iter);

/**
 * Retrieve an iterator to the list of ports from the given device
 * @param[in]       device      The input device
 * @param[out]      iter        Iterator to the list of ports for @p device
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p iter is NULL
 */
nvsdmRet_t nvsdmDeviceGetPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter);

/**
 * Retrieve an iterator to the list of ports in the network.
 * @param[out]      iter        Pointer to an opaque port iterator handle. Use
 *                              the @ref nvsdmGetNextPort or @ref nvsdmIteratePorts functions
 *                              to iterate over the list.
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p iter  is NULL
 */
nvsdmRet_t nvsdmGetAllPorts(nvsdmPortIter_t *iter);

/**
 * Retrieve the next valid port handle in the list being iterated over by @p iter.
 * @param[in]       iter        Iterator to the list being iterated over.
 * @param[out]      port        Reference to the next valid port handle, or NULL if no such handle exists.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p iter or @p port are NULL
 */
nvsdmRet_t nvsdmGetNextPort(nvsdmPortIter_t iter, nvsdmPort_t *port);

/**
 * Iterate over a list of ports, calling @p callback on each
 * @param[in]       iter        Iterator to the list being iterated over.
 * @param[in]       callback    Callback function to invoke for each port instance.
 * @param[in]       cbData      User-supplied data to pass to @p callback
 *
 * @returns         @ref NVSDM_SUCCESS on success
 *                  any error return codes from @p callback
 */
nvsdmRet_t nvsdmIteratePorts(nvsdmPortIter_t iter, nvsdmRet_t (*callback)(nvsdmPort_t const, void *), void *cbData);

/**
 * Retrieve the device that port belongs to
 * @param[in]       port        The input port
 * @param[out]      device      The device that @p port belongs to
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid port, or if @p device is NULL
 */
nvsdmRet_t nvsdmPortGetDevice(nvsdmPort_t const port, nvsdmDevice_t *device);

/**
 * Return the nvsdm port handle connected to port, or NULL if port is not connected.
 * @param[in]       port        The input port
 * @param[out]      remote      The remote part, or NULL if @p port is not connected.
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid nvsdm port handle, or if @p remote is NULL
 */
nvsdmRet_t nvsdmPortGetRemote(nvsdmPort_t const port, nvsdmPort_t *remote);

/**
 * Retrieve miscellaneous info about a port, including the port status and physical status, link speed and link width. Refer to
 * @ref nvsdmPortInfo_t for a full list of the information retrieved.
 * @param[in]       port        The target port
 * @param[out]      info        The retrieved port info
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port handle, or if @p info is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @p info->version is not valid
 */
nvsdmRet_t nvsdmPortGetInfo(nvsdmPort_t const port, nvsdmPortInfo_t *info);

/**
 * Set port state field for the given port
 * @param[in]       port        The target port
 * @param[out]      state       The desired port state, one of @ref nvsdmPortState_t
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port handle, or if @p stateu
 *                                                             is not a valid port state
 */
nvsdmRet_t nvsdmPortSetState(nvsdmPort_t const port, unsigned state);

/**
 * Retrieve a list of nvsdm device handles connected to device via one or more ports
 * @param[in]       device      The input device handle
 * @param[out]      iter        Iterator to the list of nvsdm device handles connected to @p device
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p iter is NULL
 */
nvsdmRet_t nvsdmDeviceGetConnectedDevices(nvsdmDevice_t const device, nvsdmDeviceIter_t *iter);

/**
 * Retrieve a list of nvsdm port handles connected to the ports on device
 * @param[in]       device      The input device handle
 * @param[out]      iter        Iterator to the list of nvsdm port handles connected to the ports on @p device
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p iter is NULL
 */
nvsdmRet_t nvsdmDeviceGetConnectedPorts(nvsdmDevice_t const device, nvsdmPortIter_t *iter);

/**
 * Reset the iterator to point to the first port in the list.
 * @param[in]       iter        Iterator to the list being reset.
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 */
nvsdmRet_t nvsdmResetPortList(nvsdmPortIter_t iter);

/**
 * Retrieve information corresponding to the SwitchInfo MAD attribute from the switch referenced by device.
 * Refer to @ref nvsdmSwitchInfo_t for a full list of the information retrieved.
 * @param[in]       device      Device handle for the target switch
 * @param[out]      info        The retrieved switch info
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p device is not a valid nvsdm device handle, or refers
 *                                                             to a non-switch device, or if @p info is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @p info->version is not valid
 */
nvsdmRet_t nvsdmDeviceGetSwitchInfo(nvsdmDevice_t const device, nvsdmSwitchInfo_t *info);

/* *****************************************
 * Telemetry functions
 * *****************************************
 */

/**
 * Retrieve telemetry values. On return, status code in individual entries in @p telemVals describe success/failure when
 * retrieving the given telemetry value.
 * @param[in]       port        The port from which to retrieve telemetry values
 * @param[in,out]   param       Structure wrapping the array of telemetry counters to be read.
 *
 * @returns         @ref NVSDM_SUCCESS                       on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED           if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                           prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG             if @p device is not a valid nvsdm port handle, or if @p param is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED   if @p param->version is not valid
 */
nvsdmRet_t nvsdmPortGetTelemetryValues(nvsdmPort_t const port, nvsdmTelemParam_t *param);

/**
 * Retrieve telemetry values from a ConnectX HCA. On return, status code in individual entries
 * in @p telemVals describe success/failure when retrieving the given telemetry value.
 * @param[in]       device      The device from which to retrieve telemetry values
 * @param[in,out]   param       Structure wrapping the array of telemetry counters to be read.
 *
 * @returns         @ref NVSDM_SUCCESS                       on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED           if the nvsdm interface was not initialized
 *                                                           i.e. @p nvsdmInitialize was not called
 *                                                           prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG             if @p device is not a valid nvsdm device handle,
 *                                                           or if @p param is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED   if @p param->version is not valid
 */
nvsdmRet_t nvsdmDeviceGetTelemetryValues(nvsdmDevice_t const device, nvsdmTelemParam_t *param);

/**
 * Reset telemetry counters. On return, status code for individual reset operations is encoded in the corresponding
 * entry of @p telemVals.
 * @param[in]       port        The port whose telemetry counters need to be reset
 * @param[in,out]   param       Structure wrapping the array of telemetry counters to be reset.
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED       if the nvsdm interface was not initialized i.e.
 *                                                       @ref nvsdmInitialize was not called prior
 *                                                       to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p param is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED   if @p param->version is not valid
 */
nvsdmRet_t nvsdmPortResetTelemetryCounters(nvsdmPort_t const port, nvsdmTelemParam_t *param);

/* *****************************************
 * Routing functions
 * *****************************************
 */
/**
 * Structure encoding Adaptive Routing Info
 */
typedef struct
{
    unsigned version;
    /* Offset 0x40 */
    uint32_t enabled : 1;
    uint32_t isARNSupported : 1;
    uint32_t isFRNSupported : 1;
    uint32_t isFRSupported : 1;
    uint32_t FREnabled : 1;
    uint32_t RNXmitEnabled : 1;
    uint32_t isARTrialSupported : 1;
    uint32_t reserved1 : 1;
    uint32_t subGrpsActive : 4;
    uint32_t adaptiveRoutingGroupTableCopySupported : 4;
    uint32_t directionNumSupported : 8;
    uint32_t reserved2 : 1;
    uint32_t IS4Mode : 1;
    uint32_t GLBGroups : 1;
    uint32_t bySLCap : 1;
    uint32_t bySLFn : 1;
    uint32_t byTranspCap : 1;
    uint32_t dynCapCalcSup : 1;
    uint32_t reserved3 : 1;
    /* Offset 0x44 */
    uint32_t groupCap : 12;
    uint32_t groupTop : 12;
    uint32_t groupTableCap : 4;
    uint32_t stringWidthCap : 4;
    /* Offset 0x48 */
    uint32_t ARVersionCap : 4;
    uint32_t RNVerCap : 4;
    uint32_t subGrpsSupported : 8;
    uint32_t enableBySLMask : 16;
    /* Offset 0x4C */
    uint32_t byTransportDisable : 8;
    uint32_t reserved4 : 24;
    /* Offset 0x50 */
    uint32_t ageingTimeValue;
    /* Offset 0x54 */
    uint32_t reserved5 : 12;
    uint32_t PFRNEnabled : 1;
    uint32_t reserved6: 1;
    uint32_t WHBFEn : 1;
    uint32_t bySLHBFEn : 1;
    uint32_t reserved7 : 9;
    uint32_t isPFRNSupported : 1;
    uint32_t isNVLHBFSupported : 1;
    uint32_t isBTHDQPHashSupported : 1;
    uint32_t isDCETHHashSupported : 1;
    uint32_t isSymmetricHashSupported : 1;
    uint32_t isWHBFSupported : 1;
    uint32_t isHBFSupported : 1;
    /* Offset 0x58 */
    uint32_t enableBySLMaskHBF : 16;
    uint32_t reserved8 : 13;
    uint32_t WHBFGranularity : 3;
} nvsdmARInfo_v1_t;
typedef nvsdmARInfo_v1_t nvsdmARInfo_t;
#define nvsdmARInfo_v1 NVSDM_STRUCT_VERSION(nvsdmARInfo, 1)

/**
 * Retrieve Adaptive Routing information from the given port
 * @param[in]       port        The target port
 * @param[in,out]   arInfo      Versioned structure containing adaptive routing information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref arInfo is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref arInfo->version is not valid
 */
nvsdmRet_t nvsdmPortGetARInfo(nvsdmPort_t const port, nvsdmARInfo_t *arInfo);

/**
 * Structure encoding Adaptive Routing Group Tables
 */
typedef struct
{
    unsigned version;
    /* Input param */
    union {
        struct {
            uint32_t arSubGroup : 12;
            uint32_t arGroup : 2;
            uint32_t reserved1 : 2;
            uint32_t groupTable : 4;
            uint32_t reserved2 : 4;
            uint32_t privateLFTID : 8;
        };
        uint32_t attrMod; // Attribute Modifier to retrieve Group Table information; see VS MAD spec section 2.6.2 (page 78)
    };
    /* Data output */
    struct {
        uint8_t arSubGroup0[32];
        uint8_t arSubGroup1[32];
    };
} nvsdmARGroupTable_v1_t;
typedef nvsdmARGroupTable_v1_t nvsdmARGroupTable_t;
#define nvsdmARGroupTable_v1 NVSDM_STRUCT_VERSION(nvsdmARGroupTable, 1)

/**
 * Retrieve Adaptive Routing Group Table from the given port
 * @param[in]       port            The target port
 * @param[in,out]   arGroupTable    Versioned structure passing attribute modifier (input)
 *                                  and containing adaptive routing group table information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref arGroupTable is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref arGroupTable->version is not valid
 */
nvsdmRet_t nvsdmPortGetARGroupTable(nvsdmPort_t const port,
                                      nvsdmARGroupTable_t *arGroupTable);

typedef struct
{
    uint32_t slid : 1;
    uint32_t dlid : 1;
    uint32_t sl : 1;
    uint32_t vl : 1;
    uint32_t pipeID : 1;
    uint32_t nvlrhHash : 8;
    uint32_t inputPort : 1;
    uint32_t reserved : 15;
} nvsdmNVLHBFConfigFieldsEnable_v1_t;

typedef struct
{
    unsigned version;
    /* Input param */
    union {
        struct {
            uint32_t portOrProfileNumber : 16; /* Port number if @ref profileNumberEn is zero; profile number otherwise */
            uint32_t reserved : 15;
            uint32_t profileNumberEn : 1; /* Either 0 or 1 */
        };
        uint32_t attrMod;
    };
    /* Output data */
    struct {
        /* Offset 0x40 */
        uint32_t reserved1 : 28;
        uint32_t hashType : 4;
        /* Offset 0x44 */
        uint32_t reserved2;
        /* Offset 0x48 */
        uint32_t packetHashBitmask : 8;
        uint32_t reserved3 : 24;
        /* Offset 0x4C */
        uint32_t reserved4;
        /* Offset 0x50 */
        uint32_t seed;
        /* Offset 0x54, 0x58 */
        union {
            struct {
                nvsdmNVLHBFConfigFieldsEnable_v1_t fieldsEnableUC;
                nvsdmNVLHBFConfigFieldsEnable_v1_t fieldsEnableMC;
            };
            uint64_t fieldsEnable;
        };
    };
} nvsdmNVLHBFConfig_v1_t;
typedef nvsdmNVLHBFConfig_v1_t nvsdmNVLHBFConfig_t;
#define nvsdmNVLHBFConfig_v1 NVSDM_STRUCT_VERSION(nvsdmNVLHBFConfig, 1)

/**
 * Retrieve NVLHBFConfig from the given port
 * @param[in]       port            The target port
 * @param[in,out]   config          Versioned structure passing attribute modifier (input)
 *                                  and containing NVLHBF Config information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref config is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref config->version is not valid
 */
nvsdmRet_t nvsdmPortGetNVLHBFConfig(nvsdmPort_t const port, nvsdmNVLHBFConfig_t *config);

typedef struct
{
    unsigned version;
    /* Output data */
    struct {
        /* Offset 0x40 */
        uint32_t globalLIDStartRange : 20;
        uint32_t reserved1 : 12;
        /* Offset 0x44 */
        uint32_t globalLIDRangeCap : 20;
        uint32_t reserved2 : 12;
        /* OFfset 0x48 */
        uint32_t globalLIDRangeTop : 20;
        uint32_t reserved3 : 12;
        /* Offset 0x4C */
        uint32_t alidRangeStart : 20;
        uint32_t reserved4 : 12;
        /* Offset 0x50 */
        uint32_t alidRangeCap : 20;
        uint32_t reserved5 : 12;
        /* Offset 0x54 */
        uint32_t alidRangeTop : 20;
        uint32_t reserved6 : 12;
        /* Offset 0x58 */
        uint32_t localPlaneLIDRangeStart : 20;
        uint32_t reserved7 : 12;
        /* Offset 0x5C */
        uint32_t localPlaneLIDRangeCap : 20;
        uint32_t reserved8 : 12;
        /* Offset 0x60 */
        uint32_t localPlaneLIDRangeTop : 20;
        uint32_t reserved9 : 12;
    };
} nvsdmLFTSplit_v1_t;
typedef nvsdmLFTSplit_v1_t nvsdmLFTSplit_t;
#define nvsdmLFTSplit_v1 NVSDM_STRUCT_VERSION(nvsdmLFTSplit, 1)

/**
 * Retrieve LFT Split from the given port
 * @param[in]       port            The target port
 * @param[in,out]   split           Versioned structure LFT Split information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref split is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref split->version is not valid
 */
nvsdmRet_t nvsdmPortGetLFTSplit(nvsdmPort_t const port, nvsdmLFTSplit_t *split);

/**
 * Set LFT Split for the given port
 * @param[in]       port            The target port
 * @param[in]       split           Versioned structure LFT Split information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref split is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref split->version is not valid
 */
nvsdmRet_t nvsdmPortSetLFTSplit(nvsdmPort_t const port, nvsdmLFTSplit_t const *split);

typedef struct
{
    unsigned version;
    /* Input param */
    uint32_t blockID;
    /* Output data */
    struct {
        uint32_t anycastLID : 20;
        uint32_t reserved : 4;
        uint32_t properties : 8;
    } anycastLIDBlock[16];
} nvsdmAnycastLIDInfo_v1_t;
typedef nvsdmAnycastLIDInfo_v1_t nvsdmAnycastLIDInfo_t;
#define nvsdmAnycastLIDInfo_v1 NVSDM_STRUCT_VERSION(nvsdmAnycastLIDInfo, 1)

/**
 * Retrieve Anycast LID information for the given port
 * @param[in]       port            The target port
 * @param[in,out]   info            Versioned structure passing block ID (input)
 *                                  and containing Anycast LID information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @p info is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @p info->version is not valid
 */
nvsdmRet_t nvsdmPortGetAnycastLIDInfo(nvsdmPort_t const port, nvsdmAnycastLIDInfo_t *info);

typedef struct
{
    /* DWord 1 */
    uint32_t highAmplitude : 16;
    uint32_t lowAmplitude : 16;

    /* DWord 2 */
    uint32_t positiveBound : 8;
    uint32_t reserved1 : 8;
    uint32_t negativeBound : 8;
    uint32_t reserved2 : 8;

    /* DWord 3 */
    uint32_t selectedOptionTX : 8;
    uint32_t selectedOptionRX : 8;
    uint32_t ampCalibration : 8;
    uint32_t reserved3 : 8;
} perLaneInfo_v1_t;

#define NUM_SMP_EYEOPEN_LANES_V1 4

typedef struct
{
    unsigned version;
    /* Input param */
    uint32_t groupOfLanes;
    /* Output data */
    uint32_t linkWidthActive : 8;
    uint32_t reserved1 : 8;
    uint32_t autoNegoState : 4;
    uint32_t reserved2: 12;
    perLaneInfo_v1_t perLaneInfo[NUM_SMP_EYEOPEN_LANES_V1];
    uint32_t reserved3[3];
} nvsdmSMPEyeOpen_v1_t;
typedef nvsdmSMPEyeOpen_v1_t nvsdmSMPEyeOpen_t;
#define nvsdmSMPEyeOpen_v1 NVSDM_STRUCT_VERSION(nvsdmSMPEyeOpen, 1)

/**
 * Retrieve SMP_EyeOpen information for the given port
 * @param[in]       port            The target port
 * @param[in,out]   eyeOpen         Versioned structure passing group of lanes (input)
 *                                  and containing SMP_EyeOpen information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref eyeOpen is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref eyeOpen->version is not valid
 */
nvsdmRet_t nvsdmPortGetSMPEyeOpen(nvsdmPort_t const port, nvsdmSMPEyeOpen_t *eyeOpen);

typedef struct
{
    uint32_t groupNumber : 12;
    uint32_t reserved1 : 4;
    uint32_t legacyModePort : 8;
    uint32_t tableNumber : 4;
    uint32_t reserved2 : 2;
    uint32_t lidState : 2;
} nvsdmARLFTRev1PerLID_v1_t;

#define NUM_LIDS_PER_ARLFT 16

typedef struct
{
    unsigned version;
    /* Input param */
    union {
        struct {
            uint32_t blockID : 24;
            uint32_t privateLFTID : 8;
        };
        uint32_t attrMod;
    };
    /* Input/Output data */
    nvsdmARLFTRev1PerLID_v1_t lids[NUM_LIDS_PER_ARLFT];
} nvsdmARLFTRev1_v1_t;
typedef nvsdmARLFTRev1_v1_t nvsdmARLFTRev1_t;
#define nvsdmARLFTRev1_v1 NVSDM_STRUCT_VERSION(nvsdmARLFTRev1, 1)

/**
 * Retrieve ARLFTRev1 information for the given port
 * @param[in]       port            The target port
 * @param[in,out]   arlft           Versioned structure passing block ID and private LFTID (input)
 *                                  and containing ARLFTRev1 information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref arlft is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref arlft->version is not valid
 */
nvsdmRet_t nvsdmPortGetARLFTRev1(nvsdmPort_t const port, nvsdmARLFTRev1_t *arlft);

/**
 * Set ARLFTRev1 information for the given port
 * @param[in]       port            The target port
 * @param[in]       arlft           Versioned structure passing ARLFTRev1 information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref arlft is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref arlft->version is not valid
 */
nvsdmRet_t nvsdmPortSetARLFTRev1(nvsdmPort_t const port, nvsdmARLFTRev1_t const *arlft);

typedef struct
{
    unsigned version;
    /* Input params */
    union {
        struct {
            uint32_t reserved1 : 16; /* Port number, will be filled in automatically */
            uint32_t ingressBlockID : 8;
            uint32_t egressBlockID : 8; /* Note: always '0' for QM3, per VS MAD */
        };
        uint32_t attrMod;
    };
    /* Input/Output data */
    uint32_t vlMask : 16;           /* RW */
    uint32_t mcEnable : 1;          /* RW */
    uint32_t ucEnable : 1;          /* RW */
    uint32_t reserved : 14;

    uint8_t ingressPortMask[16];   /* Index, 128b */

    uint8_t egressPortMask[32];    /* RW, 256b */
} nvsdmRailFilterConfig_v1_t;
typedef nvsdmRailFilterConfig_v1_t nvsdmRailFilterConfig_t;
#define nvsdmRailFilterConfig_v1 NVSDM_STRUCT_VERSION(nvsdmRailFilterConfig, 1)

/**
 * Retrieve RailFilterConfig information for the given port
 * @param[in]       port            The target port
 * @param[in,out]   config          Versioned structure passing ingress/egress block ID (input)
 *                                  and containing RailFilterConfig information (output)
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref config is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref config->version is not valid
 */
nvsdmRet_t nvsdmPortGetRailFilterConfig(nvsdmPort_t const port, nvsdmRailFilterConfig_t *config);

/**
 * Set RailFilterConfig information for the given port
 * @param[in]       port            The target port
 * @param[in]       config          Versioned structure passing RailFilter Information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref config is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref config->version is not valid
 */
nvsdmRet_t nvsdmPortSetRailFilterConfig(nvsdmPort_t const port, nvsdmRailFilterConfig_t const *config);

typedef struct
{
    uint32_t fomMeasurement : 4;
    uint32_t reserved1 : 28;

    uint32_t initialFOM : 8;
    uint32_t reserved2 : 8;
    uint32_t fomMode : 3;
    uint32_t reserved3 : 13;

    uint32_t lowerEye : 8;
    uint32_t midEye : 8;
    uint32_t upperEye : 8;
    uint32_t lastFOM : 8;

    uint32_t nerrsMin : 8;
    uint32_t reserved4 : 4;
    uint32_t nblksMax : 4;
    uint32_t reserved5 : 14;
    uint32_t fomExtConfCap : 1;
    uint32_t fomExtConf : 1;

    uint32_t reserved6[2];
} nvsdmSLRGPageData7Or5nm_v1_t;

typedef struct
{
    unsigned version;

    uint32_t testModeInternal : 1;
    uint32_t allLanes : 1;
    uint32_t reserved : 2;
    uint32_t portType : 4;
    uint32_t lane : 4;
    uint32_t lpMSBReserved : 2;
    uint32_t pnatReserved : 2;
    uint32_t localPortReserved : 8;
    uint32_t ver : 4;
    uint32_t status : 4;

    union {
        nvsdmSLRGPageData7Or5nm_v1_t pageData;
        uint8_t data[36];
    };
} nvsdmSLRG_v1_t;
typedef nvsdmSLRG_v1_t nvsdmSLRG_t;
#define nvsdmSLRG_v1 NVSDM_STRUCT_VERSION(nvsdmSLRG, 1)

/**
 * Retrieve Serdes Lane Receive Grade (SLRG) Information from the given port
 * @param[in]       port            The target port
 * @param[out]      slrg            Versioned structure containing SLRG information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref slrg is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref slrg->version is not valid
 */
nvsdmRet_t nvsdmPortGetSLRG(nvsdmPort_t const port, nvsdmSLRG_t *slrg);

/**
 * Set Serdes Lane Receive Grade (SLRG) Information for the given port
 * @param[in]       port            The target port
 * @param[out]      slrg            Versioned structure containing SLRG information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref slrg is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref slrg->version is not valid
 */
nvsdmRet_t nvsdmPortSetSLRG(nvsdmPort_t const port, nvsdmSLRG_t const *slrg);

typedef struct
{
    unsigned version;

    uint32_t planeInd : 4;
    uint32_t portType : 4;
    uint32_t opMod : 1;
    uint32_t applyIM : 1;
    uint32_t linkupFlow : 2;
    uint32_t lpMSBReserved : 2;
    uint32_t reserved1 : 2;
    uint32_t localPortReserved : 8;
    uint32_t reserved2 : 8;

    uint32_t lbEn : 12;
    uint32_t reserved3 : 4;
    uint32_t lbCap : 12;
    uint32_t reserved4 : 4;
} nvsdmPPLR_v1_t;
typedef nvsdmPPLR_v1_t nvsdmPPLR_t;
#define nvsdmPPLR_v1 NVSDM_STRUCT_VERSION(nvsdmPPLR, 1)

/**
 * Retrieve PPLR Information from the given port
 * @param[in]       port            The target port
 * @param[out]      pplr            Versioned structure containing PPLR information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref pplr is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref pplr->version is not valid
 */
nvsdmRet_t nvsdmPortGetPPLR(nvsdmPort_t const port, nvsdmPPLR_t *pplr);

/**
 * Set PPLR Information for the given port
 * @param[in]       port            The target port
 * @param[in]       pplr            Versioned structure containing PPLR information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref pplr is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref pplr->version is not valid
 */
nvsdmRet_t nvsdmPortSetPPLR(nvsdmPort_t const port, nvsdmPPLR_t const *pplr);

typedef struct
{
    unsigned version;

    uint32_t lbfMode : 3;
    uint32_t reserved1 : 9;
    uint32_t lpMSBReserved : 2;
    uint32_t reserved2 : 2;
    uint32_t localPortReserved : 8;
    uint32_t reserved3 : 8;

    uint32_t reserved4;
} nvsdmPLBF_v1_t;
typedef nvsdmPLBF_v1_t nvsdmPLBF_t;
#define nvsdmPLBF_v1 NVSDM_STRUCT_VERSION(nvsdmPLBF, 1)

/**
 * Retrieve PLBF Information from the given port
 * @param[in]       port            The target port
 * @param[out]      plbf            Versioned structure containing PLBF information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref plbf is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref plbf->version is not valid
 */
nvsdmRet_t nvsdmPortGetPLBF(nvsdmPort_t const port, nvsdmPLBF_t *PLBF);

/**
 * Set PLBF Information for the given port
 * @param[in]       port            The target port
 * @param[out]      plbf            Versioned structure containing PLBF information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref plbf is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref plbf->version is not valid
 */
nvsdmRet_t nvsdmPortSetPLBF(nvsdmPort_t const port, nvsdmPLBF_t const *PLBF);

typedef struct
{
    unsigned version;

    uint32_t operStatus : 4;
    uint32_t planeInd : 4;
    uint32_t adminStatus : 4;
    uint32_t lpMSBReserved : 2;
    uint32_t pnat : 2;
    uint32_t localPortReserved : 8;
    uint32_t swid : 8;

    uint32_t e : 2;
    uint32_t reserved1 : 2;
    uint32_t physicalStateStatus : 4;
    uint32_t fd : 1;
    uint32_t sleepCap : 1;
    uint32_t lockEn : 1;
    uint32_t reserved2 : 1;
    uint32_t psE : 4;
    uint32_t logicalStateStatus : 3;
    uint32_t reserved3 : 1;
    uint32_t lsE : 4;
    uint32_t reserved4 : 4;
    uint32_t eePs : 1;
    uint32_t eeLs : 1;
    uint32_t ee : 1;
    uint32_t ase : 1;

    uint32_t reserved5 : 24;
    uint32_t lockMode : 6;
    uint32_t reserved6 : 2;

    uint32_t reserved7;
} nvsdmPAOS_v1_t;
typedef nvsdmPAOS_v1_t nvsdmPAOS_t;
#define nvsdmPAOS_v1 NVSDM_STRUCT_VERSION(nvsdmPAOS, 1)

/**
 * Retrieve PAOS Information from the given port
 * @param[in]       port            The target port
 * @param[out]      paos            Versioned structure containing PAOS information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref paos is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref paos->version is not valid
 */
nvsdmRet_t nvsdmPortGetPAOS(nvsdmPort_t const port, nvsdmPAOS_t *paos);

/**
 * Set PAOS Information for the given port
 * @param[in]       port            The target port
 * @param[in]       paos            Versioned structure containing PAOS information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref paos is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref paos->version is not valid
 */
nvsdmRet_t nvsdmPortSetPAOS(nvsdmPort_t const port, nvsdmPAOS_t const *paos);

typedef struct
{
    unsigned version;

    uint32_t dstPort : 16;
    uint32_t mc : 1;
    uint32_t msl : 1;
    uint32_t dr : 1;
    uint32_t reserved1 : 1;
    uint32_t protocol : 3;
    uint32_t reserved2 : 1;
    uint32_t numPckt : 8;

    uint32_t pcktLength : 14;
    uint32_t reserved3 : 2;
    uint32_t fc : 1;
    uint32_t reserved4 : 3;
    uint32_t payload : 2;
    uint32_t reserved5 : 2;
    uint32_t numPcktHigh : 8;

    uint8_t header[60];

    uint8_t reserved6[20];

    uint32_t userPayload;

    uint32_t slVector : 16;
    uint32_t reserved7 : 16;

    uint8_t mcPorts[64];
} nvsdmMOPIR_v1_t;
typedef nvsdmMOPIR_v1_t nvsdmMOPIR_t;
#define nvsdmMOPIR_v1 NVSDM_STRUCT_VERSION(nvsdmMOPIR, 1)

/**
 * Execute MOPIR Information from the given port
 * @param[in]       port            The target port
 * @param[out]      mopir           Versioned structure containing MOPIR information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref mopir is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref mopir->version is not valid
 */
nvsdmRet_t nvsdmPortExecuteMOPIR(nvsdmPort_t const port, nvsdmMOPIR_t *MOPIR);

#define NVSDM_LFT_NUM_PORT_BLOCK_ELEMENTS 64 /* 512 bits */
typedef struct
{
    unsigned version;
    /* Input param */
    unsigned blockID;
    /* Input/Output */
    uint8_t portBlockElement[NVSDM_LFT_NUM_PORT_BLOCK_ELEMENTS];
} nvsdmLFT_v1_t;
typedef nvsdmLFT_v1_t nvsdmLFT_t;
#define nvsdmLFT_v1 NVSDM_STRUCT_VERSION(nvsdmLFT, 1)

/**
 * Retrieve Linear Forwarding Table ("LFT") Information from the given port
 * @param[in]       port            The target port
 * @param[out]      lft             Versioned structure containing LFT information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref lft is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref lft->version is not valid
 */
nvsdmRet_t nvsdmPortGetLFT(nvsdmPort_t const port, nvsdmLFT_t *lft);

/**
 * Set Linear Forwarding Table ("LFT") Information for the given port
 * @param[in]       port            The target port
 * @param[out]      lft             Versioned structure containing LFT information
 *
 * @returns         @ref NVSDM_SUCCESS                         on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED             if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                              prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG               if @p port is not a valid nvsdm port  handle, or if @ref lft is NULL
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED     if @ref lft->version is not valid
 */
nvsdmRet_t nvsdmPortSetLFT(nvsdmPort_t const port, nvsdmLFT_t const *lft);

/* *****************************************
 * Generic MAD/UMAD functions
 * *****************************************
 */
/**
 * Send a generic MAD packet after first encapsulating it in a UMAD header and retrieve the response.
 * @param[in]       dest        The destination port
 * @param[in]       mgmtClass   The management class of the request.
 * @param[out]      outBuff     Array holding the response received. Size must be NVSDM_MAD_PACKET_SIZE bytes long.
 * @param[in]       mad         The MAD packet to send. Size must be NVSDM_MAD_PACKET_SIZE bytes long.
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED       if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                      prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p dest is not a valid nvsdm port handle, or if any of @p outBuff,
 *                                                      @p bytesRead or @p mad are NULL
 *                  @ref NVSDM_ERROR_UNKNOWN             an unknown error
 */
nvsdmRet_t nvsdmPortExecuteMAD(nvsdmPort_t const dest, int mgmtClass, uint8_t *outBuff,  uint8_t const *mad);

/* *****************************************
 * Display/print functions
 * *****************************************
 */
/**
 * Get a string description for the given device. Terminated with '\\0'. The format of the string is as follows:
 * Name = \<device_name\>, DevID = \<device_ID\>, VendorID = \<vendor_ID\>, GUID = \<hexadecimal_GUID\>
 * @param[in]       device      The input device
 * @param[out]      str         The string representation of @p device
 * @param[in]       strSize     The max size of the @p str array
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED         if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                          prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p device is not a valid nvsdm device handle
 * @returns         @ref NVSDM_ERROR_INSUFFICIENT_SIZE     if @p strSize is not large enough to contain the full string description (including the
 *                                                          trailing '\\0' character). Minimum size required is NVSDM_DEV_INFO_ARRAY_SIZE.
 */
nvsdmRet_t nvsdmDeviceToString(nvsdmDevice_t const device, char str[], unsigned int strSize);

/**
 * Retrieve the name of the device, as corresponding to the IB "desc" field.
 * @param[in]       device      The input device
 * @param[out]      str         The name of @p device
 * @param[in]       strSize     The max size of the @p name array
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p device is not a valid nvsdm device handle, or if @p str is NULL
 * @returns         @ref NVSDM_ERROR_INSUFFICIENT_SIZE     if @p strSize is less than @ref NVSDM_DESC_ARRAY_SIZE.
 */
nvsdmRet_t nvsdmDeviceGetName(nvsdmDevice_t const device, char str[], unsigned int strSize);
/**
 * Retrieve a unique shortened name of the target device. This shortened name will have the prefix "CA", "SW" or "RO"
 * (corresponding to the device being one of an HCA, a SWitch or a ROuter, respectively), followed by a monotonically increasing
 * integer. Examples of names include: "CA-01, CA-02, SW-01 and RO-1". This shortened form of the name is especially useful
 * when printing out a topology matrix, for instance.
 * @param[in]       device      The input device
 * @param[out]      str         The unique shortened name of @p device
 * @param[in]       strSize     The max size of the @p name array
 *
 * @returns         @ref NVSDM_SUCCESS                     on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG           if @p device is not a valid nvsdm device handle, or if @p str is NULL
 * @returns         @ref NVSDM_ERROR_INSUFFICIENT_SIZE     if @p strSize is less than @ref NVSDM_DESC_ARRAY_SIZE.
 */
nvsdmRet_t nvsdmDeviceGetShortName(nvsdmDevice_t const device, char str[], unsigned int strSize);

/**
 * Get a string description for the given port. Terminated with '\\0'. The format of the string is as follows:
 * portId = \<port_id\>, LID = \<port_LID\>, GUID = \<hexadecimal_GUID\>
 * @param[in]       port        The input port
 * @param[out]      str         The string representation of @p port
 * @param[in]       strSize     The max size of the @p str array
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED       if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                        prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid nvsdm port handle
 * @returns         @ref NVSDM_ERROR_INSUFFICIENT_SIZE   if @p strSize is less than @ref NVSDM_PORT_INFO_ARRAY_SIZE.
 *
 */
nvsdmRet_t nvsdmPortToString(nvsdmPort_t const port, char str[], unsigned int strSize);

/* *****************************************
 * Misc helper functions
 * *****************************************
 */

/**
 * Retrieve the port number from port
 * @param[in]       port        The port whose number needs to be retrieved.
 * @param[out]      num         The retrieved port number
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port or @p num are NULL.
 */
nvsdmRet_t nvsdmPortGetNum(nvsdmPort_t const port, unsigned int *num);

/**
 * Retrieve the LID for the given port
 * @param[in]       port        The input port
 * @param[out]      lid         The LID for @p port
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED       if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                      prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid nvsdm port handle, or if @p lid is NULL
 */
nvsdmRet_t nvsdmPortGetLID(nvsdmPort_t const port, uint16_t *lid);

/**
 * Retrieve the GID for the given port
 * @param[in]       port        The input port
 * @param[out]      gid         The GID for @p port
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_UNINITIALIZED       if the nvsdm interface was not initialized i.e. @ref nvsdmInitialize was not called
 *                                                      prior to calling this function
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid nvsdm port handle, or if @p gid is NULL
 */
nvsdmRet_t nvsdmPortGetGID(nvsdmPort_t const port, uint8_t gid[16]);

/**
 * Retrieve the GUID for the given port
 * @param[in]       port        The input port
 * @param[out]      guid        The GUID for @p port
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p port is not a valid nvsdm port handle, or if @p GUID is NULL
 */
nvsdmRet_t nvsdmPortGetGUID(nvsdmPort_t const port, uint64_t *guid);

/**
 * Retrieve the GUID for the given device
 * @param[in]       device      The input device
 * @param[out]      guid        The GUID for @p device
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p GUID is NULL
 */
nvsdmRet_t nvsdmDeviceGetGUID(nvsdmDevice_t const device, uint64_t *guid);

/**
 * Retrieve the LID for the given device
 * @param[in]       device      The input device
 * @param[out]      lid         The LID for @p device
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p lid is NULL
 */
nvsdmRet_t nvsdmDeviceGetLID(nvsdmDevice_t const device, uint16_t *lid);

/**
 * Retrieve device ID.
 * @param[in]       device      The target device
 * @param[out]      devID       The retrieved device ID
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p devID is NULL
 */
nvsdmRet_t nvsdmDeviceGetDevID(nvsdmDevice_t const device, uint16_t *devID);

/**
 * Retrieve vendor ID of the given device.
 * @param[in]       device      The target device
 * @param[out]      vendorID    The retrieved vendor ID
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p vendorID is NULL
 */
nvsdmRet_t nvsdmDeviceGetVendorID(nvsdmDevice_t const device, uint32_t *vendorID);

/**
 * Retrieve device type.
 * @param[in]       device      The target device
 * @param[out]      type        The retrieved device type, one of @ref nvsdmDevType
 *
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p type is NULL
 */
nvsdmRet_t nvsdmDeviceGetType(nvsdmDevice_t const device, unsigned int *type);

/**
 * Get the health status for the given HCA nvsdmDevice if supported.
 * @param[in]       device      The target device
 * @param[out]      status      The retrieved device health status, one of @ref nvsdmDeviceHealthType_t
 * @returns         @ref NVSDM_SUCCESS                   on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG         if @p device is not a valid nvsdm device handle, or if @p type is NULL
 * @returns         @ref NVSDM_ERROR_NOT_SUPPORTED       if @p device does not support querying health status
 */
nvsdmRet_t nvsdmDeviceGetHealthStatus(nvsdmDevice_t const device, nvsdmDeviceHealthStatus_t *status);

/**
 * Retrieve the nvsdm library version number
 * @param[out]          version         Reference to the version number
 *
 * @returns             @ref NVSDM_SUCCESS                    on success
 * @returns             @ref NVSDM_ERROR_INVALID_ARG          if @p version is NULL
 */
nvsdmRet_t nvsdmGetVersion(uint64_t *version);

/**
 * Generic structure to encode extended i.e. 32b version numbers.
 */
typedef struct
{
    unsigned version;               //!< Struct version number
    uint32_t majorVersion;          //!< The major version number
    uint32_t minorVersion;          //!< The minor version number
    uint32_t patchVersion;          //!< The patch version number; also called "sub minor" in some domains
} nvsdmVersionInfo_v1_t;
typedef nvsdmVersionInfo_v1_t nvsdmVersionInfo_t;
#define nvsdmVersionInfo_v1 NVSDM_STRUCT_VERSION(nvsdmVersionInfo, 1)

/**
 * Retrieve firmware version info for the \p device. Only supported for devices of type @ref NVSDM_DEV_TYPE_SWITCH.
 * @param[in]       device      The target device
 * @param[out]      version     Contains the firmware version for the given \p device
 *
 * @returns         @ref NVSDM_SUCCESS                      on success
 * @returns         @ref NVSDM_ERROR_INVALID_ARG            if \p version is NULL
 * @returns         @ref NVSDM_ERROR_NOT_SUPPORTED          if \p device is not a supported device
 *                                                          (i.e. is not a switch)
 * @returns         @ref NVSDM_ERROR_VERSION_NOT_SUPPORTED  if \p version->version is not valid
 */
nvsdmRet_t nvsdmDeviceGetFirmwareVersion(nvsdmDevice_t const device, nvsdmVersionInfo_t *version);

/**
 * Get the range of interface versions supported by the underlying library. The library is
 * guaranteed to support every interface version in the interfal [versionFrom, versionTo]
 * (ie. both inclusive).
 * @param[out]          versionFrom         The minimum interface version supported by the library
 * @param[out]          versionTo           The maximum interface version supported by the library
 * @returns             @ref NVSDM_SUCCESS                   on success
 * @returns             @ref NVSDM_ERROR_INVALID_ARG         if any of @p versionFrom or @p versionTo are NULL
 */
nvsdmRet_t nvsdmGetSupportedInterfaceVersionRange(uint64_t *versionFrom, uint64_t *versionTo);

/**
 * Retrieve current library logging level. Use @ref nvsdmSetLogLevel to set the logging level
 * @param[out]          logLevel            The current library logging level
 * @returns             @ref NVSDM_SUCCESS                   on success
 * @returns             @ref NVSDM_ERROR_INVALID_ARG         if @p logLevel is NULL
 */
nvsdmRet_t nvsdmGetLogLevel(uint32_t *logLevel);

/**
 * Set logging level in the library to one of @ref nvsdmLogLevel. This controls the types of
 * messages printed out (to stdout) within the library. Set to @ref NVSDM_LOG_LEVEL_FATAL to turn off all
 * but the most important messages (i.e. the ones reporting and/or describing catastrophic failures).
 * Set to @ref NVSDM_LOG_LEVEL_INFO to print out all messages, including informational ones. Defaults to
 * @ref NVSDM_LOG_LEVEL_ERROR.
 * @param[in]           logLevel            The desired logging level. One of @ref nvsdmLogLevel.
 * @returns             @ref NVSDM_SUCCESS                   on success
 * @returns             @ref NVSDM_ERROR_INVALID_ARG         if @p logLevel is not one of enum nvsdmLogLevel
 */
nvsdmRet_t nvsdmSetLogLevel(uint32_t logLevel);

/**
 * Redirect all log messages to the specified file.
 * @param[in]           filePath            (Relative or absolute) Path to the desired log file
 * @returns             @ref NVSDM_SUCCESS                   on success
 * @returns             @ref NVSDM_ERROR_INVALID_ARG         if @p filePath is NULL
 * @returns             @ref NVSDM_ERROR_FILE_OPEN_FAILED    if the log file could not be opened for any reason
 */
nvsdmRet_t nvsdmSetLogFile(char const *filePath);

/**
 * Return a string describing the error code contained in @p ret
 * @param[in]           ret             The error code
 * @returns             The string describing @p ret, or NULL on error
 */
char const *nvsdmGetErrorString(nvsdmRet_t ret);

#ifdef __cplusplus
}
#endif
#endif // __NVSDM_H__

#pragma GCC diagnostic pop
