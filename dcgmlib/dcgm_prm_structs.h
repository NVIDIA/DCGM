/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @deprecated This file and structures within it are expected to be removed in the near future.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/* Helper macros for structure field definitions */
#if !defined(CONCAT_IMPL) && !defined(CONCAT) && !defined(RESERVED_FIELD)
#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y)      CONCAT_IMPL(x, y)
#define RESERVED_FIELD(len)              \
    unsigned CONCAT(reserved_, __LINE__) \
        : len
#endif

// Constants for PRM registers and groups
namespace
{
auto constexpr PRM_REG_PPCNT                 = 0x5008;
auto constexpr PPCNT_GRP_RECOVERY            = 0x1A;
auto constexpr PPCNT_GRP_PHYSICAL_LAYER_CTRS = 0x12;
auto constexpr PPCNT_GRP_PLR                 = 0x22;
auto constexpr PRM_REG_PPRM                  = 0x5059;

auto constexpr PRM_LOCAL_PORT_MIN = 1;
auto constexpr PRM_LOCAL_PORT_MAX = 1024;
}; // namespace

/*
 * PPCNT Ports Performance Counter Register Structure
 *
 * This structure represents the PPCNT register (ID 0x5008) used to access
 * physical layer statistics from GPU network interfaces. The register contains
 * header fields for port identification and configuration, followed by an array
 * of 64-bit counter values organized into different counter groups.
 *
 * Group 0x22 (PLR_COUNTERS_GROUP) contains Physical Layer Retransmission statistics
 * Group 0x12 (PHYSICAL_LAYER_COUNTERS) contains Physical Layer error statistics
 * Group 0x1A (PHYSICAL_LAYER_RECOVERY_COUNTERS) contains recovery event statistics
 */
typedef struct
{
    /* DWord 1 */
    unsigned grp        : 6;
    unsigned reserved1  : 2;
    unsigned port_type  : 4;
    unsigned lp_msb     : 2;
    unsigned pnat       : 2;
    unsigned local_port : 8;
    unsigned swid       : 8;

    /* DWord 2 */
    unsigned prio_tc      : 5;
    unsigned grp_profile  : 3;
    unsigned plane_ind    : 4;
    unsigned reserved2    : 17;
    unsigned counters_cap : 1;
    unsigned lp_gl        : 1;
    unsigned clr          : 1;

    /* DWord 3--63: Counter data payload */
    unsigned counter_set[62];
} nvmlPpcnt_t;


namespace
{
static_assert(sizeof(nvmlPpcnt_t) == 256,
              "nvmlPpcnt_t size mismatch - expected 256 bytes (64 DWords), possible padding detected");
static_assert(offsetof(nvmlPpcnt_t, counter_set) == 0x8,
              "nvmlPpcnt_t size mismatch: expected 0x8 offset for counter_set");
} // namespace

/*
 * Physical Layer Recovery Counters Group Structure - Group 0x1A
 *
 * This structure represents the Physical Layer Recovery Counters Group (ID 0x1A)
 * used to access physical layer recovery statistics from GPU network interfaces.
 * Contains recovery event counts, timing information, and FEC error statistics
 * collected during recovery operations.
 *
 * @deprecated This structure is expected to be removed in the near future.
 */
typedef struct
{
    unsigned total_successful_recovery_events;                   //!< Total successful recovery events
    unsigned unintentional_link_down_events;                     //!< Unintentional link down events
    unsigned intentional_link_down_events;                       //!< Intentional link down events
    unsigned time_in_last_host_logical_recovery;                 //!< Time in last host logical recovery
    unsigned time_in_last_host_serdes_feq_recovery;              //!< Time in last host SerDes FEQ recovery
    unsigned time_in_last_module_tx_disable_recovery;            //!< Time in last module TX disable recovery
    unsigned time_in_last_module_datapath_full_toggle_recovery;  //!< Time in last module datapath full toggle recovery
    unsigned total_time_in_host_logical_recovery;                //!< Total time in host logical recovery
    unsigned total_time_in_host_serdes_feq_recovery;             //!< Total time in host SerDes FEQ recovery
    unsigned total_time_in_module_tx_disable_recovery;           //!< Total time in module TX disable recovery
    unsigned total_time_in_module_datapath_full_toggle_recovery; //!< Total time in module datapath full toggle recovery
    unsigned total_host_logical_recovery_count;                  //!< Total host logical recovery count
    unsigned total_host_serdes_feq_recovery_count;               //!< Total host SerDes FEQ recovery count
    unsigned total_module_tx_disable_recovery_count;             //!< Total module TX disable recovery count
    unsigned total_module_datapath_full_toggle_recovery_count;   //!< Total module datapath full toggle recovery count
    unsigned total_host_logical_successful_recovery_count;       //!< Total host logical successful recovery count
    unsigned total_host_serdes_feq_successful_recovery_count;    //!< Total host SerDes FEQ successful recovery count
    unsigned total_module_tx_disable_successful_recovery_count;  //!< Total module TX disable successful recovery count
    unsigned total_module_datapath_full_toggle_successful_recovery_count; //!< Total module datapath full toggle
                                                                          //!< successful recovery count
    unsigned time_since_last_recovery;                                    //!< Time since last recovery
    unsigned last_host_logical_recovery_attempts_count;                   //!< Last host logical recovery attempts count
    unsigned last_host_serdes_feq_attempts_count;                         //!< Last host SerDes FEQ attempts count

    /* Offset 0x58: Bitfield containing time between recoveries */
    unsigned time_between_last_2_recoveries : 16; //!< Time between last 2 recoveries (16-bit)
    RESERVED_FIELD(16);

    /* Offset 0x5C: FEC uncorrectable error statistics during recovery */
    unsigned last_rs_fec_uncorrectable_during_recovery_high;  //!< Last RS-FEC uncorrectable during recovery (high)
    unsigned last_rs_fec_uncorrectable_during_recovery_low;   //!< Last RS-FEC uncorrectable during recovery (low)
    unsigned total_rs_fec_uncorrectable_during_recovery_high; //!< Total RS-FEC uncorrectable during recovery (high)
    unsigned total_rs_fec_uncorrectable_during_recovery_low;  //!< Total RS-FEC uncorrectable during recovery (low)

    /* Offset 0x6C - 0xF0: Reserved padding to reach structure end at 0xF4 */
    unsigned unused2[34]; //!< Reserved padding (34 DWords)
} nvmlPhysicalLayerRecoveryCounters_t;

namespace
{
static_assert(sizeof(nvmlPhysicalLayerRecoveryCounters_t) == 244,
              "nvmlPhysicalLayerRecoveryCounters_t size mismatch - expected 244 bytes");
static_assert(
    offsetof(nvmlPhysicalLayerRecoveryCounters_t, total_rs_fec_uncorrectable_during_recovery_low) == 0x5C + (3 * 4),
    "nvmlPhysicalLayerRecoveryCounters_t size mismatch: expected 0x5C + (3 * 4) offset for total_rs_fec_uncorrectable_during_recovery_low");
} // namespace

/*
 * Physical Layer Counters Group Structure - Group 0x12
 *
 * This structure represents the Physical Layer Counters Group (ID 0x12)
 * used to access physical layer error and FEC statistics from GPU network interfaces.
 * Contains 64-bit counters for various error types, FEC corrections, and recovery events.
 * All high/low pairs represent 64-bit values split across two 32-bit fields.
 *
 * @deprecated This structure is expected to be removed in the near future.
 */
typedef struct
{
    unsigned time_since_last_clear_high;             //!< Time since last clear (high 32-bit)
    unsigned time_since_last_clear_low;              //!< Time since last clear (low 32-bit)
    unsigned symbol_errors_high;                     //!< Symbol errors (high 32-bit)
    unsigned symbol_errors_low;                      //!< Symbol errors (low 32-bit)
    unsigned sync_headers_errors_high;               //!< Sync header errors (high 32-bit)
    unsigned sync_headers_errors_low;                //!< Sync header errors (low 32-bit)
    unsigned edpl_bip_errors_lane0_high;             //!< EDPL BIP errors lane 0 (high 32-bit)
    unsigned edpl_bip_errors_lane0_low;              //!< EDPL BIP errors lane 0 (low 32-bit)
    unsigned edpl_bip_errors_lane1_high;             //!< EDPL BIP errors lane 1 (high 32-bit)
    unsigned edpl_bip_errors_lane1_low;              //!< EDPL BIP errors lane 1 (low 32-bit)
    unsigned edpl_bip_errors_lane2_high;             //!< EDPL BIP errors lane 2 (high 32-bit)
    unsigned edpl_bip_errors_lane2_low;              //!< EDPL BIP errors lane 2 (low 32-bit)
    unsigned edpl_bip_errors_lane3_high;             //!< EDPL BIP errors lane 3 (high 32-bit)
    unsigned edpl_bip_errors_lane3_low;              //!< EDPL BIP errors lane 3 (low 32-bit)
    unsigned fc_fec_corrected_blocks_lane0_high;     //!< FC-FEC corrected blocks lane 0 (high 32-bit)
    unsigned fc_fec_corrected_blocks_lane0_low;      //!< FC-FEC corrected blocks lane 0 (low 32-bit)
    unsigned fc_fec_corrected_blocks_lane1_high;     //!< FC-FEC corrected blocks lane 1 (high 32-bit)
    unsigned fc_fec_corrected_blocks_lane1_low;      //!< FC-FEC corrected blocks lane 1 (low 32-bit)
    unsigned fc_fec_corrected_blocks_lane2_high;     //!< FC-FEC corrected blocks lane 2 (high 32-bit)
    unsigned fc_fec_corrected_blocks_lane2_low;      //!< FC-FEC corrected blocks lane 2 (low 32-bit)
    unsigned fc_fec_corrected_blocks_lane3_high;     //!< FC-FEC corrected blocks lane 3 (high 32-bit)
    unsigned fc_fec_corrected_blocks_lane3_low;      //!< FC-FEC corrected blocks lane 3 (low 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane0_high; //!< FC-FEC uncorrectable blocks lane 0 (high 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane0_low;  //!< FC-FEC uncorrectable blocks lane 0 (low 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane1_high; //!< FC-FEC uncorrectable blocks lane 1 (high 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane1_low;  //!< FC-FEC uncorrectable blocks lane 1 (low 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane2_high; //!< FC-FEC uncorrectable blocks lane 2 (high 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane2_low;  //!< FC-FEC uncorrectable blocks lane 2 (low 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane3_high; //!< FC-FEC uncorrectable blocks lane 3 (high 32-bit)
    unsigned fc_fec_uncorrectable_blocks_lane3_low;  //!< FC-FEC uncorrectable blocks lane 3 (low 32-bit)
    unsigned rs_fec_corrected_blocks_high;           //!< RS-FEC corrected blocks (high 32-bit)
    unsigned rs_fec_corrected_blocks_low;            //!< RS-FEC corrected blocks (low 32-bit)
    unsigned rs_fec_uncorrectable_blocks_high;       //!< RS-FEC uncorrectable blocks (high 32-bit)
    unsigned rs_fec_uncorrectable_blocks_low;        //!< RS-FEC uncorrectable blocks (low 32-bit)
    unsigned rs_fec_no_errors_blocks_high;           //!< RS-FEC no error blocks (high 32-bit)
    unsigned rs_fec_no_errors_blocks_low;            //!< RS-FEC no error blocks (low 32-bit)
    RESERVED_FIELD(32);                              //!< Reserved field
    RESERVED_FIELD(32);                              //!< Reserved field
    unsigned rs_fec_corrected_symbols_total_high;    //!< RS-FEC corrected symbols total (high 32-bit)
    unsigned rs_fec_corrected_symbols_total_low;     //!< RS-FEC corrected symbols total (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane0_high;    //!< RS-FEC corrected symbols lane 0 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane0_low;     //!< RS-FEC corrected symbols lane 0 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane1_high;    //!< RS-FEC corrected symbols lane 1 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane1_low;     //!< RS-FEC corrected symbols lane 1 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane2_high;    //!< RS-FEC corrected symbols lane 2 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane2_low;     //!< RS-FEC corrected symbols lane 2 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane3_high;    //!< RS-FEC corrected symbols lane 3 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane3_low;     //!< RS-FEC corrected symbols lane 3 (low 32-bit)
    unsigned link_down_events;                       //!< Link down events
    unsigned successful_recovery_events;             //!< Successful recovery events
    unsigned rs_fec_corrected_symbols_lane4_high;    //!< RS-FEC corrected symbols lane 4 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane4_low;     //!< RS-FEC corrected symbols lane 4 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane5_high;    //!< RS-FEC corrected symbols lane 5 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane5_low;     //!< RS-FEC corrected symbols lane 5 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane6_high;    //!< RS-FEC corrected symbols lane 6 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane6_low;     //!< RS-FEC corrected symbols lane 6 (low 32-bit)
    unsigned rs_fec_corrected_symbols_lane7_high;    //!< RS-FEC corrected symbols lane 7 (high 32-bit)
    unsigned rs_fec_corrected_symbols_lane7_low;     //!< RS-FEC corrected symbols lane 7 (low 32-bit)
    RESERVED_FIELD(32);                              //!< Reserved field
    RESERVED_FIELD(32);                              //!< Reserved field
    RESERVED_FIELD(32);                              //!< Reserved field
    RESERVED_FIELD(32);                              //!< Reserved field
} nvmlPhysicalLayerCounters_t;

namespace
{
static_assert(sizeof(nvmlPhysicalLayerCounters_t) == 248,
              "nvmlPhysicalLayerCounters_t size mismatch - expected 248 bytes");
static_assert(offsetof(nvmlPhysicalLayerCounters_t, rs_fec_corrected_symbols_lane7_low) == 0xE4,
              "nvmlPhysicalLayerCounters_t size mismatch: expected 0xe4 offset for rs_fec_corrected_symbols_lane7_low");
} // namespace

/**
 * PLR Counters Group Structure - Group 0x22
 *
 * @deprecated This structure is expected to be removed in the near future.
 */
typedef struct
{
    /* Offsets 0x00 - 0x44 */
    unsigned plr_rcv_codes_high;                          //!< PLR received codes (high 32-bit)
    unsigned plr_rcv_codes_low;                           //!< PLR received codes (low 32-bit)
    unsigned plr_rcv_code_err_high;                       //!< PLR received code errors (high 32-bit)
    unsigned plr_rcv_code_err_low;                        //!< PLR received code errors (low 32-bit)
    unsigned plr_rcv_uncorrectable_code_high;             //!< PLR received uncorrectable codes (high 32-bit)
    unsigned plr_rcv_uncorrectable_code_low;              //!< PLR received uncorrectable codes (low 32-bit)
    unsigned plr_xmit_codes_high;                         //!< PLR transmit codes (high 32-bit)
    unsigned plr_xmit_codes_low;                          //!< PLR transmit codes (low 32-bit)
    unsigned plr_xmit_retry_codes_high;                   //!< PLR transmit retry codes (high 32-bit)
    unsigned plr_xmit_retry_codes_low;                    //!< PLR transmit retry codes (low 32-bit)
    unsigned plr_xmit_retry_events_high;                  //!< PLR transmit retry events (high 32-bit)
    unsigned plr_xmit_retry_events_low;                   //!< PLR transmit retry events (low 32-bit)
    unsigned plr_sync_events_high;                        //!< PLR sync events (high 32-bit)
    unsigned plr_sync_events_low;                         //!< PLR sync events (low 32-bit)
    unsigned plr_codes_loss_high;                         //!< PLR codes loss (high 32-bit)
    unsigned plr_codes_loss_low;                          //!< PLR codes loss (low 32-bit)
    unsigned plr_xmit_retry_events_within_t_sec_max_high; //!< PLR transmit retry events within t_sec_max (high 32-bit)
    unsigned plr_xmit_retry_events_within_t_sec_max_low;  //!< PLR transmit retry events within t_sec_max (low 32-bit)

    /* Offset 0x48 */
    unsigned reserved[3];

    /* Offsets 0x54 - 0xF8: Reserved/padding (last word at 0xF4-0xF8) */
    uint32_t reserved2[41];
} nvmlPlrCounters_t;

namespace
{
static_assert(sizeof(nvmlPlrCounters_t) == 0xF8,
              "nvmlPlrCounters_t size mismatch: expected 248 bytes (0xF8) with last 32-bit word at offset 0xF4");
static_assert(offsetof(nvmlPlrCounters_t, plr_xmit_retry_events_within_t_sec_max_low) == 0x44,
              "nvmlPlrCounters_t size mismatch: expected 0x44 offset for plr_xmit_retry_events_within_t_sec_max_low");
} // namespace

/*
 * PPRM (Port Phy Recovery Mode Register) Structure - Register 0x5059
 *
 * This structure represents the PPRM register used to configure and monitor
 * physical layer recovery modes for GPU network interfaces. It contains
 * configuration fields for various recovery mechanisms and operational status.
 *
 * @deprecated This structure is expected to be removed in the near future.
 */
typedef struct
{
    /* DWord 1: Port identification and configuration */
    RESERVED_FIELD(2);
    unsigned ovrd_no_neg_bhvr : 2; //!< Override no negotiation behavior
    RESERVED_FIELD(4);
    unsigned plane_ind  : 4; //!< Plane index
    unsigned lp_msb     : 2; //!< Local port MSB
    unsigned pnat       : 2; //!< Port NAT
    unsigned local_port : 8; //!< Local port number
    RESERVED_FIELD(8);

    /* DWord 2: Recovery capabilities and configuration */
    unsigned recovery_types_cap : 8; //!< Recovery types capabilities
    RESERVED_FIELD(16);
    unsigned no_neg_bhvr            : 4; //!< No negotiation behavior
    unsigned wd_logic_re_lock_res   : 2; //!< Watchdog logic re-lock resolution
    unsigned oper_logic_re_lock_res : 2; //!< Operational logic re-lock resolution

    /* DWord 3: Recovery mechanism configuration */
    unsigned module_datapath_full_toggle : 4; //!< Module datapath full toggle
    unsigned module_tx_disable           : 4; //!< Module TX disable
    unsigned host_serdes_feq             : 4; //!< Host SerDes FEQ
    unsigned host_logic_re_lock          : 4; //!< Host logic re-lock
    RESERVED_FIELD(16);

    /* DWord 4: Timeout configuration */
    unsigned link_down_timeout : 16; //!< Link down timeout
    RESERVED_FIELD(16);

    /* DWord 5 */
    unsigned link_down_timeout_oper : 16; //!< Link down timeout operational
    RESERVED_FIELD(16);

    /* DWord 6: Draining timeout */
    unsigned draining_timeout : 8; //!< Draining timeout
    RESERVED_FIELD(24);

    /* DWord 7: Operational recovery status */
    unsigned oper_recovery : 8; //!< Operational recovery status
    RESERVED_FIELD(24);

    /* DWord 8: Watchdog configuration */
    unsigned wd_module_full_toggle : 8; //!< Watchdog module full toggle
    unsigned wd_module_tx_disable  : 8; //!< Watchdog module TX disable
    unsigned wd_host_serdes_feq    : 8; //!< Watchdog host SerDes FEQ
    unsigned wd_host_logic_re_lock : 8; //!< Watchdog host logic re-lock

    /* DWord 9: Operational timing */
    unsigned oper_time_module_full_toggle : 8; //!< Operational time module full toggle
    unsigned oper_time_module_tx_disable  : 8; //!< Operational time module TX disable
    unsigned oper_time_host_serdes_feq    : 8; //!< Operational time host SerDes FEQ
    unsigned oper_time_host_logic_re_lock : 8; //!< Operational time host logic re-lock

    // DWords 10-17 (Offsets 0x24 - 0x4c): Reserved
    unsigned reserved[9];
} nvmlPprm_t;

namespace
{
static_assert(sizeof(nvmlPprm_t) == 72,
              "nvmlPprm_t size mismatch - expected 72 bytes (18 DWords), possible padding detected");
} // namespace
