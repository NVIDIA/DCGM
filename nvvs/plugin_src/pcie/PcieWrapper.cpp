/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmStringHelpers.h"
#include "Pcie.h"
#include "dcgm_fields.h"

#include <PluginCommon.h>
#include <PluginInterface.h>
#include <PluginLib.h>
#include <PluginStrings.h>

extern "C" {

unsigned int const *GetFieldIds(size_t &fieldCount)
{
    static const unsigned int PCIE_PLUGIN_FIELDS[]
        = { // PCIe replay and recovery error tracking
            DCGM_FI_DEV_PCIE_REPLAY_TOTAL,
            DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL,
            DCGM_FI_DEV_SXID_NON_FATAL_ERROR,
            DCGM_FI_DEV_SXID_FATAL_ERROR,
            DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW,
            DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL,
            DCGM_FI_DEV_PCIE_CORRECTABLE_ERROR_TOTAL,

            // NVLink 4 and below supported counters
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL,
            DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL,
            DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL,
            DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL,

            // NVLink 5+ supported fieldIds - packet level and symbol level error tracking
            DCGM_FI_DEV_NVLINK_RECOVERY_EVENT_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL,
            DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL,
            DCGM_FI_DEV_NVLINK_TX_PACKET_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_PACKET_TOTAL,
            DCGM_FI_DEV_NVLINK_TX_BYTES_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_BYTES_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL,
            DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL
          };

    fieldCount = std::size(PCIE_PLUGIN_FIELDS);
    return PCIE_PLUGIN_FIELDS;
}

unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}

dcgmReturn_t GetPluginInfo(unsigned int /* pluginInterfaceVersion */, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    // parameterNames must be null terminated
    const char *parameterNames[] = { PCIE_STR_TEST_PINNED,
                                     PCIE_STR_TEST_UNPINNED,
                                     PCIE_STR_TEST_P2P_ON,
                                     PCIE_STR_TEST_P2P_OFF,
                                     PCIE_STR_NVSWITCH_NON_FATAL_CHECK,
                                     PCIE_STR_IS_ALLOWED,
                                     PCIE_STR_INTS_PER_COPY,
                                     PCIE_STR_ITERATIONS,
                                     PCIE_STR_MIN_BANDWIDTH,
                                     PCIE_STR_MAX_LATENCY,
                                     PCIE_STR_MIN_PCI_GEN,
                                     PCIE_STR_MIN_PCI_WIDTH,
                                     PCIE_STR_MAX_PCIE_REPLAYS,
                                     PCIE_STR_MAX_MEMORY_CLOCK,
                                     PCIE_STR_MAX_GRAPHICS_CLOCK,
                                     PCIE_STR_CRC_ERROR_THRESHOLD,
                                     PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED,
                                     PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED,
                                     PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED,
                                     PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED,
                                     PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED,
                                     PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED,
                                     PCIE_SUBTEST_P2P_BW_P2P_ENABLED,
                                     PCIE_SUBTEST_P2P_BW_P2P_DISABLED,
                                     PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED,
                                     PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED,
                                     PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED,
                                     PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED,
                                     PCIE_SUBTEST_P2P_LATENCY_P2P_ENABLED,
                                     PCIE_SUBTEST_P2P_LATENCY_P2P_DISABLED,
                                     PCIE_STR_TEST_NVLINK_STATUS,
                                     PCIE_STR_TEST_BROKEN_P2P,
                                     SMSTRESS_STR_TEST_DURATION,
                                     SMSTRESS_STR_TARGET_PERF,
                                     SMSTRESS_STR_TEMPERATURE_MAX,
                                     SMSTRESS_STR_USE_DGEMM,
                                     SMSTRESS_STR_MATRIX_DIM,
                                     PCIE_STR_AER_THRESHOLD,
                                     PCIE_STR_TEST_WITH_GEMM,
                                     PCIE_STR_DISABLE_TESTS,
                                     PCIE_STR_GPU_NVLINKS_EXPECTED_UP,
                                     PCIE_STR_NVSWITCH_NVLINKS_EXPECTED_UP,
                                     PCIE_STR_PARALLEL_BW_CHECK_DURATION,
                                     PCIE_STR_DONT_BIND_NUMA,
                                     PCIE_STR_MAX_NVLINK_RECOVERY_ERRORS,
                                     PCIE_STR_MAX_PCIE_CORRECTABLE_ERRORS,
                                     nullptr };
    char const *description      = "This plugin will exercise the PCIe bus for a given list of GPUs.";
    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamBool,  DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool,  DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,   DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,   DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool,  DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool,  DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool,  DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamInt,
            DcgmPluginParamBool,  DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamBool,  DcgmPluginParamString,
            DcgmPluginParamInt,   DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamBool,  DcgmPluginParamInt,
            DcgmPluginParamFloat, DcgmPluginParamNone };

    DCGM_CASSERT(sizeof(parameterNames) / sizeof(const char *) == sizeof(paramTypes) / sizeof(const dcgmPluginValue_t),
                 1);

    unsigned int paramCount = 0;

    info->numTests = 1;
    for (; parameterNames[paramCount] != nullptr; paramCount++)
    {
        SafeCopyTo(info->tests[0].validParameters[paramCount].parameterName, parameterNames[paramCount]);
        info->tests[0].validParameters[paramCount].parameterType = paramTypes[paramCount];
    }

    info->tests[0].numValidParameters = paramCount;

    SafeCopyTo(info->pluginName, static_cast<char const *>(PCIE_PLUGIN_NAME));
    SafeCopyTo(info->description, description);
    SafeCopyTo(info->tests[0].testName, static_cast<char const *>(PCIE_PLUGIN_NAME));
    SafeCopyTo(info->tests[0].description, description);
    SafeCopyTo(info->tests[0].testCategory, PCIE_PLUGIN_CATEGORY);
    info->tests[0].targetEntityGroup = DCGM_FE_GPU;

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback,
                              dcgmDiagPluginAttr_v1 const *pluginAttr,
                              HangDetectMonitor *monitor)
{
    if (statFieldIds != nullptr)
    {
        size_t fieldCount;
        unsigned int const *fieldIds = GetFieldIds(fieldCount);

        statFieldIds->numFieldIds = fieldCount;
        std::copy(fieldIds, fieldIds + fieldCount, statFieldIds->fieldIds);
    }

    BusGrind *bg = new BusGrind(handle);
    *userData    = bg;

    bg->SetPluginAttr(pluginAttr);
    bg->SetHangDetectMonitor(monitor);
    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, bg->GetDisplayName());

    return DCGM_ST_OK;
}

void RunTest(char const *testName,
             unsigned int /* timeout */,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             dcgmDiagPluginEntityList_v1 const *entityInfo,
             void *userData)
{
    auto *bg = static_cast<BusGrind *>(userData);
    bg->Go(testName, entityInfo, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (testName != nullptr && customStats != nullptr)
    {
        auto *bg = static_cast<BusGrind *>(userData);
        bg->PopulateCustomStats(testName, *customStats);
    }
}

void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData)
{
    auto *bg = static_cast<BusGrind *>(userData);
    bg->GetResults(testName, entityResults);
}

} // END extern "C"
