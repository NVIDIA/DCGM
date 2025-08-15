/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
                                     nullptr };
    char const *description      = "This plugin will exercise the PCIe bus for a given list of GPUs.";
    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,  DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamInt,
            DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamBool,  DcgmPluginParamString,
            DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamBool,  DcgmPluginParamInt,
            DcgmPluginParamNone };

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
                              dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    if (statFieldIds != nullptr)
    {
        int fieldCount                       = 0;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL; // Previously unchecked
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL; // Previously unchecked
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS;

        statFieldIds->numFieldIds = fieldCount;
    }

    BusGrind *bg = new BusGrind(handle);
    *userData    = bg;

    bg->SetPluginAttr(pluginAttr);
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
