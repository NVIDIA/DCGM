/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "Pcie.h"

#include <PluginInterface.h>
#include <PluginStrings.h>

extern "C" {

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
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
                                     nullptr };

    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,  DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamBool,
            DcgmPluginParamNone };
    DCGM_CASSERT(sizeof(parameterNames) / sizeof(const char *) == sizeof(paramTypes) / sizeof(const dcgmPluginValue_t),
                 1);

    unsigned int paramCount = 0;

    for (; parameterNames[paramCount] != nullptr; paramCount++)
    {
        snprintf(info->validParameters[paramCount].parameterName,
                 sizeof(info->validParameters[paramCount].parameterName),
                 "%s",
                 parameterNames[paramCount]);
        info->validParameters[paramCount].parameterType = paramTypes[paramCount];
    }

    info->numValidParameters = paramCount;

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", PCIE_PLUGIN_NAME);
    snprintf(info->testGroup, sizeof(info->testGroup), "Perf");
    snprintf(info->description,
             sizeof(info->description),
             "This plugin will exercise the PCIe bus for a given list of GPUs.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData)
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
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL; // Previously unchecked
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL; // Previously unchecked
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS;
        statFieldIds->fieldIds[fieldCount++] = DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS;

        statFieldIds->numFieldIds = fieldCount;
    }

    BusGrind *bg = new BusGrind(handle, gpuInfo);
    *userData    = bg;

    return DCGM_ST_OK;
}

void RunTest(unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto bg = (BusGrind *)userData;
    bg->Go(numParameters, testParameters);
}


void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (customStats != nullptr)
    {
        auto bg = (BusGrind *)userData;
        bg->PopulateCustomStats(*customStats);
    }
}

void RetrieveResults(dcgmDiagResults_t *results, void *userData)
{
    auto bg = (BusGrind *)userData;
    bg->GetResults(results);
}

} // END extern "C"
