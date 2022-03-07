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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PcieMain.h"
#include <cstdlib>
#include <cstring>

/*****************************************************************************/
BusGrind::BusGrind(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_handle(handle)
    , m_gpuInfo()
{
    m_infoStruct.testIndex        = DCGM_PCI_INDEX;
    m_infoStruct.shortDescription = "This plugin will exercise the PCIe bus for a given list of GPUs.";
    m_infoStruct.testGroups       = "Perf";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = PCIE_PLUGIN_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(PCIE_STR_TEST_PINNED, "True");
    tp->AddString(PCIE_STR_TEST_UNPINNED, "True");
    tp->AddString(PCIE_STR_TEST_P2P_ON, "True");
    tp->AddString(PCIE_STR_TEST_P2P_OFF, "True");
    tp->AddString(PCIE_STR_TEST_BROKEN_P2P, "True");
    tp->AddString(PS_LOGFILE, "stats_pcie.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);

    tp->AddString(PCIE_STR_IS_ALLOWED, "False");

    tp->AddDouble(PCIE_STR_MAX_PCIE_REPLAYS, 80.0);

    tp->AddDouble(PCIE_STR_MAX_MEMORY_CLOCK, 0.0);
    tp->AddDouble(PCIE_STR_MAX_GRAPHICS_CLOCK, 0.0);
    // CRC_ERROR_THRESHOLD is the number of CRC errors per second, per RM recommendation
    tp->AddDouble(PCIE_STR_CRC_ERROR_THRESHOLD, 100.0);
    tp->AddString(PCIE_STR_NVSWITCH_NON_FATAL_CHECK, "False");

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_PCI_GEN, 1.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_PCI_WIDTH, 1.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_PCI_GEN, 1.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_PCI_WIDTH, 1.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_ITERATIONS, 5000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_MAX_LATENCY, 100000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_ITERATIONS, 5000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_MAX_LATENCY, 100000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, 10000000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_LATENCY_P2P_ENABLED, PCIE_STR_ITERATIONS, 5000.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_LATENCY_P2P_DISABLED, PCIE_STR_ITERATIONS, 5000.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB, 4096.0);

    m_infoStruct.defaultTestParameters = tp;

    if (gpuInfo == nullptr)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "No GPU information specified.");
        AddError(d);
    }
    else
    {
        InitializeForGpuList(*gpuInfo);
        m_gpuInfo = *gpuInfo;
    }
}

/*****************************************************************************/
void BusGrind::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct)
{
    if (UsingFakeGpus())
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(NVVS_RESULT_PASS);
        return;
    }

    TestParameters testParameters(*(m_infoStruct.defaultTestParameters));
    testParameters.SetFromStruct(numParameters, tpStruct);

    if (!testParameters.GetBoolFromString(PCIE_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, PCIE_PLUGIN_NAME);
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }

    int st = main_entry(m_gpuInfo, this, &testParameters);
    if (main_should_stop)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(NVVS_RESULT_SKIP);
    }
    else if (st)
    {
        // Fatal error in plugin or test could not be initialized
        SetResult(NVVS_RESULT_FAIL);
    }
}

/*****************************************************************************/
dcgmHandle_t BusGrind::GetHandle()
{
    return m_handle;
}
