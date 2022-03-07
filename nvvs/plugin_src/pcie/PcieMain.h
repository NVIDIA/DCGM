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
#ifndef PCIEMAIN_H
#define PCIEMAIN_H

#include "DcgmRecorder.h"
#include "NvvsDeviceList.h"
#include "Pcie.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include "cuda_runtime.h"
#include <PluginDevice.h>

#define PCIE_MAX_GPUS 16

/*****************************************************************************/

/* Struct to hold global information for this test */
class BusGrindGlobals
{
public:
    TestParameters *testParameters; /* Parameters passed in from the framework */

    /* Cached parameters */
    bool test_pinned;
    bool test_unpinned;
    bool test_p2p_on;
    bool test_p2p_off;
    bool test_broken_p2p;

    BusGrind *busGrind; /* Plugin handle for setting status */

    std::vector<PluginDevice *> gpu; /* Per-gpu information */
    DcgmRecorder *m_dcgmRecorder;

    bool m_dcgmCommErrorOccurred;
    bool m_printedConcurrentGpuErrorMessage;

    BusGrindGlobals()
        : testParameters(nullptr)
        , test_pinned(false)
        , test_unpinned(false)
        , test_p2p_on(false)
        , test_p2p_off(false)
        , test_broken_p2p(true)
        , busGrind(nullptr)
        , gpu()
        , m_dcgmRecorder(nullptr)
        , m_dcgmCommErrorOccurred(false)
        , m_printedConcurrentGpuErrorMessage(false)
    {}
};

/*****************************************************************************/
int main_entry(const dcgmDiagPluginGpuList_t &gpuList, BusGrind *busGrind, TestParameters *testParameters);

/*****************************************************************************/

#endif // PCIEMAIN_H
