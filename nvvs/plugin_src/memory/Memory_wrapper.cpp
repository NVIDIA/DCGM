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
#include "Memory_wrapper.h"
#include "PluginStrings.h"
#include "memory_plugin.h"

/*****************************************************************************/
Memory::Memory(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_handle(handle)
    , m_gpuInfo()

{
    m_infoStruct.testIndex        = DCGM_MEMORY_INDEX;
    m_infoStruct.shortDescription = "This plugin will test the memory of a given GPU.";
    m_infoStruct.testGroups       = "";
    m_infoStruct.selfParallel     = false;
    m_infoStruct.logFileTag       = MEMORY_PLUGIN_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    tp->AddString(MEMORY_STR_IS_ALLOWED, "False");
    tp->AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75.0");
    tp->AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_DURATION, 1.0);
    tp->AddDouble(MEMORY_L1TAG_STR_TEST_LOOPS, 0);
    tp->AddDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS, 1024);
    tp->AddDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN, 8192);
    tp->AddString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES, "True");
    tp->AddDouble(MEMORY_L1TAG_STR_L1_CACHE_SIZE_KB_PER_SM, 0.0);
    tp->AddString(PS_LOGFILE, "stats_memory.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_infoStruct.defaultTestParameters = tp;

    if (gpuInfo == nullptr)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "No GPU information specified");
        AddError(d);
    }
    else
    {
        m_gpuInfo = *gpuInfo;
        InitializeForGpuList(*gpuInfo);
    }
}


/*****************************************************************************/
void Memory::Go(TestParameters *testParameters, const dcgmDiagGpuInfo_t &gpu)
{
    // UNUSED function. Delete when the Plugin Interface's extra methods are eliminated.
    dcgmDiagGpuList_t list = {};
    list.gpuCount          = 1;
    list.gpus[0]           = gpu;

    if (!testParameters->GetBoolFromString(MEMORY_STR_IS_ALLOWED))
    {
        DcgmError d { gpu.gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "Memory");
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }
    //    main_entry(gpu, this, testParameters);
}

/*****************************************************************************/
void Memory::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct)
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

    if (testParameters.GetBoolFromString(MEMORY_STR_IS_ALLOWED) == false)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "Memory");
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }

    for (unsigned int i = 0; i < m_gpuInfo.numGpus; i++)
    {
        main_entry(m_gpuInfo.gpus[i], this, &testParameters);
    }
}

/*****************************************************************************/
dcgmHandle_t Memory::GetHandle()
{
    return m_handle;
}
