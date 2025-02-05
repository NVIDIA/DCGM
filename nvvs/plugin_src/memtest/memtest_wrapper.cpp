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
#include "memtest_wrapper.h"
#include "Memtest.h"
#include "PluginStrings.h"
#include "dcgm_fields.h"
#include "memtest_plugin.h"

/*****************************************************************************/
MemtestPlugin::MemtestPlugin(dcgmHandle_t handle)
    : m_handle(handle)
{
    m_infoStruct.testIndex        = DCGM_MEMTEST_INDEX;
    m_infoStruct.shortDescription = "This plugin will test the memory of a given GPU.";
    m_infoStruct.testCategories   = "";
    m_infoStruct.selfParallel     = false;
    m_infoStruct.logFileTag       = MEMTEST_PLUGIN_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(MEMTEST_STR_USE_MAPPED_MEM, "False");
    tp->AddString(MEMTEST_STR_IS_ALLOWED, "True");
    tp->AddString(MEMTEST_STR_TEST_DURATION, "600");
    tp->AddString(MEMTEST_STR_TEST0, "False");
    tp->AddString(MEMTEST_STR_TEST1, "False");
    tp->AddString(MEMTEST_STR_TEST2, "False");
    tp->AddString(MEMTEST_STR_TEST3, "False");
    tp->AddString(MEMTEST_STR_TEST4, "False");
    tp->AddString(MEMTEST_STR_TEST5, "False");
    tp->AddString(MEMTEST_STR_TEST6, "False");
    tp->AddString(MEMTEST_STR_TEST7, "True");
    tp->AddString(MEMTEST_STR_TEST8, "False");
    tp->AddString(MEMTEST_STR_TEST9, "False");
    tp->AddString(MEMTEST_STR_TEST10, "True");
    tp->AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
void MemtestPlugin::Go(std::string const &testName,
                       dcgmDiagPluginEntityList_v1 const *entityInfo,
                       unsigned int numParameters,
                       dcgmDiagPluginTestParameter_t const *tpStruct)
{
    int st;

    if (testName != GetMmeTestTestName())
    {
        log_error("failed to test due to unknown test name [{}].", testName);
        return;
    }

    if (!entityInfo)
    {
        log_error("failed to test due to entityInfo is nullptr.");
        return;
    }

    InitializeForEntityList(testName, *entityInfo);

    if (UsingFakeGpus(testName))
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(3); // Sync with test_dcgm_diag.py->injection_offset for error injection
        SetResult(testName, NVVS_RESULT_PASS);
        return;
    }

    Memtest *memtest = 0;

    TestParameters testParameters(*(m_infoStruct.defaultTestParameters));
    if (testParameters.SetFromStruct(numParameters, tpStruct) != TP_ST_OK)
    {
        DCGM_LOG_WARNING << "Plugin parameters invalid";
    }

    if (testParameters.GetBoolFromString(MEMTEST_STR_IS_ALLOWED) == false)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, "Memtest");
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, testParameters.GetString(PS_IGNORE_ERROR_CODES));

    int numGpus = 0;
    for (unsigned idx = 0; idx < entityInfo->numEntities; ++idx)
    {
        if (entityInfo->entities[idx].entity.entityGroupId == DCGM_FE_GPU)
        {
            numGpus += 1;
        }
    }

    if (numGpus)
    {
        memtest = new Memtest(&testParameters, this);

        st = memtest->Run(GetHandle(), *entityInfo);
        if (main_should_stop)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
            AddError(testName, d);
            SetResult(testName, NVVS_RESULT_SKIP);
        }
        else if (st)
        {
            // Fatal error in plugin or test could not be initialized
            SetResult(testName, NVVS_RESULT_FAIL);
        }
    }
    else
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EMPTY_GPU_LIST, d);
        AddError(testName, d);
        SetResult(testName, NVVS_RESULT_FAIL);
    }

    delete memtest;
}

/*****************************************************************************/
dcgmHandle_t MemtestPlugin::GetHandle() const
{
    return m_handle;
}

std::string MemtestPlugin::GetMmeTestTestName() const
{
    return MEMTEST_PLUGIN_NAME;
}