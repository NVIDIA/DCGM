/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "ContextCreatePlugin.h"
#include "ContextCreate.h"
#include "PluginStrings.h"

ContextCreatePlugin::ContextCreatePlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_handle(handle)
    , m_gpuInfo()
{
    TestParameters *tp;
    m_infoStruct.testIndex        = DCGM_CONTEXT_CREATE_INDEX;
    m_infoStruct.shortDescription = "This plugin will attempt to create a CUDA context.";
    m_infoStruct.testGroups       = "";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = CTXCREATE_PLUGIN_NAME;

    // Populate default test parameters
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(CTXCREATE_IGNORE_EXCLUSIVE, "False");
    tp->AddString(CTXCREATE_IS_ALLOWED, "True");
    tp->AddString(PS_LOGFILE, "stats_context.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_infoStruct.defaultTestParameters = tp;

    if (gpuInfo != nullptr)
    {
        m_gpuInfo = *gpuInfo;
        InitializeForGpuList(CTXCREATE_PLUGIN_NAME, *gpuInfo);
    }
    else
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "No GPU information specified");
        AddError(CTXCREATE_PLUGIN_NAME, d);
    }
}

void ContextCreatePlugin::Go(std::string const &testName,
                             unsigned int numParameters,
                             const dcgmDiagPluginTestParameter_t *tpStruct)
{
    // UNUSED function. Delete when the Plugin Interface's extra methods are eliminated.
    TestParameters testParameters(*m_infoStruct.defaultTestParameters);
    testParameters.SetFromStruct(numParameters, tpStruct);

    if (!testParameters.GetBoolFromString(CTXCREATE_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, CTXCREATE_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    if (m_gpuInfo.numGpus)
    {
        ContextCreate cc(&testParameters, this, GetHandle());

        int st = cc.Run(m_gpuInfo);
        if (!st)
        {
            SetResult(testName, NVVS_RESULT_PASS);
        }
        else if (main_should_stop)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
            AddError(testName, d);
            SetResult(testName, NVVS_RESULT_SKIP);
        }
        else if (st == CONTEXT_CREATE_SKIP)
        {
            SetResult(testName, NVVS_RESULT_SKIP);
        }
        else
        {
            SetResult(testName, NVVS_RESULT_FAIL);
        }
    }
    else
    {
        SetResult(testName, NVVS_RESULT_FAIL);
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EMPTY_GPU_LIST, d);
        AddError(testName, d);
    }
}

dcgmHandle_t ContextCreatePlugin::GetHandle()
{
    return m_handle;
}
