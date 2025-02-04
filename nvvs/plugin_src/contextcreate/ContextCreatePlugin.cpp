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
#include "ContextCreatePlugin.h"
#include "ContextCreate.h"
#include "PluginInterface.h"
#include "PluginStrings.h"
#include "dcgm_fields.h"

ContextCreatePlugin::ContextCreatePlugin(dcgmHandle_t handle)
    : m_handle(handle)
    , m_entityInfo(std::make_unique<dcgmDiagPluginEntityList_v1>())
{
    TestParameters *tp;
    m_infoStruct.testIndex        = DCGM_CONTEXT_CREATE_INDEX;
    m_infoStruct.shortDescription = "This plugin will attempt to create a CUDA context.";
    m_infoStruct.testCategories   = "";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = CTXCREATE_PLUGIN_NAME;

    // Populate default test parameters
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(CTXCREATE_IGNORE_EXCLUSIVE, "False");
    tp->AddString(CTXCREATE_IS_ALLOWED, "True");
    tp->AddString(PS_LOGFILE, "stats_context.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    tp->AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = tp;
}

void ContextCreatePlugin::Go(std::string const &testName,
                             dcgmDiagPluginEntityList_v1 const *entityInfo,
                             unsigned int numParameters,
                             dcgmDiagPluginTestParameter_t const *tpStruct)
{
    if (testName != GetCtxCreateTestName())
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

    // UNUSED function. Delete when the Plugin Interface's extra methods are eliminated.
    TestParameters testParameters(*m_infoStruct.defaultTestParameters);
    testParameters.SetFromStruct(numParameters, tpStruct);

    if (!testParameters.GetBoolFromString(CTXCREATE_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, CTXCREATE_PLUGIN_NAME);
        AddInfo(GetCtxCreateTestName(), d.GetMessage());
        SetResult(GetCtxCreateTestName(), NVVS_RESULT_SKIP);
        return;
    }


    auto const &gpuList = m_tests.at(testName).GetGpuList();
    if (!gpuList.empty())
    {
        ParseIgnoreErrorCodesParam(testName, testParameters.GetString(PS_IGNORE_ERROR_CODES));

        ContextCreate cc(&testParameters, this, GetHandle());

        int st = cc.Run(*entityInfo);
        if (!st)
        {
            SetResult(GetCtxCreateTestName(), NVVS_RESULT_PASS);
        }
        else if (main_should_stop)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
            AddError(GetCtxCreateTestName(), d);
            SetResult(GetCtxCreateTestName(), NVVS_RESULT_SKIP);
        }
        else if (st == CONTEXT_CREATE_SKIP)
        {
            SetResult(GetCtxCreateTestName(), NVVS_RESULT_SKIP);
        }
        else
        {
            SetResult(GetCtxCreateTestName(), NVVS_RESULT_FAIL);
        }
    }
    else
    {
        SetResult(GetCtxCreateTestName(), NVVS_RESULT_FAIL);
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_EMPTY_GPU_LIST, d);
        AddError(GetCtxCreateTestName(), d);
    }
}

dcgmHandle_t ContextCreatePlugin::GetHandle()
{
    return m_handle;
}

std::string ContextCreatePlugin::GetCtxCreateTestName() const
{
    return CTXCREATE_PLUGIN_NAME;
}