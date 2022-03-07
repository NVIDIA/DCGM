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
#include <sstream>

#include "ContextCreate.h"
#include "ContextCreatePlugin.h"
#include "PluginStrings.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"

ContextCreate::ContextCreate(TestParameters *testParameters, Plugin *plugin, dcgmHandle_t handle)
    : m_plugin(plugin)
    , m_testParameters(testParameters)
    , m_device()
    , m_dcgmRecorder(0)
    , m_dcgmHandle(handle)
    , m_dcgmGroup()
{}

ContextCreate::~ContextCreate()
{}

bool ContextCreate::GpusAreNonExclusive()
{
    dcgmConfig_t current[DCGM_GROUP_MAX_ENTITIES];
    unsigned int actualSize = 0;
    dcgmReturn_t ret        = m_dcgmGroup.GetConfig(current, DCGM_GROUP_MAX_ENTITIES, actualSize);

    if (ret != DCGM_ST_OK)
    {
        std::string err = m_dcgmHandle.RetToString(ret);
        m_plugin->AddInfo(err);
        PRINT_DEBUG("%s", "%s", err.c_str());
        return false;
    }

    for (unsigned int i = 0; i < actualSize; i++)
    {
        if (current[i].computeMode == DCGM_CONFIG_COMPUTEMODE_PROHIBITED)
        {
            std::stringstream err;
            err << "GPU " << current[i].gpuId << " is in prohibited mode, so we must skip this test.";
            m_plugin->AddInfo(err.str());
            PRINT_DEBUG("%s", "%s", err.str().c_str());
            return false;
        }
        else if (current[i].computeMode == DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS)
        {
            std::stringstream err;
            err << "GPU " << current[i].gpuId << " is in exclusive mode, so we must skip this test.";
            m_plugin->AddInfo(err.str());
            PRINT_DEBUG("%s", "%s", err.str().c_str());
            return false;
        }
    }

    return true;
}

std::string ContextCreate::Init(const dcgmDiagPluginGpuList_t &gpuList)
{
    std::vector<unsigned int> gpuVec;

    for (size_t i = 0; i < gpuList.numGpus; i++)
    {
        ContextCreateDevice *ccd = 0;

        try
        {
            ccd = new ContextCreateDevice(
                gpuList.gpus[i].gpuId, gpuList.gpus[i].attributes.identifiers.pciBusId, m_plugin, m_dcgmHandle);
        }
        catch (DcgmError &d)
        {
            if (ccd)
            {
                delete ccd;
            }

            m_plugin->AddError(d);
            return d.GetMessage();
        }
        catch (std::runtime_error &re)
        {
            if (ccd)
            {
                delete ccd;
            }

            DcgmError d { gpuList.gpus[i].gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, re.what());
            return d.GetMessage();
        }

        m_device.push_back(ccd);
        gpuVec.push_back(gpuList.gpus[i].gpuId);
    }

    dcgmReturn_t ret = m_dcgmGroup.Init(m_dcgmHandle.GetHandle(), "context_create_group", gpuVec);
    if (ret != DCGM_ST_OK)
    {
        return m_dcgmHandle.RetToString(ret);
    }

    return "";
}

void ContextCreate::Cleanup()
{
    for (size_t i = 0; i < m_device.size(); i++)
    {
        delete m_device[i];
    }
    m_device.clear();

    // Cleanup the group first because it needs a handle to take care of itself
    m_dcgmGroup.Cleanup();
    m_dcgmHandle.Cleanup();
}

int ContextCreate::CanCreateContext()
{
    int created = CTX_CREATED;
    CUresult cuSt;
    std::stringstream err;
    std::string error;

    for (size_t i = 0; i < m_device.size(); i++)
    {
        cuSt = cuCtxCreate(&m_device[i]->cuContext, 0, m_device[i]->cuDevice);

        if (cuSt == CUDA_SUCCESS)
        {
            cuCtxDestroy(m_device[i]->cuContext);
            continue;
        }
        else if (cuSt == CUDA_ERROR_UNKNOWN)
        {
            err << "GPU " << m_device[i]->gpuId << " is in prohibted mode; skipping test.";
            m_plugin->AddInfo(err.str());
            PRINT_DEBUG("%s", "%s", err.str().c_str());
            created |= CTX_SKIP;
        }
        else
        {
            const char *errStr;
            cuGetErrorString(cuSt, &errStr);
            DcgmError d { m_device[i]->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, d, m_device[i]->gpuId, errStr);
            m_plugin->AddErrorForGpu(m_device[i]->gpuId, d);
            PRINT_DEBUG("%s", "%s", error.c_str());
            created |= CTX_FAIL;
        }
    }

    return created;
}

int ContextCreate::Run(const dcgmDiagPluginGpuList_t &gpuList)
{
    std::string error = Init(gpuList);

    if (error.size() != 0)
    {
        PRINT_ERROR("%s", "%s", error.c_str());
        return CONTEXT_CREATE_FAIL;
    }

    int rc = CanCreateContext();
    if ((rc & CTX_FAIL) != 0)
    {
        // cuCtxCreate() only gives us a special error for prohibited mode, not exclusive, so check
        // the GPU mode here.
        if (GpusAreNonExclusive() == false)
        {
            // Test should be skipped
            return CONTEXT_CREATE_SKIP;
        }
        else
            return CONTEXT_CREATE_FAIL;
    }
    else if (rc)
    {
        if (m_testParameters->GetString(CTXCREATE_IGNORE_EXCLUSIVE) == "True")
            return CONTEXT_CREATE_FAIL;
        else
            return CONTEXT_CREATE_SKIP;
    }

    return CONTEXT_CREATE_PASS;
}
