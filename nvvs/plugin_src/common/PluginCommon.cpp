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
#include "PluginCommon.h"
#include "DcgmError.h"
#include <DcgmLogging.h>
#include <NvvsCommon.h>

#include <sstream>

/*****************************************************************************/
void CheckAndSetResult(Plugin *p,
                       std::string const &testName,
                       const std::vector<unsigned int> &gpuList,
                       size_t i,
                       bool passed,
                       const std::vector<DcgmError> &errorList,
                       bool &allPassed,
                       bool dcgmCommError)
{
    if (passed)
    {
        p->SetResultForGpu(testName, gpuList[i], NVVS_RESULT_PASS);
    }
    else
    {
        allPassed = false;
        p->SetResultForGpu(testName, gpuList[i], NVVS_RESULT_FAIL);
        for (size_t j = 0; j < errorList.size(); j++)
        {
            DcgmError d { errorList[j] };
            d.SetGpuId(gpuList[i]);
            p->AddError(testName, d);
        }
    }

    if (dcgmCommError)
    {
        for (size_t j = i; j < gpuList.size(); j++)
        {
            p->SetResultForGpu(testName, gpuList[j], NVVS_RESULT_FAIL);
        }
    }
}

/*****************************************************************************/
bool IsSmallFrameBufferModeSet(void)
{
    const char *str = getenv("__DCGM_DIAG_SMALL_FB_MODE");
    if (str != nullptr && str[0] == '1')
    {
        return true;
    }
    else
    {
        return false;
    }
}