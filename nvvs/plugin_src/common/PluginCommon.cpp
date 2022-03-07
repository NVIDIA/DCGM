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
#include "PluginCommon.h"
#include "DcgmError.h"
#include <NvvsCommon.h>

#include <sstream>

/*****************************************************************************/
void CheckAndSetResult(Plugin *p,
                       const std::vector<unsigned int> &gpuList,
                       size_t i,
                       bool passed,
                       const std::vector<DcgmError> &errorList,
                       bool &allPassed,
                       bool dcgmCommError)
{
    if (passed)
    {
        p->SetResultForGpu(gpuList[i], NVVS_RESULT_PASS);
    }
    else
    {
        allPassed = false;
        p->SetResultForGpu(gpuList[i], NVVS_RESULT_FAIL);
        for (size_t j = 0; j < errorList.size(); j++)
        {
            p->AddErrorForGpu(gpuList[i], errorList[j]);
        }
    }

    if (dcgmCommError)
    {
        for (size_t j = i; j < gpuList.size(); j++)
        {
            p->SetResultForGpu(gpuList[j], NVVS_RESULT_FAIL);
        }
    }
}
