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
#ifndef _NVVS_NVVS_Plugin_common_H_
#define _NVVS_NVVS_Plugin_common_H_

#include "Plugin.h"

/********************************************************************/
/*
 * Sets the result for the GPU based on the value of 'passed'. Logs warnings from 'errorList' if the result is fail.
 * Sets 'allPassed' to false if the result is failed for this GPU.
 *
 * 'i' is the index of the GPU in 'gpuList'; the GPU ID is the value in the vector (i.e. gpuList[i] is GPU ID).
 * If 'dcgmCommError' is true, sets result for all GPUs starting at index 'i' to fail and adds a warning to
 * 'errorListAllGpus'.
 */
void CheckAndSetResult(Plugin *p,
                       const std::vector<unsigned int> &gpuList,
                       size_t i,
                       bool passed,
                       const std::vector<DcgmError> &errorList,
                       bool &allPassed,
                       bool dcgmCommError);

#endif // _NVVS_NVVS_Plugin_common_H_
