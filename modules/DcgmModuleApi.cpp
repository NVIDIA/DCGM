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


#include "DcgmLogging.h"
#include "DcgmProtobuf.h"
#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include "dcgm_test_apis.h" /* DCGM_EMBEDDED_HANDLE */
#include <vector>

/*****************************************************************************/
/* Stubs */
dcgmReturn_t processModuleCommandAtHostEngine(dcgmHandle_t pDcgmHandle,
                                              dcgm_module_command_header_t *moduleCommand,
                                              size_t maxResponseSize               = 0,
                                              std::unique_ptr<DcgmRequest> request = nullptr,
                                              unsigned int timeout                 = 60000);

/*****************************************************************************/
DCGM_PUBLIC_API dcgmReturn_t dcgmModuleSendBlockingFixedRequest(dcgmHandle_t pDcgmHandle,
                                                                dcgm_module_command_header_t *moduleCommand,
                                                                size_t maxResponseSize,
                                                                std::unique_ptr<DcgmRequest> request,
                                                                unsigned int timeout)
{
    if (!moduleCommand)
        return DCGM_ST_BADPARAM;

    if (moduleCommand->length < sizeof(*moduleCommand))
    {
        PRINT_ERROR("%u", "Bad module param length %u", moduleCommand->length);
        return DCGM_ST_BADPARAM;
    }
    if (moduleCommand->moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Bad module ID %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }

    return processModuleCommandAtHostEngine(pDcgmHandle, moduleCommand, maxResponseSize, std::move(request), timeout);
}

/*****************************************************************************/
