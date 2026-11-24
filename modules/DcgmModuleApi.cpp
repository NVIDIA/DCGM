/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include "dcgm_test_apis.h" /* DCGM_EMBEDDED_HANDLE */
#include <DcgmModuleTransport.hpp>
#include <vector>

/*****************************************************************************/
/**
 * Send a blocking module request to the DCGM host engine.
 *
 * @param[in] pDcgmHandle     Connection handle to the DCGM host engine.
 * @param[in] moduleCommand   Module command header and payload. Results are written back here.
 * @param[in] maxResponseSize Maximum size of the response buffer.
 * @param[in] request         Optional request used to peek at messages as they arrive.
 * @param[in] timeout         Timeout in milliseconds. Currently used only for remote requests.
 *
 * @return DCGM_ST_OK on success, DCGM_ST_BADPARAM if the command is invalid, or a DCGM error
 *         status from the host engine.
 */
DCGM_PUBLIC_API dcgmReturn_t dcgmModuleSendBlockingFixedRequest(dcgmHandle_t pDcgmHandle,
                                                                dcgm_module_command_header_t *moduleCommand,
                                                                size_t maxResponseSize,
                                                                std::unique_ptr<DcgmRequest> request,
                                                                unsigned int timeout)
{
    if (!moduleCommand)
    {
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand->length < sizeof(*moduleCommand))
    {
        log_error("Bad module param length {}", moduleCommand->length);
        return DCGM_ST_BADPARAM;
    }
    if (moduleCommand->moduleId >= DcgmModuleIdCount)
    {
        log_error("Bad module ID {}", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }

    return processModuleCommandAtHostEngine(pDcgmHandle, moduleCommand, maxResponseSize, std::move(request), timeout);
}

/*****************************************************************************/
