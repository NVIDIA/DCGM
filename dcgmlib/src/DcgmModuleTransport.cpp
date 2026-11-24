/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmModuleTransport.hpp"

#include <DcgmRequest.h>
#include <dcgm_test_apis.h>

#include <utility>

dcgmReturn_t processModuleCommandAtRemoteHostEngine(dcgmHandle_t pDcgmHandle,
                                                    dcgm_module_command_header_t *moduleCommand,
                                                    std::unique_ptr<DcgmRequest> request,
                                                    unsigned int timeout,
                                                    size_t maxResponseSize);

dcgmReturn_t processModuleCommandAtEmbeddedHostEngine(dcgm_module_command_header_t *moduleCommand,
                                                      std::unique_ptr<DcgmRequest> request);

dcgmReturn_t processModuleCommandAtHostEngineDefault(dcgmHandle_t pDcgmHandle,
                                                     dcgm_module_command_header_t *moduleCommand,
                                                     size_t maxResponseSize,
                                                     std::unique_ptr<DcgmRequest> request,
                                                     unsigned int timeout)
{
    if (pDcgmHandle != static_cast<dcgmHandle_t>(DCGM_EMBEDDED_HANDLE))
    {
        // coverity[overrun]
        return processModuleCommandAtRemoteHostEngine(
            pDcgmHandle, moduleCommand, std::move(request), timeout, maxResponseSize);
    }

    // coverity[overrun]
    return processModuleCommandAtEmbeddedHostEngine(moduleCommand, std::move(request));
}
