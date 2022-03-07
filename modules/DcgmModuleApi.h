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
#ifndef DCGMMODULEAPI_H
#define DCGMMODULEAPI_H

#include "DcgmProtocol.h"
#include "DcgmRequest.h"
#include "dcgm_module_structs.h"
#include <dcgm_api_export.h>

/*****************************************************************************/
/*
 * Helper to send a blocking module request to the DCGM host engine
 *
 * moduleCommand should be both an input and output structure of parameters to
 * your request and results from your request. All module commands
 * (EX dcgm_vgpu_msg_start_v1, dcgm_vgpu_msg_shutdown_v1) have this structure
 * at the front.
 * If request is provided, then that class's ProcessMessage class can be used
 * to peek at messages as they are received for this request and possibly notify
 * clients about state changes. Note that doing this will leave the request open
 * until it's removed from the host engine or connection.
 *
 * Note: Timeout is currently only used for remote requests.
 */
DCGM_PUBLIC_API dcgmReturn_t dcgmModuleSendBlockingFixedRequest(dcgmHandle_t pDcgmHandle,
                                                                dcgm_module_command_header_t *moduleCommand,
                                                                size_t maxResponseSize,
                                                                std::unique_ptr<DcgmRequest> request = nullptr,
                                                                unsigned int timeout                 = 60000);

/*****************************************************************************/

#endif // DCGMMODULEAPI_H
