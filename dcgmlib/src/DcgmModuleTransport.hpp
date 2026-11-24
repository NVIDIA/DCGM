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

#pragma once

#include <dcgm_module_structs.h>

#include <memory>

class DcgmRequest;

/**
 * Send a module command through the configured host-engine transport entry point.
 *
 * @param[in] pDcgmHandle     Connection handle to the DCGM host engine.
 * @param[in] moduleCommand   Module command header and payload. Results are written back here.
 * @param[in] maxResponseSize Maximum size of the response buffer.
 * @param[in] request         Optional request whose ownership is transferred to the transport.
 * @param[in] timeout         Timeout in milliseconds. Used only for remote requests.
 *
 * @return DCGM_ST_OK on success, or a dcgmReturn_t error from the selected transport, including
 *         connection, timeout, and host-engine command failures.
 *
 * @note Thread safety is provided by the selected transport. Callers must not concurrently access
 *       moduleCommand while this call is in progress.
 */
dcgmReturn_t processModuleCommandAtHostEngine(dcgmHandle_t pDcgmHandle,
                                              dcgm_module_command_header_t *moduleCommand,
                                              size_t maxResponseSize,
                                              std::unique_ptr<DcgmRequest> request,
                                              unsigned int timeout);

/**
 * Send a module command through the standard remote or embedded host-engine transport.
 *
 * @param[in] pDcgmHandle     Connection handle selecting the remote or embedded transport.
 * @param[in] moduleCommand   Module command header and payload. Results are written back here.
 * @param[in] maxResponseSize Maximum size of the response buffer.
 * @param[in] request         Optional request whose ownership is transferred to the transport.
 * @param[in] timeout         Timeout in milliseconds. Used only for remote requests.
 *
 * @return DCGM_ST_OK on success, or a dcgmReturn_t error from the selected transport, including
 *         connection, timeout, and host-engine command failures.
 *
 * @note Thread safety is provided by the selected transport. Callers must not concurrently access
 *       moduleCommand while this call is in progress.
 */
dcgmReturn_t processModuleCommandAtHostEngineDefault(dcgmHandle_t pDcgmHandle,
                                                     dcgm_module_command_header_t *moduleCommand,
                                                     size_t maxResponseSize,
                                                     std::unique_ptr<DcgmRequest> request,
                                                     unsigned int timeout);
