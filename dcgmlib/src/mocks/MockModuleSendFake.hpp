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
#include <dcgm_structs.h>

#include <cstddef>

namespace DcgmNs::Testing
{
// Arbitrary test-only sentinel distinct from null, embedded, and real connection handles.
inline constexpr dcgmHandle_t FakeModuleSendHandle = static_cast<dcgmHandle_t>(0xA51);

/**
 * Dispatch a module command through the test module-send fake.
 *
 * @param[in] pDcgmHandle     Handle associated with the fake transport request.
 * @param[in] moduleCommand   Module command header and payload to process.
 * @param[in] maxResponseSize Maximum size of the response buffer.
 * @param[in] timeout         Request timeout in milliseconds.
 *
 * @return DCGM_ST_OK on success, DCGM_ST_BADPARAM for a null command, or the configured
 *         transport or command error.
 */
dcgmReturn_t DispatchModuleSendFake(dcgmHandle_t pDcgmHandle,
                                    dcgm_module_command_header_t *moduleCommand,
                                    size_t maxResponseSize,
                                    unsigned int timeout);
} // namespace DcgmNs::Testing
