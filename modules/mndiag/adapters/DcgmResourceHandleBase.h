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

#pragma once

#include <dcgm_structs.h>

/**
 * @brief Base interface for resource handling
 *
 * This abstract class defines the interface for resource handle
 * implementations, allowing for dependency injection in tests
 */
class DcgmResourceHandleBase
{
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~DcgmResourceHandleBase() = default;

    /**
     * @brief Get initialization result
     *
     * @return dcgmReturn_t The initialization result code
     */
    virtual dcgmReturn_t GetInitResult() = 0;
};
