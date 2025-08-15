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

#include "DcgmResourceHandleBase.h"
#include <DcgmCoreProxy.h>
#include <DcgmResourceHandle.h>

/**
 * @brief Adapter for real DcgmResourceHandle
 *
 * This adapter wraps the concrete DcgmResourceHandle implementation
 * to provide the ResourceHandleBase interface
 */
class DcgmResourceHandleAdapter : public DcgmResourceHandleBase
{
public:
    /**
     * @brief Construct a new Resource Handle Adapter
     *
     * @param coreProxy The core proxy used to initialize the resource handle
     */
    explicit DcgmResourceHandleAdapter(DcgmCoreProxy &coreProxy)
        : m_handle(coreProxy)
    {}

    /**
     * @brief Get the initialization result from the underlying handle
     *
     * @return dcgmReturn_t The initialization result code
     */
    dcgmReturn_t GetInitResult() override
    {
        return m_handle.GetInitResult();
    }

private:
    DcgmResourceHandle m_handle; ///< The wrapped resource handle
};
