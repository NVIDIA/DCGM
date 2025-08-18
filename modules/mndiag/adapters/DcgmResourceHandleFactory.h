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

#include "DcgmResourceHandleAdapter.h"
#include "DcgmResourceHandleFactoryBase.h"

/**
 * @brief Default implementation of ResourceHandleFactoryBase
 *
 * Creates real DcgmResourceHandle objects wrapped in ResourceHandleAdapter.
 */
class DcgmResourceHandleFactory : public DcgmResourceHandleFactoryBase
{
public:
    /**
     * @brief Create a resource handle adapter with a real DcgmResourceHandle
     *
     * @param coreProxy The core proxy used to initialize the resource handle
     * @return std::unique_ptr<DcgmResourceHandleBase> The created resource handle adapter
     */
    std::unique_ptr<DcgmResourceHandleBase> CreateDcgmResourceHandle(DcgmCoreProxy &coreProxy) override
    {
        return std::make_unique<DcgmResourceHandleAdapter>(coreProxy);
    }
};
