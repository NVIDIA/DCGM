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

#include <DcgmCoreProxy.h>

class DcgmResourceHandle
{
public:
    explicit DcgmResourceHandle(DcgmCoreProxy &proxy);
    ~DcgmResourceHandle();

    // Get the initialization result
    dcgmReturn_t GetInitResult() const
    {
        return m_initResult;
    }

    // Get the token value
    unsigned int GetToken() const
    {
        return m_token;
    }

    // Move semantics
    DcgmResourceHandle(DcgmResourceHandle &&other) noexcept;
    DcgmResourceHandle &operator=(DcgmResourceHandle &&other) noexcept;

    // Prevent copying
    DcgmResourceHandle(DcgmResourceHandle const &)            = delete;
    DcgmResourceHandle &operator=(DcgmResourceHandle const &) = delete;

private:
    DcgmCoreProxy &m_proxy;
    unsigned int m_token;
    dcgmReturn_t m_initResult;
    bool m_isValid;
};