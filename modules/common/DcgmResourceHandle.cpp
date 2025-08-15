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

#include "DcgmResourceHandle.h"
#include <stdexcept>

DcgmResourceHandle::DcgmResourceHandle(DcgmCoreProxy &proxy)
    : m_proxy(proxy)
    , m_token(0)
    , m_initResult(DCGM_ST_GENERIC_ERROR)
    , m_isValid(false)
{
    m_initResult = m_proxy.ReserveResources(m_token);
    if (m_initResult == DCGM_ST_OK)
    {
        m_isValid = true;
        log_debug("Resource handle created with token {}", m_token);
    }
    else if (m_initResult == DCGM_ST_IN_USE)
    {
        log_error("Resources are currently in use by another diagnostic run. Please try again later.");
    }
}

DcgmResourceHandle::~DcgmResourceHandle()
{
    if (m_isValid)
    {
        m_proxy.FreeResources(m_token);
        m_token      = 0;
        m_initResult = DCGM_ST_GENERIC_ERROR;
        m_isValid    = false;
    }
}

DcgmResourceHandle::DcgmResourceHandle(DcgmResourceHandle &&other) noexcept
    : m_proxy(other.m_proxy)
    , m_token(other.m_token)
    , m_initResult(other.m_initResult)
    , m_isValid(other.m_isValid)
{
    other.m_token      = 0;
    other.m_initResult = DCGM_ST_GENERIC_ERROR;
    other.m_isValid    = false;
}

DcgmResourceHandle &DcgmResourceHandle::operator=(DcgmResourceHandle &&other) noexcept
{
    if (this != &other)
    {
        if (m_isValid)
        {
            m_proxy.FreeResources(m_token);
        }

        m_token      = other.m_token;
        m_initResult = other.m_initResult;
        m_isValid    = other.m_isValid;

        other.m_token      = 0;
        other.m_initResult = DCGM_ST_GENERIC_ERROR;
        other.m_isValid    = false;
    }
    return *this;
}
