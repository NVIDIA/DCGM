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
#pragma once

#include <dcgm_structs.h>
#include <stdexcept>


namespace DcgmNs
{
class DcgmException : std::runtime_error
{
public:
    explicit DcgmException(dcgmReturn_t errorCode) noexcept
        : runtime_error("")
        , m_errorCode(errorCode)
    {}

    char const *what() const noexcept override
    {
        return errorString(m_errorCode);
    }

    dcgmReturn_t GetErrorCode() const
    {
        return m_errorCode;
    }

private:
    dcgmReturn_t m_errorCode;
};
} // namespace DcgmNs
