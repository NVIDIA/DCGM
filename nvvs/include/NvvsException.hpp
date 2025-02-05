// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "NvvsCommon.h"
#include "NvvsExitCode.h"

#include <exception>
#include <stdexcept>


namespace DcgmNs::Nvvs
{

/**
 * @brief Exception class for Nvvs that includes an error code
 */
class NvvsException : public std::runtime_error
{
public:
    [[maybe_unused]] NvvsException(const std::string &what_arg, nvvsReturn_t errorCode)
        : std::runtime_error(what_arg)
    {
        m_errorCode = errorCode;
    }

    [[maybe_unused]] explicit NvvsException(const std::string &what_arg)
        : std::runtime_error(what_arg)
    {}

    [[maybe_unused]] explicit NvvsException(const char *what_arg)
        : std::runtime_error(what_arg)
    {}

    [[nodiscard]] nvvsReturn_t GetErrorCode() const
    {
        return m_errorCode;
    }

private:
    nvvsReturn_t m_errorCode;
};
} //namespace DcgmNs::Nvvs
