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

#include "SubreaperGuard.hpp"

#include <DcgmLogging.h>


namespace DcgmNs::Common::Subprocess::Detail
{

SubreaperGuard::SubreaperGuard()
{
    if (prctl(PR_GET_CHILD_SUBREAPER, &m_oldValue, 0, 0, 0) == -1)
    {
        auto err = errno;
        log_error("Failed to get child subreaper status: {}", strerror(err));
        return;
    }
    if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == -1)
    {
        auto err = errno;
        log_error("Failed to set child subreaper: {}", strerror(err));
        m_oldValue = -1;
        return;
    }
}

SubreaperGuard::~SubreaperGuard()
{
    if (m_oldValue != -1)
    {
        if (prctl(PR_SET_CHILD_SUBREAPER, m_oldValue, 0, 0, 0) == -1)
        {
            auto err = errno;
            log_error("Failed to restore child subreaper: {}", strerror(err));
        }
    }
}

} //namespace DcgmNs::Common::Subprocess::Detail