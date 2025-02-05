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

#include <cerrno>
#include <csignal>
#include <cstring>
#include <sys/prctl.h>

namespace DcgmNs::Common::Subprocess::Detail
{
/**
 * @brief Sets subreaper for children processes.
 * The old subreaper status is preserved and can be restored on destruction.
 *
 * @note See the \c PR_SET_CHILD_SUBREAPER documentation
 *       https://man7.org/linux/man-pages/man2/prctl.2.html for details.
 */
class SubreaperGuard
{
public:
    SubreaperGuard();
    ~SubreaperGuard();

private:
    int m_oldValue = -1;
};
} //namespace DcgmNs::Common::Subprocess::Detail