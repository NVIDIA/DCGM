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

#include <csignal>


namespace DcgmNs::Common::Subprocess::Detail
{

/**
 * @brief Disables SIGCHLD for the current process so that if the parent process dies, the child process does not
 * become a zombie. The old SIGCHLD handler is saved and is restored on destruction.
 */
class SigChldGuard
{
public:
    SigChldGuard();
    ~SigChldGuard();

private:
    struct sigaction m_oldValue = {};
};

} //namespace DcgmNs::Common::Subprocess::Detail
