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

#include "SigChldGuard.hpp"

#include <cassert>


namespace DcgmNs::Common::Subprocess::Detail
{

SigChldGuard::SigChldGuard()
{
    struct sigaction sa = {};
    sa.sa_handler       = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    assert(sigaction(SIGCHLD, &sa, &m_oldValue) == 0);
}

SigChldGuard::~SigChldGuard()
{
    assert(sigaction(SIGCHLD, &m_oldValue, nullptr) == 0);
}

} //namespace DcgmNs::Common::Subprocess::Detail