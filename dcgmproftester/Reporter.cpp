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
#include "Reporter.h"
#include "DcgmLogging.h"
#include <mutex>
#include <sstream>
#include <thread>

namespace DcgmNs::ProfTester
{
void ReporterBase::checkThread(void)
{
    static std::mutex mutex;
    static std::thread::id m_thread_id;
    static bool init = false;

    if (!init)
    {
        mutex.lock();

        if (!init)
        {
            m_thread_id = std::this_thread::get_id();
            init        = true;
        }
    }
    else
    {
        assert(std::this_thread::get_id() == m_thread_id);
    }
}

void ReporterBase::log(std::stringstream &buffer, plog::Severity severity, ReporterBase::Flags flag)
{
    if (flag == ReporterBase::new_line)
    {
        checkThread();
        std::cout << '\n';

        // We do not log empty lines.
        if (!buffer.str().empty())
        {
            LOG_(BASE_LOGGER, severity) << buffer.str();
            buffer.str("");
        }
    }
}

/*
 * Reporter wrappers around DCGM logs (to output to the log and console)
 * should be defined here for the various log levels.
 */

Reporter info_reporter(plog::info);
Reporter warn_reporter(plog::warning);
Reporter error_reporter(plog::error);

} // namespace DcgmNs::ProfTester
