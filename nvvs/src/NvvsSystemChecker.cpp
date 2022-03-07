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
#include <fstream>
#include <iostream>
#include <sstream>

#include "DcgmLogging.h"
#include "NvvsSystemChecker.h"

// Observed measurements show that the diagnostic will monopolize the CPUs at times, so we
// should recommend against training at anything .5 and above.
const double LOADAVG_THRESHOLD = .5;

std::string NvvsSystemChecker::CheckSystemInterference()
{
    m_error.clear();

    CheckCpuActivity();

    return m_error;
}

void NvvsSystemChecker::CheckCpuActivity()
{
    std::ifstream loadfile("/proc/loadavg");
    std::string line;
    double data[3];
    if (std::getline(loadfile, line))
    {
        std::istringstream iss(line);

        for (int i = 0; i < 3; i++)
            iss >> data[i];

        if (data[0] >= LOADAVG_THRESHOLD)
        {
            std::stringstream error;
            error << "Loadavg should be below " << LOADAVG_THRESHOLD << " to train the diagnostic, but is " << data[0]
                  << ".";
            m_error = error.str();
        }
    }
    else
    {
        PRINT_DEBUG("", "Unable to read a line from /proc/loadavg: please check the syslog.");
    }
}
