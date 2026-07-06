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

#include "MnDiagProcessUtils.h"
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <string_view>
#include <tclap/ArgException.h>
#include <vector>

namespace DcgmNs::Common::ProcessUtils
{
std::vector<std::pair<pid_t, std::string>> GetMpiProcessInfo(CommandExecutor *executor)
{
    std::vector<std::pair<pid_t, std::string>> processInfo;

    // Runtime dependency injection for testing
    SystemCommandExecutor defaultExecutor;
    if (!executor)
    {
        executor = &defaultExecutor;
    }

    std::string cmd = "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits";
    // Example output:
    // 3505755, /usr/libexec/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm

    log_debug("Searching for MPI processes running on GPUs using command: {}", cmd);

    try
    {
        std::string output = executor->ExecuteCommand(cmd);
        std::istringstream processOutput(output);

        std::string line;
        while (std::getline(processOutput, line))
        {
            std::vector<std::string> tokens;
            dcgmTokenizeString(line, ",", tokens);
            if (tokens.size() == 2)
            {
                pid_t pid               = std::stoi(tokens[0]);
                std::string processName = tokens[1];
                processName.erase(0, processName.find_first_not_of(" \t"));
                processName.erase(processName.find_last_not_of(" \t") + 1);
                log_debug("Found process with PID: {}, name: {}", pid, processName);
                processInfo.emplace_back(pid, processName);
            }
            else
            {
                log_error("Invalid line in nvidia-smi output: {}", line);
            }
        }
    }
    catch (std::exception const &e)
    {
        log_error("Failed to execute command: {}", e.what());
        return {};
    }

    log_debug("Found {} MPI processes running on GPUs", processInfo.size());
    return processInfo;
}

} // namespace DcgmNs::Common::ProcessUtils
