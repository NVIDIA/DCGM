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
#include <csignal>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <string_view>
#include <tclap/ArgException.h>
#include <thread>
#include <vector>

namespace DcgmNs::Common::ProcessUtils
{
template <typename DurationType>
dcgmReturn_t StopProcess(pid_t pid, unsigned int maxSigtermAttempts, DurationType sigtermRetryDelay)
{
    if (!IsProcessRunning(pid))
    {
        // No process to stop or process doesn't exist
        return DCGM_ST_INSTANCE_NOT_FOUND;
    }
    unsigned int kill_count = 0;
    bool sigkilled          = false;

    while (kill_count <= maxSigtermAttempts && kill(pid, 0) == 0)
    {
        if (kill_count < maxSigtermAttempts)
        {
            kill(pid, SIGTERM);
        }
        else
        {
            if (!sigkilled)
            {
                log_error("Unable to kill process with {} SIGTERM attempts, escalating to SIGKILL. pid: {}",
                          maxSigtermAttempts,
                          pid);
            }
            kill(pid, SIGKILL);
            sigkilled = true;
        }
        if (kill_count == 0)
        {
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(sigtermRetryDelay);
        }
        kill_count++;
    }
    if (kill_count >= maxSigtermAttempts && kill(pid, 0) == 0)
    {
        log_error("Giving up attempting to kill process {} after {} retries.", pid, maxSigtermAttempts);
        return DCGM_ST_CHILD_NOT_KILLED;
    }
    return DCGM_ST_OK;
}

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

bool IsProcessRunning(pid_t pid)
{
    return pid > 0 && kill(pid, 0) == 0;
}

template dcgmReturn_t StopProcess<std::chrono::seconds>(pid_t, unsigned int, std::chrono::seconds);
template dcgmReturn_t StopProcess<std::chrono::milliseconds>(pid_t, unsigned int, std::chrono::milliseconds);

} // namespace DcgmNs::Common::ProcessUtils
