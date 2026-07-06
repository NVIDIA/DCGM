/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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


/*
Common helper functions and classes relating to DCGM Multi Node GPU Diagnostics
*/

#include <DcgmLogging.h>
#include <DcgmUtilities.h>
#include <atomic>
#include <chrono>
#include <dcgm_structs.h>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace DcgmNs::Common::ProcessUtils
{
using DcgmNs::Utils::IsProcessRunning;
using DcgmNs::Utils::StopProcess;

/**
 * @brief Command executor interface
 */
class CommandExecutor
{
public:
    virtual ~CommandExecutor()                                 = default;
    virtual std::string ExecuteCommand(std::string const &cmd) = 0;
};

/**
 * @brief Production implementation of CommandExecutor
 */
class SystemCommandExecutor : public CommandExecutor
{
public:
    std::string ExecuteCommand(std::string const &cmd) override
    {
        constexpr size_t MAX_OUTPUT_SIZE = 4 * 1024; // 4KB limit

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe)
        {
            throw std::runtime_error("popen() failed for command: " + cmd);
        }

        std::ostringstream oss;
        char buffer[256];
        size_t totalBytesRead = 0;

        while (fgets(buffer, sizeof(buffer), pipe.get()))
        {
            size_t bufferLen = std::strlen(buffer);
            if (totalBytesRead > MAX_OUTPUT_SIZE)
            {
                log_error("Command output exceeded maximum size of {} bytes", MAX_OUTPUT_SIZE);
                break;
            }
            oss << buffer;
            totalBytesRead += bufferLen;
        }
        // Check if loop ended due to error vs. EOF
        if (ferror(pipe.get()))
        {
            throw std::runtime_error("Error reading from pipe for command: " + cmd);
        }
        return oss.str();
    }
};

/**
 * @brief Get information about MPI launched processes occupying GPUs
 *
 * @param executor Command executor to use (default: nullptr)
 * @return Vector of pairs containing PID and process name
 */
std::vector<std::pair<pid_t, std::string>> GetMpiProcessInfo(CommandExecutor *executor = nullptr);

} // namespace DcgmNs::Common::ProcessUtils
