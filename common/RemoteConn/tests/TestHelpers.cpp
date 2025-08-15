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

#include "TestHelpers.hpp"

#include <DcgmUtilities.h>

enum class ServerType : uint8_t
{
    Tcp,
    Uds,
};

void StartNcServers(ServerType serverType,
                    DcgmChildProcessManager &childProcessManager,
                    std::vector<std::string> const &ports,
                    std::vector<ChildProcessHandle_t> &tcpServers)
{
    for (auto port : ports)
    {
        boost::filesystem::path tcpServerPath = boost::process::search_path("nc");
        std::vector<char const *> tcpServerArgs;
        if (serverType == ServerType::Tcp)
        {
            tcpServerArgs = { "-l", port.c_str() };
        }
        else
        {
            tcpServerArgs = { "-l", "-k", "-U", port.c_str() };
        }
        dcgmChildProcessParams_t const params = {
            .version       = dcgmChildProcessParams_version1,
            .executable    = tcpServerPath.string().c_str(),
            .args          = tcpServerArgs.data(),
            .numArgs       = tcpServerArgs.size(),
            .env           = nullptr,
            .numEnv        = 0,
            .userName      = nullptr,
            .dataChannelFd = -1,
        };
        ChildProcessHandle_t tcpServer = INVALID_CHILD_PROCESS_HANDLE;
        int pid                        = -1;
        auto ret                       = childProcessManager.Spawn(params, tcpServer, pid);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(tcpServer != INVALID_CHILD_PROCESS_HANDLE);
        REQUIRE(pid > 0);
        tcpServers.emplace_back(tcpServer);

        // Wait until the process is running
        dcgmChildProcessStatus_t status = {};
        status.version                  = dcgmChildProcessStatus_version1;
        DcgmNs::Utils::WaitFor(
            [&]() { return childProcessManager.GetStatus(tcpServer, status) == DCGM_ST_OK && status.running; },
            std::chrono::microseconds(100));
    }
}

void StartTcpServers(DcgmChildProcessManager &childProcessManager,
                     std::vector<unsigned int> const &ports,
                     std::vector<ChildProcessHandle_t> &tcpServers)
{
    std::vector<std::string> portsStr;
    for (auto port : ports)
    {
        portsStr.emplace_back(std::to_string(port));
    }
    StartNcServers(ServerType::Tcp, childProcessManager, portsStr, tcpServers);
}

void StartUdsServers(DcgmChildProcessManager &childProcessManager,
                     std::vector<std::string> const &ports,
                     std::vector<ChildProcessHandle_t> &tcpServers)
{
    StartNcServers(ServerType::Uds, childProcessManager, ports, tcpServers);
}
