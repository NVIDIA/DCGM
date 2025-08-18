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

#include <DcgmChildProcessManager.hpp>
#include <SSHTunnelManager.hpp>
#include <catch2/catch_all.hpp>

template <typename T>
void SetChildProcessFuncs(T &mgr, DcgmChildProcessManager &childProcessManager)
{
    DcgmNs::Common::RemoteConn::detail::ChildProcessFuncs const childProcessFuncs = {
        .Spawn = [&childProcessManager](
                     auto &&...args) { return childProcessManager.Spawn(std::forward<decltype(args)>(args)...); },
        .GetStatus
        = [&childProcessManager](
              auto &&...args) { return childProcessManager.GetStatus(std::forward<decltype(args)>(args)...); },
        .GetStdErrHandle
        = [&childProcessManager](
              auto &&...args) { return childProcessManager.GetStdErrHandle(std::forward<decltype(args)>(args)...); },
        .GetStdOutHandle
        = [&childProcessManager](
              auto &&...args) { return childProcessManager.GetStdOutHandle(std::forward<decltype(args)>(args)...); },
        .GetDataChannelHandle =
            [&childProcessManager](auto &&...args) {
                return childProcessManager.GetDataChannelHandle(std::forward<decltype(args)>(args)...);
            },
        .Stop                                                                     = [&childProcessManager](
                    auto &&...args) { return childProcessManager.Stop(std::forward<decltype(args)>(args)...); },
        .Wait                                                                     = [&childProcessManager](
                    auto &&...args) { return childProcessManager.Wait(std::forward<decltype(args)>(args)...); },
        .Destroy                                                                  = [&childProcessManager](
                       auto &&...args) { return childProcessManager.Destroy(std::forward<decltype(args)>(args)...); },
    };
    mgr.SetChildProcessFuncs(&childProcessFuncs);
}

void StartTcpServers(DcgmChildProcessManager &childProcessManager,
                     std::vector<unsigned int> const &ports,
                     std::vector<ChildProcessHandle_t> &tcpServers);
void StartUdsServers(DcgmChildProcessManager &childProcessManager,
                     std::vector<std::string> const &ports,
                     std::vector<ChildProcessHandle_t> &tcpServers);
