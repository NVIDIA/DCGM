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
#pragma once

#include "dcgm_structs.h"

#include <array>
#include <filesystem>
#include <memory>
#include <set>
#include <string>

#include <cstdint>
#include <sys/un.h>

/**
 * Class to represent Host Engine command line arguments
 */
class HostEngineCommandLine
{
public:
    HostEngineCommandLine()  = default;
    ~HostEngineCommandLine() = default;

    HostEngineCommandLine(HostEngineCommandLine &&) = default;
    HostEngineCommandLine &operator=(HostEngineCommandLine &&) = default;
    [[nodiscard]] std::string const &GetUnixSocketPath() const; //!< Host Engine Unix domain socket path
    [[nodiscard]] bool IsConnectionTcp() const;                 //!< Flag to indicate that connection is TCP/IP
    [[nodiscard]] bool ShouldTerminate() const;                 //!< Flag to indicate that daemon should be terminated
    [[nodiscard]] std::uint16_t GetPort() const;                //!< Host Engine port number
    [[nodiscard]] bool ShouldDaemonize() const;                 //!< Flag to daemonize
    [[nodiscard]] std::string const &GetBindInterface() const;  //!< IP address to bind to. "" = all interfaces

    //! PID filename to use to prevent more than one Host Engine instance from running
    [[nodiscard]] std::string const &GetPidFilePath() const;
    [[nodiscard]] std::string const &GetLogLevel() const;    //!< Requested logging level
    [[nodiscard]] std::string const &GetLogFileName() const; //!< Log file name
    [[nodiscard]] bool IsLogRotate() const;                  //!< Flag to rotate log file

    //! Get modules to blacklist
    [[nodiscard]] std::set<dcgmModuleId_t> const &GetBlacklistedModules() const;

private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl *) const;
    };
    std::unique_ptr<Impl, ImplDeleter> m_pimpl;
    friend HostEngineCommandLine ParseCommandLine(int, char *[]);
};

HostEngineCommandLine ParseCommandLine(int argc, char *argv[]);
