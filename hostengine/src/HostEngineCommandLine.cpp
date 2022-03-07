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
#include "HostEngineCommandLine.h"

#include "HostEngineOutput.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_structs_internal.h>

#include <tclap/ArgException.h>
#include <tclap/CmdLine.h>
#include <tclap/Constraint.h>
#include <tclap/SwitchArg.h>
#include <tclap/ValueArg.h>

#include <iostream>
#include <set>
#include <stdexcept>

struct HostEngineCommandLine::Impl
{
    std::string m_hostEngineSockPath;        /*!< Host engine Unix domain socket path */
    std::string m_logLevel;                  /*!< Logging level string */
    std::string m_hostEngineBindInterfaceIp; /*!< IP address to bind to. "" = all interfaces */
    std::string m_logFileName;               /*!< Log file name */
    //! PID filename to use to prevent more than one nv-hostengine daemon instance from running
    std::string m_pidFilePath;

    std::set<dcgmModuleId_t> m_blacklistModules; /*!< Modules to blacklist */

    std::uint16_t m_hostEnginePort; /*!< Host engine port number */

    bool m_isHostEngineConnTCP; /*!< Flag to indicate that connection is TCP */
    bool m_isTermHostEngine;    /*!< Terminate Daemon */
    bool m_shouldDaemonize;     /*!< Has the user requested that we do not daemonize? 1=yes. 0=no */
    bool m_isLogRotate;         /*!< Rotate log file */
};

void HostEngineCommandLine::ImplDeleter::operator()(HostEngineCommandLine::Impl *ptr) const
{
    delete ptr;
}

std::string const &HostEngineCommandLine::GetUnixSocketPath() const
{
    return m_pimpl->m_hostEngineSockPath;
}

bool HostEngineCommandLine::IsConnectionTcp() const
{
    return m_pimpl->m_isHostEngineConnTCP;
}

bool HostEngineCommandLine::ShouldTerminate() const
{
    return m_pimpl->m_isTermHostEngine;
}

std::uint16_t HostEngineCommandLine::GetPort() const
{
    return m_pimpl->m_hostEnginePort;
}

bool HostEngineCommandLine::ShouldDaemonize() const
{
    return m_pimpl->m_shouldDaemonize;
}

std::string const &HostEngineCommandLine::GetBindInterface() const
{
    return m_pimpl->m_hostEngineBindInterfaceIp;
}

std::string const &HostEngineCommandLine::GetPidFilePath() const
{
    return m_pimpl->m_pidFilePath;
}

std::string const &HostEngineCommandLine::GetLogLevel() const
{
    return m_pimpl->m_logLevel;
}

bool HostEngineCommandLine::IsLogRotate() const
{
    return m_pimpl->m_isLogRotate;
}

std::string const &HostEngineCommandLine::GetLogFileName() const
{
    return m_pimpl->m_logFileName;
}

std::set<dcgmModuleId_t> const &HostEngineCommandLine::GetBlacklistedModules() const
{
    return m_pimpl->m_blacklistModules;
}

namespace
{
using namespace std::string_literals;

std::set<dcgmModuleId_t> ParseBlacklist(std::string const &value)
{
    std::set<dcgmModuleId_t> result;
    auto tokens = dcgmTokenizeString(value, ",");

    for (auto const &token : tokens)
    {
        auto moduleId = std::stoi(token);
        result.insert(static_cast<dcgmModuleId_t>(moduleId));
    }
    return result;
}

std::string ParseBindIp(std::string const &value)
{
    if (value == "all"s || value == "ALL"s)
    {
        return ""s;
    }
    return value;
}

class BlacklistModulesConstraint : public TCLAP::Constraint<std::string>
{
public:
    std::string description() const override
    {
        return "Validate that --blacklist-modules has proper format and values"s;
    }

    std::string shortID() const override
    {
        return "MODULEID[,MODULEID...]"s;
    }

    bool check(std::string const &value) const override
    {
        auto tokens = dcgmTokenizeString(value, ",");
        if (tokens.empty())
        {
            return false;
        }

        for (auto const &token : tokens)
        {
            auto moduleId = std::stoi(token);
            if (moduleId <= 0 || moduleId >= DcgmModuleIdCount)
            {
                return false;
            }
        }

        return true;
    }
};


} // namespace

namespace TCLAP
{
template <class T>
/**
 * A class that represents a named argument with an optional value.
 *
 * This argument may be seen as a combination of a SwitchArg and a ValueArg.
 * If it's used as a SwitchArg without any value - the default value will be used.
 * Other than that, the behavior is similar to the ValueArg type.
 * Most of the implementation is copied from the ValueArg.h directly.
 * Example command lines:
 *          -d                                  - in this case default value /tmp/nv-hostengine will be used
 *          -d /tmp/nv-hostengine.sock          - in this case given /tmp/nv-hostengine.sock will be used
 *      In both cases the argument will be 'set', which means .isSet() returns true.
 */
class OptionalValueArg : public ValueArg<T>
{
protected:
    using ValueArg<T>::_ignoreable;
    using ValueArg<T>::_hasBlanks;
    using ValueArg<T>::_alreadySet;
    using ValueArg<T>::_xorSet;
    using ValueArg<T>::_checkWithVisitor;
    using ValueArg<T>::_value;
    using ValueArg<T>::_default;
    using ValueArg<T>::_constraint;
    using ValueArg<T>::trimFlag;
    using ValueArg<T>::argMatches;
    using ValueArg<T>::toString;

    enum class ExtractValueResult : std::uint8_t
    {
        Ok,
        Empty,
        NextArgument
    };

    ExtractValueResult _extractValue(const std::string &val)
    {
        /*
         * For empty values or missed values we need to construct/use default value.
         * Also, we need to not accidentally consume an argument that is following the current one,
         * so anything that looks like an argument (starts with '-') we just ignore.
         */
        if (val.empty())
        {
            return ExtractValueResult::Empty;
        }
        if (val[0] == '-')
        {
            return ExtractValueResult::NextArgument;
        }

        try
        {
            ExtractValue(_value, val, typename ArgTraits<T>::ValueCategory());
        }
        catch (ArgParseException &e)
        {
            throw ArgParseException(e.error(), toString());
        }

        if (_constraint != nullptr)
        {
            if (!_constraint->check(_value))
            {
                throw(CmdLineParseException(
                    "Value '" + val + "' does not meet constraint: " + _constraint->description(), toString()));
            }
        }

        return ExtractValueResult::Ok;
    }

public:
    using ValueArg<T>::ValueArg;
    bool processArg(int *i, std::vector<std::string> &args) override
    {
        if (_ignoreable && Arg::ignoreRest())
        {
            return false;
        }

        if (_hasBlanks(args[*i]))
        {
            return false;
        }

        std::string flag = args[*i];

        std::string value {};
        trimFlag(flag, value);

        if (argMatches(flag))
        {
            if (_alreadySet)
            {
                if (_xorSet)
                {
                    throw(CmdLineParseException("Mutually exclusive argument already set!", toString()));
                }

                throw(CmdLineParseException("Argument already set!", toString()));
            }

            if (Arg::delimiter() != ' ' && value.empty())
            {
                throw(ArgParseException("Couldn't find delimiter for this argument!", toString()));
            }

            if (value.empty())
            {
                ++(*i);
                if (static_cast<unsigned int>(*i) < args.size())
                {
                    auto const res = _extractValue(args[*i]);
                    switch (res)
                    {
                        case ExtractValueResult::Ok:
                            break; // do nothing;
                        case ExtractValueResult::NextArgument:
                            --(*i);
                            [[fallthrough]];
                        case ExtractValueResult::Empty:
                            _value = _default;
                            break;
                    }
                }
                else
                {
                    _value = _default;
                }
            }
            else
            {
                _extractValue(value);
            }

            _alreadySet = true;
            _checkWithVisitor();
            return true;
        }

        return false;
    }
};
} // namespace TCLAP

HostEngineCommandLine ParseCommandLine(int argc, char *argv[])
{
    using TCLAP::CmdLine;
    using TCLAP::OptionalValueArg;
    using TCLAP::SwitchArg;
    using TCLAP::UnlabeledValueArg;
    using TCLAP::ValueArg;

    std::unique_ptr<HostEngineCommandLine::Impl> impl(new HostEngineCommandLine::Impl);

    try
    {
        static const auto prologue = "\nNVIDIA Data Center GPU Manager (DCGM)"
                                     "\nRuns as a background process to manage GPUs on the node."
                                     "\nProvides interface to address queries from DCGMI tool."
                                     "\n";

        static const auto epilogue = "Please email cudatools@nvidia.com with any questions, bug reports, etc.\n";

        auto stdOut  = HostEngineOutput(prologue, epilogue, GetBuildInfo());
        auto cmdLine = CmdLine("", ' ', std::string(DcgmNs::DcgmBuildInfo().GetVersion()), true);
        cmdLine.setOutput(&stdOut);

        auto termArg = SwitchArg("t", "term", "Terminates Host Engine [Best Effort]", cmdLine, /*default*/ false);

        auto portArg = ValueArg<std::uint16_t>("p",
                                               "port",
                                               "Specify the port for the Host Engine",
                                               /*req*/ false,
                                               /*default*/ 5555,
                                               /*typedesc*/ "PORT",
                                               cmdLine);

        auto domainSockArg
            = OptionalValueArg<std::string>("d",
                                            "domain-socket",
                                            "Specify the Unix domain socket path for host engine."
                                            "\nNo TCP listening port is opened when this option is specified.",
                                            /*req*/ false,
                                            /*default*/ "/tmp/nv-hostengine",
                                            /*typedesc*/ "SOCKET_PATH",
                                            cmdLine);

        auto daemonizeArg = SwitchArg("n",
                                      "no-daemon",
                                      "Tell the host engine not to daemonize on start-up",
                                      cmdLine,
                                      /*default*/ false);

        auto bindIpArg = ValueArg<std::string>("b",
                                               "bind-interface",
                                               "Specify the IP address of the network interface that"
                                               " the host engine should listen on."
                                               "\n\tALL = bind to all interfaces."
                                               "\n\tDefault: 127.0.0.1.",
                                               /*req*/ false,
                                               /*default*/ "127.0.0.1",
                                               /*typedesc*/ "IP_ADDRESS",
                                               cmdLine);

        auto pidFileArg = ValueArg<std::string>("",
                                                "pid",
                                                "Specify the PID filename nv-hostengine should use"
                                                " to ensure that only one instance is running.",
                                                /*req*/ false,
                                                /*default*/ "/var/run/nvhostengine.pid",
                                                /*typedesc*/ "FILENAME",
                                                cmdLine);

        auto logLevelArg
            = ValueArg<std::string>("",
                                    "log-level",
                                    "Specify the logging level. Default: " DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY
                                    "\n\tNONE  - Disable logging"
                                    "\n\tFATAL - Set log level to FATAL only"
                                    "\n\tERROR - Set log level to ERROR and above"
                                    "\n\tWARN  - Set log level to WARNING and above"
                                    "\n\tINFO  - Set log level to INFO and above"
                                    "\n\tDEBUG - Set log level to DEBUG and above",
                                    /*req*/ false,
                                    /*default*/ "", // Default filled downstream
                                    /*typedesc*/ "LEVEL",
                                    cmdLine);

        auto logFileArg
            = ValueArg<std::string>("f",
                                    "log-filename",
                                    "Specify the filename nv-hostengine should use to dump logging information."
                                    "\nDefault: " DCGM_LOGGING_DEFAULT_HOSTENGINE_FILE,
                                    /*req*/ false,
                                    /*default*/ "", // Default filled downstream
                                    /*typedesc*/ "FILENAME",
                                    cmdLine);

        auto logRotateArg = SwitchArg("",
                                      "log-rotate",
                                      "Rotate the log file if the log file with the same name already exists.",
                                      cmdLine,
                                      /*default*/ false);

        auto blacklistConstraint = BlacklistModulesConstraint {};

        auto blacklistArg
            = ValueArg<std::string>("",
                                    "blacklist-modules",
                                    "Blacklist DCGM modules from being run by the hostengine."
                                    "\nPass a comma-separated list of module IDs like 1,2,3."
                                    "\nModule IDs are available in dcgm_structs.h as DcgmModuleId constants.",
                                    /*req*/ false,
                                    /*default*/ "",
                                    &blacklistConstraint,
                                    cmdLine);

        cmdLine.parse(argc, argv);

        impl->m_hostEngineSockPath        = domainSockArg.getValue();
        impl->m_logLevel                  = logLevelArg.getValue();
        impl->m_hostEngineBindInterfaceIp = ParseBindIp(bindIpArg.getValue());
        impl->m_logFileName               = logFileArg.getValue();
        impl->m_pidFilePath               = pidFileArg.getValue();
        impl->m_blacklistModules          = ParseBlacklist(blacklistArg.getValue());
        impl->m_hostEnginePort            = portArg.getValue();
        impl->m_isHostEngineConnTCP       = not domainSockArg.isSet();
        impl->m_isTermHostEngine          = termArg.getValue();
        impl->m_shouldDaemonize           = not daemonizeArg.getValue();
        impl->m_isLogRotate               = logRotateArg.getValue();
    }
    catch (TCLAP::ArgException const &ex)
    {
        std::cerr << "Argument parsing error: " << ex.error() << " for argument ";
        if (ex.argId().length() > 10)
        {
            std::cerr << ex.argId().substr(10);
        }
        else
        {
            std::cerr << ex.argId();
        }
        std::cerr << std::endl;
        throw std::runtime_error("An error occured trying to parse the command line.");
    }

    HostEngineCommandLine result;
    result.m_pimpl.reset(impl.release());
    return result;
}
