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
#ifndef DCGMI_CLI_PARSER_H
#define DCGMI_CLI_PARSER_H

#include "dcgmi_common.h"
#include <map>


/*
 * This class is meant to handle all of the command line parsing for NVSMI
 */
class CommandLineParser
{
public:
    // entry point to start CL processing
    // only accepts a subsytem name
    static dcgmReturn_t ProcessCommandLine(int argc, char const *const *argv);

#ifndef DCGMI_TESTS
private:
#endif
    struct StaticConstructor
    {
        StaticConstructor();
    };
    // map of subsystem function pointers
    static inline std::map<std::string, dcgmReturn_t (*)(int argc, char const *const *argv)> m_functionMap;
    static inline StaticConstructor m_staticConstructor;

    // subsystem CL processing
    static dcgmReturn_t ProcessQueryCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessPolicyCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessGroupCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessFieldGroupCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessConfigCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessHealthCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessDiagCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessStatsCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessTopoCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessIntrospectCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessNvlinkCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessDmonCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessModuleCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessProfileCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessSettingsCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessVersionInfoCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessAdminCommandLine(int argc, char const *const *argv);
    static dcgmReturn_t ProcessMnDiagCommandLine(int argc, char const *const *argv);
    static unsigned int CheckGroupIdArgument(const std::string &groupId);

    // Helper to validate the clocks event mask parameter
    static void ValidateClocksEventMask(const std::string &clocksEventMask);

    static std::string ConcatenateParameters(std::vector<std::string> const &parameters);
    static std::vector<std::string> GetHostListVector(std::string_view hostList);
    static std::string ExpandRange(std::string const &range);

    static void ValidateParameters(const std::string &parameters);

    // Totals the requested test durations and throws an error if they exceed the requested timeout
    static void CheckTestDurationAndTimeout(const std::string &parameters, unsigned int timeoutSeconds);

    // Helper methods for host list validation
    static void NormalizeUnixSocketPath(std::string &hostname, std::string_view host);
    static void NormalizeIpAddress(std::string &hostname, std::string_view host);

    // Helper method to validate the parameters for the mnDiag command
    static void ProcessAndValidateMnDiagParams(int argc,
                                               char const *const *argv,
                                               dcgmRunMnDiag_v1 &drmnd,
                                               std::string &hostEngineAddressValue,
                                               bool &hostAddressWasOverridden,
                                               bool &jsonOutput);
};

#endif // DCGMI_CLI_PARSER_H
