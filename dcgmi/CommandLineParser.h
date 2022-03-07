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

private:
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
    static unsigned int CheckGroupIdArgument(const std::string &groupId);

    // Helper to validate the throttle mask parameter
    static void ValidateThrottleMask(const std::string &throttleMask);

#ifdef DEBUG
    static dcgmReturn_t ProcessAdminCommandLine(int argc, char const *const *argv);
#endif
};

#endif // DCGMI_CLI_PARSER_H
