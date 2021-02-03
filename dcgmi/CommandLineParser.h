/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <list>
#include <map>


/*
 * This class is meant to handle all of the command line parsing for NVSMI
 */
class CommandLineParser
{
public:
    // ctor/dtor
    CommandLineParser();
    ~CommandLineParser()
    {}

    // entry point to start CL processing
    // only accepts a subsytem name
    int processCommandLine(int argc, char *argv[]);

    // map of subsystem function pointers
    std::map<std::string, int (CommandLineParser::*)(int argc, char *argv[])> mFunctionMap;

private:
    // subsystem CL processing
    int processQueryCommandLine(int argc, char *argv[]);
    int processPolicyCommandLine(int argc, char *argv[]);
    int processGroupCommandLine(int argc, char *argv[]);
    int processFieldGroupCommandLine(int argc, char *argv[]);
    int processConfigCommandLine(int argc, char *argv[]);
    int processHealthCommandLine(int argc, char *argv[]);
    int processDiagCommandLine(int argc, char *argv[]);
    int processStatsCommandLine(int argc, char *argv[]);
    int processTopoCommandLine(int argc, char *argv[]);
    int processIntrospectCommandLine(int argc, char *argv[]);
    int processNvlinkCommandLine(int argc, char *argv[]);
    int processDmonCommandLine(int argc, char *argv[]);
    int processModuleCommandLine(int argc, char *argv[]);
    int processProfileCommandLine(int argc, char *argv[]);
    int processSettingsCommandLine(int argc, char *argv[]);
    int processVersionInfoCommandLine(int argc, char *argv[]);
    unsigned int CheckGroupIdArgument(const std::string &groupId) const;

    // Helper to validate the throttle mask parameter
    void ValidateThrottleMask(const std::string &throttleMask);

#ifdef DEBUG
    int processAdminCommandLine(int argc, char *argv[]);
#endif
};

typedef std::map<std::string, int (CommandLineParser::*)(int argc, char *argv[])>::iterator functionIterator;

#endif // DCGMI_CLI_PARSER_H
