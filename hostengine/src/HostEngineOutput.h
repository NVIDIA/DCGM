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

#include <tclap/CmdLineInterface.h>
#include <tclap/CmdLineOutput.h>

#include <iostream>
#include <string>

class HostEngineOutput : public TCLAP::CmdLineOutput
{
public:
    HostEngineOutput(std::string prologue, std::string epilogue, std::string version, std::size_t maxWidth = 100);
    void usage(TCLAP::CmdLineInterface &c) override;
    void version(TCLAP::CmdLineInterface &c) override;
    void failure(TCLAP::CmdLineInterface &c, TCLAP::ArgException &e) override;

private:
    std::string m_prologue;
    std::string m_epilogue;
    std::string m_version;
    std::size_t m_maxWidth;

    void PrintShortUsage(TCLAP::CmdLineInterface &cmdLine, std::ostream &os);
    void PrintLongUsage(TCLAP::CmdLineInterface &cmdLine, std::ostream &os);
};