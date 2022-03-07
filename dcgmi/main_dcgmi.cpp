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
#include "CommandLineParser.h"
#include "dcgm_agent.h"
#include "dcgm_test_apis.h"
#include "timelib.h"

#include <DcgmLogging.h>
#include <DcgmSettings.h>

#include <csignal>
#include <iostream>
#include <sstream>
#include <stdexcept>


/*****************************************************************************/
void sig_handler(int signum)
{
    exit(128 + signum); // Exit with UNIX fatal error signal code for the received signal
}

/*****************************************************************************
 * This method provides mechanism to register Sighandler callbacks for
 * SIGHUP, SIGINT, SIGQUIT, and SIGTERM
 *****************************************************************************/
int InstallCtrlHandler()
{
    if (signal(SIGHUP, sig_handler) == SIG_ERR)
    {
        return -1;
    }
    if (signal(SIGINT, sig_handler) == SIG_ERR)
    {
        return -1;
    }
    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
    {
        return -1;
    }
    if (signal(SIGTERM, sig_handler) == SIG_ERR)
    {
        return -1;
    }

    return 0;
}


int main(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;

    const std::string logFile
        = DcgmLogging::getLogFilenameFromArgAndEnv("", DCGM_LOGGING_DEFAULT_DCGMI_FILE, DCGM_ENV_LOG_PREFIX);

    const std::string logSeverity
        = DcgmLogging::getLogSeverityFromArgAndEnv("", DCGM_LOGGING_DEFAULT_DCGMI_SEVERITY, DCGM_ENV_LOG_PREFIX);

    DcgmLogging::init(logFile.c_str(),
                      DcgmLogging::severityFromString(logSeverity.c_str(), DcgmLoggingSeverityWarning));

    DCGM_LOG_INFO << "Initialized DCGMI logger";

    // Install the signal handler
    InstallCtrlHandler();

    try
    {
        result = CommandLineParser::ProcessCommandLine(argc, argv);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        result = static_cast<dcgmReturn_t>(-2);
    }

    dcgmShutdown();
    return result;
}
