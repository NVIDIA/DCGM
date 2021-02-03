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

CommandLineParser *g_cl;

/*****************************************************************************/
void sig_handler(int signum)
{
    delete g_cl;
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
    int ret = 0;
    dcgmReturn_t result;

    const std::string logFile
        = DcgmLogging::getLogFilenameFromArgAndEnv("", DCGM_LOGGING_DEFAULT_DCGMI_FILE, DCGM_ENV_LOG_PREFIX);

    const std::string logSeverity
        = DcgmLogging::getLogSeverityFromArgAndEnv("", DCGM_LOGGING_DEFAULT_DCGMI_SEVERITY, DCGM_ENV_LOG_PREFIX);

    DcgmLogging::init(logFile.c_str(),
                      DcgmLogging::severityFromString(logSeverity.c_str(), DcgmLoggingSeverityWarning));

    DCGM_LOG_INFO << "Initialized DCGMI logger";

    g_cl = new CommandLineParser();

    // Install the signal handler
    InstallCtrlHandler();

    try
    {
        result = (dcgmReturn_t)g_cl->processCommandLine(argc, argv);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        ret = -2;
        goto cleanup;
    }

    // Check if any errors thrown in dcgmi
    if (DCGM_ST_OK != result)
    {
        ret = result;
        goto cleanup;
    }


cleanup:
    delete g_cl;
    dcgmShutdown();
    return ret;
}
