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
#include "JsonOutput.h"
#include "NvidiaValidationSuite.h"
#include "NvvsCommon.h"
#include "Plugin.h"
#include <string>

namespace
{
const size_t SUCCESS                   = 0;
const size_t ERROR_IN_COMMAND_LINE     = 1;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

using namespace DcgmNs::Nvvs;

/*****************************************************************************/
static void main_sig_handler(int signo)
{
    switch (signo)
    {
        case SIGINT:
        case SIGQUIT:
        case SIGKILL:
        case SIGTERM:
            PRINT_ERROR("%d", "Received signal %d. Requesting stop.", signo);
            main_should_stop          = 1;
            nvvsCommon.mainReturnCode = MAIN_RET_ERROR; /* Still counts as an error */
            break;


        case SIGUSR1:
        case SIGUSR2:
            PRINT_ERROR("%d", "Ignoring SIGUSRn (%d)", signo);
            break;

        case SIGHUP:
            /* This one is usually used to tell a process to reread its config.
             * Treating this as a no-op for now
             */
            PRINT_ERROR("", "Ignoring SIGHUP");
            break;

        default:
            PRINT_ERROR("%d", "Received unknown signal %d. Ignoring", signo);
            break;
    }
}

void OutputMainError(const std::string &error)
{
    if (nvvsCommon.jsonOutput == false)
    {
        std::cerr << error << std::endl;
    }
    else
    {
        Json::Value jv;
        jv[NVVS_NAME][NVVS_VERSION_STR]   = DRIVER_MAJOR_VERSION;
        jv[NVVS_NAME][NVVS_RUNTIME_ERROR] = error;
        std::cerr << jv.toStyledString() << std::endl;
    }

    PRINT_ERROR("%s", "Got runtime_error: %s", error.c_str());
    PRINT_ERROR("%llx", "Global error mask is: 0x%064llx", nvvsCommon.errorMask);
}

/*****************************************************************************/
int main(int argc, char **argv)
{
    NvidiaValidationSuite *nvvs = NULL;

    struct sigaction sigHandler;
    sigHandler.sa_handler = main_sig_handler;
    sigemptyset(&sigHandler.sa_mask);
    sigHandler.sa_flags       = 0;
    nvvsCommon.mainReturnCode = MAIN_RET_OK; /* Gets set by NvidiaValidationSuite constructor, but not until later */

    /* Install signal handlers */
    sigaction(SIGINT, &sigHandler, NULL);
    sigaction(SIGTERM, &sigHandler, NULL);


    try
    {
        // declare new NVVS object
        nvvs              = new NvidiaValidationSuite();
        std::string error = nvvs->Go(argc, argv);
        if (error.size())
        {
            OutputMainError(error);
        }
    }
    catch (std::runtime_error &e)
    {
        OutputMainError(e.what());

        if (nvvs)
            delete (nvvs);
        nvvsCommon.mainReturnCode = MAIN_RET_ERROR;
        return nvvsCommon.mainReturnCode;
    }
    catch (std::exception &e)
    {
        OutputMainError(e.what());

        if (nvvs)
            delete nvvs; /* This deletes the logger, so no more PRINT_ macros after this */
        nvvsCommon.mainReturnCode = MAIN_RET_ERROR;
        return nvvsCommon.mainReturnCode; // ERROR_UNHANDLED_EXCEPTION would cause a core dump
    }

    delete nvvs; /* This deletes the logger, so no more PRINT_ macros after this */
    return nvvsCommon.mainReturnCode;
}
