/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmBuildInfo.hpp>
#include <NvvsException.hpp>

#include <memory>
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
            log_error("Received signal {}. Requesting stop.", signo);
            main_should_stop          = 1;
            nvvsCommon.mainReturnCode = NVVS_ST_SUCCESS; /* Still counts as an error */
            break;


        case SIGUSR1:
        case SIGUSR2:
            log_error("Ignoring SIGUSRn ({})", signo);
            break;

        case SIGHUP:
            /* This one is usually used to tell a process to reread its config.
             * Treating this as a no-op for now
             */
            log_error("Ignoring SIGHUP");
            break;

        default:
            log_error("Received unknown signal {}. Ignoring", signo);
            break;
    }
}

void OutputMainError(const std::string &error, nvvsReturn_t errorCode = NVVS_ST_GENERIC_ERROR)
{
    if (!nvvsCommon.jsonOutput)
    {
        std::cerr << error << std::endl;
    }
    else
    {
        ::Json::Value jv;
        jv[NVVS_NAME][NVVS_VERSION_STR]   = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
        jv[NVVS_NAME][NVVS_RUNTIME_ERROR] = error;
        jv[NVVS_NAME][NVVS_ERROR_CODE]    = errorCode;
        std::cout << jv.toStyledString() << std::endl;
    }

    log_error("Got runtime_error: {}. Error code: ", error, errorCode);
    log_error("Global error mask is: {:#064x}", nvvsCommon.errorMask);
}

/*****************************************************************************/
int main(int argc, char **argv)
{
    std::unique_ptr<NvidiaValidationSuite> nvvs;

    struct sigaction sigHandler = {};

    sigHandler.sa_handler = main_sig_handler;
    sigemptyset(&sigHandler.sa_mask);
    sigHandler.sa_flags       = 0;
    nvvsCommon.mainReturnCode = NVVS_ST_SUCCESS; /* Set by NvidiaValidationSuite constructor, but not until later */

    /* Install signal handlers */
    sigaction(SIGINT, &sigHandler, nullptr);
    sigaction(SIGTERM, &sigHandler, nullptr);

    try
    {
        // declare new NVVS object
        nvvs              = std::make_unique<NvidiaValidationSuite>();
        std::string error = nvvs->Go(argc, argv);
        if (!error.empty())
        {
            OutputMainError(error);
        }
    }
    catch (NvvsException const &ex)
    {
        OutputMainError(ex.what(), ex.GetErrorCode());
        nvvsCommon.mainReturnCode = ex.GetErrorCode();
    }
    catch (std::runtime_error &e)
    {
        OutputMainError(e.what());
        nvvsCommon.mainReturnCode = NVVS_ST_GENERIC_ERROR;
    }
    catch (std::exception &e)
    {
        OutputMainError(e.what());
        nvvsCommon.mainReturnCode = NVVS_ST_SUCCESS;
    }

    return nvvsCommon.mainReturnCode;
}
