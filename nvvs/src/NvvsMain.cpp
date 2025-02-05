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

#include "NvidiaValidationSuite.h"
#include "NvvsCommon.h"
#include "Plugin.h"
#include <DcgmBuildInfo.hpp>
#include <DcgmNvvsResponseWrapper.h>
#include <FdChannelClient.h>
#include <NvvsException.hpp>
#include <NvvsExitCode.h>
#include <dcgm_structs.h>

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
/* DCGM-3723, DCGM-4026: Do not perform logging in signal handlers. */
static void main_sig_handler(int signo)
{
    switch (signo)
    {
        case SIGINT:
        case SIGQUIT:
        case SIGKILL:
        case SIGTERM:
            main_should_stop.store(1);
            nvvsCommon.mainReturnCode = NVVS_ST_SUCCESS; /* Still counts as an error */
            break;


        case SIGUSR1:
        case SIGUSR2:
            break;

        case SIGHUP:
            /* This one is usually used to tell a process to reread its config.
             * Treating this as a no-op for now
             */
            break;

        default:
            break;
    }
}

void OutputMainError(const std::string &error, nvvsReturn_t errorCode = NVVS_ST_GENERIC_ERROR)
{
    DcgmNvvsResponseWrapper diagResponse;
    if (auto ret = diagResponse.SetVersion(nvvsCommon.diagResponseVersion); ret != DCGM_ST_OK)
    {
        std::string const errMsg
            = fmt::format("failed to set version [{}], err: [{}].", nvvsCommon.diagResponseVersion, ret);
        log_error(errMsg);
        std::cerr << errMsg << std::endl;
        return;
    }
    diagResponse.SetSystemError(error, DCGM_FR_INTERNAL);

    if (nvvsCommon.channelFd == -1)
    {
        fmt::print(stderr, "Got runtime_error: {}. Error code: {}\n", error, errorCode);
    }
    else
    {
        if (!FdChannelClient(nvvsCommon.channelFd).Write(diagResponse.RawBinaryBlob()))
        {
            log_error("failed to write diag response to caller.");
        }
    }

    log_error("Got runtime_error: {}. Error code: {}", error, errorCode);
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
