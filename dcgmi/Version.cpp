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
/*
 * File:   Version.cpp
 */

#include <Command.h>
#include <DcgmBuildInfo.hpp>
#include <Version.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>
#include <dcgmi_common.h>

VersionInfo::VersionInfo(std::string hostname)
{
    DcgmNs::DcgmBuildInfo buildInfo;
    std::cout << buildInfo;

    mHostName = std::move(hostname);

    /* supress connection errors */
    mSilent  = true;
    mTimeout = 100;
}

dcgmReturn_t VersionInfo::DoExecuteConnected()
{
    dcgmVersionInfo_t versionInfo = {};

    versionInfo.version = dcgmVersionInfo_version;
    dcgmReturn_t ret    = dcgmHostengineVersionInfo(mNvcmHandle, &versionInfo);

    if (DCGM_ST_OK != ret)
    {
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    DcgmNs::DcgmBuildInfo serverBuildInfo = DcgmNs::DcgmBuildInfo(versionInfo.rawBuildInfoString);
    std::cout << "\nHostengine build info:\n";
    std::cout << serverBuildInfo;

    return DCGM_ST_OK;
}

/*****************************************************************************/