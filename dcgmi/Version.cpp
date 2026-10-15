/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "Version.h"

#include "dcgmi_common.h"

#include <DcgmBuildInfo.hpp>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <iostream>
#include <utility>


VersionInfo::VersionInfo(std::string hostname, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
{
    m_hostName = std::move(hostname);

    /* suppress connection errors */
    m_silent  = true;
    m_timeout = 100;
}

static void PrintLocalBuildInfo()
{
    // Local build info
    static DcgmNs::DcgmBuildInfo buildInfo;
    std::cout << "\nLocal build info:\n";
    std::cout << buildInfo;
}

dcgmReturn_t VersionInfo::DoExecuteConnected()
{
    PrintLocalBuildInfo();

    dcgmVersionInfo_t versionInfo = {};

    versionInfo.version = dcgmVersionInfo_version;
    dcgmReturn_t ret    = m_dcgmOps->dcgmHostengineVersionInfo(m_dcgmHandle, &versionInfo);

    if (DCGM_ST_OK != ret)
    {
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    // Hostengine build info
    DcgmNs::DcgmBuildInfo serverBuildInfo = DcgmNs::DcgmBuildInfo(versionInfo.rawBuildInfoString);
    std::cout << "\nHostengine build info:\n";
    std::cout << serverBuildInfo;

    return DCGM_ST_OK;
}

dcgmReturn_t VersionInfo::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    PrintLocalBuildInfo();
    std::cout << "\nHostengine build info not available: could not connect to hostengine at " << m_hostName << ": "
              << errorString(connectionStatus) << std::endl;
    return connectionStatus;
}

/*****************************************************************************/
