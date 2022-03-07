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
#include "DcgmiSettings.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgmi_common.h"
#include <DcgmLogging.h>
#include <sstream>


/*****************************************************************************/
DcgmiSettingsSetLoggingSeverity::DcgmiSettingsSetLoggingSeverity(const std::string &hostname,
                                                                 const std::string &targetLogger,
                                                                 const std::string &targetSeverity,
                                                                 const bool outputAsJson)
{
    m_hostName      = hostname;
    mTargetLogger   = targetLogger;
    mTargetSeverity = targetSeverity;
    m_json          = outputAsJson;
}

/*****************************************************************************/
dcgmReturn_t DcgmiSettingsSetLoggingSeverity::DoExecuteConnected()
{
    dcgmSettingsSetLoggingSeverity_t logging = {};

    if (!DcgmLogging::isValidSeverity(mTargetSeverity.c_str()))
    {
        std::cout << "Error: Invalid severity string" << std::endl;
        return DCGM_ST_BADPARAM;
    }

    if (!DcgmLogging::isValidLogger(mTargetLogger))
    {
        std::cout << "Error: Invalid logger string" << std::endl;
        return DCGM_ST_BADPARAM;
    }

    logging.targetLogger = DcgmLogging::loggerFromString(mTargetLogger, BASE_LOGGER);
    logging.targetSeverity
        = (DcgmLoggingSeverity_t)DcgmLogging::severityFromString(mTargetSeverity.c_str(), DcgmLoggingSeverityWarning);

    return dcgmHostengineSetLoggingSeverity(m_dcgmHandle, &logging);
}
