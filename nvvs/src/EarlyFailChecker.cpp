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

#include <DcgmLogging.h>
#include <EarlyFailChecker.h>
#include <sstream>

EarlyFailChecker::EarlyFailChecker(TestParameters *tp,
                                   bool failEarly,
                                   unsigned long failCheckInterval,
                                   const dcgmDiagPluginEntityList_v1 &entityInfos)
    : m_testParameters(tp)
    , m_failEarly(failEarly)
    , m_failCheckInterval(failCheckInterval)
{
    for (unsigned int i = 0; i < entityInfos.numEntities; i++)
    {
        m_entityInfos.push_back(entityInfos.entities[i]);
    }
}

nvvsPluginResult_t EarlyFailChecker::CheckCommonErrors(unsigned long checkTime,
                                                       unsigned long startTime,
                                                       DcgmRecorder &dcgmRecorder)

{
    nvvsPluginResult_t result = NVVS_RESULT_PASS;

    if (m_failEarly == false)
    {
        // Nothing to do here, just pass all the time
        return result;
    }

    if (checkTime - m_lastCheckTime > m_failCheckInterval)
    {
        m_lastCheckTime = checkTime;
        std::vector<DcgmError> ignoredErrors;
        dcgmRecorder.CheckCommonErrors(*m_testParameters, startTime, result, m_entityInfos, m_errors, ignoredErrors);
        if (result == NVVS_RESULT_FAIL)
        {
            std::stringstream buf;
            buf << "Errors: ";
            bool first = true;
            for (auto &&error : m_errors)
            {
                if (first)
                {
                    buf << "'" << error.GetMessage() << "'";
                    first = false;
                }
                else
                {
                    buf << ", '" << error.GetMessage() << "'";
                }
            }

            DCGM_LOG_ERROR << "Stopping early due to error(s) detected: \n" << buf.str();
        }
    }

    return result;
}

std::vector<DcgmError> EarlyFailChecker::GetErrors() const
{
    return m_errors;
}
