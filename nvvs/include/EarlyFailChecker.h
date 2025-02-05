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

#pragma once

#include "DcgmRecorder.h"
#include "TestParameters.h"

#include <DcgmError.h>
#include <timelib.h>
#include <vector>

class EarlyFailChecker
{
public:
    /*************************************************************************/
    EarlyFailChecker(TestParameters *tp,
                     bool failEarly,
                     unsigned long failCheckInterval,
                     const dcgmDiagPluginEntityList_v1 &entityInfos);

    /*************************************************************************/
    nvvsPluginResult_t CheckCommonErrors(unsigned long checkTime, unsigned long startTime, DcgmRecorder &dcgmRecorder);

    /*************************************************************************/
    std::vector<DcgmError> GetErrors() const;

protected:
    TestParameters *m_testParameters  = nullptr;
    bool m_failEarly                  = false;
    unsigned long m_failCheckInterval = 0;
    unsigned long m_lastCheckTime     = ULONG_MAX;
    std::vector<DcgmError> m_errors;
    std::vector<dcgmDiagPluginEntityInfo_v1> m_entityInfos;
};
