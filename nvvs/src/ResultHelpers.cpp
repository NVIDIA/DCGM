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

#include "ResultHelpers.h"

dcgmDiagResult_t GetOverallDiagResult(dcgmDiagEntityResults_v2 const &entityResults)
{
    if (entityResults.numErrors > 0)
    {
        return DCGM_DIAG_RESULT_FAIL;
    }

    unsigned int const numResults = std::min(static_cast<unsigned int>(std::size(entityResults.results)),
                                             static_cast<unsigned int>(entityResults.numResults));
    unsigned int warnCount        = 0;
    unsigned int skipCount        = 0;
    unsigned int passCount        = 0;
    for (unsigned int i = 0; i < numResults; ++i)
    {
        if (entityResults.results[i].result == DCGM_DIAG_RESULT_FAIL)
        {
            return DCGM_DIAG_RESULT_FAIL;
        }
        warnCount += (entityResults.results[i].result == DCGM_DIAG_RESULT_WARN);
        skipCount += (entityResults.results[i].result == DCGM_DIAG_RESULT_SKIP);
        passCount += (entityResults.results[i].result == DCGM_DIAG_RESULT_PASS);
    }

    if (warnCount > 0)
    {
        return DCGM_DIAG_RESULT_WARN;
    }
    if (passCount > 0)
    {
        return DCGM_DIAG_RESULT_PASS;
    }
    if (skipCount > 0)
    {
        return DCGM_DIAG_RESULT_SKIP;
    }
    return DCGM_DIAG_RESULT_NOT_RUN;
}
