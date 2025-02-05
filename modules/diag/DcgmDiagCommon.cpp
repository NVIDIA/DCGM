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
#include <sstream>
#include <vector>

#include "DcgmDiagCommon.h"
#include "DcgmStringHelpers.h"
#include "EntityListHelpers.h"

bool is_valid_gpu_list(const std::string &gpuList)
{
    std::vector<std::string> gpuIdVec;
    dcgmTokenizeString(gpuList, ",", gpuIdVec);
    for (size_t i = 0; i < gpuIdVec.size(); i++)
    {
        for (size_t j = 0; j < gpuIdVec[i].size(); j++)
        {
            if (!isdigit(gpuIdVec[i][j]))
            {
                // found a non-digit in the comma-separated list
                return false;
            }
        }
    }
    return true;
}

bool is_valid_expected_num_entities(dcgmRunDiag_v10 const &drd,
                                    std::string const &expectedNumEntities,
                                    std::string &error)
{
    if (expectedNumEntities.size() > DCGM_EXPECTED_ENTITIES_LEN)
    {
        error = fmt::format("Error: expectedNumEntities size [{}] larger than the maximum allowable size [{}].",
                            expectedNumEntities.size(),
                            DCGM_EXPECTED_ENTITIES_LEN);
        return false;
    }

    if (!expectedNumEntities.empty())
    {
        // If the groupId option is used, only DCGM_GROUP_ALL_GPUS is supported,
        // or else the default entityIds value.
        if ((drd.groupId != DCGM_GROUP_NULL && drd.groupId != DCGM_GROUP_ALL_GPUS)
            || (drd.groupId == DCGM_GROUP_NULL && std::string_view(drd.entityIds) != "*,cpu:*"))
        {
            error = fmt::format(
                "Error: expectedNumEntities can only be used with DCGM_GROUP_ALL_GPUS, or the default entityIds parameter.");
            return false;
        }
    }
    return true;
}

dcgmReturn_t dcgm_diag_common_populate_run_diag(dcgmRunDiag_v10 &drd,
                                                const std::string &testNames,
                                                const std::string &parms,
                                                const std::string &configFileContents,
                                                const std::string &fakeGpuList,
                                                const std::string &gpuList,
                                                bool verbose,
                                                bool statsOnFail,
                                                const std::string &debugLogFile,
                                                const std::string &statsPath,
                                                unsigned int debugLevel,
                                                const std::string &clocksEventMask,
                                                unsigned int groupId,
                                                bool failEarly,
                                                unsigned int failCheckInterval,
                                                unsigned int timeout,
                                                std::string const &entityIds,
                                                std::string const &expectedNumEntities,
                                                unsigned int watchFrequency,
                                                std::string const &ignoreErrorCodes,
                                                std::string &error)
{
    std::stringstream errbuf;
    // testNames might be a simple 1, 2, 3, or 4 for short, medium, or long. Don't both in that case
    if (testNames.size() > 1)
    {
        std::vector<std::string> testNamesVec;
        dcgmTokenizeString(testNames, ",", testNamesVec);

        for (size_t namesIndex = 0; namesIndex < testNamesVec.size(); namesIndex++)
        {
            snprintf(
                drd.testNames[namesIndex], sizeof(drd.testNames[namesIndex]), "%s", testNamesVec[namesIndex].c_str());
        }
        drd.validate = DCGM_POLICY_VALID_NONE;
    }
    else if (testNames == "1")
    {
        drd.validate = DCGM_POLICY_VALID_SV_SHORT;
    }
    else if (testNames == "2")
    {
        drd.validate = DCGM_POLICY_VALID_SV_MED;
    }
    else if (testNames == "3")
    {
        drd.validate = DCGM_POLICY_VALID_SV_LONG;
    }
    else if (testNames == "4")
    {
        drd.validate = DCGM_POLICY_VALID_SV_XLONG;
    }
    else
    {
        // The only valid 1 character -r options are 1, 2, or 3
        errbuf << "Invalid value '" << testNames << "' specified for -r.";
        error = errbuf.str();
        return DCGM_ST_BADPARAM;
    }

    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        dcgmTokenizeString(parms, ";", parmsVec);

        for (size_t parmsIndex = 0; parmsIndex < parmsVec.size(); parmsIndex++)
        {
            snprintf(drd.testParms[parmsIndex], sizeof(drd.testParms[parmsIndex]), "%s", parmsVec[parmsIndex].c_str());
        }
    }

    if (fakeGpuList.empty() == false)
    {
        if (is_valid_gpu_list(fakeGpuList) == false)
        {
            errbuf << "Invalid fake gpu list '" << fakeGpuList
                   << "'. The list must be a comma-separated list of numbers only.";
            error = errbuf.str();

            return DCGM_ST_BADPARAM;
        }

        snprintf(drd.fakeGpuList, sizeof(drd.fakeGpuList), "%s", fakeGpuList.c_str());
    }

    if (gpuList.empty() == false)
    {
        if (is_valid_gpu_list(gpuList) == false)
        {
            errbuf << "Invalid gpu list '" << gpuList << "'. The list must be a comma-separated list of numbers only.";
            error = errbuf.str();

            return DCGM_ST_BADPARAM;
        }

        snprintf(drd.entityIds, sizeof(drd.entityIds), "%s", gpuList.c_str());
    }
    else
    {
        // entity-id has a default value
        if (entityIds.empty() || entityIds.size() > DCGM_ENTITY_ID_LIST_LEN)
        {
            error = fmt::format("Error: Invalid entity id: [{}]", entityIds);
            return DCGM_ST_BADPARAM;
        }
        SafeCopyTo(drd.entityIds, entityIds.data());
    }

    if (statsOnFail)
    {
        drd.flags |= DCGM_RUN_FLAGS_STATSONFAIL;
    }

    if (verbose)
    {
        drd.flags |= DCGM_RUN_FLAGS_VERBOSE;
    }

    drd.debugLevel = debugLevel;

    snprintf(drd.debugLogFile, sizeof(drd.debugLogFile), "%s", debugLogFile.c_str());
    snprintf(drd.statsPath, sizeof(drd.statsPath), "%s", statsPath.c_str());

    drd.groupId = (dcgmGpuGrp_t)(long long)groupId;

    /* Parameters for newer versions of dcgmRunDiag_t */
    dcgm_diag_common_set_config_file_contents(configFileContents, drd);

    snprintf(drd.clocksEventMask, sizeof(drd.clocksEventMask), "%s", clocksEventMask.c_str());
    if (failEarly) // v5 and newer
    {
        drd.flags |= DCGM_RUN_FLAGS_FAIL_EARLY;
        drd.failCheckInterval = failCheckInterval;
    }

    drd.timeoutSeconds = timeout;

    if (!is_valid_expected_num_entities(drd, expectedNumEntities, error))
    {
        return DCGM_ST_BADPARAM;
    }
    SafeCopyTo(drd.expectedNumEntities, expectedNumEntities.data());

    if (watchFrequency < 100000 || watchFrequency > 60000000)
    {
        error = fmt::format("Error: watch frequency can only be set between 100000 and 60000000 microseconds");
        return DCGM_ST_BADPARAM;
    }
    drd.watchFrequency = watchFrequency;

    SafeCopyTo(drd.ignoreErrorCodes, ignoreErrorCodes.data());

    return DCGM_ST_OK;
}

void dcgm_diag_common_set_config_file_contents(const std::string &configFileContents, dcgmRunDiag_v10 &drd)
{
    snprintf(drd.configFileContents, sizeof(drd.configFileContents), "%s", configFileContents.c_str());
}
