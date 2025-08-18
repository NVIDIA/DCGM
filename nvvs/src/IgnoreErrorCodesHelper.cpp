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

#include "IgnoreErrorCodesHelper.h"
#include "dcgm_errors.h"

static std::unordered_set<unsigned int> const defaultValidErrorCodes {
    DCGM_FR_THERMAL_VIOLATIONS, DCGM_FR_THERMAL_VIOLATIONS_TS, DCGM_FR_XID_ERROR,
    DCGM_FR_UNCONTAINED_ERROR,  DCGM_FR_ROW_REMAP_FAILURE,     DCGM_FR_PENDING_ROW_REMAP,
};
static char const defaultEntitySeparator = ';';

std::string ParseIgnoreErrorCodesString(std::string_view inputString,
                                        gpuIgnoreErrorCodeMap_t &gpuErrorCodeMap,
                                        std::vector<unsigned int> const &validGpuIndices,
                                        std::optional<std::unordered_set<unsigned int>> const &validErrorCodes,
                                        std::optional<char> entitySeparator)
{
    gpuErrorCodeMap.clear();
    if (inputString.empty())
    {
        return "";
    }

    char localEntitySeparator      = defaultEntitySeparator;
    std::string invalidParamErrStr = "Invalid --ignoreErrorCodes parameter string. Run nvvs --help for usage info.";
    std::unordered_set<unsigned int> localValidErrorCodes = defaultValidErrorCodes;
    std::unordered_set<unsigned int> validGpuSet;
    for (auto gpu : validGpuIndices)
    {
        validGpuSet.insert(gpu);
    }
    // Use input set of error codes if provided
    if (validErrorCodes.has_value())
    {
        localValidErrorCodes = validErrorCodes.value();
    }
    if (entitySeparator.has_value())
    {
        localEntitySeparator = entitySeparator.value();
    }
    auto multiEntityErrStr = DcgmNs::Split(inputString, localEntitySeparator);
    for (auto const &entityErrStr : multiEntityErrStr)
    {
        bool allEntities   = false;
        bool allErrorCodes = false;
        unsigned int gpuId;
        std::string entityStr, errStr;
        std::unordered_set<unsigned int> inputErrorCodes = {};

        auto entityErrStrSplit = DcgmNs::Split(entityErrStr, ':');
        if (entityErrStrSplit.size() > 2)
        {
            return invalidParamErrStr;
        }
        else if (entityErrStrSplit.size() == 1) // only error codes list provided
        {
            allEntities   = true;
            allErrorCodes = false;
            errStr        = entityErrStr;
        }
        else // both entity and error codes list provided
        {
            entityStr     = entityErrStrSplit[0];
            errStr        = entityErrStrSplit[1];
            allEntities   = (entityStr == "*");
            allErrorCodes = (errStr == "*");
        }

        if (allEntities && allErrorCodes)
        {
            // only *:* should be specified as input, if used
            if (multiEntityErrStr.size() > 1)
            {
                return fmt::format("Only '*:*' should be specified as input if provided. {}", invalidParamErrStr);
            }
        }
        else if (allEntities)
        {
            // only *:<error codes> should be specified as input, if used
            if (multiEntityErrStr.size() > 1)
            {
                return fmt::format("Only '*:<error codes list>' should be specified as input if provided. {}",
                                   invalidParamErrStr);
            }
        }

        // Parse the entity string if provided
        if (!allEntities)
        {
            // only gpu supported at this time
            if (!dcgmStrToLower(std::string(entityStr)).starts_with("gpu"))
            {
                return fmt::format("Only GPU entities are supported at this time. {}", invalidParamErrStr);
            }
            // remove "gpu" from the start of the string
            std::string_view gpuIdStr = entityStr.substr(3);
            try
            {
                // only gpus from the provided set are allowed
                gpuId = std::stoul(gpuIdStr.data());
                dcgmGroupEntityPair_t entity {
                    .entityGroupId = DCGM_FE_GPU,
                    .entityId      = gpuId,
                };
                if (!validGpuSet.contains(gpuId))
                {
                    return fmt::format("GPU index {} invalid. {}", gpuId, invalidParamErrStr);
                }
                if (gpuErrorCodeMap.contains(entity))
                {
                    return fmt::format("GPU index {} repeated. {}", gpuId, invalidParamErrStr);
                }
            }
            catch (const std::exception &)
            {
                return fmt::format("Invalid entities. {}", invalidParamErrStr);
            }
        }

        // Parse the error codes string if provided
        if (!allErrorCodes)
        {
            auto errStrSplit = DcgmNs::Split(errStr, ',');
            for (auto ec : errStrSplit)
            {
                try
                {
                    auto ecNum = std::stoul(ec.data());
                    if (!localValidErrorCodes.contains(ecNum))
                    {
                        return fmt::format("Error code {} invalid. {}", ecNum, invalidParamErrStr);
                    }
                    inputErrorCodes.insert(ecNum);
                }
                catch (const std::exception &e)
                {
                    return fmt::format("Invalid error code. {}", invalidParamErrStr);
                }
            }
        }

        // Populate the map
        if (allEntities && allErrorCodes)
        {
            for (auto gpuId : validGpuSet)
            {
                dcgmGroupEntityPair_t entity {
                    .entityGroupId = DCGM_FE_GPU,
                    .entityId      = gpuId,
                };
                gpuErrorCodeMap[entity] = localValidErrorCodes;
            }
        }
        else if (allEntities)
        {
            for (auto gpuId : validGpuSet)
            {
                dcgmGroupEntityPair_t entity {
                    .entityGroupId = DCGM_FE_GPU,
                    .entityId      = gpuId,
                };
                gpuErrorCodeMap[entity] = inputErrorCodes;
            }
        }
        else if (allErrorCodes)
        {
            dcgmGroupEntityPair_t entity {
                .entityGroupId = DCGM_FE_GPU,
                .entityId      = gpuId,
            };
            gpuErrorCodeMap[entity] = localValidErrorCodes;
        }
        else
        {
            dcgmGroupEntityPair_t entity {
                .entityGroupId = DCGM_FE_GPU,
                .entityId      = gpuId,
            };
            gpuErrorCodeMap[entity] = std::move(inputErrorCodes);
        }
    }
    return "";
}
