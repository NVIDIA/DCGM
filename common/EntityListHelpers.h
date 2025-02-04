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

#include "DcgmUtilities.h"
#include "MigIdParser.hpp"

#include <dcgm_structs.h>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DcgmNs
{

using EntityMap = std::unordered_map<DcgmNs::ParseResult, dcgmGroupEntityPair_t>;

EntityMap &operator<<(EntityMap &entityMap, dcgmMigHierarchyInfo_v2 const &info);

/**
 * This function acquires list of valid Entity Ids for GPUs and MIG instances from a hostengine and tries to
 * parse the provided string with entities list validating that every entity is known to the hostengine.
 * @param dcgmHandle A handle to a live connection to the hostengine
 * @param ids Comma separated list of entity ids
 * @return Tuple of two things: <br>
 *          1) vector of validated entities <br>
 *          2) a comma separated list of unrecognized entities.
 */
[[nodiscard]] std::tuple<std::vector<dcgmGroupEntityPair_t>, std::string> TryParseEntityList(dcgmHandle_t dcgmHandle,
                                                                                             std::string const &ids);

/**
 * This function acquires list of valid Entity Ids for GPUs and MIG instances from a hostengine and tries to
 * parse the provided string with entities list validating that every entity is known to the hostengine.
 * @param entities Entity map contains system GPU information.
 * @param ids Comma separated list of entity ids
 * @return Tuple of two things: <br>
 *          1) vector of validated entities <br>
 *          2) a comma separated list of unrecognized entities.
 */
[[nodiscard]] std::tuple<std::vector<dcgmGroupEntityPair_t>, std::string> TryParseEntityList(EntityMap entities,
                                                                                             std::string const &ids);

namespace detail
{
    struct GroupEntityPairHasher
    {
        size_t operator()(dcgmGroupEntityPair_t const &value) const
        {
            return Utils::Hash::CompoundHash(value.entityGroupId, value.entityId);
        }
    };

    struct GroupEntityPairEq
    {
        constexpr bool operator()(dcgmGroupEntityPair_t const &left, dcgmGroupEntityPair_t const &right) const
        {
            return std::tie(left.entityGroupId, left.entityId) == std::tie(right.entityGroupId, right.entityId);
        }
    };

    /**
     * @brief Result type for the `HandleWildcard()` function
     * @see `HandleWildcard()` for details about each value meaning.
     */
    enum class HandleWildcardResult
    {
        Handled,   /*!< Wildcards were found */
        Unhandled, /*!< No Wildcards were found */
        Error,     /*!< Error during parsing */
    };

    using EntityGroupContainer = std::unordered_set<dcgmGroupEntityPair_t, GroupEntityPairHasher, GroupEntityPairEq>;

    /**
     * @brief Handles cases if \a value has wildcards in one of its fields.
     *
     * @param[in]   value       Parsed Entity
     * @param[in]   entities    Entities that will be used to satisfy wildcards
     * @param[out]  result      Final list of entity pairs after the wildcards are unrolled
     * @return \c `HandleWildcardResult::Handled` - if wildcards were found and the \a result was updated
     * @return \c `HandleWildcardResult::Unhandled` - no wildcards were found and the \a result was unchanged
     * @return \c `HandleWildcardResult::Error` - an error happened in the process. The state of the \a result is
     *                                            undefined
     */
    HandleWildcardResult HandleWildcard(ParseResult const &value,
                                        EntityMap const &entities,
                                        EntityGroupContainer &result);
} //namespace detail

/**
 * Fills in a lookup table from input
 * @param migHierarchy GPU instance hierachy.
 * @param gpuIdUuids GPU ID and UUID pairs in this system.
 * @return A map from ParseResult to entity pairs.
 * @note This function writes to stdout in case of errors
 */
[[nodiscard]] EntityMap PopulateEntitiesMap(dcgmMigHierarchy_v2 const &migHierarchy,
                                            std::vector<std::pair<unsigned, std::string>> const &gpuIdUuids);

/**
 * Parse entity list string to entity group vector.
 *
 * @param[in] entityList        Entity list
 * @param[out] entityGroups     Parsed results
 * @return A string to indicate error message.
 */
std::string EntityListParser(std::string const &entityList, std::vector<dcgmGroupEntityPair_t> &entityGroups);

/**
 * Parse entity list string to entity group vector. This function will use dcmgHandle to get the system current known
 * GPU and MIG instances.
 *
 * @param dcgmHandle A handle to a live connection to the hostengine
 * @param[in] entityList        Entity list
 * @param[out] entityGroups     Parsed results
 * @return A string to indicate error message.
 */
std::string EntityListWithMigAndUuidParser(dcgmHandle_t dcgmHandle,
                                           std::string const &entityList,
                                           std::vector<dcgmGroupEntityPair_t> &entityGroups);


std::vector<std::uint32_t> ParseEntityIdsAndFilterGpu(dcgmMigHierarchy_v2 const &migHierarchy,
                                                      std::vector<std::pair<unsigned, std::string>> const &gpuIdUuids,
                                                      std::string_view entityIds);

/**
 * Parse number of entities string for the number of GPUs.
 *
 * @param[in] expectedNumEntities supported string format "gpu:N"
 * @param[out] gpuCount the number of GPUs parsed from the string
 * @return A string with error message if any.
 */
std::string ParseExpectedNumEntitiesForGpus(std::string const &expectedNumEntities, unsigned int &gpuCount);

void TopologicalSort(dcgmMigHierarchy_v2 &hierarchy);

} //namespace DcgmNs