/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmVariantHelper.hpp>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include "DcgmStringHelpers.h"
#include "MigIdParser.hpp"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgmi_common.h"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <sys/ioctl.h>


/*******************************************************************************/
dcgmReturn_t dcgmi_parse_entity_list_string(std::string const &input, std::vector<dcgmGroupEntityPair_t> &entityList)
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::stringstream ss(input);
    dcgmGroupEntityPair_t insertElem;
    std::string entityIdStr;
    std::vector<std::string> tokens;

    /* Divide the string into a vector of substrings by comma */
    std::string delimStr = ",";

    /* This is expecting input to be strings like:
       "0,1,2" or
       "gpu:0,nvswitch:3,2" (gpu 0, nvswitch 3, gpu 2) */
    dcgmTokenizeString(input, delimStr, tokens);

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        entityIdStr = tokens[i];
        if (entityIdStr.empty())
        {
            SHOW_AND_LOG_ERROR << "Error: Comma without a value detected at token " << i + 1 << " of " << input;
            return DCGM_ST_BADPARAM;
        }

        /* Default to GPUs in the case that the entityGroupId isn't specified */
        insertElem.entityGroupId = DCGM_FE_GPU;

        /* See if this has an entityGroup on the front */
        size_t colonPos = entityIdStr.find_first_of(":");
        if (colonPos != std::string::npos)
        {
            switch (entityIdStr.at(0))
            {
                case 'g':
                case 'G':
                    insertElem.entityGroupId = DCGM_FE_GPU;
                    break;
                case 'n':
                case 'N':
                    insertElem.entityGroupId = DCGM_FE_SWITCH;
                    break;
                case 'v':
                case 'V':
                    insertElem.entityGroupId = DCGM_FE_VGPU;
                    break;
                case 'i':
                case 'I':
                    insertElem.entityGroupId = DCGM_FE_GPU_I;
                    break;
                case 'c':
                case 'C':
                    insertElem.entityGroupId = DCGM_FE_GPU_CI;
                    break;
                case 'l':
                case 'L':
                    insertElem.entityGroupId = DCGM_FE_LINK;
                    break;

                default:
                    SHOW_AND_LOG_ERROR << "Error: invalid entity type: '" << entityIdStr
                                       << "'. Expected gpu/ci/i/vgpu/nvswitch.";
                    return DCGM_ST_BADPARAM;
            }

            /* Move past the colon */
            entityIdStr = entityIdStr.substr(colonPos + 1);
        }

        if (entityIdStr.empty())
        {
            SHOW_AND_LOG_ERROR << "Error: empty entityId detected in " << entityIdStr;
            return DCGM_ST_BADPARAM;
        }

        /** Add an item
         * They are encoded as a 32 bit unsigned int with a type, GPU ID, and
         * link index. Do we encode them literally for now?
         */
        if (isdigit(entityIdStr.at(0)))
        {
            insertElem.entityId = std::stol(entityIdStr);
            entityList.push_back(insertElem);
        }
        else
        {
            SHOW_AND_LOG_ERROR << "Error: Expected numerical entityId instead of " << entityIdStr;
            return DCGM_ST_BADPARAM;
        }
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t dcgmi_create_entity_group(dcgmHandle_t dcgmHandle,
                                       dcgmGroupType_t groupType,
                                       dcgmGpuGrp_t *groupId,
                                       std::vector<dcgmGroupEntityPair_t> &entityList)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid          = (unsigned int)getpid();
    char groupName[32]          = { 0 };
    unsigned int i;

    snprintf(groupName, sizeof(groupName) - 1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmGroupCreate(dcgmHandle, groupType, groupName, groupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << "Got error while creating a GPU group: " << dcgmReturn << " " << errorString(dcgmReturn);
        return dcgmReturn;
    }

    for (i = 0; i < entityList.size(); i++)
    {
        dcgmReturn = dcgmGroupAddEntity(dcgmHandle, *groupId, entityList[i].entityGroupId, entityList[i].entityId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            SHOW_AND_LOG_ERROR << "Error: Got error " << dcgmReturn << " " << errorString(dcgmReturn) << " while "
                               << "adding " << DcgmFieldsGetEntityGroupString(entityList[i].entityGroupId) << " "
                               << entityList[i].entityId << " to our entity group.";
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
bool dcgmi_entity_group_id_is_special(std::string &groupIdStr, dcgmGroupType_t *groupType, dcgmGpuGrp_t *groupId)
{
    if (groupIdStr == "all_gpus")
    {
        *groupType = DCGM_GROUP_DEFAULT;
        *groupId   = (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS;
        return true;
    }
    else if (groupIdStr == "all_nvswitches")
    {
        *groupType = DCGM_GROUP_DEFAULT_NVSWITCHES;
        *groupId   = (dcgmGpuGrp_t)DCGM_GROUP_ALL_NVSWITCHES;
        return true;
    }
    else
    {
        *groupType = DCGM_GROUP_EMPTY;
        return false;
    }
}

/*******************************************************************************/
dcgmReturn_t dcgmi_parse_field_id_list_string(std::string input, std::vector<unsigned short> &fieldIds, bool validate)
{
    std::vector<std::string> tokens;
    unsigned int i;
    unsigned short fieldId;
    dcgm_field_meta_p fieldMeta;

    /* Divide the string into a vector of substrings by comma */
    std::string delimStr = ",";
    dcgmTokenizeString(input, delimStr, tokens);

    /* Convert each token into an unsigned integer */
    for (i = 0; i < tokens.size(); i++)
    {
        fieldId = atoi(tokens[i].c_str());
        if (!fieldId && (!tokens[i].size() || tokens[i].at(0) != '0'))
        {
            SHOW_AND_LOG_ERROR << "Error: Expected numerical fieldId. Got '" << tokens[i] << "' instead.";
            return DCGM_ST_BADPARAM;
        }

        if (validate)
        {
            fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                SHOW_AND_LOG_ERROR << "Error: Got invalid field ID. '" << fieldId
                                   << "'. See dcgm_fields.h for a list of valid field IDs.";
                return DCGM_ST_BADPARAM;
            }
        }

        fieldIds.push_back(fieldId);
    }

    return DCGM_ST_OK;
}

/*******************************************************************************/
dcgmReturn_t dcgmi_create_field_group(dcgmHandle_t dcgmHandle,
                                      dcgmFieldGrp_t *groupId,
                                      std::vector<unsigned short> &fieldIds)
{
    dcgmReturn_t dcgmReturn;
    static int numGroupsCreated = 0;
    unsigned int myPid          = (unsigned int)getpid();
    char groupName[32]          = { 0 };

    snprintf(groupName, sizeof(groupName) - 1, "dcgmi_%u_%d", myPid, ++numGroupsCreated);

    dcgmReturn = dcgmFieldGroupCreate(dcgmHandle, fieldIds.size(), &fieldIds[0], groupName, groupId);
    if (dcgmReturn != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << "Got error while creating a Field Group: " << errorString(dcgmReturn);
    }

    return dcgmReturn;
}

/*******************************************************************************/
const char *dcgmi_parse_hostname_string(const char *hostName, bool *isUnixSocketAddress, bool logOnError)
{
    if (!strncmp(hostName, "unix://", 7))
    {
        *isUnixSocketAddress = true;
        /* Looks like a unix socket. Do some validation */
        if (strlen(hostName) < 8)
        {
            if (logOnError)
            {
                SHOW_AND_LOG_ERROR << "Missing hostname after \"unix://\".";
            }

            return nullptr;
        }
        else
            return &hostName[7];
    }

    /* No "unix://". Treat like a regular hostname */
    *isUnixSocketAddress = false;
    return hostName;
}

/*******************************************************************************/
namespace DcgmNs
{
bool operator==(dcgmGroupEntityPair_t const &left, dcgmGroupEntityPair_t const &right) noexcept
{
    return std::tie(left.entityGroupId, left.entityId) == std::tie(right.entityGroupId, right.entityId);
}

struct EntityMapTypedSentinel
{};

template <typename T>
struct EntityMapTypedIterator : std::iterator<std::forward_iterator_tag, T>
{
    EntityMapTypedIterator() = default;

    explicit EntityMapTypedIterator(EntityMap const &map)
    {
        m_it  = cbegin(map);
        m_end = cend(map);

        while (m_it != m_end && !std::holds_alternative<T>(m_it->first))
        {
            ++m_it;
        }
    }

    auto const &operator*() const noexcept
    {
        return *m_it;
    }

    auto const *operator->() const noexcept
    {
        return m_it.operator->();
    }

    bool operator==(EntityMapTypedSentinel const &) const
    {
        return m_it == m_end;
    }

    bool operator==(EntityMapTypedIterator const &other) const
    {
        return m_it == other.it;
    }

    template <typename Y>
    bool operator!=(Y const &other) const
    {
        return !(*this == other);
    }

    EntityMapTypedIterator &operator++()
    {
        if (m_it != m_end)
        {
            do
            {
                ++m_it;
            } while (m_it != m_end && !std::holds_alternative<T>(m_it->first));
        }
        return *this;
    }

private:
    EntityMap::const_iterator m_it;
    EntityMap::const_iterator m_end;
};

template <typename T>
auto typed_begin(EntityMap const &map)
{
    return EntityMapTypedIterator<T> { map };
}

auto typed_end(EntityMap const &)
{
    return EntityMapTypedSentinel {};
}

EntityMap &operator<<(EntityMap &entityMap, dcgmMigHierarchyInfo_v2 const &info)
{
    auto const uuid = CutUuidPrefix(info.info.gpuUuid);
    switch (info.entity.entityGroupId)
    {
        case DCGM_FE_GPU_I:
            entityMap.insert_or_assign(ParsedGpuI { std::string { uuid }, info.info.nvmlInstanceId }, info.entity);
            entityMap.insert_or_assign(ParsedGpuI { std::to_string(info.info.nvmlGpuIndex), info.info.nvmlInstanceId },
                                       info.entity);

            /* We add GPUs with both UUID and index to the map as the 0/0/0 and UUID/0/0 are identical */
            entityMap.insert_or_assign(ParsedGpu { std::string { uuid } }, info.parent);
            entityMap.insert_or_assign(ParsedGpu { std::to_string(info.info.nvmlGpuIndex) }, info.parent);

            break;

        case DCGM_FE_GPU_CI:
            entityMap.insert_or_assign(
                ParsedGpuCi { std::string { uuid }, info.info.nvmlInstanceId, info.info.nvmlComputeInstanceId },
                info.entity);
            entityMap.insert_or_assign(ParsedGpuCi { std::to_string(info.info.nvmlGpuIndex),
                                                     info.info.nvmlInstanceId,
                                                     info.info.nvmlComputeInstanceId },
                                       info.entity);

            /* We add GPUs with both UUID and index to the map as the 0/0/0 and UUID/0/0 are identical */
            entityMap.insert_or_assign(ParsedGpu { std::string { uuid } }, info.parent);
            entityMap.insert_or_assign(ParsedGpu { std::to_string(info.info.nvmlGpuIndex) }, info.parent);

            break;

        default:
            SHOW_AND_LOG_ERROR << "Unexpected entity group in the MIG hierarchy results. GroupId: "
                               << info.entity.entityGroupId;
            break;
    }

    return entityMap;
}

/**
 * Fills in a lookup table for entity ids for GPUs and MIG devices known to the hostengine
 * @param dcgmHandle A handle associated with a live connection to the hostengine.
 * @return A map from ParseResult to entity pairs.
 * @note This function writes to stdout in case of errors
 */
[[nodiscard]] static EntityMap PopulateEntitiesMap(dcgmHandle_t dcgmHandle)
{
    using namespace DcgmNs;

    EntityMap entityMap;

    // Mig Hierarchy

    dcgmMigHierarchy_v2 migHierarchy {};
    migHierarchy.version = dcgmMigHierarchy_version2;

    auto ret = dcgmGetGpuInstanceHierarchy(dcgmHandle, &migHierarchy);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_DEBUG << "Failed to collect MIG hierarchy information. "
                          "EntityIds associated with MIG devices will not be possible to specify. "
                       << "Result: " << ret << " " << errorString(ret);
    }
    else
    {
        for (size_t idx = 0; idx < migHierarchy.count; ++idx)
        {
            auto const &instance = migHierarchy.entityList[idx];
            entityMap << instance;
        }
    }

    // GPU entities

    std::array<dcgm_field_eid_t, DCGM_MAX_NUM_DEVICES> entities {};
    int numItems = DCGM_MAX_NUM_DEVICES;

    ret = dcgmGetEntityGroupEntities(dcgmHandle, DCGM_FE_GPU, entities.data(), &numItems, 0);
    if (ret != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << "Unable to collect GPU entities. "
                              "EntityIds associated with GPUs may not be possible to specify. "
                           << "Result: " << ret << " " << errorString(ret);
    }
    else
    {
        for (size_t idx = 0; idx < (size_t)numItems; ++idx)
        {
            entityMap.insert_or_assign(ParsedGpu { std::to_string(entities[idx]) },
                                       dcgmGroupEntityPair_t { DCGM_FE_GPU, entities[idx] });

            dcgmDeviceAttributes_t deviceAttributes {};
            deviceAttributes.version = dcgmDeviceAttributes_version3;

            ret = dcgmGetDeviceAttributes(dcgmHandle, entities[idx], &deviceAttributes);
            if (ret != DCGM_ST_OK)
            {
                SHOW_AND_LOG_ERROR << "Unable to collect GPU attributes for GpuId " << entities[idx]
                                   << ". It may be impossible to specify GPU UUID as entity id. "
                                   << "Result: " << ret << " " << errorString(ret);
                continue;
            }

            auto const uuid = CutUuidPrefix(deviceAttributes.identifiers.uuid);
            entityMap.insert_or_assign(ParseInstanceId(uuid), dcgmGroupEntityPair_t { DCGM_FE_GPU, entities[idx] });
        }
    }

    return entityMap;
}

namespace detail
{
    HandleWildcardResult HandleWildcard(ParseResult const &value,
                                        EntityMap const &entities,
                                        EntityGroupContainer &result)
    {
        return std::visit(overloaded(
                              [&entities, &result](ParsedGpu const &val) {
                                  if (!val.gpuUuid.IsWildcarded())
                                  {
                                      return HandleWildcardResult::Unhandled;
                                  }
                                  for (auto it = typed_begin<ParsedGpu>(entities); it != typed_end(entities); ++it)
                                  {
                                      result.insert(it->second);
                                  }
                                  return HandleWildcardResult::Handled;
                              },
                              [&entities, &result](ParsedGpuI const &val) {
                                  if (!val.gpuUuid.IsWildcarded() && !val.instanceId.IsWildcarded())
                                  {
                                      return HandleWildcardResult::Unhandled;
                                  }
                                  for (auto it = typed_begin<ParsedGpuI>(entities); it != typed_end(entities); ++it)
                                  {
                                      auto const &entity = std::get<ParsedGpuI>(it->first);

                                      bool const isGpuMatched      = compare(entity.gpuUuid, val.gpuUuid);
                                      bool const isInstanceMatched = compare(entity.instanceId, val.instanceId);

                                      if (isGpuMatched && isInstanceMatched)
                                      {
                                          result.insert(it->second);
                                      }
                                  }
                                  return HandleWildcardResult::Handled;
                              },
                              [&entities, &result](ParsedGpuCi const &val) {
                                  if (!val.gpuUuid.IsWildcarded() && !val.instanceId.IsWildcarded()
                                      && !val.computeInstanceId.IsWildcarded())
                                  {
                                      return HandleWildcardResult::Unhandled;
                                  }
                                  for (auto it = typed_begin<ParsedGpuCi>(entities); it != typed_end(entities); ++it)
                                  {
                                      auto const &entity           = std::get<ParsedGpuCi>(it->first);
                                      bool const isGpuMatched      = compare(entity.gpuUuid, val.gpuUuid);
                                      bool const isInstanceMatched = compare(entity.instanceId, val.instanceId);
                                      bool const isCInstanceMatched
                                          = compare(entity.computeInstanceId, val.computeInstanceId);

                                      if (isGpuMatched && isInstanceMatched && isCInstanceMatched)
                                      {
                                          result.insert(it->second);
                                      }
                                  }
                                  return HandleWildcardResult::Handled;
                              },
                              [&](auto const &) { return HandleWildcardResult::Error; }),
                          value);
    }
} // namespace detail


[[nodiscard]] std::tuple<std::vector<dcgmGroupEntityPair_t>, std::string> TryParseEntityList(dcgmHandle_t dcgmHandle,
                                                                                             std::string const &Ids)
{
    detail::EntityGroupContainer entityList;
    std::string rejectedIds;

    auto entities = DcgmNs::PopulateEntitiesMap(dcgmHandle);
    /*
     * Note (nkonyuchenko): entities will contain duplicates of GPUs as PopulateEntitiesMap adds GPUs with indices and
     * UUIDs as gpuUuid values. This is automatically handled by using EntityGroupContainer which is an unordered_set.
     */

    {
        auto tokens = Split(Ids, ',');
        std::vector<std::string_view> rejectedTokens;
        rejectedTokens.reserve(tokens.size());

        for (auto const &token : tokens)
        {
            /// For wildcard cases, we need to understand which part is wildcarded.
            ///     *       - all GPUs
            ///     */*     - all MIG GPU instances
            ///     */*/*   - all Compute Instances
            ///     To add all possible entities, we will have to specify "*,*/*,*/*/*"
            ///     0/*     - all GPU instance on GPU 0
            ///     0/*/0   - all Compute Instances 0 on all GPU Instance on GPU 0

            auto parsedResult = ParseInstanceId(token);
            if (std::holds_alternative<ParsedUnknown>(parsedResult))
            {
                rejectedTokens.push_back(token);
                continue;
            }

            auto result = detail::HandleWildcard(parsedResult, entities, entityList);

            switch (result)
            {
                case detail::HandleWildcardResult::Error:
                {
                    DCGM_LOG_DEBUG << fmt::format("Unable to parse token {}", token);
                    rejectedTokens.push_back(token);
                }
                break;

                case detail::HandleWildcardResult::Unhandled:
                {
                    auto it = entities.find(parsedResult);
                    if (it == entities.end())
                    {
                        DCGM_LOG_DEBUG << "Specified entity ID is valid but unknown: " << token
                                       << ". ParsedResult: " << parsedResult;
                        rejectedTokens.push_back(token);
                        continue;
                    }

                    entityList.insert(it->second);
                }
                break;

                case detail::HandleWildcardResult::Handled:
                    continue;

                default:
                    DCGM_LOG_ERROR << "Unexpected value of HandleWildcardResult: " << static_cast<std::int32_t>(result);
            }
        }

        rejectedIds = Join(rejectedTokens, ",");
    }

    std::vector<dcgmGroupEntityPair_t> result;
    result.reserve(entityList.size());
    for (auto it = entityList.begin(); it != entityList.end();)
    {
        result.push_back(entityList.extract(it++).value());
    }

    /*
     * This is not a topological sort. Just for better output of the dcgmi group -l command
     */
    std::sort(begin(result), end(result), [](auto const &left, auto const &right) {
        return std::tie(left.entityGroupId, left.entityId) < std::tie(right.entityGroupId, right.entityId);
    });

    return std::make_tuple(std::move(result), std::move(rejectedIds));
}

namespace Terminal
{
    bool IsTTY()
    {
        return !!isatty(fileno(stdout));
    }

    std::optional<TermDimensions> GetTermDimensions()
    {
        if (!IsTTY())
        {
            return std::nullopt;
        }
        winsize w = {};
        ioctl(fileno(stdout), TIOCGWINSZ, &w);
        if (w.ws_col == 0 || w.ws_row == 0)
        {
            return std::nullopt;
        }
        return TermDimensions { w.ws_row, w.ws_col };
    }
} // namespace Terminal

} // namespace DcgmNs
