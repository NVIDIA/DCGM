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

#include "EntityListHelpers.h"
#include "CpuHelpers.h"
#include "DcgmGroupEntityPairHelpers.h"
#include "DcgmLogging.h"
#include "DcgmStringHelpers.h"
#include "dcgm_agent.h"
#include <algorithm>
#include <sstream>
#include <tuple>
#include <unordered_set>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace DcgmNs;

namespace
{

bool AllDigits(std::string const &str)
{
    if (str.empty())
    {
        return false;
    }

    for (auto const c : str)
    {
        if (!isdigit(c))
        {
            return false;
        }
    }
    return true;
}

} //namespace

namespace DcgmNs
{

struct EntityMapTypedSentinel
{};

template <typename T>
struct EntityMapTypedIterator
{
    using iterator_category = std::forward_iterator_tag;
    using value_type        = T;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T *;
    using reference         = T &;

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

            break;

        default:
            log_error("Unexpected entity group in the MIG hierarchy results. GroupId: {}", info.entity.entityGroupId);
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
    // Get only supported (active) devices from the CM
    constexpr unsigned int flags = DCGM_GEGE_FLAG_ONLY_SUPPORTED;

    ret = dcgmGetEntityGroupEntities(dcgmHandle, DCGM_FE_GPU, entities.data(), &numItems, flags);
    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to collect GPU entities. "
                  "EntityIds associated with GPUs may not be possible to specify. "
                  "Result: {}, {}",
                  ret,
                  errorString(ret));
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
                log_error("Unable to collect GPU attributes for GpuId {}"
                          ". It may be impossible to specify GPU UUID as entity id. "
                          "Result: {}, {}",
                          entities[idx],
                          ret,
                          errorString(ret));
                continue;
            }

            auto const uuid = CutUuidPrefix(deviceAttributes.identifiers.uuid);
            entityMap.insert_or_assign(ParseInstanceId(uuid), dcgmGroupEntityPair_t { DCGM_FE_GPU, entities[idx] });
        }
    }

    return entityMap;
}

/**
 * Fills in a lookup table from input
 * @param migHierarchy GPU instance hierachy.
 * @param gpuIdUuids GPU ID and UUID pairs in this system.
 * @return A map from ParseResult to entity pairs.
 * @note This function writes to stdout in case of errors
 */
[[nodiscard]] EntityMap PopulateEntitiesMap(dcgmMigHierarchy_v2 const &migHierarchy,
                                            std::vector<std::pair<unsigned, std::string>> const &gpuIdUuids)
{
    using namespace DcgmNs;

    EntityMap entityMap;

    for (size_t idx = 0; idx < migHierarchy.count; ++idx)
    {
        auto const &instance = migHierarchy.entityList[idx];
        entityMap << instance;
    }

    for (size_t idx = 0; idx < gpuIdUuids.size(); ++idx)
    {
        entityMap.insert_or_assign(ParsedGpu { std::to_string(gpuIdUuids[idx].first) },
                                   dcgmGroupEntityPair_t { DCGM_FE_GPU, gpuIdUuids[idx].first });

        auto const uuid = CutUuidPrefix(gpuIdUuids[idx].second);
        entityMap.insert_or_assign(ParseInstanceId(uuid), dcgmGroupEntityPair_t { DCGM_FE_GPU, gpuIdUuids[idx].first });
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
                                                                                             std::string const &ids)
{
    auto entities = DcgmNs::PopulateEntitiesMap(dcgmHandle);
    return TryParseEntityList(std::move(entities), ids);
}

[[nodiscard]] std::tuple<std::vector<dcgmGroupEntityPair_t>, std::string> TryParseEntityList(EntityMap entities,
                                                                                             std::string const &ids)
{
    detail::EntityGroupContainer entityList;
    std::string rejectedIds;

    /*
     * Note (nkonyuchenko): entities will contain duplicates of GPUs as PopulateEntitiesMap adds GPUs with indices and
     * UUIDs as gpuUuid values. This is automatically handled by using EntityGroupContainer which is an unordered_set.
     */

    {
        auto tokens = Split(ids, ',');
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

std::string EntityListParser(std::string const &entityList, std::vector<dcgmGroupEntityPair_t> &entityGroups)
{
    std::stringstream ss(entityList);
    dcgmGroupEntityPair_t insertElem;
    std::string entityIdStr;
    std::vector<std::string> tokens;
    std::string const delimStr = ",";

    /* This is expecting input to be strings like:
       "0,1,2" or
       "gpu:0,nvswitch:3,2" (gpu 0, nvswitch 3, gpu 2) */
    dcgmTokenizeString(entityList, delimStr, tokens);

    for (size_t i = 0; i < tokens.size(); ++i)
    {
        entityIdStr = tokens[i];
        if (entityIdStr.empty())
        {
            std::string err
                = fmt::format("Error: Comma without a value detected at token {} of input {}", i + 1, entityList);
            log_error(err);
            return err;
        }

        insertElem.entityGroupId = DCGM_FE_GPU;

        size_t colonPos = entityIdStr.find_first_of(":");
        if (colonPos != std::string::npos)
        {
            std::string lowered = entityIdStr.substr(0, colonPos);
            std::transform(
                lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
            switch (lowered.at(0))
            {
                case 'g':
                    insertElem.entityGroupId = DCGM_FE_GPU;
                    break;
                case 'n':
                    insertElem.entityGroupId = DCGM_FE_SWITCH;
                    break;
                case 'v':
                    insertElem.entityGroupId = DCGM_FE_VGPU;
                    break;
                case 'i':
                    insertElem.entityGroupId = DCGM_FE_GPU_I;
                    break;
                case 'c':
                {
                    static std::string const CPU("cpu");
                    static std::string const CORE("core");
                    static std::string const CX("cx");

                    if (lowered == CPU)
                    {
                        insertElem.entityGroupId = DCGM_FE_CPU;
                    }
                    else if (lowered == CORE)
                    {
                        insertElem.entityGroupId = DCGM_FE_CPU_CORE;
                    }
                    else if (lowered == CX)
                    {
                        insertElem.entityGroupId = DCGM_FE_CONNECTX;
                    }
                    else
                    {
                        insertElem.entityGroupId = DCGM_FE_GPU_CI;
                    }
                    break;
                }
                case 'l':
                    insertElem.entityGroupId = DCGM_FE_LINK;
                    break;

                default:
                    std::string err = fmt::format("Error: invalid entity type: {}", entityIdStr);
                    log_error(err);
                    return err;
            }

            entityIdStr = entityIdStr.substr(colonPos + 1);
        }

        if (entityIdStr.empty())
        {
            std::string err = fmt::format("Error: empty entityId detected in {}", entityIdStr);
            log_error(err);
            return err;
        }

        if (entityIdStr.at(0) == '{')
        {
            size_t closeBracket = entityIdStr.find('}');

            if (closeBracket == std::string::npos)
            {
                std::string err = fmt::format("A malformed range with no close bracket was specified: {}", entityIdStr);
                log_error(err);
                return err;
            }

            std::vector<unsigned int> entityIds;
            dcgmReturn_t ret = DcgmNs::ParseRangeString(entityIdStr.substr(1, closeBracket - 1), entityIds);
            if (ret != DCGM_ST_OK)
            {
                std::string err = fmt::format("A malformed range was specified: {}", entityIdStr);
                log_error(err);
                return err;
            }

            for (auto const &entityId : entityIds)
            {
                insertElem.entityId = entityId;
                entityGroups.push_back(insertElem);
            }
        }
        else if (entityIdStr == "*")
        {
            if (insertElem.entityGroupId != DCGM_FE_CPU)
            {
                std::string err = fmt::format("Entity type: [{}] does not support wildcard.", insertElem.entityGroupId);
                log_error(err);
                return err;
            }

            CpuHelpers cpuHelpers;

            if (cpuHelpers.GetVendor() != CpuHelpers::GetNvidiaVendorName() && !CpuHelpers::SupportNonNvidiaCpu())
            {
                continue;
            }

            for (unsigned int cpuId : cpuHelpers.GetCpuIds())
            {
                insertElem.entityId = cpuId;
                entityGroups.push_back(insertElem);
            }
        }
        else
        {
            if (AllDigits(entityIdStr))
            {
                insertElem.entityId = std::stol(entityIdStr);
                entityGroups.push_back(insertElem);
            }
            else
            {
                std::string err = fmt::format("Error: Expected numerical entityId instead of {}", entityIdStr);
                log_error(err);
                return err;
            }
        }
    }
    return "";
}

std::string ParseExpectedNumEntitiesForGpus(std::string const &expectedNumEntities, unsigned int &gpuCount)
{
    gpuCount = 0;
    if (expectedNumEntities.empty())
    {
        return "";
    }

    std::string gpuCountStr;
    size_t colonPos = expectedNumEntities.find_first_of(":");
    if (colonPos != std::string::npos)
    {
        std::string loweredStr = expectedNumEntities.substr(0, colonPos);
        std::transform(
            loweredStr.begin(), loweredStr.end(), loweredStr.begin(), [](unsigned char c) { return std::tolower(c); });

        if (!loweredStr.starts_with("gpu"))
        {
            std::string err = fmt::format("Parameter expectedNumEntities {} format incorrect, cannot be parsed.",
                                          expectedNumEntities);
            return err;
        }
        gpuCountStr = expectedNumEntities.substr(colonPos + 1);
    }
    if (!AllDigits(gpuCountStr))
    {
        std::string err
            = fmt::format("Parameter expectedNumEntities {} format incorrect, cannot be parsed.", expectedNumEntities);
        return err;
    }
    try
    {
        gpuCount = stol(gpuCountStr);
    }
    catch (const std::exception &e)
    {
        std::string err = fmt::format("Parameter expectedNumEntities {} format incorrect, cannot be parsed. Error: {}",
                                      expectedNumEntities,
                                      e.what());
        return err;
    }
    return "";
}

std::string EntityListWithMigAndUuidParser(dcgmHandle_t dcgmHandle,
                                           std::string const &entityList,
                                           std::vector<dcgmGroupEntityPair_t> &entityGroups)
{
    auto [gpuEntities, rejectedIds] = DcgmNs::TryParseEntityList(dcgmHandle, entityList);

    // Fallback to old method
    std::vector<dcgmGroupEntityPair_t> oldEntityList;

    /* Convert the string to a list of entities */
    auto err = DcgmNs::EntityListParser(rejectedIds, oldEntityList);
    if (!err.empty())
    {
        return err;
    }

    std::move(begin(gpuEntities), end(gpuEntities), std::back_inserter(entityGroups));
    std::move(begin(oldEntityList), end(oldEntityList), std::back_inserter(entityGroups));
    return "";
}

std::vector<std::uint32_t> ParseEntityIdsAndFilterGpu(dcgmMigHierarchy_v2 const &migHierarchy,
                                                      std::vector<std::pair<unsigned, std::string>> const &gpuIdUuids,
                                                      std::string_view entityIds)
{
    DcgmNs::EntityMap entities = DcgmNs::PopulateEntitiesMap(migHierarchy, gpuIdUuids);

    std::vector<dcgmGroupEntityPair_t> entityGroups;
    auto [gpuEntities, rejectedIds] = DcgmNs::TryParseEntityList(std::move(entities), std::string(entityIds));

    // Fallback to old method
    std::vector<dcgmGroupEntityPair_t> oldEntityList;

    /* Convert the string to a list of entities */
    auto err = DcgmNs::EntityListParser(rejectedIds, oldEntityList);
    if (!err.empty())
    {
        log_error("failed to parse entity ids: {}", entityIds);
        return {};
    }

    std::move(begin(gpuEntities), end(gpuEntities), std::back_inserter(entityGroups));
    std::move(begin(oldEntityList), end(oldEntityList), std::back_inserter(entityGroups));

    std::vector<std::uint32_t> gpuList;
    gpuList.reserve(DCGM_MAX_NUM_DEVICES);
    for (auto const &entity : entityGroups)
    {
        if (entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        gpuList.push_back(entity.entityId);
    }
    return gpuList;
}

void TopologicalSort(dcgmMigHierarchy_v2 &hierarchy)
{
    /*
     * In the dcgmMigHierarchy_v2 we store only GPU_I and GPU_CI. There are no GPUs.
     * To sort them properly, we need the following order:
     *      GPU 0
     *          GPU_I 0
     *              GPU_CI 0
     *              GPU_CI 1
     *          GPU_I 1
     *              GPU_CI 0
     *              GPU_CI 1
     *      GPU 1
     *          ...
     * Where indices after GPU_CI and GPU_I are NVML indices, not entityIds.
     */

    std::sort(&hierarchy.entityList[0],
              &hierarchy.entityList[hierarchy.count],
              [&](dcgmMigHierarchyInfo_v2 const &left, dcgmMigHierarchyInfo_v2 const &right) {
                  return std::tie(left.info.nvmlGpuIndex,
                                  left.info.nvmlInstanceId,
                                  left.entity.entityGroupId,
                                  left.info.nvmlComputeInstanceId)
                         < std::tie(right.info.nvmlGpuIndex,
                                    right.info.nvmlInstanceId,
                                    right.entity.entityGroupId,
                                    right.info.nvmlComputeInstanceId);
              });
}

} //namespace DcgmNs
