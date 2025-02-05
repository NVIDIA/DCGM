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

#include "DeviceMonitor.h"
#include "Query.h"
#include "dcgm_fields.h"
#include "dcgm_structs_internal.h"
#include "dcgm_test_apis.h"

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <EntityListHelpers.h>

#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_set>


#define MAX_PRINT_HEADER_COUNT 14  /*!< Defines after how many lines a header should be printed again. */
#define FLOAT_VAL_PREC         3   /*!< Precision of floating-point metric values */
#define MAX_KEEP_SAMPLES       2   /*!< Defines how many samples the hostengine should keep in the Cache Manager */
#define MAX_KEEP_AGE           0.0 /*!< Enforce quota via MAX_KEEP_SAMPLES */
#define DEFAULT_COUNT          0 /*!< Magic number. Default value for the number of iterations that means run forever */
#define PADDING                2U

static const char cNA[] = "N/A";

/*****************************************************************************/
/* This is used by the signal handler to let the device monitor know that
   we've received a signal from a ctrl-c... etc that we should stop */
static std::atomic<bool> gDeviceMonitorShouldStop = false;

fmt::string_view format_as(dcgm_field_entity_group_t entityGroupId)
{
    fmt::string_view val = "UNKNOWN";
    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
            val = "GPU";
            break;
        case DCGM_FE_VGPU:
            val = "vGPU";
            break;
        case DCGM_FE_SWITCH:
            val = "Switch";
            break;
        case DCGM_FE_GPU_I:
            val = "GPU-I";
            break;
        case DCGM_FE_GPU_CI:
            val = "GPU-CI";
            break;
        case DCGM_FE_LINK:
            val = "LINK";
            break;
        case DCGM_FE_CPU:
            val = "CPU";
            break;
        case DCGM_FE_CPU_CORE:
            val = "CPU Core";
            break;
        case DCGM_FE_CONNECTX:
            val = "CX";
            break;
        case DCGM_FE_NONE:
            /**
             * We used to display this as "GPU" if there is no non-GLOBAL
             * entity to combine global data with to display. We did this
             * since GLOBAL data was still reported per GPU. But, current
             * code should not be sending DCGM_FE_NONE, so we tag it as
             * "UNKNOWN".
             */
            break;
        case DCGM_FE_COUNT:
            break;
    }

    return val;
}


template <>
struct std::hash<dcgmi_entity_pair_t>
{
    size_t operator()(dcgmi_entity_pair_t const &val) const
    {
        return DcgmNs::Utils::Hash::CompoundHash(val.entityGroupId, val.entityId);
    }
};

/**
 * Constructor : Initialize the members of class Device Info
 */
DeviceMonitor::DeviceMonitor(std::string hostname,
                             std::string requestedEntityIds,
                             std::string grpIdStr,
                             std::string fldIds,
                             int fieldGroupId,
                             std::chrono::milliseconds updateDelay,
                             int numOfIterations,
                             bool listOptionMentioned)
    : m_requestedEntityIds(std::move(requestedEntityIds))
    , m_groupIdStr(std::move(grpIdStr))
    , m_fieldIdsStr(std::move(fldIds))
    , m_fieldGrpId_int(fieldGroupId)
    , m_delay(updateDelay)
    , m_count(numOfIterations)
    , m_myGroupId(0)
    , m_fieldGroupId(0)
    , m_list(listOptionMentioned)
    , m_widthArray()
{
    gDeviceMonitorShouldStop = false;

    m_hostName = std::move(hostname);
}

/**
 * @brief Populates fields details.
 *
 * Information like Long/Short names preserved for future usage.
 * @note This function assumes that the \c DcgmFieldInit() was already called.
 *
 * @sa \c DcgmFieldInit()
 */
void DeviceMonitor::PopulateFieldDetails()
{
    for (unsigned short fieldId = 0; fieldId < DCGM_FI_MAX_FIELDS; fieldId++)
    {
        auto const *dcgmFormat_s = DcgmFieldGetById(fieldId);
        if (dcgmFormat_s == nullptr)
        {
            continue;
        }
        m_fieldDetails.emplace_back(dcgmFormat_s->tag, dcgmFormat_s->valueFormat->shortName, fieldId);
    }
}

/**
 * @brief Formats and displays the details for all known fields.
 *
 * Used to output the result of \c "-l" command line option.
 */

/**
 * @brief A string view which length is strictly shorter that the \c maxLength.
 *
 * If the original string_view is longer than \c maxLength, then the origin will be truncated and an '...'
 * will be added at the end.
 */
struct FixedSizeString
{
    std::string_view origin; /*!< Original string view for outputting */
    std::size_t maxLength;   /*!< Desired max length of the result output */
};

/**
 * @brief The formatter that implement truncation logic and adds '...' in case of too long string value.
 */
template <>
struct fmt::formatter<FixedSizeString> : fmt::formatter<std::string_view>
{
    template <typename FormatContext>
    auto format(FixedSizeString const &value, FormatContext &ctx)
    {
        if (value.origin.length() <= value.maxLength)
        {
            return fmt::formatter<std::string_view>::format(value.origin, ctx);
        }

        return fmt::formatter<std::string_view>::format(
            fmt::format("{}...", value.origin.substr(0, value.maxLength - 3)), ctx);
    }
};

struct TableDimensions
{
    std::uint16_t longNameWidth;
    std::uint16_t shortNameWidth;
    std::uint16_t fieldIdWidth;
    [[nodiscard]] constexpr inline std::uint16_t GetWidth() const noexcept
    {
        return longNameWidth + shortNameWidth + fieldIdWidth;
    }
};

static inline void PrintFieldsDetailHeader(TableDimensions const &dims)
{
    /* Outputs:
     * __________________________________________
     * Long Name          Short Name   Field ID
     * __________________________________________
     */
    using namespace fmt::literals;
    fmt::print("{0:_<{width}}\n"
               "{long:<{longSize}}{short:<{shortSize}}{fieldId:<{fieldIdSize}}\n"
               "{0:_<{width}}\n",
               "",
               "width"_a       = dims.GetWidth(),
               "long"_a        = "Long Name",
               "short"_a       = "Short Name",
               "fieldId"_a     = "Field ID",
               "longSize"_a    = dims.longNameWidth,
               "shortSize"_a   = dims.shortNameWidth,
               "fieldIdSize"_a = dims.fieldIdWidth);
}

static inline void PrintFieldDetails(TableDimensions const &dims, FieldDetails const &value)
{
    using namespace fmt::literals;
    fmt::print("{long:<{longSize}s} {short:<{shortSize}s}{fieldId:>d}\n",
               "long"_a      = FixedSizeString { value.m_longName, dims.longNameWidth },
               "longSize"_a  = dims.longNameWidth,
               "short"_a     = value.m_shortName,
               "shortSize"_a = dims.shortNameWidth,
               "fieldId"_a   = value.m_fieldId);
}

void DeviceMonitor::DisplayFieldDetails()
{
    using namespace DcgmNs::Terminal;
    auto termSize        = GetTermDimensions().value_or(TermDimensions {});
    auto const width     = std::min<std::uint16_t>(120, termSize.cols);
    auto const tableDims = TableDimensions { std::max<std::uint16_t>(20, width - 30), 20, 10 };

    PrintFieldsDetailHeader(tableDims);
    for (auto const &fieldDetail : m_fieldDetails)
    {
        PrintFieldDetails(tableDims, fieldDetail);
    }
    fflush(stdout);
}

dcgmReturn_t DeviceMonitor::DoExecuteConnected()
{
    if (m_list)
    {
        PopulateFieldDetails();
        DisplayFieldDetails();
        return DCGM_ST_OK;
    }

    return CreateFieldGroupAndGpuGroup();
}

/**
 * @brief SIGINT (Ctrl-C) Signal handler
 * @ref <a href="https://en.cppreference.com/w/c/program/signal">signal</a>
 */
void killHandler(int sig)
{
    signal(sig, SIG_IGN);
    gDeviceMonitorShouldStop.store(true, std::memory_order_relaxed);
}

/**
 * In this function we simply print out the information in the callback of
 * the  watched field. This ia called as a callback to dcgmGetLatestValues_v2
 * function.
 *
 * @param[in] entityGroupId Group of the entity this field value set belongs to.
 * @param[in] entityId      Entity this field value set belongs to.
 * @param[in] values        The values under watch.
 * @param[in] numValues     Total number of values under watch.
 * @param[in] userdata      The generic data which needs to be passed to function. The void pointer at the end allows a
 *                          pointer to be passed to this function. Here we know that we are passing in a map, so we can
 *                          cast it as such. This pointer is also useful if we need a reference to something else inside
 *                          your function.
 * @return  0 on success; 1 on failure
 */
static int ListFieldValues(dcgm_field_entity_group_t entityGroupId,
                           dcgm_field_eid_t entityId,
                           dcgmFieldValue_v1 *values,
                           int numValues,
                           void *userdata)
{
    dcgmi_entity_pair_t entityKey;

    // Note: this is a pointer to a map and we cast it to map below.
    auto &entityStats = *(DeviceMonitor::EntityStats *)userdata;

    entityKey.entityGroupId = entityGroupId;
    entityKey.entityId      = entityId;

    for (int i = 0; i < numValues; i++)
    {
        // Including a switch statement here for handling different
        // types of values (except binary blobs).
        auto fieldId            = values[i].fieldId;
        dcgm_field_meta_p field = DcgmFieldGetById(fieldId);

        if (field == nullptr)
        {
            SHOW_AND_LOG_ERROR << fmt::format("Unknown Field ID: {}\n", values[i].fieldId);
            return 1;
        }

        switch (field->fieldType)
        {
            case DCGM_FT_BINARY:
            {
                entityStats[entityKey].emplace_back(EntityStatItem(fieldId, cNA));
                break;
            }
            case DCGM_FT_DOUBLE:
            {
                try
                {
                    auto const &val = values[i].value.dbl;
                    std::string value;

                    if (DCGM_FP64_IS_BLANK(val))
                    {
                        value = cNA;
                    }
                    else
                    {
                        value = fmt::format("{:.{}F}", val, FLOAT_VAL_PREC);
                    }
                    entityStats[entityKey].emplace_back(EntityStatItem(fieldId, std::move(value)));
                }
                catch (fmt::format_error const &e)
                {
                    SHOW_AND_LOG_ERROR << "Formatting error: " << e.what();
                    entityStats[entityKey].emplace_back(EntityStatItem(fieldId, cNA));
                }

                break;
            }
            case DCGM_FT_INT64:
            case DCGM_FT_TIMESTAMP:
            {
                try
                {
                    auto const &val = values[i].value.i64;
                    std::string value;
                    if (DCGM_INT64_IS_BLANK(val))
                    {
                        value = cNA;
                    }
                    else
                    {
                        value = fmt::to_string(val);
                    }
                    entityStats[entityKey].emplace_back(EntityStatItem(fieldId, std::move(value)));
                }
                catch (fmt::format_error const &e)
                {
                    SHOW_AND_LOG_ERROR << "Formatting error: " << e.what();
                    entityStats[entityKey].emplace_back(EntityStatItem(fieldId, cNA));
                }

                break;
            }
            case DCGM_FT_STRING:
            {
                entityStats[entityKey].emplace_back(EntityStatItem(fieldId, values[i].value.str));
                break;
            }
            default:
                SHOW_AND_LOG_ERROR << fmt::format("Error in field types: {}. Exiting\n", values[i].fieldType);
                // Error, return > 0 error code.
                return 1;
        }
    }
    // Program executed correctly. Return 0 to notify DCGM (callee) that it
    // was successful.
    return 0;
}

template <typename T>
void PrintMetricsRow(dcgmi_entity_pair_t const &entity,
                     std::vector<EntityStatItem> const &values,
                     T const &widths,
                     std::vector<unsigned short> &fieldIds)
{
    using namespace fmt::literals;
    fmt::print("{entityPair:{colWidth}}",
               "entityPair"_a = fmt::format("{} {}", entity.entityGroupId, entity.entityId),
               "colWidth"_a   = widths[0]);

    size_t valuesIdx = 0;

    for (size_t fieldIdx = 0; fieldIdx < fieldIds.size(); fieldIdx++)
    {
        if ((valuesIdx < values.size()) && (values[valuesIdx].fieldId == fieldIds[fieldIdx]))
        {
            auto const width = widths[fieldIdx + 1];

            if (width == 0)
            {
                /**
                 * There were no field metadata, so no room allocated in the
                 * output table for actual value.
                 */
                fmt::print("{:{}}", "", PADDING);
            }
            else
            {
                fmt::print("{:{}s}", FixedSizeString { values[valuesIdx].value, width }, width + PADDING);
            }

            valuesIdx++;
        }
        else
        {
            auto const width = widths[fieldIdx + 1];

            fmt::print("{:{}s}", FixedSizeString { "", width }, width + PADDING);
        }
    }

    fmt::print("\n");
}

/**
 * MergeGpuAndMigEntities merges GPUs in a group with MIG GIs in the group in
 * an order with GPUs primary and GPU GIs secondary.
 *
 * @param dcgmGroupInfo:  a reference to a DCGM entity group.
 * @param migHierarchy:   a const reference to the MIG hierarchy.
 *
 * @return dcgmReturn_t:  DCGM return code.
 */
dcgmReturn_t DeviceMonitor::MergeGpuAndMigEntities(dcgmGroupInfo_t &dcgmGroupInfo,
                                                   dcgmMigHierarchy_v2 const &migHierarchy)
{
    /**
     * Map MIG entities to their MIG hierarchy entries.
     */

    std::map<std::tuple<decltype(migHierarchy.entityList[0].entity.entityGroupId),
                        decltype(migHierarchy.entityList[0].entity.entityId)>,
             std::decay_t<decltype(migHierarchy.entityList[0])>>
        migMap;

    for (unsigned int migEntityIdx = 0; migEntityIdx < migHierarchy.count; ++migEntityIdx)
    {
        dcgmMigHierarchyInfo_v2 migEntity = migHierarchy.entityList[migEntityIdx];

        migMap[std::tie(migEntity.entity.entityGroupId, migEntity.entity.entityId)] = migEntity;
    }

    /**
     * The comparison depends on whether the entity is a MIG entity or not
     * (like an NvSwitch, or CPU).
     */

    std::sort(dcgmGroupInfo.entityList, dcgmGroupInfo.entityList + dcgmGroupInfo.count, [&](auto &left, auto &right) {
        auto const &migEntity = migHierarchy.entityList[0];

        using CompareType = std::tuple<decltype(migEntity.entity.entityGroupId),
                                       decltype(migEntity.info.nvmlGpuIndex),
                                       decltype(migEntity.info.nvmlInstanceId),
                                       decltype(migEntity.entity.entityGroupId),
                                       decltype(migEntity.info.nvmlComputeInstanceId)>;

        CompareType leftItem  = std::make_tuple(DCGM_FE_NONE, 0U, 0U, DCGM_FE_NONE, 0U),
                    rightItem = std::make_tuple(DCGM_FE_NONE, 0U, 0U, DCGM_FE_NONE, 0U);

        if (migMap.contains(std::tie(left.entityGroupId, left.entityId)))
        {
            leftItem = std::make_tuple(DCGM_FE_GPU,
                                       migMap[std::tie(left.entityGroupId, left.entityId)].info.nvmlGpuIndex,
                                       migMap[std::tie(left.entityGroupId, left.entityId)].info.nvmlInstanceId,
                                       migMap[std::tie(left.entityGroupId, left.entityId)].entity.entityGroupId,
                                       migMap[std::tie(left.entityGroupId, left.entityId)].info.nvmlComputeInstanceId);
        }
        else
        {
            leftItem = std::make_tuple(left.entityGroupId, left.entityId, 0U, DCGM_FE_NONE, 0U);
        }

        if (migMap.contains(std::tie(right.entityGroupId, right.entityId)))
        {
            rightItem
                = std::make_tuple(DCGM_FE_GPU,
                                  migMap[std::tie(right.entityGroupId, right.entityId)].info.nvmlGpuIndex,
                                  migMap[std::tie(right.entityGroupId, right.entityId)].info.nvmlInstanceId,
                                  migMap[std::tie(right.entityGroupId, right.entityId)].entity.entityGroupId,
                                  migMap[std::tie(right.entityGroupId, right.entityId)].info.nvmlComputeInstanceId);
        }
        else
        {
            rightItem = std::make_tuple(right.entityGroupId, right.entityId, 0U, DCGM_FE_NONE, 0U);
        }

        return leftItem < rightItem;
    });

    return DCGM_ST_OK;
}

static bool HasGpuEntity(dcgmGroupInfo_t &dcgmGroupInfo)
{
    for (unsigned idx = 0; idx < dcgmGroupInfo.count; ++idx)
    {
        if (dcgmGroupInfo.entityList[idx].entityGroupId == DCGM_FE_GPU
            || dcgmGroupInfo.entityList[idx].entityGroupId == DCGM_FE_GPU_I
            || dcgmGroupInfo.entityList[idx].entityGroupId == DCGM_FE_GPU_CI)
        {
            return true;
        }
    }
    return false;
}

/**
 * GetSortedEntities: return a list of MIG and non-MIG entities to watch.
 *
 * @param  sortedEntities: a references to a dcgmGroupInfo_t.
 *
 * @return dcgmReturn_t: DCGM return code.
 */
dcgmReturn_t DeviceMonitor::GetSortedEntities(dcgmGroupInfo_t &dcgmGroupInfo)
{
    /* Get Group ID info to indentify GPUs monitored that might not be
     * MIG-enabled.
     */

    dcgmGroupInfo.version = dcgmGroupInfo_version;

    if (auto const ret = dcgmGroupGetInfo(m_dcgmHandle, m_myGroupId, &dcgmGroupInfo); ret != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << fmt::format(
            "Unable to retrieve information about group. Error: {}: {}",
            ret,
            (ret == DCGM_ST_NOT_CONFIGURED ? "The Group is not found" : errorString(ret)));

        return ret;
    }

    auto hierarchyV2  = dcgmMigHierarchy_v2 {};
    hierarchyV2.count = 0;

    if (HasGpuEntity(dcgmGroupInfo))
    {
        hierarchyV2.version = dcgmMigHierarchy_version2;
        auto ret            = dcgmGetGpuInstanceHierarchy(m_dcgmHandle, &hierarchyV2);
        if (ret != DCGM_ST_OK)
        {
            SHOW_AND_LOG_ERROR << fmt::format(
                "Unable to get GPU topology information. The entities in the output may be unsorted. Error: {}: {}",
                ret,
                errorString(ret));
        }
    }

    return MergeGpuAndMigEntities(dcgmGroupInfo, hierarchyV2);
}

/**
 * lockWatchAndUpdate : The function sets the watches on the field group
 *                      for the mentioned Gpu group.
 * @return dcgmReturn_t : result - success or error
 */

dcgmReturn_t DeviceMonitor::LockWatchAndUpdate()
{
    int running          = 0;
    int decrement        = 0;
    int printHeaderCount = 0;

    std::unique_ptr<dcgmGroupInfo_t> dcgmGroupInfo = std::make_unique<dcgmGroupInfo_t>();

    /* Install a signal handler to catch ctrl-c */
    signal(SIGINT, &killHandler);

    auto ret = GetSortedEntities(*(dcgmGroupInfo));

    if (ret != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << fmt::format("Unable to get DCGM sorted entities. Error: {}: {}", ret, errorString(ret));

        return ret;
    }

    // set the watch on field group id for Group of gpus.
    dcgmReturn_t result = dcgmWatchFields(m_dcgmHandle,
                                          m_myGroupId,
                                          m_fieldGroupId,
                                          std::chrono::duration_cast<std::chrono::microseconds>(m_delay).count(),
                                          MAX_KEEP_AGE,
                                          MAX_KEEP_SAMPLES);

    // Check result to see if DCGM operation was successful.
    if (result != DCGM_ST_OK)
    {
        const char *e = errorString(result);

        switch (result)
        {
            case DCGM_ST_REQUIRES_ROOT:
                e = "Unable to watch one or more of the requested fields because doing so requires the host engine to be running as root.";
                break;

            case DCGM_ST_INSUFFICIENT_DRIVER_VERSION:
                e = "Unable to watch one or more of the requested fields because doing so requires a newer driver version than what is currently installed.";
                break;

            default:
                break; /* Intentional fall-through. e is initialized before this switch statement */
        }

        SHOW_AND_LOG_ERROR << fmt::format("Error setting watches. Result: {}: {}\n", result, e);
        return result;
    }

    dcgmUpdateAllFields(m_dcgmHandle, 1);

    // Set the loop to running forever and set decrement check to false as there will be
    // no decrement in m_count when no m_count value is mentioned. Loop runs forever in default
    // condition.
    if (m_count == DEFAULT_COUNT)
    {
        running   = 1;
        decrement = false;
    }
    // The running variable is initialized with mentioned value in command and decrement flag is set to
    // true as the running variable is decremented with each iteration.
    else
    {
        running   = m_count;
        decrement = true;
    }

    while (running && !gDeviceMonitorShouldStop.load(std::memory_order_relaxed))
    {
        EntityStats entityStats;

        // entityStats should be preallocated before calling dcgmGetLatestValues_v2.
        // Otherwise, reallocation may happen in the libdcgm context using wrong memory allocator.
        entityStats.reserve(dcgmGroupInfo->count ? dcgmGroupInfo->count : DCGM_MAX_NUM_DEVICES);

        result = dcgmGetLatestValues_v2(m_dcgmHandle, m_myGroupId, m_fieldGroupId, &ListFieldValues, &entityStats);

        // Check the result to see if our DCGM operation was successful.
        if (result != DCGM_ST_OK)
        {
            SHOW_AND_LOG_ERROR << fmt::format(
                "Error getting values information. Return {}: {}", result, errorString(result));
            return result;
        }

        if (entityStats.empty())
        {
            SHOW_AND_LOG_ERROR << "No data returned from the hostengine.";
            return DCGM_ST_NO_DATA; /* Propagate this to any callers */
        }

        for (unsigned int idx = 0; idx < dcgmGroupInfo->count; idx++)
        {
            auto const &entity = dcgmGroupInfo->entityList[idx];

            dcgmi_entity_pair_t entityPair { .entityGroupId = entity.entityGroupId, .entityId = entity.entityId };

            auto it = entityStats.find(entityPair);
            if (it == entityStats.end())
            {
                /**
                 * If we specify entities but not request fields from those
                 * entities, nothing will be returned.
                 *
                 * For example:
                 *
                 * dcgmi dmon -i 0 -e 1,2,3,4,5
                 *
                 * Global data for GPU 0 will be returned but no GPU-specific
                 * data to render the global data with. In this case, we render
                 * the global data alone.
                 */
                continue;
            }

            PrintMetricsRow(it->first, it->second, m_widthArray, m_fieldIds);
        }

        fflush(stdout);

        // We decrement only when count value is not default and the loop is not supposed to run forever.
        if (decrement)
        {
            running--;
            if (running <= 0)
            {
                break;
            }
        }

        std::this_thread::sleep_for(m_delay);
        if (printHeaderCount == MAX_PRINT_HEADER_COUNT)
        {
            PrintHeaderForOutput();
            printHeaderCount = 0;
        }
        printHeaderCount++;
    }

    if (gDeviceMonitorShouldStop)
    {
        fmt::print("\n\ndmon was stopped due to receiving a signal\n");
        fflush(stdout);
    }

    return result;
}

/**
 * printHeaderForOutput - Directs the header to Console output stream.
 */
void DeviceMonitor::PrintHeaderForOutput() const
{
    fmt::print("{}\n"
               "{}\n",
               m_header,
               m_headerUnit);
    fflush(stdout);
}

/**
 * The function initializes the header for tabular output with alignment by
 * setting width and using the field Ids to create header.
 *
 * @param fieldIds      The vector containing field ids.
 * @param numFields     Number of fields int the vector.
 */
void DeviceMonitor::SetHeaderForOutput(unsigned short fieldIds[], unsigned int const numFields)
{
    fmt::memory_buffer headBuffer;
    fmt::memory_buffer unitBuffer;
    headBuffer.reserve(512);
    unitBuffer.reserve(512);
    constexpr int firstFieldWidth = 11;

    /*
     * #Entity sName   sName    sName
     * ID
     */
    fmt::format_to(std::back_inserter(headBuffer), "{:<{}}", "#Entity", firstFieldWidth);
    fmt::format_to(std::back_inserter(unitBuffer), "{:<{}}", "ID", firstFieldWidth);

    m_widthArray[0] = firstFieldWidth;

    for (std::uint32_t i = 0; i < numFields; ++i)
    {
        auto const *fieldMeta   = DcgmFieldGetById(fieldIds[i]);
        auto const *valueFormat = fieldMeta ? fieldMeta->valueFormat : nullptr;
        auto const width        = (valueFormat ? valueFormat->width : 0U) + 6;
        m_widthArray[i + 1]     = width;
        fmt::format_to(std::back_inserter(headBuffer),
                       "{:<{}}",
                       FixedSizeString { valueFormat ? valueFormat->shortName : "", width },
                       width + PADDING);
        fmt::format_to(std::back_inserter(unitBuffer),
                       "{:<{}}",
                       FixedSizeString { valueFormat ? valueFormat->unit : "", width },
                       width + PADDING);
    }

    m_header     = fmt::to_string(headBuffer);
    m_headerUnit = fmt::to_string(unitBuffer);
}

std::vector<dcgmGroupEntityPair_t> helperParseCpuEntities(std::string const &inputCpuIds)
{
    std::vector<dcgmGroupEntityPair_t> cpuIds;
    auto cpuIdStrings = DcgmNs::Split(inputCpuIds, ',');
    for (auto cpu : cpuIdStrings)
    {
        unsigned int cpuId = strtoul(std::string(cpu).c_str(), nullptr, 10);
        cpuIds.emplace_back(dcgmGroupEntityPair_t { DCGM_FE_CPU, cpuId });
    }
    return cpuIds;
}

dcgmReturn_t DeviceMonitor::CreateEntityGroupFromEntityList()
{
    std::vector<dcgmGroupEntityPair_t> entityList;

    if (m_requestedEntityIds != "-1")
    {
        auto err = DcgmNs::EntityListWithMigAndUuidParser(m_dcgmHandle, m_requestedEntityIds, entityList);
        if (!err.empty())
        {
            SHOW_AND_LOG_ERROR << err;
            return DCGM_ST_BADPARAM;
        }
    }

    /* Create a group based on this list of entities */
    return dcgmi_create_entity_group(m_dcgmHandle, DCGM_GROUP_EMPTY, &m_myGroupId, entityList);
}

dcgmReturn_t DeviceMonitor::ValidateOrCreateEntityGroup()
{
    dcgmGroupType_t groupType = DCGM_GROUP_EMPTY;
    /**
     * Check if m_requestedEntityIds is set or not. If set, we create
     * a group including the devices mentioned with flag.
     */
    if (m_requestedEntityIds != "-1")
    {
        return CreateEntityGroupFromEntityList();
    }

    /* If no group ID or entity list was specified, assume all GPUs to act like nvidia-smi dmon */
    if (m_groupIdStr == "-1")
    {
        std::vector<dcgmGroupEntityPair_t> entityList; /* Empty List */
        return dcgmi_create_entity_group(m_dcgmHandle, DCGM_GROUP_DEFAULT, &m_myGroupId, entityList);
    }

    bool const groupIdIsSpecial = dcgmi_entity_group_id_is_special(m_groupIdStr, &groupType, &m_myGroupId);
    if (groupIdIsSpecial)
    {
        /* m_myGroupId was already set to the correct group ID of the special group */
        return DCGM_ST_OK;
    }

    int groupIdAsInt = 0;
    try
    {
        groupIdAsInt = std::stoi(m_groupIdStr);
    }
    catch (std::exception const & /*ex*/)
    {
        SHOW_AND_LOG_ERROR << fmt::format("Expected a numerical groupId. Instead, got '{}'", m_groupIdStr);
        return DCGM_ST_BADPARAM;
    }

    m_myGroupId = static_cast<dcgmGpuGrp_t>(groupIdAsInt);

    /* Try to get a handle to the group the user specified */
    std::unique_ptr<dcgmGroupInfo_t> dcgmGroupInfo = std::make_unique<dcgmGroupInfo_t>();
    dcgmGroupInfo->version                         = dcgmGroupInfo_version;

    if (auto const ret = dcgmGroupGetInfo(m_dcgmHandle, m_myGroupId, dcgmGroupInfo.get()); ret != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << fmt::format(
            "Enable to retrieve information about group. Error: {}: {}",
            ret,
            (ret == DCGM_ST_NOT_CONFIGURED ? "The Group is not found" : errorString(ret)));

        return ret;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DeviceMonitor::ValidateOrCreateFieldGroup()
{
    dcgmFieldGroupInfo_t fieldGroupInfo;

    if (m_fieldIdsStr != "-1")
    {
        if (auto const ret = dcgmi_parse_field_id_list_string(m_fieldIdsStr, m_fieldIds, true); ret != DCGM_ST_OK)
        {
            return ret;
        }

        if (auto const ret = dcgmi_create_field_group(m_dcgmHandle, &m_fieldGroupId, m_fieldIds); ret != DCGM_ST_OK)
        {
            return ret;
        }

        SetHeaderForOutput(&m_fieldIds[0], m_fieldIds.size());
        return DCGM_ST_OK;
    }

    if (m_fieldGrpId_int == -1)
    {
        SHOW_AND_LOG_ERROR << fmt::format("Either field group ID or a list of field IDs must be provided. "
                                          "See dcgmi dmon --help for usage.");
    }

    memset(&fieldGroupInfo, 0, sizeof(fieldGroupInfo));
    fieldGroupInfo.version = dcgmFieldGroupInfo_version;

    fieldGroupInfo.fieldGroupId = static_cast<dcgmFieldGrp_t>(m_fieldGrpId_int);

    if (auto const ret = dcgmFieldGroupGetInfo(m_dcgmHandle, &fieldGroupInfo); ret != DCGM_ST_OK)
    {
        SHOW_AND_LOG_ERROR << fmt::format(
            "Unable to retrieve information about field group {}. Error: {}: {}",
            m_fieldGrpId_int,
            ret,
            (ret == DCGM_ST_NOT_CONFIGURED ? "The Field Group is not found" : errorString(ret)));

        return DCGM_ST_BADPARAM;
    }

    m_fieldIds.clear();
    m_fieldIds.reserve(fieldGroupInfo.numFieldIds);
    for (unsigned int i = 0; i < fieldGroupInfo.numFieldIds; i++)
    {
        m_fieldIds.push_back(fieldGroupInfo.fieldIds[i]);
    }

    SetHeaderForOutput(fieldGroupInfo.fieldIds, fieldGroupInfo.numFieldIds);
    m_fieldGroupId = static_cast<dcgmFieldGrp_t>(m_fieldGrpId_int);
    return DCGM_ST_OK;
}

dcgmReturn_t DeviceMonitor::CreateFieldGroupAndGpuGroup()
{
    dcgmReturn_t dcgmReturn = ValidateOrCreateEntityGroup();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    /**
     * The Gpu group has been created and Group ID has been set now. We create a
     * field group either along with fieldIDs mentioned or the fields mentioned.
     */
    dcgmReturn = ValidateOrCreateFieldGroup();
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    PrintHeaderForOutput();

    dcgmReturn = LockWatchAndUpdate();
    return dcgmReturn;
}
