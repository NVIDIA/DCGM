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
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "DcgmLogging.h"
#include "DcgmRecorder.h"
#include "DcgmUtilities.h"
#include "NvvsCommon.h"
#include "PluginStrings.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "timelib.h"

const long long defaultFrequency = 5000000; // update each field every 5 seconds (a million microseconds)

errorType_t standardErrorFields[] = { { DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, TS_STR_SBE_ERROR_THRESHOLD },
                                      { DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, nullptr },
                                      { DCGM_FI_DEV_XID_ERRORS, nullptr },
                                      { DCGM_FI_DEV_PCIE_REPLAY_COUNTER, PCIE_STR_MAX_PCIE_REPLAYS },
                                      { DCGM_FI_DEV_ROW_REMAP_PENDING, nullptr },
                                      { DCGM_FI_DEV_ROW_REMAP_FAILURE, nullptr },
                                      { DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS, nullptr },
                                      { 0, nullptr } };

unsigned short standardInfoFields[] = { DCGM_FI_DEV_GPU_TEMP,          DCGM_FI_DEV_GPU_UTIL,
                                        DCGM_FI_DEV_POWER_USAGE,       DCGM_FI_DEV_SM_CLOCK,
                                        DCGM_FI_DEV_MEM_CLOCK,         DCGM_FI_DEV_POWER_VIOLATION,
                                        DCGM_FI_DEV_THERMAL_VIOLATION, DCGM_FI_DEV_CLOCKS_EVENT_REASONS,
                                        DCGM_FI_DEV_MEMORY_TEMP,       0 };

dcgmReturn_t DcgmRecorderBase::GetCurrentFieldValue(unsigned int, unsigned short, dcgmFieldValue_v2 &, unsigned int)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmRecorderBase::GetFieldSummary(dcgmFieldSummaryRequest_t &)
{
    return DCGM_ST_NOT_SUPPORTED;
}

std::string DcgmRecorderBase::GetGpuUtilizationNote(unsigned int, timelib64_t)
{
    return "";
}

DcgmRecorder::DcgmRecorder()
    : m_fieldIds()
    , m_gpuIds()
    , m_fieldGroupId(0)
    , m_dcgmHandle()
    , m_dcgmGroup()
    , m_dcgmSystem()
    , m_valuesHolder()
    , m_nextValuesSinceTs(0)
    , m_watchFrequency(defaultFrequency)

{}

DcgmRecorder::DcgmRecorder(dcgmHandle_t handle)
    : m_fieldIds()
    , m_gpuIds()
    , m_fieldGroupId(0)
    , m_dcgmHandle(handle)
    , m_dcgmGroup()
    , m_dcgmSystem()
    , m_valuesHolder()
    , m_nextValuesSinceTs(0)
    , m_watchFrequency(defaultFrequency)

{
    m_dcgmSystem.Init(handle);
}

DcgmRecorder::DcgmRecorder(DcgmRecorder &&other) noexcept
    : m_fieldIds(other.m_fieldIds)
    , m_gpuIds(other.m_gpuIds)
    , m_fieldGroupId(other.m_fieldGroupId)
    , m_dcgmHandle(std::move(other.m_dcgmHandle))
    , m_dcgmGroup(std::move(other.m_dcgmGroup))
    , m_dcgmSystem(other.m_dcgmSystem)
    , m_valuesHolder(other.m_valuesHolder)
    , m_nextValuesSinceTs(other.m_nextValuesSinceTs)
    , m_watchFrequency(other.m_watchFrequency)
    , m_ignoreErrorCodes(other.m_ignoreErrorCodes)
{}

DcgmRecorder::~DcgmRecorder()
{
    Shutdown();
}

dcgmReturn_t DcgmRecorder::CreateGroup(const std::vector<unsigned int> &gpuIds,
                                       bool /*allGpus*/,
                                       const std::string &groupName)
{
    if (m_dcgmHandle.GetHandle() == 0)
    {
        return DCGM_ST_CONNECTION_NOT_VALID;
    }

    return m_dcgmGroup.Init(m_dcgmHandle.GetHandle(), groupName, gpuIds);
}

dcgmReturn_t DcgmRecorder::AddWatches(const std::vector<unsigned short> &fieldIds,
                                      const std::vector<unsigned int> &gpuIds,
                                      bool allGpus,
                                      const std::string &fieldGroupName,
                                      const std::string &groupName,
                                      double testDuration)

{
    dcgmReturn_t ret;
    m_fieldIds = fieldIds;
    m_gpuIds   = gpuIds;

    if (fieldIds.size() == 0 || fieldIds.size() > DCGM_FI_MAX_FIELDS)
    {
        log_error("Invalid number of field ids {} is not in range 0-{}", fieldIds.size(), DCGM_FI_MAX_FIELDS);
        return DCGM_ST_BADPARAM;
    }

    if (gpuIds.size() == 0)
    {
        log_error("Gpu Ids must contain at least 1 gpu id");
        return DCGM_ST_BADPARAM;
    }

    ret = CreateGroup(gpuIds, allGpus, groupName);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    ret = m_dcgmGroup.FieldGroupCreate(fieldIds, fieldGroupName);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    return m_dcgmGroup.WatchFields(m_watchFrequency, testDuration + 30);
}

void DcgmRecorder::GetErrorString(dcgmReturn_t ret, std::string &err)
{
    const char *tmp = errorString(ret);

    err = (tmp == nullptr) ? fmt::format("Unknown error from DCGM: {}", ret) : tmp;
}

dcgmReturn_t DcgmRecorder::Init(const std::string &hostname)
{
    dcgmReturn_t ret;

    ret = m_dcgmHandle.ConnectToDcgm(hostname);

    if (ret == DCGM_ST_OK)
    {
        m_dcgmSystem.Init(m_dcgmHandle.GetHandle());
    }

    return ret;
}

void DcgmRecorder::Init(dcgmHandle_t handle)
{
    m_dcgmSystem.Init(handle);
    m_dcgmHandle = handle;
}

dcgmReturn_t DcgmRecorder::Shutdown()
{
    if (m_dcgmHandle.GetHandle() == 0)
    {
        m_fieldGroupId = 0;
        return DCGM_ST_OK;
    }

    if (m_fieldGroupId != 0)
    {
        // coverity[check_return]
        dcgmFieldGroupDestroy(m_dcgmHandle.GetHandle(), m_fieldGroupId);
        m_fieldGroupId = 0;
    }

    m_dcgmGroup.Cleanup();

    return DCGM_ST_OK;
}

void DcgmRecorder::GetTagFromFieldId(unsigned short fieldId, std::string &tag)
{
    dcgm_field_meta_p fm = DcgmFieldGetById(fieldId);

    if (fm == 0)
    {
        std::stringstream tmp;
        tmp << fieldId;
        tag = tmp.str();
    }
    else
        tag = fm->tag;
}

void DcgmRecorder::ClearCustomData()
{
    m_customStatHolder.ClearCustomData();
}

void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, double value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}

void DcgmRecorder::SetGroupedStat(const std::string &groupName, const std::string &name, long long value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}


std::vector<dcgmTimeseriesInfo_t> DcgmRecorder::GetGroupedStat(const std::string &groupName, const std::string &name)
{
    return m_customStatHolder.GetGroupedStat(groupName, name);
}

void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

void DcgmRecorder::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

int storeValues(dcgm_field_entity_group_t entityGroupId,
                dcgm_field_eid_t entityId,
                dcgmFieldValue_v1 *values,
                int numValues,
                void *userData)
{
    if (userData == 0)
        return static_cast<int>(DCGM_ST_BADPARAM);

    DcgmValuesSinceHolder *dvsh = static_cast<DcgmValuesSinceHolder *>(userData);
    for (int i = 0; i < numValues; i++)
    {
        // Skip bad values
        if (values[i].status != DCGM_ST_OK)
            continue;

        dvsh->AddValue(entityGroupId, entityId, values[i].fieldId, values[i]);
    }

    return 0;
}

dcgmReturn_t DcgmRecorder::GetFieldValuesSince(dcgm_field_entity_group_t /*entityGroupId*/,
                                               dcgm_field_eid_t /*entityId*/,
                                               unsigned short /*fieldId*/,
                                               long long ts,
                                               bool force)
{
    // Use the timestamp to prevent asking for something that we've already grabbed, unless force is true
    if (force == true)
    {
        m_valuesHolder.ClearCache();
    }
    else if (ts < m_nextValuesSinceTs)
    {
        ts = m_nextValuesSinceTs;
    }

    dcgmReturn_t ret = m_dcgmGroup.GetValuesSince(ts, storeValues, &m_valuesHolder, &m_nextValuesSinceTs);

    return ret;
}

std::string DcgmRecorder::GetWatchedFieldsAsJson(Json::Value &jv, long long ts)
{
    std::string errStr;
    dcgmReturn_t ret = DCGM_ST_OK;

    // Make sure we have all of our values queried
    for (size_t i = 0; i < m_gpuIds.size(); i++)
    {
        for (size_t j = 0; j < m_fieldIds.size(); j++)
        {
            ret = GetFieldValuesSince(DCGM_FE_GPU, m_gpuIds[i], m_fieldIds[j], ts, true);

            if (ret != DCGM_ST_OK)
            {
                GetErrorString(ret, errStr);
                return errStr;
            }
        }
    }

    m_valuesHolder.AddToJson(jv);
    m_customStatHolder.AddCustomData(jv);

    return errStr;
}

/*
 * GPUs Json is in the format:
 *
 *  jv[GPUS] is an array of gpu ids
 *  jv[GPUS][gpuId] is a map of atributes
 *  jv[GPUS][gpuId][attrname] is an array of objects with timestamp and value
 */
std::string DcgmRecorder::GetWatchedFieldsAsString(std::string &output, long long ts)
{
    Json::Value jv;
    std::string errStr = GetWatchedFieldsAsJson(jv, ts);
    if (errStr.size() > 0)
        return errStr;

    std::stringstream buf;
    buf << "GPU Collections\n";

    Json::Value &gpuArray = jv[GPUS];
    for (unsigned int gpuIndex = 0; gpuIndex < gpuArray.size(); gpuIndex++)
    {
        buf << "\tNvml Idx " << gpuIndex << "\n";

        for (Json::Value::iterator it = gpuArray[gpuIndex].begin(); it != gpuArray[gpuIndex].end(); ++it)
        {
            std::string attrName   = it.key().asString();
            Json::Value &attrArray = gpuArray[gpuIndex][attrName];

            for (unsigned int attrIndex = 0; attrIndex < attrArray.size(); attrIndex++)
            {
                buf << "\t\t" << attrName << ": timestamp " << attrArray[attrIndex]["timestamp"];
                buf << ", val " << attrArray[attrIndex]["value"] << "\n";
            }
        }
    }

    output = buf.str();

    return errStr;
}

int DcgmRecorder::WriteToFile(const std::string &filename, int logFileType, long long testStart)
{
    m_customStatHolder.InitGpus(m_gpuIds);

    std::ofstream f;
    f.open(filename.c_str());

    if (f.fail())
    {
        log_error("Unable to open file {}: '{}'", filename, strerror(errno));
        return -1;
    }


    switch (logFileType)
    {
        case NVVS_LOGFILE_TYPE_TEXT:
        {
            std::string output;
            std::string error = GetWatchedFieldsAsString(output, testStart);
            if (error.size() == 0)
                f << output;
            else
                f << error;

            break;
        }

        case NVVS_LOGFILE_TYPE_JSON:
        default:
        {
            Json::Value jv;
            std::string error = GetWatchedFieldsAsJson(jv, testStart);

            if (error.size() == 0)
                f << jv.toStyledString();
            else
                f << error;
        }

        break;
    }

    f.close();
    return 0;
}

dcgmReturn_t DcgmRecorder::GetFieldSummary(dcgmFieldSummaryRequest_t &request)
{
    dcgmReturn_t ret;
    std::string errStr;

    request.version = dcgmFieldSummaryRequest_version1;
    ret             = dcgmGetFieldSummary(m_dcgmHandle.GetHandle(), &request);

    if (ret == DCGM_ST_NO_DATA)
    {
        // Lack of data is not an error
        ret = DCGM_ST_OK;
    }

    return ret;
}

int DcgmRecorder::GetValueIndex(unsigned short fieldId)
{
    // Default to index 0 for DCGM_SUMMARY_MAX
    int index = 0;

    switch (fieldId)
    {
        case DCGM_FI_DEV_THERMAL_VIOLATION:
            return 1; // This one should return the sum
            break;


        case DCGM_FI_DEV_ECC_SBE_VOL_TOTAL:
        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL:
        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:

            // All of these should use DCGM_SUMMARY_DIFF
            index = 2;
            break;
    }

    return index;
}

void DcgmRecorder::FormatFieldViolationError(DcgmError &d,
                                             unsigned short fieldId,
                                             unsigned int gpuId,
                                             timelib64_t startTime,
                                             int64_t intValue,
                                             double dblValue,
                                             const std::string &fieldName)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_ECC_SBE_VOL_TOTAL:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SBE_VIOLATION, d, intValue, fieldName.c_str(), gpuId);

            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DBE_VIOLATION, d, intValue, fieldName.c_str(), gpuId);

            break;

        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_REPLAY_VIOLATION, d, intValue, fieldName.c_str(), gpuId);

            break;

        case DCGM_FI_DEV_THERMAL_VIOLATION:
        {
            dcgmReturn_t ret = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, startTime, false);
            dcgmFieldValue_v1 dfv;
            memset(&dfv, 0, sizeof(dfv));
            double seconds = intValue / 1000000000.0; // The violation is reported in nanoseconds.

            if (ret == DCGM_ST_OK)
            {
                m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCKS_EVENT_REASONS, dfv, 0);
            }

            if (dfv.ts != 0) // the field value timestamp will be 0 if we couldn't find one
            {
                double timeDiff = (dfv.ts - startTime) / 1000000.0;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS_TS, d, seconds, timeDiff, gpuId);
            }
            else
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS, d, seconds, gpuId);
            }
            break;
        }

        case DCGM_FI_DEV_XID_ERRORS:
        {
            if (intValue == 95)
            {
                // XID 95 has its own error message
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNCONTAINED_ERROR, d);
            }
            else
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_XID_ERROR, d, intValue, gpuId);
            }
            break;
        }

        case DCGM_FI_DEV_ROW_REMAP_FAILURE:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ROW_REMAP_FAILURE, d, gpuId);

            break;

        case DCGM_FI_DEV_ROW_REMAP_PENDING:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_ROW_REMAP, d, gpuId);

            break;

        case DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_SXID_ERROR, d, intValue);

            break;

        default:
        {
            if (DCGM_INT64_IS_BLANK(intValue))
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION_DBL, d, dblValue, fieldName.c_str(), gpuId);
            }
            else
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION, d, intValue, fieldName.c_str(), gpuId);
            }
            break;
        }
    }
}

void DcgmRecorder::AddFieldViolationError(unsigned short fieldId,
                                          unsigned int gpuId,
                                          timelib64_t startTime,
                                          int64_t intValue,
                                          double dblValue,
                                          const std::string &fieldName,
                                          std::vector<DcgmError> &fatalErrorList,
                                          std::vector<DcgmError> &ignoredErrorList,
                                          int &st)
{
    DcgmError d { gpuId };

    FormatFieldViolationError(d, fieldId, gpuId, startTime, intValue, dblValue, fieldName);

    AddOrClearError(d, fatalErrorList, ignoredErrorList, st);
}

void DcgmRecorder::AddFieldThresholdViolationError(unsigned short fieldId,
                                                   unsigned int gpuId,
                                                   int64_t intValue,
                                                   int64_t thresholdValue,
                                                   const std::string &fieldName,
                                                   std::vector<DcgmError> &fatalErrorList,
                                                   std::vector<DcgmError> &ignoredErrorList,
                                                   int &st)
{
    DcgmError d { gpuId };
    switch (fieldId)
    {
        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:

            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_PCIE_REPLAY_THRESHOLD_VIOLATION, d, intValue, fieldName.c_str(), gpuId, thresholdValue);

            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:

            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_DBE_THRESHOLD_VIOLATION, d, intValue, fieldName.c_str(), gpuId, thresholdValue);

            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_TOTAL:

            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_SBE_THRESHOLD_VIOLATION, d, intValue, fieldName.c_str(), gpuId, thresholdValue);

            break;

        default:

            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD, d, intValue, fieldName.c_str(), gpuId, thresholdValue);

            break;
    }
    AddOrClearError(d, fatalErrorList, ignoredErrorList, st);
}

int DcgmRecorder::CheckXIDs(unsigned int gpuId,
                            timelib64_t startTime,
                            std::vector<DcgmError> &fatalErrorList,
                            std::vector<DcgmError> &ignoredErrorList)
{
    int count = DCGM_MAX_XID_INFO;
    dcgmFieldValue_v1 values[DCGM_MAX_XID_INFO];

    int st = dcgmGetMultipleValuesForField(
        m_dcgmHandle.GetHandle(), gpuId, DCGM_FI_DEV_XID_ERRORS, &count, startTime, 0, DCGM_ORDER_ASCENDING, values);

    if (st != DCGM_ST_OK)
    {
        log_error("Skipping XID check for gpu {} due to error {}.", gpuId, st);
        return st;
    }

    std::unordered_set<unsigned int> errors;

    /* gather unique XIDs */
    for (int i = 0; i < count; i++)
    {
        if (!DCGM_INT64_IS_BLANK(values[i].value.i64))
            errors.insert(values[i].value.i64);
    }

    for (const auto &error : errors)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_XID_ERROR, d, error, gpuId);
        st = DR_VIOLATION;
        AddOrClearError(d, fatalErrorList, ignoredErrorList, st);
    }

    return st;
}

int DcgmRecorder::CheckErrorFields(std::vector<unsigned short> &fieldIds,
                                   const std::vector<dcgmTimeseriesInfo_t> *failureThresholds,
                                   unsigned int gpuId,
                                   std::vector<DcgmError> &fatalErrorList,
                                   std::vector<DcgmError> &ignoredErrorList,
                                   timelib64_t startTime)
{
    int st = DR_SUCCESS;

    dcgmFieldSummaryRequest_t fsr;
    std::string error;
    memset(&fsr, 0, sizeof(fsr));
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_SUM | DCGM_SUMMARY_DIFF;
    fsr.startTime       = startTime;
    fsr.endTime         = 0;

    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        if (fieldIds[i] == DCGM_FI_DEV_XID_ERRORS)
        {
            /* XID errors handled in CheckXIDs to avoid "summary" */
            continue;
        }

        dcgm_field_meta_p fm = DcgmFieldGetById(fieldIds[i]);
        if (fm == 0)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_FIELD_TAG, d, fieldIds[i]);
            fatalErrorList.push_back(d);
            return DR_COMM_ERROR;
        }

        memset(&fsr.response, 0, sizeof(fsr.response));
        fsr.fieldId      = fieldIds[i];
        dcgmReturn_t ret = GetFieldSummary(fsr);

        if (ret == DCGM_ST_NOT_SUPPORTED)
        {
            DCGM_LOG_DEBUG << "Not checking for errors in unsupported field " << fm->tag;
            continue;
        }

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, fm->tag, gpuId);
            fatalErrorList.push_back(d);
            return DR_COMM_ERROR;
        }

        int valueIndex = GetValueIndex(fieldIds[i]);

        // Check for failure detection
        if (fm->fieldType == DCGM_FT_INT64)
        {
            if (failureThresholds == nullptr && fsr.response.values[valueIndex].i64 > 0
                && DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                st = DR_VIOLATION;
                AddFieldViolationError(fieldIds[i],
                                       gpuId,
                                       startTime,
                                       fsr.response.values[valueIndex].i64,
                                       DCGM_FP64_BLANK,
                                       fm->tag,
                                       fatalErrorList,
                                       ignoredErrorList,
                                       st);
            }
            else if (failureThresholds != nullptr
                     && fsr.response.values[valueIndex].i64 > static_cast<int64_t>((*failureThresholds)[i].val.i64)
                     && DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                st = DR_VIOLATION;
                AddFieldThresholdViolationError(fieldIds[i],
                                                gpuId,
                                                fsr.response.values[valueIndex].i64,
                                                (*failureThresholds)[i].val.i64,
                                                fm->tag,
                                                fatalErrorList,
                                                ignoredErrorList,
                                                st);
            }
        }
        else if (fm->fieldType == DCGM_FT_DOUBLE)
        {
            if (failureThresholds == nullptr && fsr.response.values[valueIndex].fp64 > 0.0
                && DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                st = DR_VIOLATION;
                AddFieldViolationError(fieldIds[i],
                                       gpuId,
                                       startTime,
                                       DCGM_INT64_BLANK,
                                       fsr.response.values[valueIndex].fp64,
                                       fm->tag,
                                       fatalErrorList,
                                       ignoredErrorList,
                                       st);
            }
            else if (failureThresholds != nullptr
                     && fsr.response.values[valueIndex].fp64 > (*failureThresholds)[i].val.fp64
                     && DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD_DBL,
                                          d,
                                          fsr.response.values[valueIndex].fp64,
                                          fm->tag,
                                          gpuId,
                                          (*failureThresholds)[i].val.fp64);
                st = DR_VIOLATION;
                AddOrClearError(d, fatalErrorList, ignoredErrorList, st);
            }
        }
        else
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNSUPPORTED_FIELD_TYPE, d, fm->tag);
            fatalErrorList.push_back(d);
            st = DR_VIOLATION;
        }
    }
    return st;
}

int DcgmRecorder::CheckErrorFields(std::vector<unsigned short> &fieldIds,
                                   const std::vector<dcgmTimeseriesInfo_t> *failureThresholds,
                                   unsigned int gpuId,
                                   long long maxTemp,
                                   std::vector<DcgmError> &fatalErrorList,
                                   std::vector<DcgmError> &ignoredErrorList,
                                   timelib64_t startTime)
{
    int st = CheckErrorFields(fieldIds, failureThresholds, gpuId, fatalErrorList, ignoredErrorList, startTime);

    std::string infoMsg;
    long long highTemp;
    int tmpSt = CheckGpuTemperature(gpuId, fatalErrorList, maxTemp, infoMsg, startTime, highTemp);
    if (tmpSt == DR_VIOLATION || (st == DR_SUCCESS && st != tmpSt))
    {
        st = tmpSt;
    }

    tmpSt = CheckXIDs(gpuId, startTime, fatalErrorList, ignoredErrorList);
    if (tmpSt == DR_VIOLATION || (st == DR_SUCCESS && st != tmpSt))
    {
        st = tmpSt;
    }

    return st;
}

dcgmReturn_t DcgmRecorder::CheckPerSecondErrorConditions(const std::vector<unsigned short> &fieldIds,
                                                         const std::vector<dcgmFieldValue_v1> &failureThreshold,
                                                         unsigned int gpuId,
                                                         std::vector<DcgmError> &errorList,
                                                         timelib64_t startTime)
{
    dcgmReturn_t st = DCGM_ST_OK;

    if (fieldIds.size() != failureThreshold.size())
    {
        log_error("One failure threshold must be specified for each field id");
        return DCGM_ST_BADPARAM;
    }

    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        std::string tag;
        GetTagFromFieldId(fieldIds[i], tag);

        // Make sure we have the timeseries data for these fields
        st = GetFieldValuesSince(DCGM_FE_GPU, gpuId, fieldIds[i], startTime, true);

        if (st == DCGM_ST_NOT_SUPPORTED)
        {
            DCGM_LOG_DEBUG << "Not checking for errors in unsupported field: " << tag;
            continue;
        }

        if (st != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, st, tag.c_str(), gpuId);
            errorList.push_back(d);
            return st;
        }

        // On error the values holder will appened to errorList
        if (m_valuesHolder.DoesValuePassPerSecondThreshold(
                fieldIds[i], failureThreshold[i], gpuId, tag.c_str(), errorList, startTime))
        {
            st = DCGM_ST_DIAG_THRESHOLD_EXCEEDED;
        }
    }

    return st;
}

dcgmHandle_t DcgmRecorder::GetHandle()
{
    return m_dcgmHandle.GetHandle();
}

void DcgmRecorder::SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value)
{
    m_customStatHolder.SetSingleGroupStat(gpuId, name, value);
}

std::vector<dcgmTimeseriesInfo_t> DcgmRecorder::GetCustomGpuStat(unsigned int gpuId, const std::string &name)
{
    return m_customStatHolder.GetCustomGpuStat(gpuId, name);
}

int DcgmRecorder::CheckGpuTemperature(unsigned int gpuId,
                                      std::vector<DcgmError> &errorList,
                                      long long maxTemp,
                                      std::string &infoMsg,
                                      timelib64_t startTime,
                                      long long &highTemp)

{
    int st = DR_SUCCESS;
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId         = DCGM_FI_DEV_GPU_TEMP;
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_AVG;
    fsr.startTime       = startTime;
    fsr.endTime         = 0;

    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "gpu temperature", gpuId);
        errorList.push_back(d);
        highTemp = 0;
        return DR_COMM_ERROR;
    }

    highTemp = fsr.response.values[0].i64;
    if (DCGM_INT64_IS_BLANK(fsr.response.values[0].i64))
    {
        highTemp = 0;
    }

    if (highTemp > maxTemp)
    {
        DcgmError d { gpuId };
        char const *tempViolationSrc = "GPU";
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, highTemp, tempViolationSrc, gpuId, maxTemp);
        errorList.push_back(d);
        st = DR_VIOLATION;
    }

    double avg = fsr.response.values[1].i64;
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(0);
    ss << "GPU " << gpuId << " temperature average:\t" << avg << " C";
    infoMsg = ss.str();

    return st;
}


int DcgmRecorder::CheckForClocksEvent(unsigned int gpuId,
                                      timelib64_t startTime,
                                      std::vector<DcgmError> &fatalErrorList,
                                      std::vector<DcgmError> &ignoredErrorList)
{
    // mask for the failures we're evaluating
    static const uint64_t failureMask = DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN | DCGM_CLOCKS_EVENT_REASON_SW_THERMAL
                                        | DCGM_CLOCKS_EVENT_REASON_HW_THERMAL | DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE;
    uint64_t mask = failureMask;

    // Update mask to ignore clocks event reasons given by the ignoreMask
    if (nvvsCommon.clocksEventIgnoreMask != DCGM_INT64_BLANK && nvvsCommon.clocksEventIgnoreMask > 0)
    {
        mask &= ~nvvsCommon.clocksEventIgnoreMask;
    }

    dcgmFieldValue_v1 dfv;
    dcgmReturn_t st = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCKS_EVENT_REASONS, startTime, true);
    int rc          = DR_SUCCESS;

    std::stringstream buf;

    if (st == DCGM_ST_NOT_SUPPORTED)
    {
        DCGM_LOG_DEBUG << "Skipping clocks event check because it is unsupported.";
        return DR_SUCCESS;
    }

    if (st != DCGM_ST_OK)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, st, "clocks event", gpuId);
        fatalErrorList.push_back(d);
        return DR_COMM_ERROR;
    }

    m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCKS_EVENT_REASONS, dfv, mask);
    int64_t maskedResult = dfv.value.i64 & mask;

    if (maskedResult)
    {
        const char *detail = NULL;
        double timeDiff    = (dfv.ts - startTime) / 1000000.0;

        if ((maskedResult & DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN))
        {
            detail = "clocks_event_reason_hw_slowdown: either the temperature is too high or there is a "
                     "power supply problem (the power brake assertion has been tripped).";
        }
        else if ((maskedResult & DCGM_CLOCKS_EVENT_REASON_SW_THERMAL))
        {
            detail = "clocks_event_reason_sw_thermal_slowdown: the GPU or its memory have reached unsafe "
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_EVENT_REASON_HW_THERMAL))
        {
            detail = "clocks_event_reason_hw_thermal_slowdown: the GPU or its memory have reached unsafe "
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE))
        {
            detail = "clocks_event_reason_hw_power_brake_slowdown: the power brake assertion has triggered. "
                     "Please check the power supply.";
        }

        if (detail != NULL)
        {
            rc = DR_VIOLATION;
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CLOCKS_EVENT_VIOLATION, d, gpuId, timeDiff, detail);
            AddOrClearError(d, fatalErrorList, ignoredErrorList, rc);
        }
    }

    return rc;
}

int DcgmRecorder::CheckForThrottling(unsigned int gpuId,
                                     timelib64_t startTime,
                                     std::vector<DcgmError> &fatalErrorList,
                                     std::vector<DcgmError> &ignoredErrorList)
{
    return CheckForClocksEvent(gpuId, startTime, fatalErrorList, ignoredErrorList);
}

int DcgmRecorder::CheckEffectiveBER(unsigned int gpuId, std::vector<DcgmError> &fatalErrorList)
{
    dcgmFieldValue_v2 value;
    dcgmReturn_t ret = GetCurrentFieldValue(gpuId, DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER, value, 0);
    if (ret == DCGM_ST_NOT_SUPPORTED || value.status == DCGM_ST_NOT_SUPPORTED)
    {
        log_debug("Skipping effective BER check because it is unsupported.");
        return DR_SUCCESS;
    }
    if (ret != DCGM_ST_OK)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "effective BER", gpuId);
        fatalErrorList.push_back(d);
        return DR_COMM_ERROR;
    }

    auto const effectiveBer = value.value.i64;
    // Effective BER is the bit error rate itself, and a rate of 0 indicates a healthy system.
    // However, in NVML the smallest value is 1.5e-254, to be able to compare and avoid precision issues
    // with the decoded value, we decode it here and compare the mantissa and exponent to 15 and 255
    // directly, rather than using the decoded field (i.e., DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT).
    auto const [mantissa, exponent, ber] = DcgmNs::Utils::NvmlBerParser(effectiveBer);
    if (mantissa != 15 && exponent != 255)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_EFFECTIVE_BER_THRESHOLD, d, ber, gpuId);
        fatalErrorList.push_back(d);
        return DR_VIOLATION;
    }

    return DR_SUCCESS;
}

dcgmReturn_t DcgmRecorder::GetCurrentFieldValue(unsigned int gpuId,
                                                unsigned short fieldId,
                                                dcgmFieldValue_v2 &value,
                                                unsigned int flags)
{
    memset(&value, 0, sizeof(value));

    return m_dcgmSystem.GetGpuLatestValue(gpuId, fieldId, flags, value);
}

int DcgmRecorder::GetLatestValuesForWatchedFields(unsigned int flags, std::vector<DcgmError> &fatalErrorList)
{
    dcgmReturn_t ret = m_dcgmSystem.GetLatestValuesForGpus(m_gpuIds, m_fieldIds, flags, storeValues, &m_valuesHolder);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "all watched fields", DcgmError::GpuIdTag::Unknown);
        fatalErrorList.push_back(d);
        return DR_COMM_ERROR;
    }

    return DR_SUCCESS;
}

std::string DcgmRecorder::GetGpuUtilizationNote(unsigned int gpuId, timelib64_t startTime)
{
    static const int UTILIZATION_THRESHOLD = 75;
    std::stringstream msg;
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId         = DCGM_FI_DEV_GPU_UTIL;
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX;
    fsr.startTime       = startTime;
    fsr.endTime         = 0;

    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        std::string error;
        GetErrorString(ret, error);
        log_error("unable to query for gpu temperature: {} for GPU {}", error, gpuId);
        return error;
    }

    if (fsr.response.values[0].i64 < UTILIZATION_THRESHOLD)
    {
        msg << "NOTE: GPU usage was only " << fsr.response.values[0].i64 << " for GPU " << gpuId
            << ". This may have caused the failure. Verify that no other processes are contending"
               " for GPU resources; if any exist, stop them and retry.";
    }

    return msg.str();
}

dcgmReturn_t DcgmRecorder::GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &attributes)
{
    memset(&attributes, 0, sizeof(attributes));
    attributes.version = dcgmDeviceAttributes_version3;
    return m_dcgmSystem.GetDeviceAttributes(gpuId, attributes);
}

void DcgmRecorder::AddDiagStats(const std::vector<dcgmDiagCustomStats_t> &customStats)
{
    m_customStatHolder.AddDiagStats(customStats);
}

long long DcgmRecorder::DetermineMaxTemp(const dcgmDiagPluginEntityInfo_v1 &entityInfo)
{
    long long const fakeMaxTemp  = 85;
    unsigned int flags           = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 maxTempVal = {};

    if (entityInfo.entity.entityGroupId != DCGM_FE_GPU)
    {
        log_debug("Function called with non-GPU entityInfo.");
        return fakeMaxTemp;
    }

    dcgmReturn_t ret = GetCurrentFieldValue(entityInfo.entity.entityId, DCGM_FI_DEV_GPU_MAX_OP_TEMP, maxTempVal, flags);

    if (ret != DCGM_ST_OK || DCGM_INT64_IS_BLANK(maxTempVal.value.i64))
    {
        DCGM_LOG_WARNING << "Cannot read the max operating temperature for GPU " << entityInfo.entity.entityId << ": "
                         << errorString(ret) << ", defaulting to the slowdown temperature";

        if (entityInfo.auxField.gpu.status == DcgmEntityStatusFake)
        {
            /* fake gpus don't report max temp */
            return fakeMaxTemp;
        }
        else
        {
            return entityInfo.auxField.gpu.attributes.thermalSettings.slowdownTemp;
        }
    }
    else
    {
        return maxTempVal.value.i64;
    }
}

void DcgmRecorder::CheckNonFatalErrors(timelib64_t startTime,
                                       dcgmGroupEntityPair_t const entity,
                                       std::vector<DcgmError> &fatalErrors,
                                       std::vector<DcgmError> &ignoredErrors)
{
    if (CheckForClocksEvent(entity.entityId, startTime, fatalErrors, ignoredErrors) == DR_COMM_ERROR)
    {
        DCGM_LOG_ERROR << "Unable to read the clocks event information from the hostengine";
    }

    std::vector<unsigned short> nfFieldIds = { DCGM_FI_DEV_THERMAL_VIOLATION };
    // error checking and reporting handled in CheckErrorFields()
    (void)CheckErrorFields(nfFieldIds, nullptr, entity.entityId, fatalErrors, ignoredErrors, startTime);
}


void DcgmRecorder::CheckCommonErrors(TestParameters &tp,
                                     timelib64_t startTime,
                                     nvvsPluginResult_t &result,
                                     std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos,
                                     std::vector<DcgmError> &fatalErrors,
                                     std::vector<DcgmError> &ignoredErrors)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> thresholds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = nullptr;
    bool needThresholds                              = false;
    dcgmTimeseriesInfo_t tsInfo                      = {};
    tsInfo.isInt                                     = true;

    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        if (standardErrorFields[i].thresholdName == nullptr)
        {
            fieldIds.push_back(standardErrorFields[i].fieldId);
            tsInfo.val.i64 = 0;
            thresholds.push_back(tsInfo);
        }
        else if (tp.HasKey(standardErrorFields[i].thresholdName))
        {
            fieldIds.push_back(standardErrorFields[i].fieldId);
            needThresholds = true;
            tsInfo.val.i64 = tp.GetDouble(standardErrorFields[i].thresholdName);
            thresholds.push_back(tsInfo);
        }
    }

    if (needThresholds)
    {
        thresholdsPtr = &thresholds;
    }

    dcgmUpdateAllFields(m_dcgmHandle.GetHandle(), 1);
    for (auto &&entityInfo : entityInfos)
    {
        if (entityInfo.entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        long long maxTemp = DetermineMaxTemp(entityInfo);
        int ret           = CheckErrorFields(
            fieldIds, thresholdsPtr, entityInfo.entity.entityId, maxTemp, fatalErrors, ignoredErrors, startTime);
        int retHBMErrorFields = CheckHBMErrorFields(entityInfo, fatalErrors, startTime);

        // ret - violation, retHBMErrorFields - violation --> ret = violation
        // ret - success, retHBMErrorFields - violation --> ret = violation
        // ret - violation, retHBMErrorFields - success --> ret = violation (does not enter the if case as ret is
        // already a violation) ret - success, retHBMErrorFields - success --> ret = success (does not enter the if case
        // as both are success)
        if (retHBMErrorFields == DR_VIOLATION || (ret == DR_SUCCESS && ret != retHBMErrorFields))
        {
            ret = retHBMErrorFields;
        }

        if (ret == DR_COMM_ERROR)
        {
            DCGM_LOG_ERROR << "Unable to read the error values from the hostengine";
            result = NVVS_RESULT_FAIL;
        }
        else if (ret == DR_VIOLATION || result == NVVS_RESULT_FAIL)
        {
            result = NVVS_RESULT_FAIL;
            CheckNonFatalErrors(startTime, entityInfo.entity, fatalErrors, ignoredErrors);
        }
    }
}

std::string DcgmRecorder::ErrorAsString(dcgmReturn_t ret)
{
    const char *str = errorString(ret);
    if (str == nullptr)
    {
        return fmt::format("Unknown error code {}", ret);
    }

    return str;
}

unsigned int DcgmRecorder::GetCudaMajorVersion()
{
    return m_dcgmSystem.GetCudaMajorVersion();
}

int DcgmRecorder::CheckHBMErrorFields(dcgmDiagPluginEntityInfo_v1 const &entityInfo,
                                      std::vector<DcgmError> &errors,
                                      timelib64_t startTime)
{
    // HBM temperature check
    // local vars
    long long maxHBMTemp;
    int skipCheck                = 0;
    dcgmFieldValue_v2 maxTempVal = {};
    int ret                      = DR_SUCCESS;

    if (entityInfo.entity.entityGroupId != DCGM_FE_GPU)
    {
        log_debug("Function called with non-GPU entityInfo.");
        return ret;
    }

    // get the max temperature
    dcgmReturn_t returnValue = GetCurrentFieldValue(
        entityInfo.entity.entityId, DCGM_FI_DEV_MEM_MAX_OP_TEMP, maxTempVal, DCGM_FV_FLAG_LIVE_DATA);

    if (returnValue != DCGM_ST_OK || DCGM_INT64_IS_BLANK(maxTempVal.value.i64))
    {
        // logging the error
        if (DCGM_INT64_IS_BLANK(maxTempVal.value.i64))
        {
            log_debug("Cannot read the max HBM operating temperature for gpuId {}: {}: INT64 value is blank.",
                      entityInfo.entity.entityId,
                      errorString(returnValue));
        }
        else
        {
            log_debug("Cannot read the max HBM operating temperature for gpuId {}: {}",
                      entityInfo.entity.entityId,
                      errorString(returnValue));
        }

        if (entityInfo.auxField.gpu.status == DcgmEntityStatusFake)
        {
            // fake gpu - hardcoding value
            maxHBMTemp = 100;
        }
        else
        {
            // read empty, skipping the check
            skipCheck = 1;
        }
    }
    else
    {
        // setting maxHBMTemp to max value read
        maxHBMTemp = maxTempVal.value.i64;
    }

    if (skipCheck == 0)
    {
        // checking if current HBM temperature is greater than max
        ret = VerifyHBMTemperature(entityInfo.entity.entityId, errors, maxHBMTemp, startTime);
    }

    return ret;
}


int DcgmRecorder::VerifyHBMTemperature(unsigned int gpuId,
                                       std::vector<DcgmError> &errorList,
                                       long long maxTemp,
                                       timelib64_t startTime)

{
    // local vars
    long long currHighTemp;

    // defaulting to success
    int st = DR_SUCCESS;

    // field struct for current HBM temperature
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId         = DCGM_FI_DEV_MEMORY_TEMP;
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_AVG;
    fsr.startTime       = startTime;
    fsr.endTime         = 0;

    // populate field
    dcgmReturn_t ret = GetFieldSummary(fsr);

    // check if field is not populated
    if (ret != DCGM_ST_OK)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "HBM temperature", gpuId);
        errorList.push_back(d);
        currHighTemp = 0;
        return DR_COMM_ERROR;
    }

    // get the curr high temp
    currHighTemp = fsr.response.values[0].i64;

    // check if nan or blank
    if (DCGM_INT64_IS_BLANK(fsr.response.values[0].i64))
    {
        currHighTemp = 0;
    }

    // check if curr high temp is lesser or greater than max HBM temp
    if (currHighTemp > maxTemp)
    {
        DcgmError d { gpuId };
        char const *tempViolationSrc = "HBM Memory on GPU";
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, currHighTemp, tempViolationSrc, gpuId, maxTemp);
        errorList.push_back(d);
        st = DR_VIOLATION;
    }

    return st;
}

void DcgmRecorder::SetWatchFrequency(long long watchFrequency)
{
    m_watchFrequency = watchFrequency;
}

void DcgmRecorder::SetIgnoreErrorCodes(gpuIgnoreErrorCodeMap_t const &ignoreErrorCodes)
{
    m_ignoreErrorCodes = ignoreErrorCodes;
}

void DcgmRecorder::AddOrClearError(DcgmError const &d,
                                   std::vector<DcgmError> &fatalErrorList,
                                   std::vector<DcgmError> &ignoredErrorList,
                                   int &retCode)
{
    auto const &entity  = d.GetEntity();
    auto const &errCode = d.GetCode();
    if (m_ignoreErrorCodes.contains(entity) && m_ignoreErrorCodes[entity].contains(errCode))
    {
        log_debug("Error code {} ignored: {}.", errCode, d.GetMessage());
        retCode = DR_SUCCESS;
        ignoredErrorList.push_back(d);
        return;
    }
    fatalErrorList.push_back(d);
}
