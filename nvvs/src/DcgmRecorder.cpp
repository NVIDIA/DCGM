/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "NvvsCommon.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "timelib.h"

const long long defaultFrequency = 1000000; // update each field every second (a million microseconds)

DcgmRecorder::DcgmRecorder()
    : m_fieldIds()
    , m_gpuIds()
    , m_fieldGroupId(0)
    , m_dcgmHandle()
    , m_dcgmGroup()
    , m_dcgmSystem()
    , m_valuesHolder()
    , m_nextValuesSinceTs(0)

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
{}

DcgmRecorder::~DcgmRecorder()
{
    Shutdown();
}

std::string DcgmRecorder::CreateGroup(const std::vector<unsigned int> &gpuIds,
                                      bool /*allGpus*/,
                                      const std::string &groupName)
{
    dcgmReturn_t ret;
    std::string errStr;

    if (m_dcgmHandle.GetHandle() == 0)
        return "Must connect to DCGM before creating a group";

    ret = m_dcgmGroup.Init(m_dcgmHandle.GetHandle(), groupName, gpuIds);

    if (ret != DCGM_ST_OK)
        errStr = m_dcgmHandle.RetToString(ret);

    return errStr;
}

std::string DcgmRecorder::AddWatches(const std::vector<unsigned short> &fieldIds,
                                     const std::vector<unsigned int> &gpuIds,
                                     bool allGpus,
                                     const std::string &fieldGroupName,
                                     const std::string &groupName,
                                     double testDuration)

{
    dcgmReturn_t ret;
    std::string errStr;
    m_fieldIds = fieldIds;
    m_gpuIds   = gpuIds;

    if (fieldIds.size() == 0 || fieldIds.size() > DCGM_FI_MAX_FIELDS)
        return "Field Ids must contain at least 1 field id";

    if (gpuIds.size() == 0)
        return "Gpu Ids must contain at least 1 gpu id";

    errStr = CreateGroup(gpuIds, allGpus, groupName);
    if (errStr.size() != 0)
        return errStr;

    ret = m_dcgmGroup.FieldGroupCreate(fieldIds, fieldGroupName);
    if (ret != DCGM_ST_OK)
    {
        GetErrorString(ret, errStr);
        return errStr;
    }

    ret = m_dcgmGroup.WatchFields(defaultFrequency, testDuration + 30);
    if (ret != DCGM_ST_OK)
    {
        GetErrorString(ret, errStr);
    }

    return errStr;
}

void DcgmRecorder::GetErrorString(dcgmReturn_t ret, std::string &err)
{
    std::stringstream err_stream;
    const char *tmp = errorString(ret);

    if (tmp == NULL)
    {
        err_stream << "Unknown error from DCGM: " << ret;
        err = err_stream.str();
    }
    else
        err = tmp;
}

std::string DcgmRecorder::Init(const std::string &hostname)
{
    std::string errStr;
    dcgmReturn_t ret;

    ret = m_dcgmHandle.ConnectToDcgm(hostname);

    if (ret != DCGM_ST_OK)
        errStr = m_dcgmHandle.GetLastError();
    else
    {
        m_dcgmSystem.Init(m_dcgmHandle.GetHandle());
    }

    return errStr;
}

std::string DcgmRecorder::Init(dcgmHandle_t handle)
{
    m_dcgmSystem.Init(handle);
    m_dcgmHandle = handle;
    return "";
}

std::string DcgmRecorder::Shutdown()
{
    std::string errStr;

    if (m_dcgmHandle.GetHandle() == 0)
    {
        m_fieldGroupId = 0;
        return errStr;
    }

    if (m_fieldGroupId != 0)
    {
        // Ignore errors
        dcgmFieldGroupDestroy(m_dcgmHandle.GetHandle(), m_fieldGroupId);
        m_fieldGroupId = 0;
    }

    m_dcgmGroup.Cleanup();

    return errStr;
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
        PRINT_ERROR("%s %s", "Unable to open file %s: '%s'", filename.c_str(), strerror(errno));
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

void DcgmRecorder::AddFieldViolationError(unsigned short fieldId,
                                          unsigned int gpuId,
                                          timelib64_t startTime,
                                          int64_t intValue,
                                          double dblValue,
                                          const char *fieldName,
                                          std::vector<DcgmError> &errorList)
{
    DcgmError d { gpuId };
    bool loggedError = false;

    // Thermal violation errors are logged in a special way
    if (fieldId == DCGM_FI_DEV_THERMAL_VIOLATION)
    {
        dcgmReturn_t ret = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, startTime, false);
        dcgmFieldValue_v1 dfv;
        memset(&dfv, 0, sizeof(dfv));
        double seconds = intValue / 1000000000.0; // The violation is reported in nanoseconds.

        if (ret == DCGM_ST_OK)
        {
            m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dfv, 0);
        }

        if (dfv.ts != 0) // the field value timestamp will be 0 if we couldn't find one
        {
            double timeDiff = (dfv.ts - startTime) / 1000000.0;
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS_TS, d, seconds, timeDiff, gpuId);
            errorList.push_back(d);
            loggedError = true;
        }
        else
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS, d, seconds, gpuId);
            errorList.push_back(d);
            loggedError = true;
        }
    }

    if (loggedError == false)
    {
        if (DCGM_INT64_IS_BLANK(intValue))
        {
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION_DBL, d, dblValue, fieldName, gpuId);
        }
        else
        {
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_VIOLATION, d, intValue, fieldName, gpuId);
        }
        errorList.push_back(d);
    }
}

int DcgmRecorder::CheckErrorFields(std::vector<unsigned short> &fieldIds,
                                   const std::vector<dcgmTimeseriesInfo_t> *failureThresholds,
                                   unsigned int gpuId,
                                   long long maxTemp,
                                   std::vector<DcgmError> &errorList,
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
        dcgm_field_meta_p fm = DcgmFieldGetById(fieldIds[i]);
        if (fm == 0)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_FIELD_TAG, d, fieldIds[i]);
            errorList.push_back(d);
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
            errorList.push_back(d);
            return DR_COMM_ERROR;
        }

        int valueIndex = GetValueIndex(fieldIds[i]);

        // Check for failure detection
        if (fm->fieldType == DCGM_FT_INT64)
        {
            if (failureThresholds == 0 && fsr.response.values[valueIndex].i64 > 0
                && DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                AddFieldViolationError(fieldIds[i],
                                       gpuId,
                                       startTime,
                                       fsr.response.values[valueIndex].i64,
                                       DCGM_FP64_BLANK,
                                       fm->tag,
                                       errorList);
                st = DR_VIOLATION;
            }
            else if (failureThresholds != 0 && fsr.response.values[valueIndex].i64 > (*failureThresholds)[i].val.i64
                     && DCGM_INT64_IS_BLANK(fsr.response.values[valueIndex].i64) == 0)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD,
                                          d,
                                          fsr.response.values[valueIndex].i64,
                                          fm->tag,
                                          gpuId,
                                          (*failureThresholds)[i].val.i64);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
        }
        else if (fm->fieldType == DCGM_FT_DOUBLE)
        {
            if (failureThresholds == 0 && fsr.response.values[valueIndex].fp64 > 0.0
                && DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                AddFieldViolationError(fieldIds[i],
                                       gpuId,
                                       startTime,
                                       DCGM_INT64_BLANK,
                                       fsr.response.values[valueIndex].fp64,
                                       fm->tag,
                                       errorList);
                st = DR_VIOLATION;
            }
            else if (failureThresholds != 0 && fsr.response.values[valueIndex].fp64 > (*failureThresholds)[i].val.fp64
                     && DCGM_FP64_IS_BLANK(fsr.response.values[valueIndex].fp64) == 0)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FIELD_THRESHOLD_DBL,
                                          d,
                                          fsr.response.values[valueIndex].fp64,
                                          fm->tag,
                                          gpuId,
                                          (*failureThresholds)[i].val.fp64);
                errorList.push_back(d);
                st = DR_VIOLATION;
            }
        }
        else
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_UNSUPPORTED_FIELD_TYPE, d, fm->tag);
            errorList.push_back(d);
            st = DR_VIOLATION;
        }
    }

    std::string infoMsg;
    long long highTemp;
    int tmpSt = CheckGpuTemperature(gpuId, errorList, maxTemp, infoMsg, startTime, highTemp);
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
        PRINT_ERROR("", "One failure threshold must be specified for each field id");
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

int DcgmRecorder::CheckThermalViolations(unsigned int gpuId, std::vector<DcgmError> &errorList, timelib64_t startTime)
{
    int st = DR_SUCCESS;
    dcgmFieldSummaryRequest_t fsr;
    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId         = DCGM_FI_DEV_THERMAL_VIOLATION;
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_SUM;
    fsr.startTime       = startTime;
    fsr.endTime         = 0;

    dcgmReturn_t ret = GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        // This may be null since we only expose thermal violations with the setting of an environmental variable
        return ret;
    }

    if (fsr.response.values[0].i64 > 0 && !DCGM_INT64_IS_BLANK(fsr.response.values[0].i64))
    {
        dcgmReturn_t ret = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, startTime, true);
        dcgmFieldValue_v1 dfv;
        memset(&dfv, 0, sizeof(dfv));

        if (ret == DCGM_ST_OK)
        {
            m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dfv, 0);
        }

        // The field value returns nanoseconds, so convert to seconds.
        double violationSeconds = fsr.response.values[0].i64 / 1000000000.0;

        if (dfv.ts != 0) // the field value timestamp will be 0 if we couldn't find one
        {
            double timeDiff = (dfv.ts - startTime) / 1000000.0;
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS_TS, d, violationSeconds, timeDiff, gpuId);
            errorList.push_back(d);
        }
        else
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THERMAL_VIOLATIONS, d, violationSeconds, gpuId);
            errorList.push_back(d);
        }

        // Thermal violations were found so, make the return indicate we found a violation
        st = DR_VIOLATION;
    }

    return st;
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
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, highTemp, gpuId, maxTemp);
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

int DcgmRecorder::CheckForThrottling(unsigned int gpuId, timelib64_t startTime, std::vector<DcgmError> &errorList)
{
    // mask for the failures we're evaluating
    static const uint64_t failureMask = DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL
                                        | DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL
                                        | DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;
    uint64_t mask = failureMask;

    // Update mask to ignore throttle reasons given by the ignoreMask
    if (nvvsCommon.throttleIgnoreMask != DCGM_INT64_BLANK && nvvsCommon.throttleIgnoreMask > 0)
    {
        mask &= ~nvvsCommon.throttleIgnoreMask;
    }

    dcgmFieldValue_v1 dfv;
    dcgmReturn_t st = GetFieldValuesSince(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, startTime, true);
    int rc          = DR_SUCCESS;

    std::stringstream buf;

    if (st == DCGM_ST_NOT_SUPPORTED)
    {
        DCGM_LOG_DEBUG << "Skipping throttling check because it is unsupported.";
        return DR_SUCCESS;
    }

    if (st != DCGM_ST_OK)
    {
        DcgmError d { gpuId };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, st, "clock throttling", gpuId);
        errorList.push_back(d);
        return DR_COMM_ERROR;
    }

    m_valuesHolder.GetFirstNonZero(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_CLOCK_THROTTLE_REASONS, dfv, mask);
    int64_t maskedResult = dfv.value.i64 & mask;

    if (maskedResult)
    {
        const char *detail = NULL;
        double timeDiff    = (dfv.ts - startTime) / 1000000.0;

        if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN))
        {
            detail = "clocks_throttle_reason_hw_slowdown: either the temperature is too high or there is a "
                     "power supply problem (the power brake assertion has been tripped).";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL))
        {
            detail = "clocks_throttle_reason_sw_thermal_slowdown: the GPU or its memory have reached unsafe "
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL))
        {
            detail = "clocks_throttle_reason_hw_thermal_slowdown: the GPU or its memory have reached unsafe "
                     "temperatures.";
        }
        else if ((maskedResult & DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE))
        {
            detail = "clocks_throttle_reason_hw_power_brake_slowdown: the power brake assertion has triggered. "
                     "Please check the power supply.";
        }

        if (detail != NULL)
        {
            rc = DR_VIOLATION;
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_THROTTLING_VIOLATION, d, gpuId, timeDiff, detail);
            errorList.push_back(d);
        }
    }

    return rc;
}

dcgmReturn_t DcgmRecorder::GetCurrentFieldValue(unsigned int gpuId,
                                                unsigned short fieldId,
                                                dcgmFieldValue_v2 &value,
                                                unsigned int flags)
{
    memset(&value, 0, sizeof(value));

    return m_dcgmSystem.GetGpuLatestValue(gpuId, fieldId, flags, value);
}

int DcgmRecorder::GetLatestValuesForWatchedFields(unsigned int flags, std::vector<DcgmError> &errorList)
{
    dcgmReturn_t ret = m_dcgmSystem.GetLatestValuesForGpus(m_gpuIds, m_fieldIds, flags, storeValues, &m_valuesHolder);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_FIELD_QUERY, d, ret, "all watched fields", m_gpuIds[0]);
        errorList.push_back(d);
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
        PRINT_ERROR("%s %u", "unable to query for gpu temperature: %s for GPU %u", error.c_str(), gpuId);
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
    attributes.version = dcgmDeviceAttributes_version2;
    return m_dcgmSystem.GetDeviceAttributes(gpuId, attributes);
}

void DcgmRecorder::AddDiagStats(const std::vector<dcgmDiagCustomStats_t> &customStats)
{
    m_customStatHolder.AddDiagStats(customStats);
}
