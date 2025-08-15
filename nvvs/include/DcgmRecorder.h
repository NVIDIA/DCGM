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
#ifndef DCGM_RECORDER_H
#define DCGM_RECORDER_H

#include <map>
#include <string>
#include <vector>

#include "CustomStatHolder.h"
#include "DcgmError.h"
#include "DcgmGroup.h"
#include "DcgmHandle.h"
#include "DcgmMutex.h"
#include "DcgmSystem.h"
#include "DcgmValuesSinceHolder.h"
#include "IgnoreErrorCodesHelper.h"
#include "TestParameters.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include <json/json.h>

#define VALUE "value"

#define DR_SUCCESS      0
#define DR_COMM_ERROR   -1
#define DR_VIOLATION    -2
#define DR_CLOCKS_EVENT -3
// Deprecated: Use DR_CLOCKS_EVENT instead
#define DR_THROTTLING DR_CLOCKS_EVENT

typedef struct
{
    unsigned short fieldId;
    const char *thresholdName;
} errorType_t;

extern errorType_t standardErrorFields[];
extern unsigned short standardInfoFields[];

class DcgmRecorderBase
{
public:
    virtual ~DcgmRecorderBase() = default;

    virtual dcgmReturn_t GetCurrentFieldValue(unsigned int gpuId,
                                              unsigned short fieldId,
                                              dcgmFieldValue_v2 &value,
                                              unsigned int flags);

    virtual dcgmReturn_t GetFieldSummary(dcgmFieldSummaryRequest_t &request);

    virtual std::string GetGpuUtilizationNote(unsigned int gpuId, timelib64_t startTime);
};

class DcgmRecorder : public DcgmRecorderBase
{
public:
    DcgmRecorder();
    DcgmRecorder(DcgmRecorder &&other) noexcept;
    explicit DcgmRecorder(dcgmHandle_t handle);
    ~DcgmRecorder();

    /*
     * Adds watches to the specified field list by creating field groups and gpu groups with the specified
     * names.
     *
     * Returns an empty string on SUCCESS or a string with an error message in it on failure
     */
    dcgmReturn_t AddWatches(const std::vector<unsigned short> &fieldIds,
                            const std::vector<unsigned int> &gpuIds,
                            bool allGpus,
                            const std::string &fieldGroupName,
                            const std::string &groupName,
                            double testDuration);

    /*
     * Initialize by connecting to the hostengine at the specified hostname
     */
    dcgmReturn_t Init(const std::string &hostname);

    /*
     * Initialize by accepting the pre-existing handle to DCGM
     */
    void Init(dcgmHandle_t handle);

    /*
     */
    dcgmReturn_t Shutdown();

    /*
     * Iterate over data
     *
     * @param fieldId - the fieldId we want to query from DCGM
     * @param ts - timestamp that we want values since
     *
     * @return DCGM_ST_* as appropriate
     */
    dcgmReturn_t CheckFieldValuesSince(unsigned short fieldId, long long ts);

    /*
     * Writes the data for the field id group to the specified file name in the specified format
     */
    int WriteToFile(const std::string &filename, int logFileType, long long testStart);

    /*
     */
    dcgmReturn_t GetFieldSummary(dcgmFieldSummaryRequest_t &request) override;

    /*
     * Populate tag with the field name for the specified field id
     */
    static void GetTagFromFieldId(unsigned short fieldId, std::string &tag);

    /*
     * Returns an integer representing the major CUDA version
     */
    unsigned int GetCudaMajorVersion();

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, double value);

    /*
     * Adds a custom timeseries statistic for GPU gpuId with the specified name and value
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, long long value);

    /*
     * Get the data associated with the custom gpu stat recorded
     */
    std::vector<dcgmTimeseriesInfo_t> GetCustomGpuStat(unsigned int gpuId, const std::string &name);

    /*
     * Adds a custom field fo GPU gpuId
     */
    void SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, double value);

    /*
     * Adds a custom statistic for group groupName with the specified name and value
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, long long value);

    /*
     * Returns vector of stored group stats
     */
    std::vector<dcgmTimeseriesInfo_t> GetGroupedStat(const std::string &groupName, const std::string &name);

    /*
     * Clears the custom data currently stored in this object
     */
    void ClearCustomData();

    /*
     * Makes a stateful query for the specified entity's fieldId since ts. If we have already stored the data
     * for the specified entity's fieldId in m_valuesHolder, then no work is done, unless force is set.
     *
     * @return:
     *
     * DCGM_ST_OK     : on success
     * DCGM_ST_*      : to call out specific errors
     */
    dcgmReturn_t GetFieldValuesSince(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     unsigned short fieldId,
                                     long long ts,
                                     bool force);

    /*
     * Checks for any non-zero entries for any of the fields specified in fieldIds between startTime and
     * endTime for GPU gpuId, and adds an error message to errorList if any violate the failureThresholds
     * or warningThresholds. warningThresholds causes a warning message to be printed even if a field
     * value does not meet the value in failureThresholds
     *
     * @return:
     *
     * DR_SUCCESS      : on success
     * DR_COMM_ERROR   : if we couldn't get the information from DCGM
     * DR_VIOLATION    : if a value was found above a failure threshold
     */
    int CheckErrorFields(std::vector<unsigned short> &fieldIds,
                         const std::vector<dcgmTimeseriesInfo_t> *failureThreshold,
                         unsigned int gpuId,
                         long long maxTemp,
                         std::vector<DcgmError> &fatalErrorList,
                         std::vector<DcgmError> &ignoredErrorList,
                         timelib64_t startTime);

    /*
     * Iterates through the stored values to check if there is ever a jump that is greater than or equal to
     * the specified threshold, and records an error in errorList if one is present.
     *
     * @return
     * DCGM_ST_OK                       : on success
     * DCGM_ST_BADPARAM                 : if not enough thresholds are offered
     * DCGM_ST_DIAG_THRESHOLD_EXCEEDED  : if a value was found above a failure threshold
     */
    dcgmReturn_t CheckPerSecondErrorConditions(const std::vector<unsigned short> &fieldIds,
                                               const std::vector<dcgmFieldValue_v1> &failureThreshold,
                                               unsigned int gpuId,
                                               std::vector<DcgmError> &errorList,
                                               timelib64_t startTime);

    /*
     * Determines the index of our dcgmFieldSummaryRequest_t we should look at for this fieldId.
     * Currently, the options are 0 for DCGM_SUMMARY_MAX and 1 for DCGM_SUMMARY_DIFF.
     *
     * Note: this may need updating as additional fields are added if those fields need to look at a different
     * summary option
     *
     * @return:
     *
     * 0        : look at DCGM_SUMMARY_MAX
     * 1        : look at DCGM_SUMMARY_DIFF
     */
    int GetValueIndex(unsigned short fieldId);

    /*
     * Checks if the GPU temperature was about the maxTemp at any point in the specified time period, and adds
     * an error message if the temperature was too high.
     * Populates infoMsg with the average temperature
     * Populates highTemp with the highest temperature recorded during the test; set to 0 on error.
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : if the temperature was ever too high
     */
    int CheckGpuTemperature(unsigned int gpuId,
                            std::vector<DcgmError> &errorList,
                            long long maxTemp,
                            std::string &infoMsg,
                            timelib64_t startTime,
                            long long &highTemp);

    dcgmHandle_t GetHandle();

    /*
     * Checks if the specified GPU reported clocks event and sets an appropriate error message if so.
     *
     * @return:
     *
     * DR_SUCCESS     : No clocks event is happening
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : Clocks event is happening
     */
    int CheckForClocksEvent(unsigned int gpuId,
                            timelib64_t startTime,
                            std::vector<DcgmError> &fatalErorList,
                            std::vector<DcgmError> &ignoredErrorList);

    /*
     * Deprecated: Use CheckForClocksEvent instead.
     */
    int CheckForThrottling(unsigned int gpuId,
                           timelib64_t startTime,
                           std::vector<DcgmError> &fatalErrorList,
                           std::vector<DcgmError> &ignoredErrorList);

    int CheckEffectiveBER(unsigned int gpuId, std::vector<DcgmError> &fatalErrorList);

    /*
     * Populates dcgmTimeseriesInfo with the current value of the specified field for the specified GPU
     * Adds an appropriate warning if we can't get the fieldId
     *
     * @return:
     *
     * DCGM_ST_OK     : on success
     * DCGM_ST_*      : if we couldn't get the information from DCGM
     */
    dcgmReturn_t GetCurrentFieldValue(unsigned int gpuId,
                                      unsigned short fieldId,
                                      dcgmFieldValue_v2 &value,
                                      unsigned int flags) override;

    /*
     * Retrieves the latest values for the watched fields (added via AddWatches).
     * @param flags - Set to 0 to get latest cached values from DCGM
     *              - Set to DCGM_FV_FLAG_LIVE_DATA to get live values from the driver
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     */
    int GetLatestValuesForWatchedFields(unsigned int flags, std::vector<DcgmError> &errorList);

    /*
     * If gpu utilization is below 75% for this GPU, then return a note prompting the user to investigate.
     * Otherwise return an empty string
     */
    std::string GetGpuUtilizationNote(unsigned int gpuId, timelib64_t startTime) override;

    /*
     * Get the attributes for the specified device
     */
    dcgmReturn_t GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &attributes);

    /*
     * Add the customStats information to the private custom stats storage.
     *
     * @param customStats - the list of diag custom stats structs containing stats information not tied
     *                      to field ids that should be added to the private storage.
     */
    void AddDiagStats(const std::vector<dcgmDiagCustomStats_t> &customStats);

    /**
     * Check the errors that should be checked for each plugin
     *
     * @param tp (I) - the test parameters for this plugin
     * @param starttime (I) - the start time for the plugin
     * @param result (O) - set to NVVS_RESULT_FAIL if a failure is detected
     *
     * @return a vector filled with each detected error
     */
    void CheckCommonErrors(TestParameters &tp,
                           timelib64_t startTime,
                           nvvsPluginResult_t &result,
                           std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos,
                           std::vector<DcgmError> &fatalErrors,
                           std::vector<DcgmError> &ignoredErrors);

    /*
     */
    long long DetermineMaxTemp(const dcgmDiagPluginEntityInfo_v1 &entityInfo);

    /*
     * Convert the dcgm return code to a string
     *
     * @return a string representing the DCGM return code
     */
    std::string ErrorAsString(dcgmReturn_t ret);


    /*
     * Checks the HBM error fields
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : if the temperature was ever too high
     */
    int CheckHBMErrorFields(dcgmDiagPluginEntityInfo_v1 const &entityInfo,
                            std::vector<DcgmError> &errors,
                            timelib64_t startTime);

    /*
     * Checks if the HBM temperature set by user is less than the max HBM temperature in the specified time period, and
     * adds an error message if the temperature was too high.
     *
     * @return:
     *
     * DR_SUCCESS     : on success
     * DR_COMM_ERROR  : if we couldn't get the information from DCGM
     * DR_VIOLATION   : if the temperature was ever too high
     */
    int VerifyHBMTemperature(unsigned int gpuId,
                             std::vector<DcgmError> &errorList,
                             long long maxTemp,
                             timelib64_t startTime);

    /*
     * Set a custom frequency to watch fields
     *
     * @param watchFrequency - the watch frequency in microseconds
     */
    void SetWatchFrequency(long long watchFrequency);

    /*
     * Set ignore error codes; to be used when checking for errors
     *
     * @param ignoreErrorCodes - Per entity set of ignore error codes
     */
    void SetIgnoreErrorCodes(gpuIgnoreErrorCodeMap_t const &ignoreErrorCodes);

    /*
     * Add the given error to the error list vector if the error code is not part of
     * the ignoreErrorCode list. Otherwise, discard the error and set retCode to
     * DR_SUCCESS.
     *
     * @param[in] d - error to be added
     * @param[out] errorList - list to which error is added
     * @param[out] retCode - return code set to success if error is ignored
     */
    void AddOrClearError(DcgmError const &d,
                         std::vector<DcgmError> &fatalErrorList,
                         std::vector<DcgmError> &ignoredErrorList,
                         int &retCode);

private:
    std::vector<unsigned short> m_fieldIds;
    std::vector<unsigned int> m_gpuIds;
    dcgmFieldGrp_t m_fieldGroupId;
    std::map<std::string, std::map<std::string, std::string>> m_groupSingleData;

    DcgmHandle m_dcgmHandle;
    DcgmGroup m_dcgmGroup;
    DcgmSystem m_dcgmSystem;

    DcgmValuesSinceHolder m_valuesHolder;

    long long m_nextValuesSinceTs;
    long long m_watchFrequency;

    CustomStatHolder m_customStatHolder;
    gpuIgnoreErrorCodeMap_t m_ignoreErrorCodes;

    /*
     * Helper method to get the watched fields as a string
     */
    std::string GetWatchedFieldsAsString(std::string &output, long long ts);

    /*
     * Helper method to log field violation errors
     */
    void AddFieldViolationError(unsigned short fieldId,
                                unsigned int gpuId,
                                timelib64_t startTime,
                                int64_t intValue,
                                double dblValue,
                                const std::string &fieldName,
                                std::vector<DcgmError> &fatalErrorList,
                                std::vector<DcgmError> &ignoredErrorList,
                                int &st);

    void AddFieldThresholdViolationError(unsigned short fieldId,
                                         unsigned int gpuId,
                                         int64_t intValue,
                                         int64_t thresholdValue,
                                         const std::string &fieldName,
                                         std::vector<DcgmError> &fatalErrorList,
                                         std::vector<DcgmError> &ignoredErrorList,
                                         int &st);

    /*
     * Helper method to get the watched fields as a json object
     */
    std::string GetWatchedFieldsAsJson(Json::Value &jv, long long ts);

    /*
     * Helper method to create a group in DCGM
     */
    dcgmReturn_t CreateGroup(const std::vector<unsigned int> &gpuIds, bool allGpus, const std::string &groupName);

    /*
     * Helper method to get an error string from a dcgmReturn_t
     */
    void GetErrorString(dcgmReturn_t ret, std::string &err);

    int CheckXIDs(unsigned int gpuId,
                  timelib64_t startTime,
                  std::vector<DcgmError> &errorList,
                  std::vector<DcgmError> &ignoredErrorList);

    /**
     * Check only the fields specified in `fieldIds`. See other overloads for additional checks.
     *
     * @return:
     *
     * DR_SUCCESS      : on success
     * DR_COMM_ERROR   : if we couldn't get the information from DCGM
     * DR_VIOLATION    : if a value was found above a failure threshold
     */
    int CheckErrorFields(std::vector<unsigned short> &fieldIds,
                         const std::vector<dcgmTimeseriesInfo_t> *failureThresholds,
                         unsigned int gpuId,
                         std::vector<DcgmError> &fatalErrorList,
                         std::vector<DcgmError> &ignoredErrorList,
                         timelib64_t startTime);

    /**
     * Check and report on non-fatal conditions.
     */
    void CheckNonFatalErrors(timelib64_t startTime,
                             dcgmGroupEntityPair_t const entityInfo,
                             std::vector<DcgmError> &fatalErrors,
                             std::vector<DcgmError> &ignoredErrors);

protected:
    /**
     * Formats the DcgmError according to the field ID for the violation.
     * A few fields have specific error codes, others use the generic message.
     *
     * @param d - the DcgmError being formatted
     * @param fieldId - the field ID for the violation
     * @param gpuId - the GPU that has this violation
     * @param intValue - either DCGM_INT64_BLANK (this field has fp64 values) or the value flagged as a violation.
     * @param dblValue - either DCGM_FP64_BLANK (this field uses int64 values) or the value flagged as a violation.
     * @param fieldName - the field's name
     */
    void FormatFieldViolationError(DcgmError &d,
                                   unsigned short fieldId,
                                   unsigned int gpuId,
                                   timelib64_t startTime,
                                   int64_t intValue,
                                   double dblValue,
                                   const std::string &fieldName);
};


#endif
