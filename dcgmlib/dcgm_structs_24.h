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
#ifndef DCGM_STRUCTS_24__H
#define DCGM_STRUCTS_24__H

#include "dcgm_structs.h"

/**
 * Maximum number of metric ID groups that can exist in DCGM
 */
#define DCGM_PROF_MAX_NUM_GROUPS_V1 10

/**
 * Maximum number of field IDs that can be in a single DCGM profiling metric group
 */
#define DCGM_PROF_MAX_FIELD_IDS_PER_GROUP_V1 8

/**
 * Structure to return all of the profiling metric groups that are available for the given groupId.
 */
typedef struct
{
    unsigned short majorId;   //!< Major ID of this metric group. Metric groups with the same majorId cannot be
                              //!< watched concurrently with other metric groups with the same majorId
    unsigned short minorId;   //!< Minor ID of this metric group. This distinguishes metric groups within the same
                              //!< major metric group from each other
    unsigned int numFieldIds; //!< Number of field IDs that are populated in fieldIds[]
    unsigned short fieldIds[DCGM_PROF_MAX_FIELD_IDS_PER_GROUP_V1]; //!< DCGM Field IDs that are part of this profiling
                                                                   //!< group. See DCGM_FI_PROF_* definitions in
                                                                   //!< dcgm_fields.h for details.
} dcgmProfMetricGroupInfo_v1;

typedef struct
{
    /** \name Input parameters
     * @{
     */
    unsigned int version; //!< Version of this request. Should be dcgmProfGetMetricGroups_version
    unsigned int unused;  //!< Not used for now. Set to 0
    dcgmGpuGrp_t groupId; //!< Group of GPUs we should get the metric groups for. These must all be the
                          //!< exact same GPU or DCGM_ST_GROUP_INCOMPATIBLE will be returned
    /**
     * @}
     */

    /** \name Output
     * @{
     */
    unsigned int numMetricGroups; //!< Number of entries in metricGroups[] that are populated
    unsigned int unused1;         //!< Not used for now. Set to 0
    dcgmProfMetricGroupInfo_v1 metricGroups[DCGM_PROF_MAX_NUM_GROUPS_V1]; //!< Info for each metric group
    /**
     * @}
     */
} dcgmProfGetMetricGroups_v2;

/**
 * Version 1 of dcgmProfGetMetricGroups_t
 */
#define dcgmProfGetMetricGroups_version2 MAKE_DCGM_VERSION(dcgmProfGetMetricGroups_v2, 2)

/**
 * Structure to pass to dcgmProfWatchFields() when watching profiling metrics
 */
typedef struct
{
    unsigned int version;        //!< Version of this request. Should be dcgmProfWatchFields_version
    dcgmGpuGrp_t groupId;        //!< Group ID representing collection of one or more GPUs. Look at \ref dcgmGroupCreate
                                 //!< for details on creating the group. Alternatively, pass in the group id as \a
                                 //!< DCGM_GROUP_ALL_GPUS to perform operation on all the GPUs. The GPUs of the group
                                 //!< must all be identical or DCGM_ST_GROUP_INCOMPATIBLE will be returned by this API.
    unsigned int numFieldIds;    //!< Number of field IDs that are being passed in fieldIds[]
    unsigned short fieldIds[16]; //!< DCGM_FI_PROF_? field IDs to watch
    long long updateFreq;        //!< How often to update this field in usec. Note that profiling metrics may need to be
                                 //!< sampled more frequently than this value. See
                                 //!< dcgmProfMetricGroupInfo_t.minUpdateFreqUsec of the metric group matching
                                 //!< metricGroupTag to see what this minimum is. If minUpdateFreqUsec < updateFreq
                                 //!< then samples will be aggregated to updateFreq intervals in DCGM's internal cache.
    double maxKeepAge;           //!< How long to keep data for every fieldId in seconds
    int maxKeepSamples;          //!< Maximum number of samples to keep for each fieldId. 0=no limit
    unsigned int flags;          //!< For future use. Set to 0 for now.
} dcgmProfWatchFields_v1;

/**
 * Version 1 of dcgmProfWatchFields_v1
 */
#define dcgmProfWatchFields_version1 MAKE_DCGM_VERSION(dcgmProfWatchFields_v1, 1)

/**
 * Structure to store the GPU hierarchy for a system
 *
 * Added in DCGM 2.0
 */
typedef struct
{
    unsigned int version;
    unsigned int count;
    dcgmMigHierarchyInfo_t entityList[DCGM_MAX_HIERARCHY_INFO];
} dcgmMigHierarchy_v1;

#define dcgmMigHierarchy_version1 MAKE_DCGM_VERSION(dcgmMigHierarchy_v1, 1)

typedef struct
{
    unsigned int version;
    unsigned int persistenceModeEnabled;
    unsigned int migModeEnabled;
} dcgmDeviceSettings_v1;

#define dcgmDevicesSettings_version1 MAKE_DCGM_VERSION(dcgmDeviceSettings_v1, 1)

/**
 * Represents attributes corresponding to a device
 */
typedef struct
{
    unsigned int version;                     //!< Version number (dcgmDeviceAttributes_version)
    dcgmDeviceSupportedClockSets_t clockSets; //!< Supported clocks for the device
    dcgmDeviceThermals_t thermalSettings;     //!< Thermal settings for the device
    dcgmDevicePowerLimits_t powerLimits;      //!< Various power limits for the device
    dcgmDeviceIdentifiers_t identifiers;      //!< Identifiers for the device
    dcgmDeviceMemoryUsage_t memoryUsage;      //!< Memory usage info for the device
    char unused[208];                         //!< Unused Space. Set to 0 for now
} dcgmDeviceAttributes_v1;

/**
 * Version 1 for \ref dcgmDeviceAttributes_v1
 */
#define dcgmDeviceAttributes_version1 MAKE_DCGM_VERSION(dcgmDeviceAttributes_v1, 1)
DCGM_CASSERT(dcgmDeviceAttributes_version1 == (long)16782628, 1);

typedef struct
{
    unsigned int version;                     //!< Version number (dcgmDeviceAttributes_version)
    dcgmDeviceSupportedClockSets_t clockSets; //!< Supported clocks for the device
    dcgmDeviceThermals_t thermalSettings;     //!< Thermal settings for the device
    dcgmDevicePowerLimits_t powerLimits;      //!< Various power limits for the device
    dcgmDeviceIdentifiers_t identifiers;      //!< Identifiers for the device
    dcgmDeviceMemoryUsage_t memoryUsage;      //!< Memory usage info for the device
    dcgmDeviceSettings_v1 settings;           //!< Basic device settings
} dcgmDeviceAttributes_v2;

/**
 * Version 2 for \ref dcgmDeviceAttributes_v2
 */
#define dcgmDeviceAttributes_version2 MAKE_DCGM_VERSION(dcgmDeviceAttributes_v2, 2)
DCGM_CASSERT(dcgmDeviceAttributes_version2 == (long)33559648, 1);

/**
 * Number of NvLink links per NvSwitch supported by DCGM pre-Hopper
 */
#define DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH_V1 36

/**
 * State of NvLink links for a NvSwitch
 */
typedef struct
{
    dcgm_field_eid_t entityId;                                              //!< Entity ID of the NvSwitch (physicalId)
    dcgmNvLinkLinkState_t linkState[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH_V1]; //!< Per-NvSwitch link states
} dcgmNvLinkNvSwitchLinkStatus_v1;

typedef struct
{
    unsigned int version; //!< Version of this request. Should be dcgmNvLinkStatus_version1
    unsigned int numGpus; //!< Number of entries in gpus[] that are populated
    dcgmNvLinkGpuLinkStatus_v1 gpus[DCGM_MAX_NUM_DEVICES]; //!< Per-GPU NvLink link statuses
    unsigned int numNvSwitches;                            //!< Number of entries in nvSwitches[] that are populated
    dcgmNvLinkNvSwitchLinkStatus_v1 nvSwitches[DCGM_MAX_NUM_SWITCHES]; //!< Per-NvSwitch link statuses
} dcgmNvLinkStatus_v1;

/**
 * Version 1 of dcgmNvLinkStatus
 */
#define dcgmNvLinkStatus_version1 MAKE_DCGM_VERSION(dcgmNvLinkStatus_v1, 1)

typedef struct
{
    unsigned int version; //!< Version of this request. Should be dcgmNvLinkStatus_version1
    unsigned int numGpus; //!< Number of entries in gpus[] that are populated
    dcgmNvLinkGpuLinkStatus_v2 gpus[DCGM_MAX_NUM_DEVICES]; //!< Per-GPU NvLink link statuses
    unsigned int numNvSwitches;                            //!< Number of entries in nvSwitches[] that are populated
    dcgmNvLinkNvSwitchLinkStatus_v1 nvSwitches[DCGM_MAX_NUM_SWITCHES]; //!< Per-NvSwitch link statuses
} dcgmNvLinkStatus_v2;

typedef struct
{
    dcgmDiagResult_t status; //!< The result of the test
    char warning[1024];      //!< Warning returned from the test, if any
    char info[1024];         //!< Information details returned from the test, if any
} dcgmDiagTestResult_v1;

/**
 * Version 2 of dcgmNvLinkStatus
 */
#define dcgmNvLinkStatus_version2 MAKE_DCGM_VERSION(dcgmNvLinkStatus_v2, 2)

#define DCGM_PER_GPU_TEST_COUNT_V6 7

/**
 * Per GPU diagnostics result structure
 */
typedef struct
{
    unsigned int gpuId;                                        //!< ID for the GPU this information pertains
    unsigned int hwDiagnosticReturn;                           //!< Per GPU hardware diagnostic test return code
    dcgmDiagTestResult_v2 results[DCGM_PER_GPU_TEST_COUNT_V6]; //!< Array with a result for each per-gpu test
} dcgmDiagResponsePerGpu_v2;

/**
 * Global diagnostics result structure v6
 *
 * Since DCGM 2.0
 */
typedef struct
{
    unsigned int version;           //!< version number (dcgmDiagResult_version)
    unsigned int gpuCount;          //!< number of valid per GPU results
    unsigned int levelOneTestCount; //!< number of valid levelOne results

    dcgmDiagTestResult_v2 levelOneResults[LEVEL_ONE_MAX_RESULTS];    //!< Basic, system-wide test results.
    dcgmDiagResponsePerGpu_v2 perGpuResponses[DCGM_MAX_NUM_DEVICES]; //!< per GPU test results
    dcgmDiagErrorDetail_t systemError;                               //!< System-wide error reported from NVVS
    char _unused[1024];                                              //!< No longer used
} dcgmDiagResponse_v6;

/**
 * Version 6 for \ref dcgmDiagResponse_v6
 */
#define dcgmDiagResponse_version6 MAKE_DCGM_VERSION(dcgmDiagResponse_v6, 6)
DCGM_CASSERT(dcgmDiagResponse_version6 == (long)0x06079090, 1);

/**
 * Identifies a level to retrieve field introspection info for
 */
typedef enum dcgmIntrospectLevel_enum
{
    DCGM_INTROSPECT_LVL_INVALID     = 0, //!< Invalid value
    DCGM_INTROSPECT_LVL_FIELD       = 1, //!< Introspection data is grouped by field ID
    DCGM_INTROSPECT_LVL_FIELD_GROUP = 2, //!< Introspection data is grouped by field group
    DCGM_INTROSPECT_LVL_ALL_FIELDS,      //!< Introspection data is aggregated for all fields
} dcgmIntrospectLevel_t;

/**
 * Identifies the retrieval context for introspection API calls.
 */
typedef struct
{
    unsigned int version;                //!< version number (dcgmIntrospectContext_version)
    dcgmIntrospectLevel_t introspectLvl; //!< Introspect Level \ref dcgmIntrospectLevel_t
    union
    {
        dcgmGpuGrp_t fieldGroupId;    //!< Only needed if \ref introspectLvl is DCGM_INTROSPECT_LVL_FIELD_GROUP
        unsigned short fieldId;       //!< Only needed if \ref introspectLvl is DCGM_INTROSPECT_LVL_FIELD
        unsigned long long contextId; //!< Overloaded way to access both fieldGroupId and fieldId
    };
} dcgmIntrospectContext_v1;

/**
 * Typedef for \ref dcgmIntrospectContext_v1
 */
typedef dcgmIntrospectContext_v1 dcgmIntrospectContext_t;

/**
 * Version 1 for \ref dcgmIntrospectContext_t
 */
#define dcgmIntrospectContext_version1 MAKE_DCGM_VERSION(dcgmIntrospectContext_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectContext_t
 */
#define dcgmIntrospectContext_version dcgmIntrospectContext_version1

/**
 * DCGM Execution time info for a set of fields
 */
typedef struct
{
    unsigned int version; //!< version number (dcgmIntrospectFieldsExecTime_version)

    long long meanUpdateFreqUsec; //!< the mean update frequency of all fields

    double recentUpdateUsec; //!< the sum of every field's most recent execution time after they
                             //!< have been normalized to \ref meanUpdateFreqUsec".
                             //!< This is roughly how long it takes to update fields every \ref meanUpdateFreqUsec

    long long totalEverUpdateUsec; //!< The total amount of time, ever, that has been spent updating all the fields
} dcgmIntrospectFieldsExecTime_v1;

/**
 * Typedef for \ref dcgmIntrospectFieldsExecTime_t
 */
typedef dcgmIntrospectFieldsExecTime_v1 dcgmIntrospectFieldsExecTime_t;

/**
 * Version 1 for \ref dcgmIntrospectFieldsExecTime_t
 */
#define dcgmIntrospectFieldsExecTime_version1 MAKE_DCGM_VERSION(dcgmIntrospectFieldsExecTime_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectFieldsExecTime_t
 */
#define dcgmIntrospectFieldsExecTime_version dcgmIntrospectFieldsExecTime_version1

/**
 * Full introspection info for field execution time
 *
 * Since DCGM 2.0
 */
typedef struct
{
    unsigned int version; //!< version number (dcgmIntrospectFullFieldsExecTime_version)

    dcgmIntrospectFieldsExecTime_v1 aggregateInfo; //!< info that includes global and device scope

    int hasGlobalInfo;                          //!< 0 means \ref globalInfo is populated, !0 means it's not
    dcgmIntrospectFieldsExecTime_v1 globalInfo; //!< info that only includes global field scope

    unsigned short gpuInfoCount;                         //!< count of how many entries in \ref gpuInfo are populated
    unsigned int gpuIdsForGpuInfo[DCGM_MAX_NUM_DEVICES]; //!< the GPU ID at a given index identifies which gpu
                                                         //!< the corresponding entry in \ref gpuInfo is from

    dcgmIntrospectFieldsExecTime_v1 gpuInfo[DCGM_MAX_NUM_DEVICES]; //!< info that is separated by the
                                                                   //!< GPU ID that the watches were for
} dcgmIntrospectFullFieldsExecTime_v2;

/**
 * typedef for \ref dcgmIntrospectFullFieldsExecTime_v1
 */
typedef dcgmIntrospectFullFieldsExecTime_v2 dcgmIntrospectFullFieldsExecTime_t;

/**
 * Version 1 for \ref dcgmIntrospectFullFieldsExecTime_t
 */
#define dcgmIntrospectFullFieldsExecTime_version2 MAKE_DCGM_VERSION(dcgmIntrospectFullFieldsExecTime_v2, 2)

/**
 * Latest version for \ref dcgmIntrospectFullFieldsExecTime_t
 */
#define dcgmIntrospectFullFieldsExecTime_version dcgmIntrospectFullFieldsExecTime_version2

/**
 * State of DCGM metadata gathering.  If it is set to DISABLED then "Metadata" API
 * calls to DCGM are not supported.
 */
typedef enum dcgmIntrospectState_enum
{
    DCGM_INTROSPECT_STATE_DISABLED = 0,
    DCGM_INTROSPECT_STATE_ENABLED  = 1
} dcgmIntrospectState_t;

/**
 * Full introspection info for field memory
 */
typedef struct
{
    unsigned int version; //!< version number (dcgmIntrospectFullMemory_version)

    dcgmIntrospectMemory_v1 aggregateInfo; //!< info that includes global and device scope

    int hasGlobalInfo;                  //!< 0 means \ref globalInfo is populated, !0 means it's not
    dcgmIntrospectMemory_v1 globalInfo; //!< info that only includes global field scope

    unsigned short gpuInfoCount;                         //!< count of how many entries in \ref gpuInfo are populated
    unsigned int gpuIdsForGpuInfo[DCGM_MAX_NUM_DEVICES]; //!< the GPU ID at a given index identifies which gpu
                                                         //!< the corresponding entry in \ref gpuInfo is from

    dcgmIntrospectMemory_v1 gpuInfo[DCGM_MAX_NUM_DEVICES]; //!< info that is divided by the
                                                           //!< GPU ID that the watches were for
} dcgmIntrospectFullMemory_v1;

/**
 * typedef for \ref dcgmIntrospectFullMemory_v1
 */
typedef dcgmIntrospectFullMemory_v1 dcgmIntrospectFullMemory_t;

/**
 * Version 1 for \ref dcgmIntrospectFullMemory_t
 */
#define dcgmIntrospectFullMemory_version1 MAKE_DCGM_VERSION(dcgmIntrospectFullMemory_v1, 1)

/**
 * Latest version for \ref dcgmIntrospectFullMemory_t
 */
#define dcgmIntrospectFullMemory_version dcgmIntrospectFullMemory_version1

DCGM_CASSERT(dcgmIntrospectFieldsExecTime_version == (long)16777248, 1);
DCGM_CASSERT(dcgmIntrospectFullFieldsExecTime_version == (long)0x020004D8, 1);
DCGM_CASSERT(dcgmIntrospectContext_version == (long)16777232, 1);

#endif // DCGM_STRUCTS_24__H