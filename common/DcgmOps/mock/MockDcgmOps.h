/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmGroupEntityPairHelpers.h>
#include <DcgmOps.h>

#include <array>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DcgmNs
{

/**
 * Mock DCGM entity with injectable field values and attributes.
 *
 * MockDcgmOps uses this type to represent entities such as GPUs, MIG instances,
 * switches, CPUs, and other DCGM entity groups in unit tests.
 */
class MockDcgmEntity
{
public:
    explicit MockDcgmEntity(dcgmGroupEntityPair_t const &entity);
    MockDcgmEntity(MockDcgmEntity const &other)            = default;
    MockDcgmEntity &operator=(const MockDcgmEntity &other) = default;

    ~MockDcgmEntity() = default;

    dcgmGroupEntityPair_t GetEntity() const;

    dcgmFieldValue_v2 GetFieldValue(unsigned short const fieldId) const;
    void InjectFieldValue(unsigned short const fieldId, dcgmFieldValue_v2 const &val);

    void SetCpuAffinityMask(std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &cpuAffinityMask);
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &GetCpuAffinityMask() const;

    void SetDevAttr(dcgmDeviceAttributes_t const &devAttr);
    dcgmDeviceAttributes_t const &GetDevAttr() const;

private:
    dcgmGroupEntityPair_t m_entity;
    std::unordered_map<unsigned short, dcgmFieldValue_v2> m_fieldValues;
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> m_cpuAffinityMask;
    dcgmDeviceAttributes_t m_devAttr;
};

/**
 * Mock field-group definition tracked by MockDcgmOps.
 */
struct MockDcgmFieldGroup
{
    std::string name;
    std::vector<unsigned short> fields;
};

/**
 * In-memory test double for the DcgmOps-compatible interface.
 *
 * Tests can configure the public state structures to control return codes,
 * output data, and recorded call information without connecting to a real
 * hostengine.
 */
class MockDcgmOps final
{
public:
    struct HostengineState
    {
        dcgmReturn_t versionReturn         = DCGM_ST_OK;
        dcgmReturn_t environmentReturn     = DCGM_ST_OK;
        dcgmReturn_t environmentResult     = DCGM_ST_OK;
        dcgmReturn_t disconnectReturn      = DCGM_ST_OK;
        dcgmReturn_t startEmbeddedReturn   = DCGM_ST_OK;
        dcgmReturn_t stopEmbeddedReturn    = DCGM_ST_OK;
        dcgmReturn_t pauseTelemetryReturn  = DCGM_ST_OK;
        dcgmReturn_t resumeTelemetryReturn = DCGM_ST_OK;
        dcgmReturn_t setLoggingReturn      = DCGM_ST_OK;
        dcgmReturn_t attachReturn          = DCGM_ST_OK;
        dcgmReturn_t detachReturn          = DCGM_ST_OK;
        int versionCallCount               = 0;
        int environmentCallCount           = 0;
        int disconnectCallCount            = 0;
        int startEmbeddedCallCount         = 0;
        int stopEmbeddedCallCount          = 0;
        int pauseTelemetryCallCount        = 0;
        int resumeTelemetryCallCount       = 0;
        int setLoggingCallCount            = 0;
        int attachCallCount                = 0;
        int detachCallCount                = 0;
        std::function<void()> shutdownCallback;
        dcgmHandle_t lastVersionHandle                = 0;
        dcgmHandle_t lastEnvironmentHandle            = 0;
        dcgmHandle_t lastDisconnectedHandle           = 0;
        dcgmHandle_t lastStoppedEmbeddedHandle        = 0;
        dcgmHandle_t lastPausedTelemetryHandle        = 0;
        dcgmHandle_t lastResumedTelemetryHandle       = 0;
        dcgmHandle_t lastSettingsHandle               = 0;
        dcgmOperationMode_t lastEmbeddedOperationMode = DCGM_OPERATION_MODE_AUTO;
        dcgmHandle_t embeddedHandle                   = 0;
        unsigned int lastVersionVersion               = 0;
        std::string rawBuildInfo;
        std::string environmentValue;
        std::string lastEnvironmentName;
        dcgmSettingsSetLoggingSeverity_t lastLogging {};
    };

    struct StatusState
    {
        dcgmReturn_t createReturn  = DCGM_ST_OK;
        dcgmReturn_t destroyReturn = DCGM_ST_OK;
        int createCallCount        = 0;
        int destroyCallCount       = 0;
        std::optional<dcgmStatus_t> handle;
        dcgmStatus_t createdHandle       = 0;
        dcgmStatus_t lastDestroyedStatus = 0;
    };

    struct DiscoveryState
    {
        dcgmReturn_t allDevicesReturn           = DCGM_ST_OK;
        dcgmReturn_t allSupportedDevicesReturn  = DCGM_ST_OK;
        dcgmReturn_t chipArchitectureReturn     = DCGM_ST_OK;
        dcgmReturn_t nvLinkStatusReturn         = DCGM_ST_OK;
        int allDevicesCallCount                 = 0;
        int allSupportedDevicesCallCount        = 0;
        int chipArchitectureCallCount           = 0;
        int nvLinkStatusCallCount               = 0;
        dcgmHandle_t lastHandle                 = 0;
        unsigned int lastGpuId                  = 0;
        dcgmChipArchitecture_t chipArchitecture = DCGM_CHIP_ARCH_HOPPER;
        std::map<unsigned int, dcgmChipArchitecture_t> chipArchitecturesByGpuId;
        std::vector<unsigned int> allDevices;
        std::vector<unsigned int> allSupportedDevices;
        dcgmNvLinkStatus_v5 nvLinkStatus {};
    };

    struct QueryState
    {
        dcgmReturn_t gpuStatusReturn    = DCGM_ST_OK;
        dcgmReturn_t cpuHierarchyReturn = DCGM_ST_OK;
        dcgmReturn_t workloadReturn     = DCGM_ST_OK;
        int gpuStatusCallCount          = 0;
        int cpuHierarchyCallCount       = 0;
        int workloadCallCount           = 0;
        dcgmHandle_t lastHandle         = 0;
        unsigned int lastGpuId          = 0;
        DcgmEntityStatus_t gpuStatus    = DcgmEntityStatusOk;
        dcgmCpuHierarchy_v2 cpuHierarchy {};
        dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
        dcgmDeviceWorkloadPowerProfilesStatus_v1 profileStatus {};
    };

    struct MigHierarchyState
    {
        using Handler = std::function<dcgmReturn_t(dcgmHandle_t, dcgmMigHierarchy_v2 *)>;

        dcgmReturn_t returnCode = DCGM_ST_OK;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        dcgmMigHierarchy_v2 hierarchy {};
        Handler handler;
    };

    struct EntitiesGetLatestValuesState
    {
        using Handler = std::function<dcgmReturn_t(dcgmHandle_t,
                                                   dcgmGroupEntityPair_t *,
                                                   unsigned int,
                                                   unsigned short *,
                                                   unsigned int,
                                                   unsigned int,
                                                   dcgmFieldValue_v2 *)>;

        dcgmReturn_t returnCode = DCGM_ST_OK;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        unsigned int lastFlags  = 0;
        std::vector<dcgmGroupEntityPair_t> lastEntities;
        std::vector<unsigned short> lastFields;
        std::optional<std::vector<dcgmFieldValue_v2>> values;
        std::map<unsigned short, dcgmReturn_t> returnByFieldId;
        Handler handler;
    };

    struct GroupManagementState
    {
        dcgmReturn_t getAllReturn = DCGM_ST_OK;
        std::optional<dcgmReturn_t> createReturn;
        std::optional<dcgmReturn_t> destroyReturn;
        std::optional<dcgmReturn_t> addReturn;
        dcgmReturn_t removeReturn     = DCGM_ST_OK;
        int getAllCallCount           = 0;
        int createCallCount           = 0;
        int destroyCallCount          = 0;
        int addCallCount              = 0;
        int removeCallCount           = 0;
        dcgmHandle_t lastHandle       = 0;
        dcgmGpuGrp_t lastGroupId      = 0;
        dcgmGroupType_t lastGroupType = DCGM_GROUP_EMPTY;
        std::string lastGroupName;
        dcgm_field_entity_group_t lastEntityGroup = DCGM_FE_NONE;
        dcgm_field_eid_t lastEntityId             = 0;
        dcgmGpuGrp_t newGroupId                   = 42;
        std::vector<dcgmGpuGrp_t> groupIds;
    };

    struct GroupInfoState
    {
        dcgmReturn_t returnCode  = DCGM_ST_OK;
        int callCount            = 0;
        dcgmHandle_t lastHandle  = 0;
        dcgmGpuGrp_t lastGroupId = 0;
        std::optional<dcgmGroupInfo_t> overrideInfo;
    };

    struct FieldGroupState
    {
        dcgmReturn_t createReturn                = DCGM_ST_OK;
        dcgmReturn_t destroyReturn               = DCGM_ST_OK;
        dcgmReturn_t getInfoReturn               = DCGM_ST_OK;
        dcgmReturn_t getAllReturn                = DCGM_ST_OK;
        dcgmReturn_t watchReturn                 = DCGM_ST_OK;
        dcgmReturn_t unwatchReturn               = DCGM_ST_OK;
        int createCallCount                      = 0;
        int destroyCallCount                     = 0;
        int getInfoCallCount                     = 0;
        int getAllCallCount                      = 0;
        int watchCallCount                       = 0;
        int unwatchCallCount                     = 0;
        dcgmHandle_t lastHandle                  = 0;
        dcgmFieldGrp_t nextFieldGroupId          = 0;
        dcgmFieldGrp_t lastDestroyedFieldGroupId = 0;
        dcgmFieldGrp_t lastRequestedInfoId       = 0;
        dcgmGpuGrp_t lastWatchedGroupId          = 0;
        dcgmFieldGrp_t lastWatchedFieldGroupId   = 0;
        dcgmGpuGrp_t lastUnwatchedGroupId        = 0;
        dcgmFieldGrp_t lastUnwatchedFieldGroupId = 0;
        long long lastUpdateFreq                 = 0;
        double lastMaxKeepAge                    = 0.0;
        int lastMaxKeepSamples                   = 0;
        std::vector<unsigned short> lastCreatedFieldIds;
        std::string lastCreatedName;
        std::unordered_map<dcgmFieldGrp_t, dcgmFieldGroupInfo_t> fieldGroupInfo;
        std::optional<dcgmAllFieldGroup_t> allFieldGroups;
    };

    struct UpdateAllFieldsState
    {
        std::optional<dcgmReturn_t> returnCode;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        int lastWaitForUpdate   = 0;
    };

    struct ValuesSinceState
    {
        dcgmReturn_t returnCode                 = DCGM_ST_OK;
        int valuesSinceV2CallCount              = 0;
        dcgmHandle_t lastHandle                 = 0;
        dcgmGpuGrp_t lastGroupId                = 0;
        dcgmFieldGrp_t lastFieldGroup           = 0;
        long long lastSinceTimestamp            = 0;
        long long nextSinceTimestamp            = 0;
        dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
        dcgm_field_eid_t entityId               = 0;
        std::optional<std::vector<dcgmFieldValue_v1>> values;
    };

    struct LatestValuesForFieldsState
    {
        dcgmReturn_t returnCode = DCGM_ST_OK;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        int lastGpuId           = 0;
        unsigned int lastCount  = 0;
        std::vector<unsigned short> lastFields;
        std::optional<std::vector<dcgmFieldValue_v1>> values;
    };

    struct MultipleValuesForFieldState
    {
        dcgmReturn_t returnCode    = DCGM_ST_OK;
        int callCount              = 0;
        dcgmHandle_t lastHandle    = 0;
        int lastGpuId              = 0;
        unsigned short lastFieldId = 0;
        long long lastStartTs      = 0;
        long long lastEndTs        = 0;
        dcgmOrder_t lastOrder      = DCGM_ORDER_ASCENDING;
        std::optional<std::vector<dcgmFieldValue_v1>> values;
    };

    struct FieldSummaryState
    {
        dcgmReturn_t returnCode = DCGM_ST_OK;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        dcgmFieldSummaryRequest_t lastRequest {};
        dcgmSummaryResponse_t response {};
        std::unordered_map<unsigned short, dcgmReturn_t> returnByFieldId;
        std::unordered_map<unsigned short, dcgmSummaryResponse_t> responseByFieldId;
    };

    struct TopologyState
    {
        dcgmReturn_t deviceReturn = DCGM_ST_OK;
        dcgmReturn_t groupReturn  = DCGM_ST_OK;
        int deviceCallCount       = 0;
        int groupCallCount        = 0;
        dcgmHandle_t lastHandle   = 0;
        unsigned int lastGpuId    = 0;
        dcgmGpuGrp_t lastGroupId  = 0;
        std::optional<dcgmDeviceTopology_t> deviceTopology;
        dcgmGroupTopology_t groupTopology {};
    };

    struct DeviceAttributesState
    {
        dcgmReturn_t returnCode = DCGM_ST_OK;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        unsigned int lastGpuId  = 0;
        std::optional<dcgmDeviceAttributes_t> overrideAttributes;
    };

    struct EntityListState
    {
        dcgmReturn_t returnCode                   = DCGM_ST_OK;
        int callCount                             = 0;
        dcgmHandle_t lastHandle                   = 0;
        dcgm_field_entity_group_t lastEntityGroup = DCGM_FE_NONE;
        unsigned int lastFlags                    = 0;
        std::optional<std::vector<dcgm_field_eid_t>> gpuEntities;
    };

    struct ConfigState
    {
        dcgmReturn_t setReturn          = DCGM_ST_OK;
        dcgmReturn_t currentGetReturn   = DCGM_ST_OK;
        dcgmReturn_t targetGetReturn    = DCGM_ST_OK;
        dcgmReturn_t workloadReturn     = DCGM_ST_OK;
        dcgmReturn_t enforceReturn      = DCGM_ST_OK;
        dcgmReturn_t vgpuSetReturn      = DCGM_ST_OK;
        int setCallCount                = 0;
        int getCallCount                = 0;
        int workloadCallCount           = 0;
        int enforceCallCount            = 0;
        int vgpuSetCallCount            = 0;
        dcgmHandle_t lastHandle         = 0;
        dcgmGpuGrp_t lastGroupId        = 0;
        dcgmStatus_t lastStatusHandle   = 0;
        dcgmConfigType_t lastConfigType = DCGM_CONFIG_CURRENT_STATE;
        int lastConfigCount             = 0;
        dcgmConfig_t lastSetConfig {};
        dcgmVgpuConfig_t lastVgpuSetConfig {};
        dcgmWorkloadPowerProfile_t lastWorkloadProfile {};
        std::vector<dcgmConfig_t> currentConfigs;
        std::vector<dcgmConfig_t> targetConfigs;
    };

    struct PolicyState
    {
        dcgmReturn_t setReturn        = DCGM_ST_OK;
        dcgmReturn_t getReturn        = DCGM_ST_OK;
        dcgmReturn_t registerReturn   = DCGM_ST_OK;
        dcgmReturn_t unregisterReturn = DCGM_ST_OK;
        int setCallCount              = 0;
        int getCallCount              = 0;
        int registerCallCount         = 0;
        int unregisterCallCount       = 0;
        dcgmHandle_t lastHandle       = 0;
        dcgmGpuGrp_t lastGroupId      = 0;
        dcgmStatus_t lastStatusHandle = 0;
        int lastPolicyGetCount        = 0;
        dcgmPolicyCondition_t lastCondition {};
        dcgmPolicyCondition_t lastRegisteredCondition {};
        fpRecvUpdates lastCallback = nullptr;
        uint64_t lastUserData      = 0;
        dcgmPolicy_t lastPolicy {};
        std::vector<dcgmPolicy_t> policies;
    };

    struct FieldPolicyState
    {
        dcgmReturn_t createReturn        = DCGM_ST_OK;
        dcgmReturn_t modifyReturn        = DCGM_ST_OK;
        dcgmReturn_t deleteReturn        = DCGM_ST_OK;
        dcgmReturn_t enableReturn        = DCGM_ST_OK;
        dcgmReturn_t disableReturn       = DCGM_ST_OK;
        dcgmReturn_t getOneReturn        = DCGM_ST_OK;
        dcgmReturn_t getAllReturn        = DCGM_ST_OK;
        dcgmReturn_t importReturn        = DCGM_ST_OK;
        dcgmReturn_t getViolationsReturn = DCGM_ST_OK;
        int createCallCount              = 0;
        int modifyCallCount              = 0;
        int deleteCallCount              = 0;
        int enableCallCount              = 0;
        int disableCallCount             = 0;
        int getOneCallCount              = 0;
        int getAllCallCount              = 0;
        int importCallCount              = 0;
        int getViolationsCallCount       = 0;
        dcgmHandle_t lastHandle          = 0;
        unsigned short lastFieldId       = 0;
        uint64_t lastPolicyId            = 0;
        unsigned int lastDeleteAll       = 0;
        int64_t lastSinceTimestamp       = 0;
        unsigned int lastRequestedCount  = 0;
        uint64_t nextPolicyId            = 1;
        std::string lastName;
        std::string lastImportPath;
        dcgmPolicyInfo_t lastModifiedPolicy {};
        std::vector<dcgmPolicyInfo_t> policies;
        std::vector<dcgm_policy_violation_t> violations;
    };

    struct HealthState
    {
        dcgmReturn_t getReturn      = DCGM_ST_OK;
        dcgmReturn_t setReturn      = DCGM_ST_OK;
        dcgmReturn_t checkReturn    = DCGM_ST_OK;
        int getCallCount            = 0;
        int setCallCount            = 0;
        int checkCallCount          = 0;
        dcgmHandle_t lastHandle     = 0;
        dcgmGpuGrp_t lastGroupId    = 0;
        dcgmHealthSystems_t systems = static_cast<dcgmHealthSystems_t>(0);
        dcgmHealthSetParams_v2 lastSetParams {};
        dcgmHealthResponse_t response {};
    };

    struct ProcessStatsState
    {
        dcgmReturn_t watchPidReturn  = DCGM_ST_OK;
        dcgmReturn_t getPidReturn    = DCGM_ST_OK;
        dcgmReturn_t startReturn     = DCGM_ST_OK;
        dcgmReturn_t stopReturn      = DCGM_ST_OK;
        dcgmReturn_t removeReturn    = DCGM_ST_OK;
        dcgmReturn_t removeAllReturn = DCGM_ST_OK;
        dcgmReturn_t getJobReturn    = DCGM_ST_OK;
        int watchPidCallCount        = 0;
        int getPidCallCount          = 0;
        int startCallCount           = 0;
        int stopCallCount            = 0;
        int removeCallCount          = 0;
        int removeAllCallCount       = 0;
        int getJobCallCount          = 0;
        dcgmHandle_t lastHandle      = 0;
        dcgmGpuGrp_t lastGroupId     = 0;
        long long lastUpdateFreq     = 0;
        double lastMaxKeepAge        = 0.0;
        int lastMaxKeepSamples       = 0;
        std::string lastJobId;
        dcgmPidInfo_t pidInfo {};
        dcgmJobInfo_t jobInfo {};
    };

    struct IntrospectState
    {
        dcgmReturn_t memoryReturn      = DCGM_ST_OK;
        dcgmReturn_t cpuReturn         = DCGM_ST_OK;
        int memoryCallCount            = 0;
        int cpuCallCount               = 0;
        dcgmHandle_t lastHandle        = 0;
        int lastMemoryWait             = 0;
        int lastCpuWait                = 0;
        unsigned int lastMemoryVersion = 0;
        unsigned int lastCpuVersion    = 0;
        dcgmIntrospectMemory_t memory {};
        dcgmIntrospectCpuUtil_t cpu {};
    };

    struct CacheManagerState
    {
        dcgmReturn_t fieldInfoReturn = DCGM_ST_OK;
        dcgmReturn_t injectReturn    = DCGM_ST_OK;
        int fieldInfoCallCount       = 0;
        int injectCallCount          = 0;
        dcgmHandle_t lastHandle      = 0;
        unsigned int lastGpuId       = 0;
        dcgmCacheManagerFieldInfo_v4_t requestedFieldInfo {};
        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        dcgmInjectFieldValue_t injectedValue {};
    };

    struct ModuleState
    {
        dcgmReturn_t denylistReturn   = DCGM_ST_OK;
        dcgmReturn_t statusesReturn   = DCGM_ST_OK;
        int denylistCallCount         = 0;
        int statusesCallCount         = 0;
        dcgmHandle_t lastHandle       = 0;
        dcgmModuleId_t lastDenylistId = DcgmModuleIdCount;
        dcgmModuleGetStatuses_t statuses {};
    };

    struct ConnectionInfo
    {
        std::string connectionString;
        dcgmConnectV3Params_t params {};
    };

    struct ConnectionState
    {
        dcgmReturn_t connectV3Return = DCGM_ST_OK;
        std::optional<dcgmReturn_t> disconnectReturn;
        int connectV3CallCount  = 0;
        int disconnectCallCount = 0;
        std::string lastConnectionString;
        dcgmConnectV3Params_t lastConnectParams {};
        dcgmHandle_t lastDisconnectedHandle = 0;
        bool useGeneratedConnectHandles     = false;
        dcgmHandle_t nextGeneratedHandle    = 1;
        std::map<dcgmHandle_t, ConnectionInfo> connectionsByHandle;
    };

    struct MultinodeRequestState
    {
        using Handler = std::function<dcgmReturn_t(dcgmHandle_t, dcgmMultinodeRequest_t *)>;

        dcgmReturn_t returnCode = DCGM_ST_NOT_SUPPORTED;
        int callCount           = 0;
        dcgmHandle_t lastHandle = 0;
        std::unordered_map<dcgmMultinodeRequestType_t, unsigned int> requestTypeCounts;
        std::unordered_map<dcgmMultinodeRequestType_t, Handler> requestTypeHandlers;
        std::map<std::pair<dcgmHandle_t, dcgmMultinodeRequestType_t>, Handler> handleRequestTypeHandlers;
    };

    // Stands in for the nvvs/diag response returned by the host engine, so that dcgmi diag argument
    // handling can be unit-tested without running an actual diagnostic. Tests set actionValidateReturn
    // to simulate the value that nvvs would have returned.
    struct ActionValidateState
    {
        dcgmReturn_t actionValidateReturn = DCGM_ST_OK;
        int actionValidateCallCount       = 0;
        dcgmHandle_t lastHandle           = 0;
    };

    struct MnDiagState
    {
        dcgmReturn_t runReturn                          = DCGM_ST_OK;
        dcgmReturn_t stopReturn                         = DCGM_ST_OK;
        int runCallCount                                = 0;
        int stopCallCount                               = 0;
        dcgmHandle_t lastRunHandle                      = 0;
        dcgmHandle_t lastStopHandle                     = 0;
        unsigned int lastRunVersion                     = 0;
        std::unique_ptr<dcgmMnDiagResponse_v2> response = std::make_unique<dcgmMnDiagResponse_v2>();
    };

    struct ProfileState
    {
        dcgmReturn_t metricGroupsReturn = DCGM_ST_OK;
        dcgmReturn_t pauseReturn        = DCGM_ST_OK;
        dcgmReturn_t resumeReturn       = DCGM_ST_OK;
        int metricGroupsCallCount       = 0;
        int pauseCallCount              = 0;
        int resumeCallCount             = 0;
        dcgmHandle_t lastHandle         = 0;
        unsigned int lastGpuId          = 0;
        dcgmProfGetMetricGroups_t metricGroups {};
    };

    MockDcgmOps();
    ~MockDcgmOps() = default;

    dcgmReturn_t dcgmInit();
    dcgmReturn_t dcgmShutdown();
    dcgmReturn_t dcgmConnect_v2(const char *ipAddress, dcgmConnectV2Params_t *connectParams, dcgmHandle_t *pDcgmHandle);
    dcgmReturn_t dcgmConnect_v3(const char *connectionString,
                                dcgmConnectV3Params_t *connectParams,
                                dcgmHandle_t *pDcgmHandle);
    dcgmReturn_t dcgmDisconnect(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *request);
    dcgmReturn_t dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle);
    dcgmReturn_t dcgmStopEmbedded(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmPauseTelemetryForDiag(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmResumeTelemetryForDiag(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmHostengineVersionInfo(dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo);
    dcgmReturn_t dcgmHostengineEnvironmentVariableInfo(dcgmHandle_t pDcgmHandle, dcgmEnvVarInfo_t *pEnvVarInfo);
    dcgmReturn_t dcgmHostengineSetLoggingSeverity(dcgmHandle_t pDcgmHandle, dcgmSettingsSetLoggingSeverity_t *logging);
    dcgmReturn_t dcgmAttachDriver(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmDetachDriver(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmStatusCreate(dcgmStatus_t *statusHandle);
    dcgmReturn_t dcgmStatusDestroy(dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                             dcgmGroupEntityPair_t entities[],
                                             unsigned int entityCount,
                                             unsigned short fields[],
                                             unsigned int fieldCount,
                                             unsigned int flags,
                                             dcgmFieldValue_v2 values[]);
    dcgmReturn_t dcgmGetGpuStatus(dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status);
    dcgmReturn_t dcgmGetAllDevices(dcgmHandle_t pDcgmHandle, unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES], int *count);
    dcgmReturn_t dcgmGetAllSupportedDevices(dcgmHandle_t pDcgmHandle,
                                            unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                            int *count);
    dcgmReturn_t dcgmGetGpuChipArchitecture(dcgmHandle_t dcgmHandle,
                                            unsigned int gpuId,
                                            dcgmChipArchitecture_t *chipArchitecture);
    dcgmReturn_t dcgmGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v5 *linkStatus);
    dcgmReturn_t dcgmGetDeviceWorkloadPowerProfileInfo(dcgmHandle_t pDcgmHandle,
                                                       unsigned int gpuId,
                                                       dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                       dcgmDeviceWorkloadPowerProfilesStatus_v1 *profileStatus);
    dcgmReturn_t dcgmGetGpuInstanceHierarchy(dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy);
    dcgmReturn_t dcgmGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy);
    dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                 dcgmGroupType_t type,
                                 const char *groupName,
                                 dcgmGpuGrp_t *pDcgmGrpId);
    dcgmReturn_t dcgmGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId);
    dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId);
    dcgmReturn_t dcgmGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                       dcgmGpuGrp_t groupId,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId);
    dcgmReturn_t dcgmGroupGetInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *pDcgmGroupInfo);
    dcgmReturn_t dcgmGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count);
    dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                      int numFieldIds,
                                      const unsigned short *fieldIds,
                                      const char *fieldGroupName,
                                      dcgmFieldGrp_t *dcgmFieldGroupId);
    dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId);
    dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId);
    dcgmReturn_t dcgmFieldGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo);
    dcgmReturn_t dcgmFieldGroupGetAll(dcgmHandle_t dcgmHandle, dcgmAllFieldGroup_t *allGroupInfo);
    dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmFieldGrp_t fieldGroupId,
                                 long long updateFreq,
                                 double maxKeepAge,
                                 int maxKeepSamples);
    dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId);
    dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate);
    dcgmReturn_t dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmFieldGrp_t fieldGroupId,
                                        dcgmFieldValueEntityEnumeration_f enumCB,
                                        void *userData);
    dcgmReturn_t dcgmGetValuesSince_v2(dcgmHandle_t pDcgmHandle,
                                       dcgmGpuGrp_t groupId,
                                       dcgmFieldGrp_t fieldGroupId,
                                       long long sinceTimestamp,
                                       long long *nextSinceTimestamp,
                                       dcgmFieldValueEntityEnumeration_f enumCB,
                                       void *userData);
    dcgmReturn_t dcgmGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request);
    dcgmReturn_t dcgmGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                               int gpuId,
                                               unsigned short fieldId,
                                               int *count,
                                               long long startTs,
                                               long long endTs,
                                               dcgmOrder_t order,
                                               dcgmFieldValue_v1 values[]);
    dcgmReturn_t dcgmGetLatestValuesForFields(dcgmHandle_t pDcgmHandle,
                                              int gpuId,
                                              unsigned short fields[],
                                              unsigned int count,
                                              dcgmFieldValue_v1 values[]);
    dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                       unsigned int gpuId,
                                       dcgmDeviceTopology_v2 *pDcgmDeviceTopology);
    dcgmReturn_t dcgmGetGroupTopology(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmGroupTopology_v2 *pDcgmGroupTopology);
    dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                         unsigned int gpuId,
                                         dcgmDeviceAttributes_t *pDcgmAttr);
    dcgmReturn_t dcgmGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                            dcgm_field_entity_group_t entityGroup,
                                            dcgm_field_eid_t *entities,
                                            int *numEntities,
                                            unsigned int flags);
    dcgmReturn_t dcgmConfigSet(dcgmHandle_t pDcgmHandle,
                               dcgmGpuGrp_t groupId,
                               dcgmConfig_t *deviceConfig,
                               dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmConfigGet(dcgmHandle_t pDcgmHandle,
                               dcgmGpuGrp_t groupId,
                               dcgmConfigType_t reqType,
                               int count,
                               dcgmConfig_t deviceConfigList[],
                               dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmConfigSetWorkloadPowerProfile(dcgmHandle_t pDcgmHandle,
                                                   dcgmWorkloadPowerProfile_t *workloadPowerProfile);
    dcgmReturn_t dcgmConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmPolicySet(dcgmHandle_t pDcgmHandle,
                               dcgmGpuGrp_t groupId,
                               dcgmPolicy_t *policy,
                               dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmPolicyGet(dcgmHandle_t pDcgmHandle,
                               dcgmGpuGrp_t groupId,
                               int count,
                               dcgmPolicy_t *policy,
                               dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmPolicyUnregister(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition);
    dcgmReturn_t dcgmPolicyCreate(dcgmHandle_t pDcgmHandle,
                                  unsigned short fieldId,
                                  const char *name,
                                  double threshold,
                                  dcgmPolicyOperator_t policyOperator,
                                  const dcgmGroupEntityPair_t *entities,
                                  unsigned int entityCount,
                                  unsigned int channelMask,
                                  unsigned int enabled,
                                  unsigned int rateLimitSec,
                                  uint64_t *policyId);
    dcgmReturn_t dcgmPolicyModify(dcgmHandle_t pDcgmHandle, const dcgmPolicyInfo_t *policy);
    dcgmReturn_t dcgmPolicyDelete(dcgmHandle_t pDcgmHandle,
                                  unsigned short fieldId,
                                  uint64_t policyId,
                                  unsigned int deleteAll);
    dcgmReturn_t dcgmPolicyEnable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId);
    dcgmReturn_t dcgmPolicyDisable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId);
    dcgmReturn_t dcgmPolicyGetOne(dcgmHandle_t pDcgmHandle,
                                  unsigned short fieldId,
                                  uint64_t policyId,
                                  dcgmPolicyInfo_t *policy);
    dcgmReturn_t dcgmPolicyGetAll(dcgmHandle_t pDcgmHandle, dcgmPolicyInfo_t *policies, unsigned int *count);
    dcgmReturn_t dcgmPolicyImport(dcgmHandle_t pDcgmHandle, const char *path);
    dcgmReturn_t dcgmPolicyGetViolations(dcgmHandle_t pDcgmHandle,
                                         int64_t sinceTimestamp,
                                         dcgm_policy_violation_t *violations,
                                         unsigned int *count);
    dcgmReturn_t dcgmHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems);
    dcgmReturn_t dcgmHealthSet_v2(dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params);
    dcgmReturn_t dcgmHealthCheck(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results);
    dcgmReturn_t dcgmWatchPidFields(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    long long updateFreq,
                                    double maxKeepAge,
                                    int maxKeepSamples);
    dcgmReturn_t dcgmGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo);
    dcgmReturn_t dcgmJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64]);
    dcgmReturn_t dcgmJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64]);
    dcgmReturn_t dcgmJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64]);
    dcgmReturn_t dcgmJobRemoveAll(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo);
    dcgmReturn_t dcgmIntrospectGetHostengineMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                        dcgmIntrospectMemory_t *memoryInfo,
                                                        int waitIfNoData);
    dcgmReturn_t dcgmIntrospectGetHostengineCpuUtilization(dcgmHandle_t pDcgmHandle,
                                                           dcgmIntrospectCpuUtil_t *cpuUtil,
                                                           int waitIfNoData);
    dcgmReturn_t dcgmModuleDenylist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId);
    dcgmReturn_t dcgmModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses);
    dcgmReturn_t dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *params);
    dcgmReturn_t dcgmEngineRun(unsigned short portNumber, char const *socketPath, unsigned int connectionType);
    dcgmReturn_t dcgmStatusPopError(dcgmStatus_t statusHandle, dcgmErrorInfo_t *errorInfo);
    dcgmReturn_t dcgmPolicyRegister_v2(dcgmHandle_t pDcgmHandle,
                                       dcgmGpuGrp_t groupId,
                                       dcgmPolicyCondition_t condition,
                                       fpRecvUpdates callback,
                                       uint64_t userData);
    dcgmReturn_t dcgmModuleIdToName(dcgmModuleId_t id, char const **name);
    dcgmReturn_t dcgmActionValidate_v2(dcgmHandle_t pDcgmHandle, dcgmRunDiag_v10 *drd, dcgmDiagResponse_v12 *response);
    dcgmReturn_t dcgmStopDiagnostic(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmRunMnDiagnostic(dcgmHandle_t pDcgmHandle,
                                     dcgmRunMnDiag_v2 const *drmnd,
                                     dcgmMnDiagResponse_v2 *response);
    dcgmReturn_t dcgmStopMnDiagnostic(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmDiagSendHeartbeat(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmProfGetSupportedMetricGroups(dcgmHandle_t pDcgmHandle, dcgmProfGetMetricGroups_t *metricGroups);
    dcgmReturn_t dcgmProfPause(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmProfResume(dcgmHandle_t pDcgmHandle);
    dcgmReturn_t dcgmGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle, dcgmCacheManagerFieldInfo_v4_t *fieldInfo);
    dcgmReturn_t dcgmInjectFieldValue(dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmInjectFieldValue_t *fieldValue);
    dcgmReturn_t dcgmVgpuConfigGet(dcgmHandle_t pDcgmHandle,
                                   dcgmGpuGrp_t groupId,
                                   dcgmConfigType_t type,
                                   int count,
                                   dcgmVgpuConfig_t *deviceConfigList,
                                   dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                   dcgmGpuGrp_t groupId,
                                   dcgmVgpuConfig_t *deviceConfig,
                                   dcgmStatus_t statusHandle);
    dcgmReturn_t dcgmGetVgpuDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                             unsigned int gpuId,
                                             dcgmVgpuDeviceAttributes_t *attributes);
    dcgmReturn_t dcgmGetVgpuInstanceAttributes(dcgmHandle_t pDcgmHandle,
                                               unsigned int vgpuId,
                                               dcgmVgpuInstanceAttributes_t *attributes);
    dcgmErrorSeverity_t dcgmErrorGetPriorityByCode(unsigned int code);
    void SetErrorPriorityByCode(unsigned int code, dcgmErrorSeverity_t severity);

    void SetConnectV3Result(dcgmReturn_t result);
    void UseGeneratedConnectHandles(dcgmHandle_t firstHandle = 1);
    int GetConnectV3CallCount() const;
    std::string GetLastConnectV3ConnectionString() const;
    dcgmConnectV3Params_t GetLastConnectV3Params() const;
    bool HasConnection(dcgmHandle_t handle) const;
    std::optional<ConnectionInfo> GetConnectionInfo(dcgmHandle_t handle) const;
    void SetDisconnectResult(dcgmReturn_t result);
    int GetDisconnectCallCount() const;

    void SetMultinodeRequestResult(dcgmReturn_t result);
    dcgmReturn_t GetMultinodeRequestResult() const;
    void SetMultinodeRequestHandler(dcgmMultinodeRequestType_t requestType, MultinodeRequestState::Handler handler);
    void SetMultinodeRequestHandler(dcgmHandle_t handle,
                                    dcgmMultinodeRequestType_t requestType,
                                    MultinodeRequestState::Handler handler);
    int GetMultinodeRequestCallCount() const;
    dcgmHandle_t GetLastMultinodeRequestHandle() const;
    unsigned int GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t requestType) const;

    void ResetApiState();

    static dcgmDeviceAttributes_t MakeDefaultDeviceAttributes();
    static dcgmCpuHierarchy_v2 MakeDefaultCpuHierarchy();
    static dcgmWorkloadPowerProfileProfilesInfo_v1 MakeDefaultWorkloadPowerProfileInfo();
    static dcgmDeviceWorkloadPowerProfilesStatus_v1 MakeDefaultWorkloadPowerProfileStatus();
    static dcgmGroupInfo_t MakeGroupInfo(std::string_view groupName,
                                         std::span<dcgmGroupEntityPair_t const> entities = {});
    static dcgmGroupInfo_t MakeGroupInfo(std::string_view groupName,
                                         std::initializer_list<dcgmGroupEntityPair_t> entities);
    void AddMockedEntity(MockDcgmEntity const &mockedEntity);

    dcgmHandle_t const m_handleId = 0xc8763;
    HostengineState m_hostEngineState;
    StatusState m_statusState;
    DiscoveryState m_discoveryState;
    QueryState m_queryState;
    MigHierarchyState m_migHierarchyState;
    EntitiesGetLatestValuesState m_entitiesGetLatestValuesState;
    GroupManagementState m_groupManagementState;
    GroupInfoState m_groupInfoState;
    FieldGroupState m_fieldGroupState;
    UpdateAllFieldsState m_updateAllFieldsState;
    ValuesSinceState m_valuesSinceState;
    LatestValuesForFieldsState m_latestValuesForFieldsState;
    MultipleValuesForFieldState m_multipleValuesForFieldState;
    FieldSummaryState m_fieldSummaryState;
    TopologyState m_topologyState;
    DeviceAttributesState m_deviceAttributesState;
    EntityListState m_entityListState;
    ConfigState m_configState;
    PolicyState m_policyState;
    FieldPolicyState m_fieldPolicyState;
    HealthState m_healthState;
    ProcessStatsState m_processStatsState;
    IntrospectState m_introspectState;
    CacheManagerState m_cacheManagerState;
    ModuleState m_moduleState;
    ConnectionState m_connectionState;
    MultinodeRequestState m_multinodeRequestState;
    ActionValidateState m_actionValidateState;
    MnDiagState m_mnDiagState;
    ProfileState m_profileState;

private:
    mutable std::mutex m_connectionMutex;
    mutable std::mutex m_multinodeRequestMutex;

    std::unordered_map<dcgmGroupEntityPair_t, MockDcgmEntity> m_entities;

    dcgmGpuGrp_t m_globalGroupId = 1;
    std::unordered_map<dcgmGpuGrp_t, dcgmGroupInfo_v3> m_groups;

    std::unordered_map<dcgmFieldGrp_t, MockDcgmFieldGroup> m_fieldGroups;
    std::unordered_map<unsigned int, dcgmErrorSeverity_t> m_errorPriorities;

    std::unordered_map<dcgmGroupEntityPair_t, std::unordered_map<dcgmGroupEntityPair_t, dcgmGpuTopologyLevel_t>>
        m_topology;
};

} //namespace DcgmNs
