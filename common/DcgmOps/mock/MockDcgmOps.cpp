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

#include "MockDcgmOps.h"

#include <DcgmStringHelpers.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace DcgmNs
{

dcgmDeviceAttributes_t MockDcgmOps::MakeDefaultDeviceAttributes()
{
    dcgmDeviceAttributes_t deviceAttributes {};
    deviceAttributes.version = dcgmDeviceAttributes_version;
    SafeCopyTo(deviceAttributes.identifiers.deviceName, "Mock GPU");
    SafeCopyTo(deviceAttributes.identifiers.brandName, "MockBrand");
    SafeCopyTo(deviceAttributes.identifiers.pciBusId, "0000:01:00.0");
    SafeCopyTo(deviceAttributes.identifiers.uuid, "GPU-mock-uuid");
    SafeCopyTo(deviceAttributes.identifiers.serial, "serial-0");
    SafeCopyTo(deviceAttributes.identifiers.inforomImageVersion, "inforom-1");
    SafeCopyTo(deviceAttributes.identifiers.vbios, "vbios-1");
    deviceAttributes.powerLimits.curPowerLimit      = 250;
    deviceAttributes.powerLimits.defaultPowerLimit  = 240;
    deviceAttributes.powerLimits.maxPowerLimit      = 300;
    deviceAttributes.powerLimits.minPowerLimit      = 150;
    deviceAttributes.powerLimits.enforcedPowerLimit = 245;
    deviceAttributes.thermalSettings.shutdownTemp   = 95;
    deviceAttributes.thermalSettings.slowdownTemp   = 90;
    return deviceAttributes;
}

dcgmCpuHierarchy_v2 MockDcgmOps::MakeDefaultCpuHierarchy()
{
    dcgmCpuHierarchy_v2 cpuHierarchy {};
    cpuHierarchy.version = dcgmCpuHierarchy_version2;
    cpuHierarchy.numCpus = 1;
    SafeCopyTo(cpuHierarchy.cpus[0].serial, "cpu-serial-0");
    cpuHierarchy.cpus[0].ownedCores.bitmask[0] = 0xfull;
    return cpuHierarchy;
}

dcgmWorkloadPowerProfileProfilesInfo_v1 MockDcgmOps::MakeDefaultWorkloadPowerProfileInfo()
{
    dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
    profilesInfo.version                           = dcgmWorkloadPowerProfileProfilesInfo_version1;
    profilesInfo.profileCount                      = 1;
    profilesInfo.workloadPowerProfile[0].version   = dcgmWorkloadPowerProfileInfo_version;
    profilesInfo.workloadPowerProfile[0].profileId = DCGM_POWER_PROFILE_COMPUTE;
    profilesInfo.workloadPowerProfile[0].priority  = 1;
    return profilesInfo;
}

dcgmDeviceWorkloadPowerProfilesStatus_v1 MockDcgmOps::MakeDefaultWorkloadPowerProfileStatus()
{
    dcgmDeviceWorkloadPowerProfilesStatus_v1 profileStatus {};
    profileStatus.version                 = dcgmDeviceWorkloadPowerProfilesStatus_version1;
    profileStatus.profileMask[0]          = 0x1;
    profileStatus.requestedProfileMask[0] = 0x1;
    profileStatus.enforcedProfileMask[0]  = 0x1;
    return profileStatus;
}

MockDcgmOps::MockDcgmOps()
{
    ResetApiState();
}

MockDcgmEntity::MockDcgmEntity(dcgmGroupEntityPair_t const &entity)
    : m_entity(entity)
{
    std::memset(m_cpuAffinityMask.data(), 0, m_cpuAffinityMask.size() * sizeof(m_cpuAffinityMask[0]));
    m_cpuAffinityMask[0] = 1;
    std::memset(&m_devAttr, 0, sizeof(m_devAttr));
}

dcgmGroupEntityPair_t MockDcgmEntity::GetEntity() const
{
    return m_entity;
}

dcgmFieldValue_v2 MockDcgmEntity::GetFieldValue(unsigned short const fieldId) const
{
    if (!m_fieldValues.contains(fieldId))
    {
        dcgmFieldValue_v2 val;
        val.status = DCGM_ST_NO_DATA;
        return val;
    }
    return m_fieldValues.at(fieldId);
}

void MockDcgmEntity::InjectFieldValue(unsigned short const fieldId, dcgmFieldValue_v2 const &val)
{
    m_fieldValues[fieldId] = val;
}

void MockDcgmEntity::SetCpuAffinityMask(
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &cpuAffinityMask)
{
    m_cpuAffinityMask = cpuAffinityMask;
}

std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &MockDcgmEntity::GetCpuAffinityMask() const
{
    return m_cpuAffinityMask;
}

void MockDcgmEntity::SetDevAttr(dcgmDeviceAttributes_t const &devAttr)
{
    m_devAttr = devAttr;
}

dcgmDeviceAttributes_t const &MockDcgmEntity::GetDevAttr() const
{
    return m_devAttr;
}

dcgmReturn_t MockDcgmOps::dcgmInit()
{
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmShutdown()
{
    if (m_hostEngineState.shutdownCallback)
    {
        m_hostEngineState.shutdownCallback();
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmConnect_v2(const char *, dcgmConnectV2Params_t *, dcgmHandle_t *pDcgmHandle)
{
    if (pDcgmHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    *pDcgmHandle = m_handleId;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmConnect_v3(const char *connectionString,
                                         dcgmConnectV3Params_t *connectParams,
                                         dcgmHandle_t *pDcgmHandle)
{
    if (pDcgmHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    std::lock_guard<std::mutex> lock(m_connectionMutex);
    m_connectionState.connectV3CallCount++;
    m_connectionState.lastConnectionString = connectionString == nullptr ? "" : connectionString;
    m_connectionState.lastConnectParams    = connectParams == nullptr ? dcgmConnectV3Params_t {} : *connectParams;

    if (m_connectionState.connectV3Return != DCGM_ST_OK)
    {
        return m_connectionState.connectV3Return;
    }

    auto const handle
        = m_connectionState.useGeneratedConnectHandles ? m_connectionState.nextGeneratedHandle++ : m_handleId;
    *pDcgmHandle = handle;
    m_connectionState.connectionsByHandle[handle]
        = { m_connectionState.lastConnectionString, m_connectionState.lastConnectParams };
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmDisconnect(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.disconnectCallCount++;
    m_hostEngineState.lastDisconnectedHandle = pDcgmHandle;

    std::lock_guard<std::mutex> lock(m_connectionMutex);
    m_connectionState.disconnectCallCount++;
    m_connectionState.lastDisconnectedHandle = pDcgmHandle;
    return m_connectionState.disconnectReturn.value_or(m_hostEngineState.disconnectReturn);
}

dcgmReturn_t MockDcgmOps::dcgmMultinodeRequest(dcgmHandle_t pDcgmHandle, dcgmMultinodeRequest_t *request)
{
    if (request == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    MultinodeRequestState::Handler handler;
    dcgmReturn_t returnCode = DCGM_ST_NOT_SUPPORTED;
    {
        std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
        m_multinodeRequestState.callCount++;
        m_multinodeRequestState.lastHandle = pDcgmHandle;
        m_multinodeRequestState.requestTypeCounts[request->requestType]++;

        auto const handleRequestKey = std::make_pair(pDcgmHandle, request->requestType);
        if (auto iter = m_multinodeRequestState.handleRequestTypeHandlers.find(handleRequestKey);
            iter != m_multinodeRequestState.handleRequestTypeHandlers.end())
        {
            handler = iter->second;
        }
        else if (auto iter = m_multinodeRequestState.requestTypeHandlers.find(request->requestType);
                 iter != m_multinodeRequestState.requestTypeHandlers.end())
        {
            handler = iter->second;
        }
        returnCode = m_multinodeRequestState.returnCode;
    }

    if (handler)
    {
        return handler(pDcgmHandle, request);
    }

    return returnCode;
}

void MockDcgmOps::SetConnectV3Result(dcgmReturn_t result)
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    m_connectionState.connectV3Return = result;
}

void MockDcgmOps::UseGeneratedConnectHandles(dcgmHandle_t firstHandle)
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    m_connectionState.useGeneratedConnectHandles = true;
    m_connectionState.nextGeneratedHandle        = firstHandle;
}

int MockDcgmOps::GetConnectV3CallCount() const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    return m_connectionState.connectV3CallCount;
}

std::string MockDcgmOps::GetLastConnectV3ConnectionString() const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    return m_connectionState.lastConnectionString;
}

dcgmConnectV3Params_t MockDcgmOps::GetLastConnectV3Params() const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    return m_connectionState.lastConnectParams;
}

bool MockDcgmOps::HasConnection(dcgmHandle_t handle) const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    return m_connectionState.connectionsByHandle.contains(handle);
}

std::optional<MockDcgmOps::ConnectionInfo> MockDcgmOps::GetConnectionInfo(dcgmHandle_t handle) const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    auto const iter = m_connectionState.connectionsByHandle.find(handle);
    if (iter == m_connectionState.connectionsByHandle.end())
    {
        return std::nullopt;
    }
    return iter->second;
}

void MockDcgmOps::SetDisconnectResult(dcgmReturn_t result)
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    m_connectionState.disconnectReturn = result;
}

int MockDcgmOps::GetDisconnectCallCount() const
{
    std::lock_guard<std::mutex> lock(m_connectionMutex);
    return m_connectionState.disconnectCallCount;
}

void MockDcgmOps::SetMultinodeRequestResult(dcgmReturn_t result)
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    m_multinodeRequestState.returnCode = result;
}

dcgmReturn_t MockDcgmOps::GetMultinodeRequestResult() const
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    return m_multinodeRequestState.returnCode;
}

void MockDcgmOps::SetMultinodeRequestHandler(dcgmMultinodeRequestType_t requestType,
                                             MultinodeRequestState::Handler handler)
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    m_multinodeRequestState.requestTypeHandlers.insert_or_assign(requestType, std::move(handler));
}

void MockDcgmOps::SetMultinodeRequestHandler(dcgmHandle_t handle,
                                             dcgmMultinodeRequestType_t requestType,
                                             MultinodeRequestState::Handler handler)
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    m_multinodeRequestState.handleRequestTypeHandlers.insert_or_assign(std::make_pair(handle, requestType),
                                                                       std::move(handler));
}

int MockDcgmOps::GetMultinodeRequestCallCount() const
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    return m_multinodeRequestState.callCount;
}

dcgmHandle_t MockDcgmOps::GetLastMultinodeRequestHandle() const
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    return m_multinodeRequestState.lastHandle;
}

unsigned int MockDcgmOps::GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t requestType) const
{
    std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
    auto const iter = m_multinodeRequestState.requestTypeCounts.find(requestType);
    if (iter == m_multinodeRequestState.requestTypeCounts.end())
    {
        return 0;
    }
    return iter->second;
}

dcgmReturn_t MockDcgmOps::dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    m_hostEngineState.startEmbeddedCallCount++;
    m_hostEngineState.lastEmbeddedOperationMode = opMode;

    if (pDcgmHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_hostEngineState.startEmbeddedReturn != DCGM_ST_OK)
    {
        return m_hostEngineState.startEmbeddedReturn;
    }

    m_hostEngineState.embeddedHandle = m_handleId;
    *pDcgmHandle                     = m_hostEngineState.embeddedHandle;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.stopEmbeddedCallCount++;
    m_hostEngineState.lastStoppedEmbeddedHandle = pDcgmHandle;
    return m_hostEngineState.stopEmbeddedReturn;
}

dcgmReturn_t MockDcgmOps::dcgmPauseTelemetryForDiag(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.pauseTelemetryCallCount++;
    m_hostEngineState.lastPausedTelemetryHandle = pDcgmHandle;
    return m_hostEngineState.pauseTelemetryReturn;
}

dcgmReturn_t MockDcgmOps::dcgmResumeTelemetryForDiag(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.resumeTelemetryCallCount++;
    m_hostEngineState.lastResumedTelemetryHandle = pDcgmHandle;
    return m_hostEngineState.resumeTelemetryReturn;
}

dcgmReturn_t MockDcgmOps::dcgmHostengineVersionInfo(dcgmHandle_t pDcgmHandle, dcgmVersionInfo_t *pVersionInfo)
{
    m_hostEngineState.versionCallCount++;
    m_hostEngineState.lastVersionHandle = pDcgmHandle;

    if (pVersionInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_hostEngineState.lastVersionVersion = pVersionInfo->version;
    if (m_hostEngineState.versionReturn != DCGM_ST_OK)
    {
        return m_hostEngineState.versionReturn;
    }

    SafeCopyTo(pVersionInfo->rawBuildInfoString, m_hostEngineState.rawBuildInfo.c_str());
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmHostengineEnvironmentVariableInfo(dcgmHandle_t pDcgmHandle, dcgmEnvVarInfo_t *pEnvVarInfo)
{
    m_hostEngineState.environmentCallCount++;
    m_hostEngineState.lastEnvironmentHandle = pDcgmHandle;

    if (pEnvVarInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_hostEngineState.lastEnvironmentName = pEnvVarInfo->envVarName;
    if (m_hostEngineState.environmentReturn != DCGM_ST_OK)
    {
        return m_hostEngineState.environmentReturn;
    }

    pEnvVarInfo->ret = m_hostEngineState.environmentResult;
    if (m_hostEngineState.environmentResult == DCGM_ST_OK)
    {
        SafeCopyTo(pEnvVarInfo->envVarValue, m_hostEngineState.environmentValue.c_str());
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmHostengineSetLoggingSeverity(dcgmHandle_t pDcgmHandle,
                                                           dcgmSettingsSetLoggingSeverity_t *logging)
{
    m_hostEngineState.setLoggingCallCount++;
    m_hostEngineState.lastSettingsHandle = pDcgmHandle;
    if (logging == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_hostEngineState.lastLogging = *logging;
    return m_hostEngineState.setLoggingReturn;
}

dcgmReturn_t MockDcgmOps::dcgmAttachDriver(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.attachCallCount++;
    m_hostEngineState.lastSettingsHandle = pDcgmHandle;
    return m_hostEngineState.attachReturn;
}

dcgmReturn_t MockDcgmOps::dcgmDetachDriver(dcgmHandle_t pDcgmHandle)
{
    m_hostEngineState.detachCallCount++;
    m_hostEngineState.lastSettingsHandle = pDcgmHandle;
    return m_hostEngineState.detachReturn;
}

dcgmReturn_t MockDcgmOps::dcgmStatusCreate(dcgmStatus_t *statusHandle)
{
    m_statusState.createCallCount++;

    if (statusHandle == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_statusState.createReturn != DCGM_ST_OK)
    {
        return m_statusState.createReturn;
    }

    m_statusState.createdHandle = m_statusState.handle.value_or(static_cast<dcgmStatus_t>(m_handleId));
    *statusHandle               = m_statusState.createdHandle;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmStatusDestroy(dcgmStatus_t statusHandle)
{
    m_statusState.destroyCallCount++;
    m_statusState.lastDestroyedStatus = statusHandle;
    return m_statusState.destroyReturn;
}

void MockDcgmOps::ResetApiState()
{
    {
        std::lock_guard<std::mutex> lock(m_connectionMutex);
        m_connectionState = {};
    }
    {
        std::lock_guard<std::mutex> lock(m_multinodeRequestMutex);
        m_multinodeRequestState = {};
    }

    m_hostEngineState              = {};
    m_statusState                  = {};
    m_discoveryState               = {};
    m_queryState                   = {};
    m_migHierarchyState            = {};
    m_entitiesGetLatestValuesState = {};
    m_groupManagementState         = {};
    m_groupInfoState               = {};
    m_fieldGroupState              = {};
    m_updateAllFieldsState         = {};
    m_valuesSinceState             = {};
    m_latestValuesForFieldsState   = {};
    m_multipleValuesForFieldState  = {};
    m_fieldSummaryState            = {};
    m_topologyState                = {};
    m_deviceAttributesState        = {};
    m_entityListState              = {};
    m_configState                  = {};
    m_policyState                  = {};
    m_healthState                  = {};
    m_processStatsState            = {};
    m_introspectState              = {};
    m_cacheManagerState            = {};
    m_moduleState                  = {};
    m_actionValidateState          = {};

    m_migHierarchyState.hierarchy.version   = dcgmMigHierarchy_version2;
    m_queryState.cpuHierarchy               = MakeDefaultCpuHierarchy();
    m_queryState.profilesInfo               = MakeDefaultWorkloadPowerProfileInfo();
    m_queryState.profileStatus              = MakeDefaultWorkloadPowerProfileStatus();
    m_discoveryState.nvLinkStatus.version   = dcgmNvLinkStatus_version5;
    m_topologyState.groupTopology.version   = dcgmGroupTopology_version;
    m_healthState.response.version          = dcgmHealthResponse_version;
    m_processStatsState.pidInfo.version     = dcgmPidInfo_version;
    m_processStatsState.jobInfo.version     = dcgmJobInfo_version;
    m_introspectState.memory.version        = dcgmIntrospectMemory_version1;
    m_introspectState.cpu.version           = dcgmIntrospectCpuUtil_version1;
    m_moduleState.statuses.version          = dcgmModuleGetStatuses_version;
    m_mnDiagState                           = {};
    m_mnDiagState.response->version         = dcgmMnDiagResponse_version2;
    m_profileState                          = {};
    m_profileState.metricGroups.version     = dcgmProfGetMetricGroups_version;
    m_mnDiagState.response->numTests        = 1;
    m_mnDiagState.response->tests[0].result = DCGM_DIAG_RESULT_PASS;
    SafeCopyTo(m_mnDiagState.response->tests[0].name, "MNUBERGEMM");
}

dcgmReturn_t MockDcgmOps::dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                      dcgmGroupEntityPair_t entities[],
                                                      unsigned int entityCount,
                                                      unsigned short fields[],
                                                      unsigned int fieldCount,
                                                      unsigned int flags,
                                                      dcgmFieldValue_v2 values[])
{
    m_entitiesGetLatestValuesState.callCount++;
    m_entitiesGetLatestValuesState.lastHandle = pDcgmHandle;
    m_entitiesGetLatestValuesState.lastFlags  = flags;
    m_entitiesGetLatestValuesState.lastEntities.clear();
    if (entities != nullptr)
    {
        m_entitiesGetLatestValuesState.lastEntities.assign(entities, entities + entityCount);
    }
    m_entitiesGetLatestValuesState.lastFields.clear();
    if (fields != nullptr)
    {
        m_entitiesGetLatestValuesState.lastFields.assign(fields, fields + fieldCount);
    }

    if (m_entitiesGetLatestValuesState.handler)
    {
        return m_entitiesGetLatestValuesState.handler(
            pDcgmHandle, entities, entityCount, fields, fieldCount, flags, values);
    }

    if (entities == nullptr || fields == nullptr || values == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_entitiesGetLatestValuesState.returnCode != DCGM_ST_OK)
    {
        return m_entitiesGetLatestValuesState.returnCode;
    }

    if (!m_entitiesGetLatestValuesState.returnByFieldId.empty())
    {
        for (unsigned int fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
        {
            if (auto it = m_entitiesGetLatestValuesState.returnByFieldId.find(fields[fieldIndex]);
                it != m_entitiesGetLatestValuesState.returnByFieldId.end() && it->second != DCGM_ST_OK)
            {
                return it->second;
            }
        }
    }

    if (m_entitiesGetLatestValuesState.values.has_value())
    {
        auto const valueCount = entityCount * fieldCount;
        auto const limit
            = std::min(valueCount, static_cast<unsigned int>(m_entitiesGetLatestValuesState.values->size()));
        for (unsigned int i = 0; i < limit; ++i)
        {
            values[i] = (*m_entitiesGetLatestValuesState.values)[i];
        }
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    if (entities == nullptr || entityCount == 0 || fields == nullptr || fieldCount == 0 || values == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    unsigned int valIdx = 0;
    for (unsigned int i = 0; i < entityCount; ++i)
    {
        auto it = m_entities.find(entities[i]);
        if (it == m_entities.end())
        {
            return DCGM_ST_BADPARAM;
        }
        auto &mockedEntity = it->second;
        for (unsigned int j = 0; j < fieldCount; ++j)
        {
            values[valIdx] = mockedEntity.GetFieldValue(fields[j]);
            if (values[valIdx].status == DCGM_ST_NO_DATA)
            {
                return DCGM_ST_NO_DATA;
            }
            valIdx += 1;
        }
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetGpuStatus(dcgmHandle_t pDcgmHandle, unsigned int gpuId, DcgmEntityStatus_t *status)
{
    m_queryState.gpuStatusCallCount++;
    m_queryState.lastHandle = pDcgmHandle;
    m_queryState.lastGpuId  = gpuId;

    if (status == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_queryState.gpuStatusReturn != DCGM_ST_OK)
    {
        return m_queryState.gpuStatusReturn;
    }

    *status = m_queryState.gpuStatus;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetAllDevices(dcgmHandle_t pDcgmHandle,
                                            unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                            int *count)
{
    m_discoveryState.allDevicesCallCount++;
    m_discoveryState.lastHandle = pDcgmHandle;

    if (gpuIdList == nullptr || count == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_discoveryState.allDevicesReturn != DCGM_ST_OK)
    {
        return m_discoveryState.allDevicesReturn;
    }

    auto const limit = std::min(m_discoveryState.allDevices.size(), static_cast<std::size_t>(DCGM_MAX_NUM_DEVICES));
    for (std::size_t i = 0; i < limit; ++i)
    {
        gpuIdList[i] = m_discoveryState.allDevices[i];
    }
    *count = static_cast<int>(limit);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetAllSupportedDevices(dcgmHandle_t pDcgmHandle,
                                                     unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                                     int *count)
{
    m_discoveryState.allSupportedDevicesCallCount++;
    m_discoveryState.lastHandle = pDcgmHandle;

    if (gpuIdList == nullptr || count == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_discoveryState.allSupportedDevicesReturn != DCGM_ST_OK)
    {
        return m_discoveryState.allSupportedDevicesReturn;
    }

    auto const limit
        = std::min(m_discoveryState.allSupportedDevices.size(), static_cast<std::size_t>(DCGM_MAX_NUM_DEVICES));
    for (std::size_t i = 0; i < limit; ++i)
    {
        gpuIdList[i] = m_discoveryState.allSupportedDevices[i];
    }
    *count = static_cast<int>(limit);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetGpuChipArchitecture(dcgmHandle_t dcgmHandle,
                                                     unsigned int gpuId,
                                                     dcgmChipArchitecture_t *chipArchitecture)
{
    m_discoveryState.chipArchitectureCallCount++;
    m_discoveryState.lastHandle = dcgmHandle;
    m_discoveryState.lastGpuId  = gpuId;

    if (chipArchitecture == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_discoveryState.chipArchitectureReturn != DCGM_ST_OK)
    {
        return m_discoveryState.chipArchitectureReturn;
    }

    if (auto const it = m_discoveryState.chipArchitecturesByGpuId.find(gpuId);
        it != m_discoveryState.chipArchitecturesByGpuId.end())
    {
        *chipArchitecture = it->second;
    }
    else
    {
        *chipArchitecture = m_discoveryState.chipArchitecture;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetNvLinkLinkStatus(dcgmHandle_t dcgmHandle, dcgmNvLinkStatus_v5 *linkStatus)
{
    m_discoveryState.nvLinkStatusCallCount++;
    m_discoveryState.lastHandle = dcgmHandle;

    if (linkStatus == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_discoveryState.nvLinkStatusReturn != DCGM_ST_OK)
    {
        return m_discoveryState.nvLinkStatusReturn;
    }

    *linkStatus = m_discoveryState.nvLinkStatus;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetDeviceWorkloadPowerProfileInfo(dcgmHandle_t pDcgmHandle,
                                                                unsigned int gpuId,
                                                                dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                                dcgmDeviceWorkloadPowerProfilesStatus_v1 *profileStatus)
{
    m_queryState.workloadCallCount++;
    m_queryState.lastHandle = pDcgmHandle;
    m_queryState.lastGpuId  = gpuId;

    if (profilesInfo == nullptr || profileStatus == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_queryState.workloadReturn != DCGM_ST_OK)
    {
        return m_queryState.workloadReturn;
    }

    *profilesInfo  = m_queryState.profilesInfo;
    *profileStatus = m_queryState.profileStatus;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetGpuInstanceHierarchy(dcgmHandle_t dcgmHandle, dcgmMigHierarchy_v2 *hierarchy)
{
    m_migHierarchyState.callCount++;
    m_migHierarchyState.lastHandle = dcgmHandle;

    if (hierarchy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_migHierarchyState.handler)
    {
        return m_migHierarchyState.handler(dcgmHandle, hierarchy);
    }
    if (m_migHierarchyState.returnCode != DCGM_ST_OK)
    {
        return m_migHierarchyState.returnCode;
    }

    *hierarchy = m_migHierarchyState.hierarchy;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetCpuHierarchy_v2(dcgmHandle_t dcgmHandle, dcgmCpuHierarchy_v2 *cpuHierarchy)
{
    m_queryState.cpuHierarchyCallCount++;
    m_queryState.lastHandle = dcgmHandle;

    if (cpuHierarchy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_queryState.cpuHierarchyReturn != DCGM_ST_OK)
    {
        return m_queryState.cpuHierarchyReturn;
    }

    *cpuHierarchy = m_queryState.cpuHierarchy;
    return DCGM_ST_OK;
}

void MockDcgmOps::AddMockedEntity(MockDcgmEntity const &mockedEntity)
{
    auto entity = mockedEntity.GetEntity();
    if (m_entities.contains(entity))
    {
        throw std::invalid_argument(
            fmt::format("Duplicated mocked entity group [{}] with id [{}].", entity.entityGroupId, entity.entityId));
    }
    m_entities.emplace(entity, mockedEntity);
}

dcgmReturn_t MockDcgmOps::dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                                unsigned int gpuId,
                                                dcgmDeviceTopology_v2 *pDcgmDeviceTopology)
{
    m_topologyState.deviceCallCount++;
    m_topologyState.lastHandle = pDcgmHandle;
    m_topologyState.lastGpuId  = gpuId;

    if (m_topologyState.deviceReturn != DCGM_ST_OK || m_topologyState.deviceTopology.has_value())
    {
        if (pDcgmDeviceTopology == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }
        if (m_topologyState.deviceReturn != DCGM_ST_OK)
        {
            return m_topologyState.deviceReturn;
        }

        *pDcgmDeviceTopology = *m_topologyState.deviceTopology;
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = gpuId };
    if (!pDcgmDeviceTopology || !m_topology.contains(entity) || !m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    auto const &mockedEntity     = m_entities.at(entity);
    auto const &mask             = mockedEntity.GetCpuAffinityMask();
    pDcgmDeviceTopology->version = dcgmDeviceTopology_version2;
    std::memcpy(pDcgmDeviceTopology->cpuAffinityMask, mask.data(), sizeof(pDcgmDeviceTopology->cpuAffinityMask));
    pDcgmDeviceTopology->numGpus = 0;
    for (auto const &[targetEntity, topology] : m_topology.at(entity))
    {
        if (targetEntity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        if (pDcgmDeviceTopology->numGpus >= std::size(pDcgmDeviceTopology->gpuPaths))
        {
            break;
        }

        auto &gpuPath                         = pDcgmDeviceTopology->gpuPaths[pDcgmDeviceTopology->numGpus];
        gpuPath                               = {};
        gpuPath.gpuId                         = targetEntity.entityId;
        gpuPath.path                          = topology;
        uint64_t const nvLinkPath             = static_cast<uint64_t>(DCGM_TOPOLOGY_PATH_NVLINK(topology)) >> 8;
        unsigned int const numConnectedNvLink = std::bit_width(nvLinkPath);
        gpuPath.localNvLinkIds                = (1ULL << numConnectedNvLink) - 1;
        pDcgmDeviceTopology->numGpus += 1;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetGroupTopology(dcgmHandle_t pDcgmHandle,
                                               dcgmGpuGrp_t groupId,
                                               dcgmGroupTopology_v2 *pDcgmGroupTopology)
{
    m_topologyState.groupCallCount++;
    m_topologyState.lastHandle  = pDcgmHandle;
    m_topologyState.lastGroupId = groupId;

    if (pDcgmGroupTopology == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_topologyState.groupReturn != DCGM_ST_OK)
    {
        return m_topologyState.groupReturn;
    }

    *pDcgmGroupTopology = m_topologyState.groupTopology;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                          dcgmGroupType_t type,
                                          const char *groupName,
                                          dcgmGpuGrp_t *pDcgmGrpId)
{
    m_groupManagementState.createCallCount++;
    char const *normalizedGroupName      = groupName == nullptr ? "" : groupName;
    m_groupManagementState.lastHandle    = pDcgmHandle;
    m_groupManagementState.lastGroupType = type;
    m_groupManagementState.lastGroupName = normalizedGroupName;

    if (m_groupManagementState.createReturn.has_value())
    {
        if (pDcgmGrpId == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }
        if (*m_groupManagementState.createReturn != DCGM_ST_OK)
        {
            return *m_groupManagementState.createReturn;
        }

        *pDcgmGrpId = m_groupManagementState.newGroupId;
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    if (pDcgmGrpId == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmGroupInfo_v3 group {};

    SafeCopyTo(group.groupName, normalizedGroupName);
    group.version = dcgmGroupInfo_version3;
    group.count   = 0;

    if (m_entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }
    for (auto const &[entity, _] : m_entities)
    {
        switch (type)
        {
            case DCGM_GROUP_EMPTY:
                break;
            case DCGM_GROUP_DEFAULT:
                if (entity.entityGroupId == DCGM_FE_GPU)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_NVSWITCHES:
                if (entity.entityGroupId == DCGM_FE_SWITCH)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_INSTANCES:
                if (entity.entityGroupId == DCGM_FE_GPU_I)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES:
                if (entity.entityGroupId == DCGM_FE_GPU_CI)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_EVERYTHING:
                group.entityList[group.count] = entity;
                group.count += 1;
                break;
        }
    }
    *pDcgmGrpId               = m_globalGroupId;
    m_groups[m_globalGroupId] = group;
    m_globalGroupId += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId)
{
    m_groupManagementState.addCallCount++;
    m_groupManagementState.lastHandle      = pDcgmHandle;
    m_groupManagementState.lastGroupId     = groupId;
    m_groupManagementState.lastEntityGroup = entityGroupId;
    m_groupManagementState.lastEntityId    = entityId;

    if (m_groupManagementState.addReturn.has_value())
    {
        return *m_groupManagementState.addReturn;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    if (!m_groups.contains(groupId))
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmGroupEntityPair_t entity { .entityGroupId = entityGroupId, .entityId = entityId };
    if (!m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    if (m_groups[groupId].count >= DCGM_GROUP_MAX_ENTITIES_V2)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    m_groups[groupId].entityList[m_groups[groupId].count] = entity;
    m_groups[groupId].count += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGroupRemoveEntity(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId)
{
    m_groupManagementState.removeCallCount++;
    m_groupManagementState.lastHandle      = pDcgmHandle;
    m_groupManagementState.lastGroupId     = groupId;
    m_groupManagementState.lastEntityGroup = entityGroupId;
    m_groupManagementState.lastEntityId    = entityId;
    return m_groupManagementState.removeReturn;
}

dcgmReturn_t MockDcgmOps::dcgmGroupAddDevice(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, unsigned int gpuId)
{
    return dcgmGroupAddEntity(pDcgmHandle, groupId, DCGM_FE_GPU, gpuId);
}

dcgmReturn_t MockDcgmOps::dcgmGroupGetInfo(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmGroupInfo_t *pDcgmGroupInfo)
{
    m_groupInfoState.callCount++;
    m_groupInfoState.lastHandle  = pDcgmHandle;
    m_groupInfoState.lastGroupId = groupId;

    if (m_groupInfoState.returnCode != DCGM_ST_OK || m_groupInfoState.overrideInfo.has_value())
    {
        if (pDcgmGroupInfo == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }
        if (m_groupInfoState.returnCode != DCGM_ST_OK)
        {
            return m_groupInfoState.returnCode;
        }

        *pDcgmGroupInfo = *m_groupInfoState.overrideInfo;
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (pDcgmGroupInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (!m_groups.contains(groupId))
    {
        return DCGM_ST_NOT_CONFIGURED;
    }

    *pDcgmGroupInfo = m_groups.at(groupId);
    return DCGM_ST_OK;
}

dcgmGroupInfo_t MockDcgmOps::MakeGroupInfo(std::string_view groupName, std::span<dcgmGroupEntityPair_t const> entities)
{
    dcgmGroupInfo_t groupInfo {};
    auto const count = std::min(entities.size(), std::size(groupInfo.entityList));

    groupInfo.version = dcgmGroupInfo_version;
    SafeCopyTo(groupInfo.groupName, std::string(groupName).c_str());
    groupInfo.count = static_cast<unsigned int>(count);
    std::copy_n(entities.begin(), count, groupInfo.entityList);
    return groupInfo;
}

dcgmGroupInfo_t MockDcgmOps::MakeGroupInfo(std::string_view groupName,
                                           std::initializer_list<dcgmGroupEntityPair_t> entities)
{
    return MakeGroupInfo(groupName, std::span<dcgmGroupEntityPair_t const>(entities.begin(), entities.size()));
}

dcgmReturn_t MockDcgmOps::dcgmGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    m_groupManagementState.getAllCallCount++;
    m_groupManagementState.lastHandle = pDcgmHandle;

    if (groupIdList == nullptr || count == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_groupManagementState.getAllReturn != DCGM_ST_OK)
    {
        return m_groupManagementState.getAllReturn;
    }

    auto const copiedCount
        = std::min(m_groupManagementState.groupIds.size(), static_cast<std::size_t>(DCGM_MAX_NUM_GROUPS));
    std::copy_n(m_groupManagementState.groupIds.begin(), copiedCount, groupIdList);
    *count = static_cast<unsigned int>(copiedCount);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                               int numFieldIds,
                                               const unsigned short *fieldIds,
                                               const char *fieldGroupName,
                                               dcgmFieldGrp_t *dcgmFieldGroupId)
{
    m_fieldGroupState.createCallCount++;
    m_fieldGroupState.lastHandle      = dcgmHandle;
    m_fieldGroupState.lastCreatedName = fieldGroupName == nullptr ? "" : fieldGroupName;
    m_fieldGroupState.lastCreatedFieldIds.clear();
    if (fieldIds != nullptr && numFieldIds > 0)
    {
        m_fieldGroupState.lastCreatedFieldIds.assign(fieldIds, fieldIds + numFieldIds);
    }

    if (fieldIds == nullptr || fieldGroupName == nullptr || dcgmFieldGroupId == nullptr || numFieldIds < 0)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_fieldGroupState.createReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.createReturn;
    }

    dcgmFieldGrp_t const newFieldGroupId = m_fieldGroupState.nextFieldGroupId;
    m_fieldGroups[newFieldGroupId].name  = fieldGroupName;
    for (int i = 0; i < numFieldIds; ++i)
    {
        m_fieldGroups[newFieldGroupId].fields.push_back(fieldIds[i]);
    }

    *dcgmFieldGroupId = newFieldGroupId;
    m_fieldGroupState.nextFieldGroupId += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    m_groupManagementState.destroyCallCount++;
    m_groupManagementState.lastHandle  = pDcgmHandle;
    m_groupManagementState.lastGroupId = groupId;

    if (m_groupManagementState.destroyReturn.has_value())
    {
        return *m_groupManagementState.destroyReturn;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    m_groups.erase(groupId);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    m_fieldGroupState.destroyCallCount++;
    m_fieldGroupState.lastHandle                = dcgmHandle;
    m_fieldGroupState.lastDestroyedFieldGroupId = dcgmFieldGroupId;

    if (m_fieldGroupState.destroyReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.destroyReturn;
    }
    m_fieldGroups.erase(dcgmFieldGroupId);
    m_fieldGroupState.fieldGroupInfo.erase(dcgmFieldGroupId);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmFieldGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    m_fieldGroupState.getInfoCallCount++;
    m_fieldGroupState.lastHandle          = dcgmHandle;
    m_fieldGroupState.lastRequestedInfoId = fieldGroupInfo == nullptr ? 0 : fieldGroupInfo->fieldGroupId;

    if (fieldGroupInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_fieldGroupState.getInfoReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.getInfoReturn;
    }

    auto const configured = m_fieldGroupState.fieldGroupInfo.find(fieldGroupInfo->fieldGroupId);
    if (configured != m_fieldGroupState.fieldGroupInfo.end())
    {
        *fieldGroupInfo = configured->second;
        return DCGM_ST_OK;
    }

    auto const created = m_fieldGroups.find(fieldGroupInfo->fieldGroupId);
    if (created == m_fieldGroups.end())
    {
        return DCGM_ST_NOT_CONFIGURED;
    }

    auto const &source      = created->second;
    fieldGroupInfo->version = dcgmFieldGroupInfo_version;
    SafeCopyTo(fieldGroupInfo->fieldGroupName, source.name.c_str());
    auto const count            = std::min(source.fields.size(), std::size(fieldGroupInfo->fieldIds));
    fieldGroupInfo->numFieldIds = static_cast<unsigned int>(count);
    std::copy_n(source.fields.begin(), count, fieldGroupInfo->fieldIds);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmFieldGroupGetAll(dcgmHandle_t dcgmHandle, dcgmAllFieldGroup_t *allGroupInfo)
{
    m_fieldGroupState.getAllCallCount++;
    m_fieldGroupState.lastHandle = dcgmHandle;

    if (allGroupInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_fieldGroupState.getAllReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.getAllReturn;
    }
    if (m_fieldGroupState.allFieldGroups.has_value())
    {
        *allGroupInfo = *m_fieldGroupState.allFieldGroups;
        return DCGM_ST_OK;
    }

    *allGroupInfo         = {};
    allGroupInfo->version = dcgmAllFieldGroup_version;
    for (auto const &[fieldGroupId, _] : m_fieldGroups)
    {
        if (allGroupInfo->numFieldGroups >= std::size(allGroupInfo->fieldGroups))
        {
            break;
        }

        allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].fieldGroupId = fieldGroupId;
        allGroupInfo->numFieldGroups += 1;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                          dcgmGpuGrp_t groupId,
                                          dcgmFieldGrp_t fieldGroupId,
                                          long long updateFreq,
                                          double maxKeepAge,
                                          int maxKeepSamples)
{
    m_fieldGroupState.watchCallCount++;
    m_fieldGroupState.lastHandle              = pDcgmHandle;
    m_fieldGroupState.lastWatchedGroupId      = groupId;
    m_fieldGroupState.lastWatchedFieldGroupId = fieldGroupId;
    m_fieldGroupState.lastUpdateFreq          = updateFreq;
    m_fieldGroupState.lastMaxKeepAge          = maxKeepAge;
    m_fieldGroupState.lastMaxKeepSamples      = maxKeepSamples;

    if (m_fieldGroupState.watchReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.watchReturn;
    }
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmFieldGrp_t fieldGroupId)
{
    m_fieldGroupState.unwatchCallCount++;
    m_fieldGroupState.lastHandle                = pDcgmHandle;
    m_fieldGroupState.lastUnwatchedGroupId      = groupId;
    m_fieldGroupState.lastUnwatchedFieldGroupId = fieldGroupId;

    if (m_fieldGroupState.unwatchReturn != DCGM_ST_OK)
    {
        return m_fieldGroupState.unwatchReturn;
    }
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate)
{
    m_updateAllFieldsState.callCount++;
    m_updateAllFieldsState.lastHandle        = pDcgmHandle;
    m_updateAllFieldsState.lastWaitForUpdate = waitForUpdate;

    if (m_updateAllFieldsState.returnCode.has_value())
    {
        return *m_updateAllFieldsState.returnCode;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                                 dcgmGpuGrp_t groupId,
                                                 dcgmFieldGrp_t fieldGroupId,
                                                 dcgmFieldValueEntityEnumeration_f enumCB,
                                                 void *userData)
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (enumCB == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }

    auto &group     = m_groups.at(groupId);
    auto fieldGroup = m_fieldGroups.at(fieldGroupId);
    for (unsigned int i = 0; i < group.count; ++i)
    {
        for (auto const fieldId : fieldGroup.fields)
        {
            std::array<dcgmFieldValue_v1, 1> values;

            dcgmGroupEntityPair_t entity
                = { .entityGroupId = group.entityList[i].entityGroupId, .entityId = group.entityList[i].entityId };

            if (!m_entities.contains(entity))
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            auto &mockedEntity  = m_entities.at(entity);
            auto const &fv      = mockedEntity.GetFieldValue(fieldId);
            values[0].fieldId   = fv.fieldId;
            values[0].fieldType = fv.fieldType;
            values[0].status    = fv.status;
            values[0].ts        = fv.ts;
            memcpy(&values[0].value, &fv.value, sizeof(values[0].value));
            values[0].version = dcgmFieldValue_version1;
            enumCB(group.entityList[i].entityGroupId,
                   group.entityList[i].entityId,
                   values.data(),
                   values.size(),
                   userData);
        }
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetValuesSince_v2(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgmFieldGrp_t fieldGroupId,
                                                long long sinceTimestamp,
                                                long long *nextSinceTimestamp,
                                                dcgmFieldValueEntityEnumeration_f enumCB,
                                                void *userData)
{
    m_valuesSinceState.valuesSinceV2CallCount++;
    m_valuesSinceState.lastHandle         = pDcgmHandle;
    m_valuesSinceState.lastGroupId        = groupId;
    m_valuesSinceState.lastFieldGroup     = fieldGroupId;
    m_valuesSinceState.lastSinceTimestamp = sinceTimestamp;

    if (nextSinceTimestamp != nullptr)
    {
        *nextSinceTimestamp = m_valuesSinceState.nextSinceTimestamp;
    }
    if (m_valuesSinceState.returnCode != DCGM_ST_OK)
    {
        return m_valuesSinceState.returnCode;
    }
    if (enumCB != nullptr && m_valuesSinceState.values.has_value())
    {
        enumCB(m_valuesSinceState.entityGroupId,
               m_valuesSinceState.entityId,
               m_valuesSinceState.values->data(),
               static_cast<int>(m_valuesSinceState.values->size()),
               userData);
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetFieldSummary(dcgmHandle_t pDcgmHandle, dcgmFieldSummaryRequest_t *request)
{
    m_fieldSummaryState.callCount++;
    m_fieldSummaryState.lastHandle = pDcgmHandle;

    if (request == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_fieldSummaryState.lastRequest = *request;

    auto const returnIt = m_fieldSummaryState.returnByFieldId.find(request->fieldId);
    if (returnIt != m_fieldSummaryState.returnByFieldId.end())
    {
        if (returnIt->second != DCGM_ST_OK)
        {
            return returnIt->second;
        }
    }
    else if (m_fieldSummaryState.returnCode != DCGM_ST_OK)
    {
        return m_fieldSummaryState.returnCode;
    }

    auto const responseIt = m_fieldSummaryState.responseByFieldId.find(request->fieldId);
    request->response
        = responseIt == m_fieldSummaryState.responseByFieldId.end() ? m_fieldSummaryState.response : responseIt->second;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetMultipleValuesForField(dcgmHandle_t pDcgmHandle,
                                                        int gpuId,
                                                        unsigned short fieldId,
                                                        int *count,
                                                        long long startTs,
                                                        long long endTs,
                                                        dcgmOrder_t order,
                                                        dcgmFieldValue_v1 values[])
{
    m_multipleValuesForFieldState.callCount++;
    m_multipleValuesForFieldState.lastHandle  = pDcgmHandle;
    m_multipleValuesForFieldState.lastGpuId   = gpuId;
    m_multipleValuesForFieldState.lastFieldId = fieldId;
    m_multipleValuesForFieldState.lastStartTs = startTs;
    m_multipleValuesForFieldState.lastEndTs   = endTs;
    m_multipleValuesForFieldState.lastOrder   = order;

    if (count == nullptr || values == nullptr || *count < 0)
    {
        return DCGM_ST_BADPARAM;
    }

    if (m_multipleValuesForFieldState.returnCode != DCGM_ST_OK)
    {
        return m_multipleValuesForFieldState.returnCode;
    }
    if (m_multipleValuesForFieldState.values.has_value())
    {
        auto const limit = std::min(*count, static_cast<int>(m_multipleValuesForFieldState.values->size()));
        for (int i = 0; i < limit; ++i)
        {
            values[i] = (*m_multipleValuesForFieldState.values)[i];
        }
        *count = limit;
        return DCGM_ST_OK;
    }

    *count = 0;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetLatestValuesForFields(dcgmHandle_t pDcgmHandle,
                                                       int gpuId,
                                                       unsigned short fields[],
                                                       unsigned int count,
                                                       dcgmFieldValue_v1 values[])
{
    m_latestValuesForFieldsState.callCount++;
    m_latestValuesForFieldsState.lastHandle = pDcgmHandle;
    m_latestValuesForFieldsState.lastGpuId  = gpuId;
    m_latestValuesForFieldsState.lastCount  = count;
    m_latestValuesForFieldsState.lastFields.clear();
    if (fields != nullptr)
    {
        m_latestValuesForFieldsState.lastFields.assign(fields, fields + count);
    }

    if (fields == nullptr || values == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_latestValuesForFieldsState.returnCode != DCGM_ST_OK)
    {
        return m_latestValuesForFieldsState.returnCode;
    }
    if (m_latestValuesForFieldsState.values.has_value())
    {
        if (m_latestValuesForFieldsState.values->size() < count)
        {
            return DCGM_ST_BADPARAM;
        }

        for (unsigned int i = 0; i < count; ++i)
        {
            values[i] = (*m_latestValuesForFieldsState.values)[i];
        }
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = static_cast<dcgm_field_eid_t>(gpuId) };
    if (!m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    auto const &mockedEntity = m_entities.at(entity);
    for (unsigned int i = 0; i < count; ++i)
    {
        auto const value    = mockedEntity.GetFieldValue(fields[i]);
        values[i].version   = dcgmFieldValue_version1;
        values[i].fieldId   = value.fieldId;
        values[i].fieldType = value.fieldType;
        values[i].status    = value.status;
        values[i].ts        = value.ts;
        std::memcpy(&values[i].value, &value.value, sizeof(values[i].value));
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                  unsigned int gpuId,
                                                  dcgmDeviceAttributes_t *pDcgmAttr)
{
    m_deviceAttributesState.callCount++;
    m_deviceAttributesState.lastHandle = pDcgmHandle;
    m_deviceAttributesState.lastGpuId  = gpuId;

    if (m_deviceAttributesState.returnCode != DCGM_ST_OK || m_deviceAttributesState.overrideAttributes.has_value())
    {
        if (pDcgmAttr == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }
        if (m_deviceAttributesState.returnCode != DCGM_ST_OK)
        {
            return m_deviceAttributesState.returnCode;
        }

        *pDcgmAttr = *m_deviceAttributesState.overrideAttributes;
        return DCGM_ST_OK;
    }

    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = gpuId };
    if (!pDcgmAttr || !m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    auto const &mockedEntity = m_entities.at(entity);
    *pDcgmAttr               = mockedEntity.GetDevAttr();
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmGetEntityGroupEntities(dcgmHandle_t dcgmHandle,
                                                     dcgm_field_entity_group_t entityGroup,
                                                     dcgm_field_eid_t *entities,
                                                     int *numEntities,
                                                     unsigned int flags)
{
    m_entityListState.callCount++;
    m_entityListState.lastHandle      = dcgmHandle;
    m_entityListState.lastEntityGroup = entityGroup;
    m_entityListState.lastFlags       = flags;

    if (m_entityListState.returnCode != DCGM_ST_OK || m_entityListState.gpuEntities.has_value())
    {
        if (entities == nullptr || numEntities == nullptr || *numEntities < 0)
        {
            return DCGM_ST_BADPARAM;
        }
        if (m_entityListState.returnCode != DCGM_ST_OK)
        {
            return m_entityListState.returnCode;
        }
        if (entityGroup != DCGM_FE_GPU)
        {
            *numEntities = 0;
            return DCGM_ST_OK;
        }

        auto const limit = std::min(*numEntities, static_cast<int>(m_entityListState.gpuEntities->size()));
        for (int i = 0; i < limit; ++i)
        {
            entities[i] = (*m_entityListState.gpuEntities)[i];
        }
        *numEntities = limit;
        return DCGM_ST_OK;
    }

    if (dcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (entities == nullptr || numEntities == nullptr || *numEntities < 0)
    {
        return DCGM_ST_BADPARAM;
    }

    int count = 0;
    for (auto const &[entity, _] : m_entities)
    {
        if (entity.entityGroupId != entityGroup)
        {
            continue;
        }
        if (count == *numEntities)
        {
            return DCGM_ST_INSUFFICIENT_SIZE;
        }
        entities[count] = entity.entityId;
        count += 1;
    }

    *numEntities = count;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmConfigGet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmConfigType_t reqType,
                                        int count,
                                        dcgmConfig_t deviceConfigList[],
                                        dcgmStatus_t statusHandle)
{
    m_configState.getCallCount++;
    m_configState.lastHandle       = pDcgmHandle;
    m_configState.lastGroupId      = groupId;
    m_configState.lastConfigType   = reqType;
    m_configState.lastConfigCount  = count;
    m_configState.lastStatusHandle = statusHandle;

    auto const &source
        = reqType == DCGM_CONFIG_CURRENT_STATE ? m_configState.currentConfigs : m_configState.targetConfigs;
    auto const result
        = reqType == DCGM_CONFIG_CURRENT_STATE ? m_configState.currentGetReturn : m_configState.targetGetReturn;
    if (deviceConfigList == nullptr || count < 0)
    {
        return DCGM_ST_BADPARAM;
    }
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    auto const limit = std::min(count, static_cast<int>(source.size()));
    for (int i = 0; i < limit; ++i)
    {
        deviceConfigList[i] = source[i];
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmConfigSet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmConfig_t *deviceConfig,
                                        dcgmStatus_t statusHandle)
{
    m_configState.setCallCount++;
    m_configState.lastHandle       = pDcgmHandle;
    m_configState.lastGroupId      = groupId;
    m_configState.lastStatusHandle = statusHandle;
    if (deviceConfig != nullptr)
    {
        m_configState.lastSetConfig = *deviceConfig;
    }
    return m_configState.setReturn;
}

dcgmReturn_t MockDcgmOps::dcgmConfigEnforce(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle)
{
    m_configState.enforceCallCount++;
    m_configState.lastHandle       = pDcgmHandle;
    m_configState.lastGroupId      = groupId;
    m_configState.lastStatusHandle = statusHandle;
    return m_configState.enforceReturn;
}

dcgmReturn_t MockDcgmOps::dcgmConfigSetWorkloadPowerProfile(dcgmHandle_t pDcgmHandle,
                                                            dcgmWorkloadPowerProfile_t *workloadPowerProfile)
{
    m_configState.workloadCallCount++;
    m_configState.lastHandle = pDcgmHandle;
    if (workloadPowerProfile != nullptr)
    {
        m_configState.lastWorkloadProfile = *workloadPowerProfile;
    }
    return m_configState.workloadReturn;
}

dcgmReturn_t MockDcgmOps::dcgmPolicySet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmPolicy_t *policy,
                                        dcgmStatus_t statusHandle)
{
    m_policyState.setCallCount++;
    m_policyState.lastHandle       = pDcgmHandle;
    m_policyState.lastGroupId      = groupId;
    m_policyState.lastStatusHandle = statusHandle;
    if (policy != nullptr)
    {
        m_policyState.lastPolicy = *policy;
    }
    return m_policyState.setReturn;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyGet(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        int count,
                                        dcgmPolicy_t *policy,
                                        dcgmStatus_t statusHandle)
{
    m_policyState.getCallCount++;
    m_policyState.lastHandle         = pDcgmHandle;
    m_policyState.lastGroupId        = groupId;
    m_policyState.lastPolicyGetCount = count;
    m_policyState.lastStatusHandle   = statusHandle;

    if (policy == nullptr || count < 0)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_policyState.getReturn != DCGM_ST_OK)
    {
        return m_policyState.getReturn;
    }

    auto const limit = std::min(count, static_cast<int>(m_policyState.policies.size()));
    for (int i = 0; i < limit; ++i)
    {
        policy[i] = m_policyState.policies[i];
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyUnregister(dcgmHandle_t pDcgmHandle,
                                               dcgmGpuGrp_t groupId,
                                               dcgmPolicyCondition_t condition)
{
    m_policyState.unregisterCallCount++;
    m_policyState.lastHandle    = pDcgmHandle;
    m_policyState.lastGroupId   = groupId;
    m_policyState.lastCondition = condition;
    return m_policyState.unregisterReturn;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyCreate(dcgmHandle_t pDcgmHandle,
                                           unsigned short fieldId,
                                           const char *name,
                                           double threshold,
                                           dcgmPolicyOperator_t policyOperator,
                                           const dcgmGroupEntityPair_t *entities,
                                           unsigned int entityCount,
                                           unsigned int channelMask,
                                           unsigned int enabled,
                                           unsigned int rateLimitSec,
                                           uint64_t *policyId)
{
    m_fieldPolicyState.createCallCount++;
    m_fieldPolicyState.lastHandle  = pDcgmHandle;
    m_fieldPolicyState.lastFieldId = fieldId;
    m_fieldPolicyState.lastName    = name == nullptr ? "" : name;

    if (policyId == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_fieldPolicyState.createReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.createReturn;
    }

    *policyId = m_fieldPolicyState.nextPolicyId++;
    dcgmPolicyInfo_t policy {};
    policy.version        = dcgmPolicyInfo_version;
    policy.policyId       = *policyId;
    policy.fieldId        = fieldId;
    policy.threshold      = threshold;
    policy.policyOperator = policyOperator;
    policy.entityCount    = std::min(entityCount, static_cast<unsigned int>(std::size(policy.entities)));
    policy.channelMask    = channelMask;
    policy.state          = enabled != 0 ? DCGM_POLICY_STATE_ENABLED : DCGM_POLICY_STATE_DISABLED;
    policy.rateLimitSec   = rateLimitSec;
    SafeCopyTo(policy.name, name);
    if (entities != nullptr)
    {
        std::copy_n(entities, policy.entityCount, policy.entities);
    }
    m_fieldPolicyState.policies.push_back(policy);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyModify(dcgmHandle_t pDcgmHandle, const dcgmPolicyInfo_t *policy)
{
    m_fieldPolicyState.modifyCallCount++;
    m_fieldPolicyState.lastHandle = pDcgmHandle;
    if (policy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    m_fieldPolicyState.lastModifiedPolicy = *policy;
    if (m_fieldPolicyState.modifyReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.modifyReturn;
    }
    auto const it
        = std::find_if(m_fieldPolicyState.policies.begin(),
                       m_fieldPolicyState.policies.end(),
                       [policy](dcgmPolicyInfo_t const &candidate) {
                           return candidate.fieldId == policy->fieldId && candidate.policyId == policy->policyId;
                       });
    if (it != m_fieldPolicyState.policies.end())
    {
        *it = *policy;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyDelete(dcgmHandle_t pDcgmHandle,
                                           unsigned short fieldId,
                                           uint64_t policyId,
                                           unsigned int deleteAll)
{
    m_fieldPolicyState.deleteCallCount++;
    m_fieldPolicyState.lastHandle    = pDcgmHandle;
    m_fieldPolicyState.lastFieldId   = fieldId;
    m_fieldPolicyState.lastPolicyId  = policyId;
    m_fieldPolicyState.lastDeleteAll = deleteAll;
    if (m_fieldPolicyState.deleteReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.deleteReturn;
    }
    if (deleteAll != 0)
    {
        m_fieldPolicyState.policies.clear();
    }
    else
    {
        std::erase_if(m_fieldPolicyState.policies, [fieldId, policyId](dcgmPolicyInfo_t const &candidate) {
            return candidate.fieldId == fieldId && candidate.policyId == policyId;
        });
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyEnable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId)
{
    m_fieldPolicyState.enableCallCount++;
    m_fieldPolicyState.lastHandle   = pDcgmHandle;
    m_fieldPolicyState.lastFieldId  = fieldId;
    m_fieldPolicyState.lastPolicyId = policyId;
    if (m_fieldPolicyState.enableReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.enableReturn;
    }
    if (auto const it = std::find_if(m_fieldPolicyState.policies.begin(),
                                     m_fieldPolicyState.policies.end(),
                                     [fieldId, policyId](dcgmPolicyInfo_t const &candidate) {
                                         return candidate.fieldId == fieldId && candidate.policyId == policyId;
                                     });
        it != m_fieldPolicyState.policies.end())
    {
        it->state = DCGM_POLICY_STATE_ENABLED;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyDisable(dcgmHandle_t pDcgmHandle, unsigned short fieldId, uint64_t policyId)
{
    m_fieldPolicyState.disableCallCount++;
    m_fieldPolicyState.lastHandle   = pDcgmHandle;
    m_fieldPolicyState.lastFieldId  = fieldId;
    m_fieldPolicyState.lastPolicyId = policyId;
    if (m_fieldPolicyState.disableReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.disableReturn;
    }
    if (auto const it = std::find_if(m_fieldPolicyState.policies.begin(),
                                     m_fieldPolicyState.policies.end(),
                                     [fieldId, policyId](dcgmPolicyInfo_t const &candidate) {
                                         return candidate.fieldId == fieldId && candidate.policyId == policyId;
                                     });
        it != m_fieldPolicyState.policies.end())
    {
        it->state = DCGM_POLICY_STATE_DISABLED;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyGetOne(dcgmHandle_t pDcgmHandle,
                                           unsigned short fieldId,
                                           uint64_t policyId,
                                           dcgmPolicyInfo_t *policy)
{
    m_fieldPolicyState.getOneCallCount++;
    m_fieldPolicyState.lastHandle   = pDcgmHandle;
    m_fieldPolicyState.lastFieldId  = fieldId;
    m_fieldPolicyState.lastPolicyId = policyId;
    if (policy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_fieldPolicyState.getOneReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.getOneReturn;
    }
    auto const it = std::find_if(m_fieldPolicyState.policies.begin(),
                                 m_fieldPolicyState.policies.end(),
                                 [fieldId, policyId](dcgmPolicyInfo_t const &candidate) {
                                     return candidate.fieldId == fieldId && candidate.policyId == policyId;
                                 });
    if (it == m_fieldPolicyState.policies.end())
    {
        return DCGM_ST_NO_DATA;
    }
    *policy = *it;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyGetAll(dcgmHandle_t pDcgmHandle, dcgmPolicyInfo_t *policies, unsigned int *count)
{
    m_fieldPolicyState.getAllCallCount++;
    m_fieldPolicyState.lastHandle = pDcgmHandle;
    if (count == nullptr || (*count > 0 && policies == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }
    m_fieldPolicyState.lastRequestedCount = *count;
    if (m_fieldPolicyState.getAllReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.getAllReturn;
    }
    auto const capacity  = *count;
    auto const available = static_cast<unsigned int>(m_fieldPolicyState.policies.size());
    auto const toCopy    = std::min(capacity, available);
    std::copy_n(m_fieldPolicyState.policies.begin(), toCopy, policies);
    *count = available;
    return available > capacity ? DCGM_ST_INSUFFICIENT_SIZE : DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyImport(dcgmHandle_t pDcgmHandle, const char *path)
{
    m_fieldPolicyState.importCallCount++;
    m_fieldPolicyState.lastHandle     = pDcgmHandle;
    m_fieldPolicyState.lastImportPath = path == nullptr ? "" : path;
    return path == nullptr ? DCGM_ST_BADPARAM : m_fieldPolicyState.importReturn;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyGetViolations(dcgmHandle_t pDcgmHandle,
                                                  int64_t sinceTimestamp,
                                                  dcgm_policy_violation_t *violations,
                                                  unsigned int *count)
{
    m_fieldPolicyState.getViolationsCallCount++;
    m_fieldPolicyState.lastHandle         = pDcgmHandle;
    m_fieldPolicyState.lastSinceTimestamp = sinceTimestamp;
    if (count == nullptr || (*count > 0 && violations == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }
    m_fieldPolicyState.lastRequestedCount = *count;
    if (m_fieldPolicyState.getViolationsReturn != DCGM_ST_OK)
    {
        return m_fieldPolicyState.getViolationsReturn;
    }
    auto const capacity  = *count;
    auto const available = static_cast<unsigned int>(m_fieldPolicyState.violations.size());
    auto const toCopy    = std::min(capacity, available);
    std::copy_n(m_fieldPolicyState.violations.begin(), toCopy, violations);
    *count = available;
    return available > capacity ? DCGM_ST_INSUFFICIENT_SIZE : DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmHealthGet(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    m_healthState.getCallCount++;
    m_healthState.lastHandle  = pDcgmHandle;
    m_healthState.lastGroupId = groupId;

    if (systems == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_healthState.getReturn != DCGM_ST_OK)
    {
        return m_healthState.getReturn;
    }

    *systems = m_healthState.systems;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmHealthSet_v2(dcgmHandle_t pDcgmHandle, dcgmHealthSetParams_v2 *params)
{
    m_healthState.setCallCount++;
    m_healthState.lastHandle = pDcgmHandle;

    if (params == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_healthState.lastGroupId   = params->groupId;
    m_healthState.lastSetParams = *params;
    return m_healthState.setReturn;
}

dcgmReturn_t MockDcgmOps::dcgmHealthCheck(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results)
{
    m_healthState.checkCallCount++;
    m_healthState.lastHandle  = pDcgmHandle;
    m_healthState.lastGroupId = groupId;

    if (results == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_healthState.checkReturn != DCGM_ST_OK)
    {
        return m_healthState.checkReturn;
    }

    *results = m_healthState.response;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmWatchPidFields(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             long long updateFreq,
                                             double maxKeepAge,
                                             int maxKeepSamples)
{
    m_processStatsState.watchPidCallCount++;
    m_processStatsState.lastHandle         = pDcgmHandle;
    m_processStatsState.lastGroupId        = groupId;
    m_processStatsState.lastUpdateFreq     = updateFreq;
    m_processStatsState.lastMaxKeepAge     = maxKeepAge;
    m_processStatsState.lastMaxKeepSamples = maxKeepSamples;
    return m_processStatsState.watchPidReturn;
}

dcgmReturn_t MockDcgmOps::dcgmGetPidInfo(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmPidInfo_t *pidInfo)
{
    m_processStatsState.getPidCallCount++;
    m_processStatsState.lastHandle  = pDcgmHandle;
    m_processStatsState.lastGroupId = groupId;

    if (pidInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_processStatsState.getPidReturn != DCGM_ST_OK)
    {
        return m_processStatsState.getPidReturn;
    }

    auto pid     = pidInfo->pid;
    *pidInfo     = m_processStatsState.pidInfo;
    pidInfo->pid = pid;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmJobStartStats(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, char jobId[64])
{
    m_processStatsState.startCallCount++;
    m_processStatsState.lastHandle  = pDcgmHandle;
    m_processStatsState.lastGroupId = groupId;
    m_processStatsState.lastJobId   = jobId == nullptr ? "" : jobId;
    return m_processStatsState.startReturn;
}

dcgmReturn_t MockDcgmOps::dcgmJobStopStats(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    m_processStatsState.stopCallCount++;
    m_processStatsState.lastHandle = pDcgmHandle;
    m_processStatsState.lastJobId  = jobId == nullptr ? "" : jobId;
    return m_processStatsState.stopReturn;
}

dcgmReturn_t MockDcgmOps::dcgmJobRemove(dcgmHandle_t pDcgmHandle, char jobId[64])
{
    m_processStatsState.removeCallCount++;
    m_processStatsState.lastHandle = pDcgmHandle;
    m_processStatsState.lastJobId  = jobId == nullptr ? "" : jobId;
    return m_processStatsState.removeReturn;
}

dcgmReturn_t MockDcgmOps::dcgmJobRemoveAll(dcgmHandle_t pDcgmHandle)
{
    m_processStatsState.removeAllCallCount++;
    m_processStatsState.lastHandle = pDcgmHandle;
    return m_processStatsState.removeAllReturn;
}

dcgmReturn_t MockDcgmOps::dcgmJobGetStats(dcgmHandle_t pDcgmHandle, char jobId[64], dcgmJobInfo_t *pJobInfo)
{
    m_processStatsState.getJobCallCount++;
    m_processStatsState.lastHandle = pDcgmHandle;
    m_processStatsState.lastJobId  = jobId == nullptr ? "" : jobId;

    if (pJobInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_processStatsState.getJobReturn != DCGM_ST_OK)
    {
        return m_processStatsState.getJobReturn;
    }

    *pJobInfo = m_processStatsState.jobInfo;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmIntrospectGetHostengineMemoryUsage(dcgmHandle_t pDcgmHandle,
                                                                 dcgmIntrospectMemory_t *memoryInfo,
                                                                 int waitIfNoData)
{
    m_introspectState.memoryCallCount++;
    m_introspectState.lastHandle     = pDcgmHandle;
    m_introspectState.lastMemoryWait = waitIfNoData;

    if (memoryInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_introspectState.lastMemoryVersion = memoryInfo->version;
    if (m_introspectState.memoryReturn != DCGM_ST_OK)
    {
        return m_introspectState.memoryReturn;
    }

    *memoryInfo = m_introspectState.memory;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmIntrospectGetHostengineCpuUtilization(dcgmHandle_t pDcgmHandle,
                                                                    dcgmIntrospectCpuUtil_t *cpuUtil,
                                                                    int waitIfNoData)
{
    m_introspectState.cpuCallCount++;
    m_introspectState.lastHandle  = pDcgmHandle;
    m_introspectState.lastCpuWait = waitIfNoData;

    if (cpuUtil == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_introspectState.lastCpuVersion = cpuUtil->version;
    if (m_introspectState.cpuReturn != DCGM_ST_OK)
    {
        return m_introspectState.cpuReturn;
    }

    *cpuUtil = m_introspectState.cpu;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmModuleDenylist(dcgmHandle_t pDcgmHandle, dcgmModuleId_t moduleId)
{
    m_moduleState.denylistCallCount++;
    m_moduleState.lastHandle     = pDcgmHandle;
    m_moduleState.lastDenylistId = moduleId;
    return m_moduleState.denylistReturn;
}

dcgmReturn_t MockDcgmOps::dcgmModuleGetStatuses(dcgmHandle_t pDcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses)
{
    m_moduleState.statusesCallCount++;
    m_moduleState.lastHandle = pDcgmHandle;

    if (moduleStatuses == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_moduleState.statusesReturn != DCGM_ST_OK)
    {
        return m_moduleState.statusesReturn;
    }

    *moduleStatuses = m_moduleState.statuses;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmStartEmbedded_v2(dcgmStartEmbeddedV2Params_v1 *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmEngineRun(unsigned short, char const *, unsigned int)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmStatusPopError(dcgmStatus_t, dcgmErrorInfo_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmPolicyRegister_v2(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgmPolicyCondition_t condition,
                                                fpRecvUpdates callback,
                                                uint64_t userData)
{
    m_policyState.registerCallCount++;
    m_policyState.lastHandle              = pDcgmHandle;
    m_policyState.lastGroupId             = groupId;
    m_policyState.lastRegisteredCondition = condition;
    m_policyState.lastCallback            = callback;
    m_policyState.lastUserData            = userData;
    return m_policyState.registerReturn;
}

dcgmReturn_t MockDcgmOps::dcgmModuleIdToName(dcgmModuleId_t, char const **)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmActionValidate_v2(dcgmHandle_t pDcgmHandle, dcgmRunDiag_v10 *, dcgmDiagResponse_v12 *)
{
    m_actionValidateState.actionValidateCallCount++;
    m_actionValidateState.lastHandle = pDcgmHandle;

    return m_actionValidateState.actionValidateReturn;
}

dcgmReturn_t MockDcgmOps::dcgmStopDiagnostic(dcgmHandle_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmRunMnDiagnostic(dcgmHandle_t pDcgmHandle,
                                              dcgmRunMnDiag_v2 const *drmnd,
                                              dcgmMnDiagResponse_v2 *response)
{
    m_mnDiagState.runCallCount++;
    m_mnDiagState.lastRunHandle  = pDcgmHandle;
    m_mnDiagState.lastRunVersion = drmnd == nullptr ? 0 : drmnd->version;

    if (drmnd == nullptr || response == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (m_mnDiagState.runReturn != DCGM_ST_OK)
    {
        return m_mnDiagState.runReturn;
    }

    *response = *m_mnDiagState.response;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmStopMnDiagnostic(dcgmHandle_t pDcgmHandle)
{
    m_mnDiagState.stopCallCount++;
    m_mnDiagState.lastStopHandle = pDcgmHandle;
    return m_mnDiagState.stopReturn;
}

dcgmReturn_t MockDcgmOps::dcgmDiagSendHeartbeat(dcgmHandle_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmProfGetSupportedMetricGroups(dcgmHandle_t pDcgmHandle,
                                                           dcgmProfGetMetricGroups_t *metricGroups)
{
    m_profileState.metricGroupsCallCount++;
    m_profileState.lastHandle = pDcgmHandle;

    if (metricGroups == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (m_profileState.metricGroupsReturn != DCGM_ST_OK)
    {
        return m_profileState.metricGroupsReturn;
    }

    auto const requestedGpuId = metricGroups->gpuId;
    *metricGroups             = m_profileState.metricGroups;
    metricGroups->gpuId       = requestedGpuId;
    m_profileState.lastGpuId  = requestedGpuId;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmProfPause(dcgmHandle_t pDcgmHandle)
{
    m_profileState.pauseCallCount++;
    m_profileState.lastHandle = pDcgmHandle;
    return m_profileState.pauseReturn;
}

dcgmReturn_t MockDcgmOps::dcgmProfResume(dcgmHandle_t pDcgmHandle)
{
    m_profileState.resumeCallCount++;
    m_profileState.lastHandle = pDcgmHandle;
    return m_profileState.resumeReturn;
}

dcgmReturn_t MockDcgmOps::dcgmGetCacheManagerFieldInfo(dcgmHandle_t pDcgmHandle,
                                                       dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
{
    m_cacheManagerState.fieldInfoCallCount++;
    m_cacheManagerState.lastHandle = pDcgmHandle;

    if (fieldInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_cacheManagerState.requestedFieldInfo = *fieldInfo;
    if (m_cacheManagerState.fieldInfoReturn != DCGM_ST_OK)
    {
        return m_cacheManagerState.fieldInfoReturn;
    }

    auto response          = m_cacheManagerState.fieldInfo;
    response.entityGroupId = fieldInfo->entityGroupId;
    response.entityId      = fieldInfo->entityId;
    response.fieldId       = fieldInfo->fieldId;
    *fieldInfo             = response;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmOps::dcgmInjectFieldValue(dcgmHandle_t pDcgmHandle,
                                               unsigned int gpuId,
                                               dcgmInjectFieldValue_t *fieldValue)
{
    m_cacheManagerState.injectCallCount++;
    m_cacheManagerState.lastHandle = pDcgmHandle;
    m_cacheManagerState.lastGpuId  = gpuId;

    if (fieldValue == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    m_cacheManagerState.injectedValue = *fieldValue;
    return m_cacheManagerState.injectReturn;
}

dcgmReturn_t MockDcgmOps::dcgmVgpuConfigGet(dcgmHandle_t,
                                            dcgmGpuGrp_t,
                                            dcgmConfigType_t,
                                            int,
                                            dcgmVgpuConfig_t *,
                                            dcgmStatus_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmVgpuConfigSet(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmVgpuConfig_t *deviceConfig,
                                            dcgmStatus_t statusHandle)
{
    m_configState.vgpuSetCallCount++;
    m_configState.lastHandle       = pDcgmHandle;
    m_configState.lastGroupId      = groupId;
    m_configState.lastStatusHandle = statusHandle;
    if (deviceConfig != nullptr)
    {
        m_configState.lastVgpuSetConfig = *deviceConfig;
    }
    return m_configState.vgpuSetReturn;
}

dcgmReturn_t MockDcgmOps::dcgmGetVgpuDeviceAttributes(dcgmHandle_t, unsigned int, dcgmVgpuDeviceAttributes_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t MockDcgmOps::dcgmGetVgpuInstanceAttributes(dcgmHandle_t, unsigned int, dcgmVgpuInstanceAttributes_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmErrorSeverity_t MockDcgmOps::dcgmErrorGetPriorityByCode(unsigned int code)
{
    if (auto const it = m_errorPriorities.find(code); it != m_errorPriorities.end())
    {
        return it->second;
    }
    return DCGM_ERROR_UNKNOWN;
}

void MockDcgmOps::SetErrorPriorityByCode(unsigned int code, dcgmErrorSeverity_t severity)
{
    m_errorPriorities[code] = severity;
}

} //namespace DcgmNs
