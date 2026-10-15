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

#include <DcgmCacheManager.h>
#include <DcgmFieldGroup.h>
#include <DcgmGroupManager.h>
#include <DcgmHostEngineHandler.h>
#include <ScopedDcgmFields.hpp>
#include <dcgm_fields.h>
#include <mock/MockDcgmCacheManager.h>

#include <atomic>
#include <cstddef>
#include <dcgm_module_structs.h>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

/**
 * MockDcgmCacheManagerBridge adapts a MockDcgmCacheManager to the DcgmCacheManager*
 * interface used by tests that exercise raw DcgmCacheManager* construction paths.
 * Only the methods DcgmGroupManager actually calls are forwarded to the mock;
 * everything else falls through to the base class.
 */
class MockDcgmCacheManagerBridge : public DcgmCacheManager
{
public:
    explicit MockDcgmCacheManagerBridge(MockDcgmCacheManager &mock)
        : DcgmCacheManager()
        , m_mock(mock)
    {}

    dcgmReturn_t GetAllEntitiesOfEntityGroup(int activeOnly,
                                             dcgm_field_entity_group_t entityGroupId,
                                             std::vector<dcgmGroupEntityPair_t> &entities) override
    {
        return m_mock.GetAllEntitiesOfEntityGroup(activeOnly, entityGroupId, entities);
    }

private:
    MockDcgmCacheManager &m_mock;
};

/**
 * TestDcgmHostEngineHandler provides a minimal, standalone DcgmHostEngineHandler
 * for unit tests via the friend-class relationship declared in
 * DcgmHostEngineHandler.h.  Shared by all dcgmlibtests translation units to
 * satisfy the ODR requirement of a single class definition.
 */
class TestDcgmHostEngineHandler
    : private ScopedDcgmFields
    , public DcgmHostEngineHandler
{
public:
    TestDcgmHostEngineHandler()
        : DcgmHostEngineHandler()
    {
        /* Initialise module metadata the way the real constructor does so that
           methods iterating m_modules work correctly in tests. */
        for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
        {
            m_modules[i].id     = static_cast<dcgmModuleId_t>(i);
            m_modules[i].status = DcgmModuleStatusNotLoaded;
            m_modules[i].ptr    = nullptr;
        }
        /* Register as the singleton so that code paths calling
           DcgmHostEngineHandler::Instance() work in tests. */
        mpHostEngineHandlerInstance = this;
    }

    ~TestDcgmHostEngineHandler()
    {
        /* Injected managers are not owned by this handler; null them so the
         * base-class destructor does not destroy them. */
        ResetCacheManager();
        mpGroupManager      = nullptr;
        mpFieldGroupManager = nullptr;
        /* Unregister the singleton so it does not dangle after this test. */
        if (mpHostEngineHandlerInstance == this)
        {
            mpHostEngineHandlerInstance = nullptr;
        }
    }

    /* Returns the number of currently tracked job records. */
    size_t GetJobCount() const
    {
        return mJobIdMap.size();
    }

    /* Directly set a module's status (bypasses load logic). */
    void SetModuleStatus(dcgmModuleId_t id, dcgmModuleStatus_t status)
    {
        m_modules[id].status = status;
    }

    /* Mark a module slot as loaded so ptr-checks in NotifyLoggingSeverityChange pass.
       Uses a non-null sentinel; the SendModuleMessage override never dereferences ptr. */
    void SetModuleLoaded(dcgmModuleId_t id)
    {
        m_modules[id].ptr = reinterpret_cast<DcgmModule *>(1);
    }

    struct CapturedModuleMessage
    {
        dcgmModuleId_t moduleId;
        unsigned int subCommand;
        unsigned int version;
        unsigned int length;
        unsigned int groupId; /* populated for DCGM_CORE_SR_GROUP_REMOVED */
        unsigned int gpuId;   /* populated for DCGM_CORE_SR_MIG_UPDATED */
    };
    std::vector<CapturedModuleMessage> m_moduleMessages;

    dcgmReturn_t SendModuleMessage(dcgmModuleId_t id, dcgm_module_command_header_t *cmd) override
    {
        if (m_modules[id].ptr == nullptr)
            return DcgmHostEngineHandler::SendModuleMessage(id, cmd);
        unsigned int capturedGroupId = 0;
        if (cmd->subCommand == DCGM_CORE_SR_GROUP_REMOVED)
            capturedGroupId = reinterpret_cast<dcgm_core_msg_group_removed_t *>(cmd)->groupId;
        unsigned int capturedGpuId = 0;
        if (cmd->subCommand == DCGM_CORE_SR_MIG_UPDATED)
            capturedGpuId = reinterpret_cast<dcgm_core_msg_mig_updated_t *>(cmd)->gpuId;
        m_moduleMessages.push_back({ id,
                                     cmd->subCommand,
                                     cmd->version,
                                     static_cast<unsigned int>(cmd->length),
                                     capturedGroupId,
                                     capturedGpuId });
        return DCGM_ST_OK;
    }

    /* Read back a module's status. */
    dcgmModuleStatus_t GetModuleStatus(dcgmModuleId_t id) const
    {
        return m_modules[id].status;
    }

    /* Expose the private RemoveAllTrackedRequests for unit testing. */
    dcgmReturn_t CallRemoveAllTrackedRequests()
    {
        return RemoveAllTrackedRequests();
    }

    /* Direct read/write of m_hostengineHealth for unit testing. */
    void SetHostEngineHealth(unsigned int value)
    {
        m_hostengineHealth = value;
    }

    /* Returns the number of currently tracked watched requests. */
    size_t GetWatchedRequestCount() const
    {
        return m_watchedRequests.size();
    }

    /* Direct write of NVML flag members for unit testing. */
    void SetUsingInjectionNvml(bool value)
    {
        m_usingInjectionNvml = value;
    }

    void SetNvmlLoaded(bool value)
    {
        m_nvmlLoaded.store(value, std::memory_order_release);
    }

    /* Inject a pre-built adapter (production or test use). */
    void SetCacheManager(DcgmCacheManagerAnyUniquePtr cm)
    {
        ReplaceCacheManager(std::move(cm));
    }

    MockDcgmCacheManager &SetMockCacheManager()
    {
        auto mock     = std::make_unique<MockDcgmCacheManager>();
        auto &mockRef = *mock;
        ReplaceCacheManager(std::move(mock));
        return mockRef;
    }

    /* Inject a real or mock group manager for tests. */
    void SetGroupManager(DcgmGroupManager *gm)
    {
        mpGroupManager = gm;
    }

    /* Inject a real or mock field group manager for tests. */
    void SetFieldGroupManager(DcgmFieldGroupManager *fgm)
    {
        mpFieldGroupManager = fgm;
    }

    void ReplaceCacheManager(DcgmCacheManagerAnyUniquePtr cm)
    {
        std::destroy_at(&mpCacheManager);
        std::construct_at(&mpCacheManager, std::move(cm));
    }

    void SetFieldGroupPidAndJobStats(dcgmFieldGrp_t id)
    {
        mFieldGroupPidAndJobStats = id;
    }

    /* Register a hook that intercepts ProcessModuleCommand for a specific
     * (moduleId, subCommand) pair.  The hook receives the raw header pointer
     * so it can cast to the concrete message struct, mutate fields, and
     * return whatever dcgmReturn_t it likes.  When no hook matches the call
     * falls through to the real DcgmHostEngineHandler::ProcessModuleCommand. */
    using ModuleCommandHook = std::function<dcgmReturn_t(dcgm_module_command_header_t *)>;
    std::map<std::pair<dcgmModuleId_t, unsigned int>, ModuleCommandHook> m_moduleCommandHooks;

    dcgmReturn_t ProcessModuleCommand(dcgm_module_command_header_t *cmd) override
    {
        auto key = std::make_pair(cmd->moduleId, cmd->subCommand);
        auto it  = m_moduleCommandHooks.find(key);
        if (it != m_moduleCommandHooks.end())
            return it->second(cmd);
        return DcgmHostEngineHandler::ProcessModuleCommand(cmd);
    }

    /* Expose RemoveUnhealthyGpus (private) for direct unit testing. */
    void CallRemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds)
    {
        RemoveUnhealthyGpus(gpuIds);
    }

    /* Expose private methods for testing */
    dcgmReturn_t GetCachedOrLiveValueForEntity(dcgmGroupEntityPair_t entity,
                                               unsigned short fieldId,
                                               DcgmFvBuffer &fvBuffer)
    {
        return DcgmHostEngineHandler::GetCachedOrLiveValueForEntity(entity, fieldId, fvBuffer);
    }

    dcgmReturn_t CallGetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
    {
        return DcgmHostEngineHandler::GetCacheManagerFieldInfo(fieldInfo);
    }

    bool GetGpusDetached() const
    {
        return m_gpusDetached.load();
    }

    void CallProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message)
    {
        DcgmHostEngineHandler::ProcessMessage(connectionId, std::move(message));
    }

    dcgmReturn_t CallWatchHostEngineFields()
    {
        return WatchHostEngineFields();
    }

    dcgmFieldGrp_t GetFieldGroup30Sec() const
    {
        return mFieldGroup30Sec;
    }

    dcgmFieldGrp_t GetFieldGroupHourly() const
    {
        return mFieldGroupHourly;
    }

    dcgmFieldGrp_t GetFieldGroupPidAndJobStats() const
    {
        return mFieldGroupPidAndJobStats;
    }

    struct CapturedIpcSend
    {
        dcgm_connection_id_t connectionId;
        std::unique_ptr<DcgmMessage> message;
    };
    std::vector<CapturedIpcSend> m_ipcSends;
    dcgmReturn_t m_sendMessageViaIpcRet = DCGM_ST_OK;

    dcgmReturn_t SendMessageViaIpc(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message) override
    {
        m_ipcSends.push_back({ connectionId, std::move(message) });
        return m_sendMessageViaIpcRet;
    }

    dcgmReturn_t CallHelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short fieldId,
                                               dcgmStatSummaryInt64_t *summary,
                                               long long startTime,
                                               long long endTime)
    {
        return HelperGetInt64StatSummary(entityGroupId, entityId, fieldId, summary, startTime, endTime);
    }

    dcgmReturn_t CallHelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short fieldId,
                                               dcgmStatSummaryInt32_t *summary,
                                               long long startTime,
                                               long long endTime)
    {
        return HelperGetInt32StatSummary(entityGroupId, entityId, fieldId, summary, startTime, endTime);
    }

    dcgmReturn_t CallHelperHealthCheck(unsigned int groupId,
                                       long long startTime,
                                       long long endTime,
                                       dcgmHealthResponse_v5 &response)
    {
        return HelperHealthCheck(groupId, startTime, endTime, response);
    }

    dcgmReturn_t CallWatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                            timelib64_t monitorIntervalUsec,
                                            double maxSampleAge,
                                            int maxKeepSamples,
                                            int activeOnly,
                                            DcgmWatcher const &watcher)
    {
        return WatchFieldGroupAllGpus(
            fieldGroupId, monitorIntervalUsec, maxSampleAge, maxKeepSamples, activeOnly, watcher);
    }

    void SetDcgmIpc(std::unique_ptr<DcgmIpc> ipc)
    {
        m_dcgmIpc = std::move(ipc);
    }
};
