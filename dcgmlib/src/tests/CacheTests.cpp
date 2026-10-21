/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <TimeLib.hpp>
#include <array>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <dcgm_agent.h>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#include <DcgmCMUtils.h>
#include <DcgmCacheManager.h>
#include <Defer.hpp>
#include <DeferredPrmCounterApply.h>
#include <ScopedDcgmFields.hpp>
#include <ScopedEnvVars.hpp>
#include <UnitTestHelpers.h>
#include <dcgm_fields.h>
#include <dcgm_prm_structs.h>
#include <nvml_injection.h>
#include <vector>

#include <arpa/inet.h>

namespace
{
/**
 * Test adapter that exposes protected cache-manager helpers for focused unit tests.
 */
class TestDcgmCacheManager : public DcgmCacheManager
{
public:
    void RunOneUpdateCycleForTest()
    {
        m_updateThreadCtx = std::make_unique<dcgmcm_update_thread_t>();
        DoOneUpdateAllFields();
    }

    using DcgmCacheManager::AddFakeComputeInstance;
    using DcgmCacheManager::AddFakeGpu;
    using DcgmCacheManager::AddFakeInstance;
    using DcgmCacheManager::AddFieldWatch;
    using DcgmCacheManager::AddOrUpdateWatcher;
    using DcgmCacheManager::AllocWatchInfo;
    using DcgmCacheManager::AllocWatchInfoTimeSeries;
    using DcgmCacheManager::AppendEntityBlob;
    using DcgmCacheManager::AppendEntityDouble;
    using DcgmCacheManager::AppendEntityInt64;
    using DcgmCacheManager::AppendEntityString;
    using DcgmCacheManager::AppendSamples;
    using DcgmCacheManager::AreAllGpuIdsSameSku;
    using DcgmCacheManager::BufferOrCacheLatestGpuValue;
    using DcgmCacheManager::CachePrmField;
    using DcgmCacheManager::CheckValidGlobalField;
    using DcgmCacheManager::CheckValidGpuField;
    using DcgmCacheManager::ClearAllEntities;
    using DcgmCacheManager::ClearEntity;
    using DcgmCacheManager::ClearGpuMigInfo;
    using DcgmCacheManager::ClearWatchInfo;
    using DcgmCacheManager::DriverVersionIsAtLeast;
    using DcgmCacheManager::EnforceWatchInfoQuota;
    using DcgmCacheManager::EntityIdToWatchKey;
    using DcgmCacheManager::EntityKeySupportsGpm;
    using DcgmCacheManager::EntityPairSupportsGpm;
    using DcgmCacheManager::FilterActiveEntities;
    using DcgmCacheManager::FreeSamples;
    using DcgmCacheManager::FreeWatchInfo;
    using DcgmCacheManager::GenerateCudaVisibleDevicesValue;
    using DcgmCacheManager::GetActiveGpuHandles;
    using DcgmCacheManager::GetAllEntitiesOfEntityGroup;
    using DcgmCacheManager::GetAllGpuInfo;
    using DcgmCacheManager::GetAllWatchObjects;
    using DcgmCacheManager::GetComputeInstanceEntityId;
    using DcgmCacheManager::GetComputeInstanceNvmlDevice;
    using DcgmCacheManager::GetEntityNvLinkLinkStatus;
    using DcgmCacheManager::GetEntityStatus;
    using DcgmCacheManager::GetEntityWatchInfo;
    using DcgmCacheManager::GetEntityWatchInfoSnapshot;
    using DcgmCacheManager::GetFMStatusFromStruct;
    using DcgmCacheManager::GetFp64SummaryData;
    using DcgmCacheManager::GetGlobalWatchInfo;
    using DcgmCacheManager::GetGpuArch;
    using DcgmCacheManager::GetGpuBrand;
    using DcgmCacheManager::GetGpuCount;
    using DcgmCacheManager::GetGpuExcludeList;
    using DcgmCacheManager::GetGpuId;
    using DcgmCacheManager::GetGpuIdForEntity;
    using DcgmCacheManager::GetGpuIds;
    using DcgmCacheManager::GetGpuStatus;
    using DcgmCacheManager::GetInstanceEntityId;
    using DcgmCacheManager::GetInstanceProfile;
    using DcgmCacheManager::GetInt64SummaryData;
    using DcgmCacheManager::GetLatestSample;
    using DcgmCacheManager::GetMigIndicesForEntity;
    using DcgmCacheManager::GetMultipleLatestLiveSamples;
    using DcgmCacheManager::GetMultipleLatestSamples;
    using DcgmCacheManager::GetNvLinkTopologyInformation;
    using DcgmCacheManager::GetNvmlDeviceFromEntityId;
    using DcgmCacheManager::GetProfModuleServicedEntities;
    using DcgmCacheManager::GetRuntimeStats;
    using DcgmCacheManager::GetSafeNvmlHandle;
    using DcgmCacheManager::GetSamples;
    using DcgmCacheManager::GetTopologyHelper;
    using DcgmCacheManager::GetValidFieldIds;
    using DcgmCacheManager::GetWorkloadPowerProfilesInfo;
    using DcgmCacheManager::GpuIdToNvmlIndex;
    using DcgmCacheManager::Init;
    using DcgmCacheManager::InitializeGpuInstances;
    using DcgmCacheManager::InitializeNvmlEventSet;
    using DcgmCacheManager::InjectSamples;
    using DcgmCacheManager::InsertBlankValue;
    using DcgmCacheManager::InsertNvmlErrorValue;
    using DcgmCacheManager::IsGpuAllowlisted;
    using DcgmCacheManager::IsGpuMigEnabled;
    using DcgmCacheManager::IsMigEnabledAnywhere;
    using DcgmCacheManager::IsModulePushedFieldId;
    using DcgmCacheManager::m_delayedMigReconfigProcessingTimestamp;
    using DcgmCacheManager::m_driverFullVersion;
    using DcgmCacheManager::m_driverMajorVersion;

    using DcgmCacheManager::m_driverVersion;
    using DcgmCacheManager::m_forceProfMetricsThroughGpm;
    using DcgmCacheManager::m_gpus;
    using DcgmCacheManager::m_haveAnyLiveSubscribers;
    using DcgmCacheManager::m_metaGroupWatchedFields;
    using DcgmCacheManager::m_numComputeInstances;
    using DcgmCacheManager::m_numGpus;
    using DcgmCacheManager::m_numInstances;
    using DcgmCacheManager::m_nvmlDriver;
    using DcgmCacheManager::m_nvmlEventSetInitialized;
    using DcgmCacheManager::m_nvmlLoaded;
    using DcgmCacheManager::m_skipDriverCalls;
    using DcgmCacheManager::ManageVgpuList;
    using DcgmCacheManager::MarkSubscribersInThreadCtx;
    using DcgmCacheManager::MergeNewlyDetectedGpuList;
    using DcgmCacheManager::NotifyMigUpdateSubscribers;
    using DcgmCacheManager::NvmlIndexToGpuId;
    using DcgmCacheManager::NvmlPostWatch;
    using DcgmCacheManager::NvmlPreWatch;
    using DcgmCacheManager::OnConnectionRemoveImpl;
    using DcgmCacheManager::pciBusGpuIdMap;
    using DcgmCacheManager::PopulateCpuAffinity;
    using DcgmCacheManager::PopulateNvLinkLinkStatus;
    using DcgmCacheManager::PrecheckWatchInfoForSamples;
    using DcgmCacheManager::ReadAndCacheFBMemoryInfo;
    using DcgmCacheManager::ReadAndCacheNvLinkBer;
    using DcgmCacheManager::ReadAndCacheNvLinkCountField;
    using DcgmCacheManager::ReadFabricManagerStatusField;
    using DcgmCacheManager::RecordXidForComputeInstance;
    using DcgmCacheManager::RecordXidForGpu;
    using DcgmCacheManager::RecordXidForGpuInstance;

    using DcgmCacheManager::RemoveEntityFieldWatch;
    using DcgmCacheManager::RemoveFieldWatch;
    using DcgmCacheManager::RemoveGlobalFieldWatch;
    using DcgmCacheManager::RemoveWatcher;
    using DcgmCacheManager::SetEntityNvLinkLinkState;
    using DcgmCacheManager::SetGpuNvLinkLinkState;
    using DcgmCacheManager::SetPracticalEntityInfo;
    using DcgmCacheManager::SetValueImpl;
    using DcgmCacheManager::SetWorkloadPowerProfile;

    using DcgmCacheManager::Start;
    using DcgmCacheManager::SubscribeForEvent;
    using DcgmCacheManager::UpdateAllFields;
    using DcgmCacheManager::UpdateNvLinkLinkState;
    using DcgmCacheManager::UpdateWatchFromWatchers;

    dcgmReturn_t RunOneUpdateCycle()
    {
        dcgmcm_update_thread_t threadCtx;
        timelib64_t earliestNextUpdate = 0;
        DcgmLockGuard dlg              = DcgmLockGuard(m_mutex);

        return ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);
    }

    std::pair<dcgmReturn_t, dcgmMutexReturn_t> RunOneUpdateCycleAndPollMutex()
    {
        dcgmcm_update_thread_t threadCtx;
        timelib64_t earliestNextUpdate = 0;
        DcgmLockGuard dlg              = DcgmLockGuard(m_mutex);
        dcgmReturn_t const ret         = ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);
        return { ret, m_mutex->Poll() };
    }
};

dcgmcm_gpu_info_t MakeDetectedGpu(unsigned int gpuId,
                                  char const *uuid,
                                  char const *busId,
                                  dcgmGpuBrandType_t brand    = DCGM_GPU_BRAND_GEFORCE,
                                  dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_AMPERE)
{
    dcgmcm_gpu_info_t gpu {};
    gpu.gpuId  = gpuId;
    gpu.status = DcgmEntityStatusOk;
    gpu.brand  = brand;
    gpu.arch   = arch;
    SafeCopyTo(gpu.uuid, uuid);
    SafeCopyTo(gpu.pciInfo.busIdLegacy, busId);
    SafeCopyTo(gpu.deviceName, "detected-gpu");
    return gpu;
}
constexpr std::array<unsigned short, 3> VgpuTypeFieldIds {
    DCGM_FI_DEV_VGPU_TYPE_NAME,
    DCGM_FI_DEV_VGPU_TYPE_CLASS,
    DCGM_FI_DEV_VGPU_TYPE_LICENSE,
};

void WatchField(TestDcgmCacheManager &cacheManager,
                unsigned int gpuId,
                unsigned short fieldId,
                DcgmWatcher const &watcher)
{
    bool wereFirstWatcher = false;
    auto ret              = cacheManager.AddFieldWatch(
        DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
    REQUIRE(ret == DCGM_ST_OK);
}

template <typename Validator>
void CheckLatestBlob(TestDcgmCacheManager &cacheManager,
                     unsigned int gpuId,
                     unsigned short fieldId,
                     Validator &&validator)
{
    dcgmcm_sample_t sample {};
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
    REQUIRE(sample.val.blob != nullptr);
    validator(sample.val.blob);
    REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
}

struct FvSubscriptionCallbackState
{
    int calls           = 0;
    int watcherCount    = 0;
    size_t elementCount = 0;
    std::array<DcgmWatcherType_t, 8> watchers {};
};

[[maybe_unused]] void FvSubscriptionCallback(DcgmFvBuffer *fvBuffer,
                                             DcgmWatcherType_t *watcherTypes,
                                             int numWatcherTypes,
                                             void *userData)
{
    auto *state = static_cast<FvSubscriptionCallbackState *>(userData);
    state->calls++;
    state->watcherCount = numWatcherTypes;
    for (int i = 0; i < numWatcherTypes && i < static_cast<int>(state->watchers.size()); i++)
    {
        state->watchers[i] = watcherTypes[i];
    }

    size_t bufferSize = 0;
    REQUIRE(fvBuffer->GetSize(&bufferSize, &state->elementCount) == DCGM_ST_OK);
}

[[maybe_unused]] bool UseOnlySamplesAtOrAfterTimestamp(timeseries_entry_p entry, void *userData)
{
    auto const cutoff = *static_cast<timelib64_t const *>(userData);
    return entry->usecSince1970 >= cutoff;
}

nvmlPRMCounterList_v1_t MakeInjectedPrmCounterResponse(nvmlPRMCounter_v1_t &counter,
                                                       uint32_t value,
                                                       nvmlReturn_t status        = NVML_SUCCESS,
                                                       nvmlValueType_t outputType = NVML_VALUE_TYPE_UNSIGNED_INT)
{
    counter                                = {};
    counter.inData.localPort               = PRM_LOCAL_PORT_MIN;
    counter.counterId                      = NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY;
    counter.counterValue.status            = status;
    counter.counterValue.outputType        = outputType;
    counter.counterValue.outputValue.uiVal = value;

    nvmlPRMCounterList_v1_t counterList = {};
    counterList.numCounters             = 1;
    counterList.counters                = &counter;
    return counterList;
}

uint32_t GetNvmlFuncCallCount(std::string_view funcName)
{
    injectNvmlFuncCallCounts_t funcCallCounts = {};
    REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
    for (unsigned int i = 0; i < funcCallCounts.numFuncs; ++i)
    {
        if (funcName == funcCallCounts.funcCallInfo[i].funcName)
        {
            return funcCallCounts.funcCallInfo[i].funcCallCount;
        }
    }
    return 0;
}

} // namespace

std::unordered_set<uint32_t> ReadEnvForFatalXids();

#if defined(NV_VMWARE)
/* No. of iterations corresponding to different sample set of vgpuIds */
#define NUM_VGPU_LISTS             5
#define TEST_MAX_NUM_VGPUS_PER_GPU 16

TEST_CASE("CacheManager: Test VgpuManageList")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    unsigned int gpuId = cm.AddFakeGpu();
    nvmlVgpuInstance_t vgpuIds[NUM_VGPU_LISTS][TEST_MAX_NUM_VGPUS_PER_GPU]
        = { { 11, 41, 52, 61, 32, 45, 91, 21, 43, 29, 19, 93, 0, 0, 0, 0 },
            { 8, 32, 45, 91, 21, 43, 29, 19, 93, 0, 0, 0, 0, 0, 0, 0 },
            { 7, 41, 52, 32, 45, 91, 21, 43, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 4, 41, 32, 91, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

    for (unsigned int i = 0; i < NUM_VGPU_LISTS; i++)
    {
        REQUIRE(cm.ManageVgpuList(gpuId, (unsigned int *)(vgpuIds + i)) == DCGM_ST_OK);

        fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_VM_NAME);
        REQUIRE(fieldMeta != nullptr);

        for (unsigned int j = 0; j < (TEST_MAX_NUM_VGPUS_PER_GPU - 1); j++)
        {
            /* Since 0 is not a valid vgpuId, so existing the loop as subsequent elements will also be zero */
            if (vgpuIds[i][j + 1] == 0)
                break;

            memset(&sample, 0, sizeof(sample));
            sample.timestamp = 0;
            switch (fieldMeta->fieldType)
            {
                case DCGM_FT_DOUBLE:
                    sample.val.d = 1.0;
                    break;

                case DCGM_FT_TIMESTAMP:
                    sample.val.i64 = timelib_usecSince1970();
                    break;

                case DCGM_FT_INT64:
                    sample.val.i64 = 1;
                    break;

                case DCGM_FT_STRING:
                    sample.val.str      = (char *)"nvidia"; /* Use static string so we don't have to alloc/free */
                    sample.val2.ptrSize = strlen(sample.val.str) + 1;
                    break;

                case DCGM_FT_BINARY:
                    /* Just insert any blob of data */
                    sample.val.blob     = &sample;
                    sample.val2.ptrSize = sizeof(sample);
                    break;

                default:
                    break;
            }

            /* Inject a fake value */
            REQUIRE(cm.InjectSamples(DCGM_FE_VGPU, vgpuIds[i][j + 1], fieldMeta->fieldId, &sample, 1) == 0);

            /* To verify retrieved sample against whatever sample which was injected in the cache */
            memset(&sample, 0, sizeof(sample));
            REQUIRE(cm.GetLatestSample(DCGM_FE_VGPU, vgpuIds[i][j + 1], fieldMeta->fieldId, &sample, 0) == DCGM_ST_OK);

            REQUIRE(std::string(sample.val.str) == std::string("nvidia"));

            REQUIRE(cm.FreeSamples(&sample, 1, fieldMeta->fieldId) == 0);
        }

        /* Inject-retrieve routine for vGPU field 'DCGM_FI_DEV_VGPU_TYPE' for single vgpuId(41) which is of int type of
         * value */
        if (i == 0)
        {
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            REQUIRE(fieldMeta != nullptr);

            sample.val.i64   = 1;
            sample.timestamp = 0;
            REQUIRE(cm.InjectSamples(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 1) == 0);

            memset(&sample, 0, sizeof(sample));

            REQUIRE(cm.GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0) == DCGM_ST_OK);

            REQUIRE(sample.val.i64 == 1);
        }

        /* To verify that no samples retrieved for a vgpuId 41 which has been removed from the List. */
        if ((i == (NUM_VGPU_LISTS - 1)) && (NUM_VGPU_LISTS != 1))
        {
            /* For vGPU field 'DCGM_FI_DEV_VGPU_VM_NAME' */
            REQUIRE(cm.GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0) == DCGM_ST_NOT_WATCHED);

            /* For vGPU field 'DCGM_FI_DEV_VGPU_TYPE' */
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            REQUIRE(fieldMeta != nullptr);

            REQUIRE(cm.GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0) == DCGM_ST_NOT_WATCHED);
        }
    }
}
#endif

TEST_CASE("CacheManager: Test GetGpuId")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    std::set<unsigned int> nonExistentInstances;
    std::set<unsigned int> nonExistentComputeInstances;
    for (unsigned int i = 0; i < 5; i++)
    {
        nonExistentInstances.insert(i);
        nonExistentComputeInstances.insert(i);
    }
    unsigned int gpuId = cm.AddFakeGpu();
    unsigned int instanceIds[2];
    instanceIds[0] = cm.AddFakeInstance(gpuId);
    instanceIds[1] = cm.AddFakeInstance(gpuId);
    unsigned int computeInstanceIds[2];
    computeInstanceIds[0] = cm.AddFakeComputeInstance(instanceIds[0]);
    computeInstanceIds[1] = cm.AddFakeComputeInstance(instanceIds[1]);

    REQUIRE(instanceIds[0] != instanceIds[1]);
    REQUIRE(instanceIds[0] / DCGM_MAX_INSTANCES_PER_GPU == gpuId);
    REQUIRE(instanceIds[1] / DCGM_MAX_INSTANCES_PER_GPU == gpuId);
    REQUIRE(computeInstanceIds[0] != computeInstanceIds[1]);
    REQUIRE(computeInstanceIds[0] / DCGM_MAX_COMPUTE_INSTANCES_PER_GPU == gpuId);
    REQUIRE(computeInstanceIds[1] / DCGM_MAX_COMPUTE_INSTANCES_PER_GPU == gpuId);

    unsigned int gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU, gpuId, gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, instanceIds[0], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);
    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, instanceIds[1], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, computeInstanceIds[0], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);
    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, computeInstanceIds[1], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    // Make sure we get a failure for the fake ones
    for (unsigned int i = 0; i < 2; i++)
    {
        nonExistentInstances.erase(instanceIds[i]);
        nonExistentComputeInstances.erase(computeInstanceIds[i]);
    }

    for (auto &&id : nonExistentInstances)
    {
        REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, id, gpuIdToSet) == DCGM_ST_INSTANCE_NOT_FOUND);
    }

    for (auto &&id : nonExistentComputeInstances)
    {
        REQUIRE(cm.GetGpuId(DCGM_FE_GPU_CI, id, gpuIdToSet) == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
    }
}

TEST_CASE("CacheManager: fake GPU instances respect per-GPU capacity")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    unsigned int const gpu0 = cm.AddFakeGpu();
    unsigned int const gpu1 = cm.AddFakeGpu();
    REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
    REQUIRE(gpu1 != DCGM_GPU_ID_BAD);

    for (unsigned int i = 0; i < DCGM_MAX_INSTANCES_PER_GPU; ++i)
    {
        unsigned int const instanceId = cm.AddFakeInstance(gpu0);
        CHECK(instanceId == gpu0 * DCGM_MAX_INSTANCES_PER_GPU + i);
    }

    REQUIRE(cm.m_gpus[gpu0].instances.size() == DCGM_MAX_INSTANCES_PER_GPU);
    CHECK(cm.m_numInstances == DCGM_MAX_INSTANCES_PER_GPU);

    CHECK(cm.AddFakeInstance(gpu0) == DCGM_ENTITY_ID_BAD);
    CHECK(cm.m_gpus[gpu0].instances.size() == DCGM_MAX_INSTANCES_PER_GPU);
    CHECK(cm.m_numInstances == DCGM_MAX_INSTANCES_PER_GPU);

    SECTION("Inject instance and check owner")
    {
        unsigned int const gpu1Instance = cm.AddFakeInstance(gpu1);
        REQUIRE(gpu1Instance == gpu1 * DCGM_MAX_INSTANCES_PER_GPU);
        CHECK(cm.m_numInstances == DCGM_MAX_INSTANCES_PER_GPU + 1);

        auto const owner = cm.GetGpuIdForEntity(DCGM_FE_GPU_I, gpu1Instance);
        REQUIRE(owner.has_value());
        CHECK(owner.value() == gpu1);
    }
}

TEST_CASE("CacheManager: fake entity query helpers")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    unsigned int const gpu0      = cm.AddFakeGpu(0x20B0, 0x145F);
    unsigned int const gpu1      = cm.AddFakeGpu(0x20B0, 0x145F);
    unsigned int const gpu2      = cm.AddFakeGpu(0x2330, 0x1626);
    unsigned int const instance0 = cm.AddFakeInstance(gpu0);
    unsigned int const instance1 = cm.AddFakeInstance(gpu1);
    unsigned int const ci0       = cm.AddFakeComputeInstance(instance0);
    unsigned int const ci1       = cm.AddFakeComputeInstance(instance1);

    SECTION("GIVEN fake GPUs WHEN GPU ids and counts are queried THEN active and all modes include them")
    {
        std::vector<unsigned int> gpuIds;

        REQUIRE(cm.GetGpuIds(0, gpuIds) == DCGM_ST_OK);
        REQUIRE(gpuIds.size() == 3);
        CHECK(gpuIds[0] == gpu0);
        CHECK(gpuIds[1] == gpu1);
        CHECK(gpuIds[2] == gpu2);
        CHECK(cm.GetGpuCount(0) == 3);

        REQUIRE(cm.GetGpuIds(1, gpuIds) == DCGM_ST_OK);
        REQUIRE(gpuIds.size() == 3);
        CHECK(cm.GetGpuCount(1) == 3);
    }

    SECTION("GIVEN fake GPU hierarchy WHEN entities are queried THEN GPUs, instances, and compute instances are listed")
    {
        std::vector<dcgmGroupEntityPair_t> entities;

        REQUIRE(cm.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 3);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU);
        CHECK(entities[0].entityId == gpu0);

        REQUIRE(cm.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_I, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU_I);
        CHECK(entities[0].entityId == instance0);
        CHECK(entities[1].entityId == instance1);

        REQUIRE(cm.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_CI, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU_CI);
        CHECK(entities[0].entityId == ci0);
        CHECK(entities[1].entityId == ci1);

        CHECK(cm.GetAllEntitiesOfEntityGroup(1, DCGM_FE_NONE, entities) == DCGM_ST_NOT_SUPPORTED);
        CHECK(entities.empty());
    }

    SECTION("GIVEN fake entities WHEN status is queried THEN owning GPU status is returned")
    {
        CHECK(cm.GetEntityStatus(DCGM_FE_GPU, gpu0) == DcgmEntityStatusFake);
        CHECK(cm.GetEntityStatus(DCGM_FE_GPU_I, instance0) == DcgmEntityStatusFake);
        CHECK(cm.GetEntityStatus(DCGM_FE_GPU_CI, ci0) == DcgmEntityStatusFake);
        CHECK(cm.GetEntityStatus(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES) == DcgmEntityStatusUnknown);
        CHECK(cm.GetEntityStatus(DCGM_FE_NONE, 0) == DcgmEntityStatusUnknown);
    }

    SECTION("GIVEN fake GPUs WHEN SKU and NVML index helpers are queried THEN valid and invalid cases are handled")
    {
        std::vector<unsigned int> sameSkuGpus { gpu0, gpu1 };
        std::vector<unsigned int> mixedSkuGpus { gpu0, gpu2 };
        std::vector<unsigned int> invalidGpus { gpu0, DCGM_MAX_NUM_DEVICES };

        CHECK(cm.AreAllGpuIdsSameSku(sameSkuGpus) == 1);
        CHECK(cm.AreAllGpuIdsSameSku(mixedSkuGpus) == 0);
        CHECK(cm.AreAllGpuIdsSameSku(invalidGpus) == 0);

        auto nvmlIndex = cm.GpuIdToNvmlIndex(gpu1);
        REQUIRE(nvmlIndex.has_value());
        CHECK(*nvmlIndex == static_cast<int>(gpu1));
        CHECK_FALSE(cm.GpuIdToNvmlIndex(DCGM_MAX_NUM_DEVICES).has_value());

        auto gpuId = cm.NvmlIndexToGpuId(*nvmlIndex);
        REQUIRE(gpuId.has_value());
        CHECK(*gpuId == gpu1);
        CHECK_FALSE(cm.NvmlIndexToGpuId(DCGM_MAX_NUM_DEVICES).has_value());
    }

    SECTION("GIVEN fake GPUs WHEN all GPU info is requested THEN cached identity data is copied")
    {
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;

        REQUIRE(cm.GetAllGpuInfo(gpuInfo) == DCGM_ST_OK);
        REQUIRE(gpuInfo.size() == 3);
        CHECK(gpuInfo[0].gpuId == gpu0);
        CHECK(gpuInfo[0].status == DcgmEntityStatusFake);
        CHECK(gpuInfo[0].nvmlIndex == gpu0);
        CHECK(gpuInfo[0].pciInfo.pciDeviceId == 0x20B0);
        CHECK(gpuInfo[2].pciInfo.pciDeviceId == 0x2330);
    }

    SECTION("GIVEN fake GPUs WHEN identity helpers are queried THEN defaults and invalid ids are handled")
    {
        dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_UNKNOWN;

        CHECK(cm.GetGpuStatus(gpu0) == DcgmEntityStatusFake);
        CHECK(cm.GetGpuStatus(DCGM_MAX_NUM_DEVICES) == DcgmEntityStatusUnknown);
        CHECK(cm.GetGpuBrand(gpu0) == DCGM_GPU_BRAND_TESLA);
        CHECK(cm.GetGpuBrand(DCGM_MAX_NUM_DEVICES) == DCGM_GPU_BRAND_UNKNOWN);
        REQUIRE(cm.GetGpuArch(gpu0, arch) == DCGM_ST_OK);
        CHECK(arch == DCGM_CHIP_ARCH_UNKNOWN);
        CHECK(cm.GetGpuArch(DCGM_MAX_NUM_DEVICES, arch) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN fake GPUs WHEN lightweight GPU helpers are called THEN local state is used")
    {
        std::vector<nvmlExcludedDeviceInfo_t> excludeList;
        dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
        dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus {};

        REQUIRE(cm.GetGpuExcludeList(excludeList) == DCGM_ST_OK);
        CHECK(excludeList.empty());
        CHECK(cm.GetWorkloadPowerProfilesInfo(gpu0, &profilesInfo, &profilesStatus) == DCGM_ST_OK);
        CHECK(cm.GetWorkloadPowerProfilesInfo(DCGM_MAX_NUM_DEVICES, &profilesInfo, &profilesStatus)
              == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN fake GPU NvLink state WHEN link helpers are called THEN validation is applied")
    {
        CHECK(cm.SetGpuNvLinkLinkState(gpu0, 0, DcgmNvLinkLinkStateUp) == DCGM_ST_OK);
        CHECK(cm.SetGpuNvLinkLinkState(gpu0, DCGM_NVLINK_MAX_LINKS_PER_GPU - 1, DcgmNvLinkLinkStateUp) == DCGM_ST_OK);
        CHECK(cm.SetGpuNvLinkLinkState(DCGM_MAX_NUM_DEVICES, 0, DcgmNvLinkLinkStateUp) == DCGM_ST_BADPARAM);
        CHECK(cm.SetGpuNvLinkLinkState(gpu0, DCGM_NVLINK_MAX_LINKS_PER_GPU, DcgmNvLinkLinkStateUp) == DCGM_ST_BADPARAM);
        CHECK(cm.SetEntityNvLinkLinkState(DCGM_FE_GPU, gpu0, 1, DcgmNvLinkLinkStateDown) == DCGM_ST_OK);
        CHECK(cm.SetEntityNvLinkLinkState(DCGM_FE_NONE, gpu0, 1, DcgmNvLinkLinkStateDown) == DCGM_ST_NOT_SUPPORTED);
    }
}

TEST_CASE("CacheManager: GPU allowlist and detected GPU merge helpers")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    SECTION("GIVEN invalid GPU id WHEN allowlist is queried THEN it is rejected")
    {
        cm.m_numGpus = 1;
        CHECK(cm.IsGpuAllowlisted(1) == 0);
    }

    SECTION("GIVEN supported and unsupported architectures WHEN allowlist is queried THEN brand thresholds apply")
    {
        cm.m_numGpus       = 3;
        cm.m_gpus[0].brand = DCGM_GPU_BRAND_TESLA;
        cm.m_gpus[0].arch  = DCGM_CHIP_ARCH_KEPLER;
        cm.m_gpus[1].brand = DCGM_GPU_BRAND_GEFORCE;
        cm.m_gpus[1].arch  = DCGM_CHIP_ARCH_KEPLER;
        cm.m_gpus[2].brand = DCGM_GPU_BRAND_GEFORCE;
        cm.m_gpus[2].arch  = DCGM_CHIP_ARCH_MAXWELL;

        CHECK(cm.IsGpuAllowlisted(0) == 1);
        CHECK(cm.IsGpuAllowlisted(1) == 0);
        CHECK(cm.IsGpuAllowlisted(2) == 1);
    }

    SECTION("GIVEN no cached GPUs WHEN detected GPUs are merged THEN empty UUID entries are skipped")
    {
        dcgmcm_gpu_info_t detected[3] {};
        detected[0] = MakeDetectedGpu(0, "GPU-A", "0000:01:00.0");
        detected[1] = MakeDetectedGpu(1, "", "0000:02:00.0");
        detected[2] = MakeDetectedGpu(2, "GPU-C", "0000:03:00.0");

        cm.MergeNewlyDetectedGpuList(detected, 3);

        REQUIRE(cm.m_numGpus == 2);
        CHECK(std::string(cm.m_gpus[0].uuid) == "GPU-A");
        CHECK(std::string(cm.m_gpus[1].uuid) == "GPU-C");
        CHECK(cm.pciBusGpuIdMap.at("0000:01:00.0") == 0);
        CHECK(cm.pciBusGpuIdMap.at("0000:03:00.0") == 2);
    }

    SECTION("GIVEN existing GPUs WHEN detected GPUs are merged THEN matches update and new GPUs append")
    {
        cm.m_numGpus = 2;
        cm.m_gpus[0] = MakeDetectedGpu(0, "GPU-OLD-A", "0000:01:00.0", DCGM_GPU_BRAND_GEFORCE, DCGM_CHIP_ARCH_MAXWELL);
        cm.m_gpus[1] = MakeDetectedGpu(1, "GPU-OLD-B", "0000:02:00.0", DCGM_GPU_BRAND_TESLA, DCGM_CHIP_ARCH_KEPLER);
        cm.m_gpus[1].status = DcgmEntityStatusLost;

        dcgmcm_gpu_info_t detected[2] {};
        detected[0] = MakeDetectedGpu(0, "GPU-OLD-B", "0000:22:00.0", DCGM_GPU_BRAND_GEFORCE, DCGM_CHIP_ARCH_AMPERE);
        detected[0].virtualizationMode = DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU;
        detected[0].supportGpm         = true;
        detected[0].migEnabled         = true;
        detected[0].ciCount            = 3;
        detected[0].maxGpcs            = 4;
        detected[0].usedGpcs           = 2;
        detected[0].ccMode             = 1;
        detected[1]                    = MakeDetectedGpu(2, "GPU-NEW-C", "0000:03:00.0");

        cm.MergeNewlyDetectedGpuList(detected, 2);

        REQUIRE(cm.m_numGpus == 3);
        CHECK(std::string(cm.m_gpus[1].uuid) == "GPU-OLD-B");
        CHECK(cm.m_gpus[1].status == DcgmEntityStatusOk);
        CHECK(cm.m_gpus[1].brand == DCGM_GPU_BRAND_GEFORCE);
        CHECK(cm.m_gpus[1].arch == DCGM_CHIP_ARCH_AMPERE);
        CHECK(cm.m_gpus[1].virtualizationMode == DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU);
        CHECK(cm.m_gpus[1].supportGpm);
        CHECK(cm.m_gpus[1].migEnabled);
        CHECK(cm.m_gpus[1].ciCount == 3);
        CHECK(cm.m_gpus[1].maxGpcs == 4);
        CHECK(cm.m_gpus[1].usedGpcs == 2);
        CHECK(cm.m_gpus[1].ccMode == 1);
        CHECK(std::string(cm.m_gpus[2].uuid) == "GPU-NEW-C");
        CHECK(cm.m_gpus[2].gpuId == 2);
        CHECK(cm.pciBusGpuIdMap.at("0000:22:00.0") == 1);
        CHECK(cm.pciBusGpuIdMap.at("0000:03:00.0") == 2);
    }
}

TEST_CASE("CacheManager: field validation helpers")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;
    unsigned int const gpuId = cm.AddFakeGpu();

    SECTION("GIVEN global field ids WHEN validating global fields THEN scope and unknown field errors are reported")
    {
        CHECK(cm.CheckValidGlobalField(DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_OK);
        CHECK(cm.CheckValidGlobalField(DCGM_FI_SYSTEM_FIELD_UNKNOWN) == DCGM_ST_UNKNOWN_FIELD);
        CHECK(cm.CheckValidGlobalField(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN GPU field ids WHEN validating GPU fields THEN scope, field, and GPU errors are reported")
    {
        CHECK(cm.CheckValidGpuField(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_OK);
        CHECK(cm.CheckValidGpuField(gpuId, DCGM_FI_SYSTEM_FIELD_UNKNOWN) == DCGM_ST_UNKNOWN_FIELD);
        CHECK(cm.CheckValidGpuField(gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_BADPARAM);
        CHECK(cm.CheckValidGpuField(DCGM_MAX_NUM_DEVICES, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN null runtime stats WHEN stats are requested THEN the helper returns without writing")
    {
        cm.GetRuntimeStats(nullptr);
    }
}

TEST_CASE("CacheManager: watch info snapshot helpers")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;
    unsigned int const gpuId = cm.AddFakeGpu();

    SECTION("GIVEN a null snapshot destination WHEN snapshot is requested THEN bad parameter is returned")
    {
        CHECK(cm.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr)
              == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN no watch info WHEN snapshot is requested THEN not watched is returned")
    {
        dcgmcm_watch_info_t snapshot {};

        CHECK(cm.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
              == DCGM_ST_NOT_WATCHED);
    }

    SECTION("GIVEN existing watch info WHEN snapshot is requested THEN a copy is returned")
    {
        dcgmcm_watch_info_p watchInfo = cm.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched           = 1;
        watchInfo->lastStatus          = NVML_ERROR_NOT_SUPPORTED;
        watchInfo->lastQueriedUsec     = 1234;
        watchInfo->monitorIntervalUsec = 5678;
        watchInfo->maxAgeUsec          = 9012;
        watchInfo->fetchCount          = 34;

        dcgmcm_watch_info_t snapshot {};
        REQUIRE(cm.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
                == DCGM_ST_OK);

        CHECK(snapshot.watchKey.entityGroupId == watchInfo->watchKey.entityGroupId);
        CHECK(snapshot.watchKey.entityId == watchInfo->watchKey.entityId);
        CHECK(snapshot.watchKey.fieldId == watchInfo->watchKey.fieldId);
        CHECK(snapshot.isWatched == 1);
        CHECK(snapshot.lastStatus == NVML_ERROR_NOT_SUPPORTED);
        CHECK(snapshot.lastQueriedUsec == 1234);
        CHECK(snapshot.monitorIntervalUsec == 5678);
        CHECK(snapshot.maxAgeUsec == 9012);
        CHECK(snapshot.fetchCount == 34);

        std::vector<dcgmcm_watch_info_p> watchers;
        cm.GetAllWatchObjects(watchers);
        REQUIRE_FALSE(watchers.empty());
        bool foundWatchInfo = false;
        for (auto const watcher : watchers)
        {
            foundWatchInfo |= (watcher == watchInfo);
        }
        CHECK(foundWatchInfo);
    }
}

TEST_CASE("CacheManager: watcher aggregation helpers")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cm;
    cm.m_nvmlLoaded.store(false, std::memory_order_release);

    dcgm_entity_key_t key {};
    cm.EntityIdToWatchKey(&key, DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    auto watchInfo = cm.AllocWatchInfo(key);
    REQUIRE(watchInfo != nullptr);
    DcgmNs::Defer freeWatchInfo([&] { cm.FreeWatchInfo(watchInfo); });

    GIVEN("empty and null watcher inputs")
    {
        dcgm_watch_watcher_info_t watcher {};
        watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 100);
        watcher.monitorIntervalUsec = 5000;
        watcher.maxAgeUsec          = 10000;

        bool wasAdded = false;

        THEN("bad parameters and empty watch state are reported")
        {
            CHECK(cm.AddOrUpdateWatcher(nullptr, &wasAdded, &watcher) == DCGM_ST_BADPARAM);
            CHECK(cm.RemoveWatcher(nullptr, &watcher) == DCGM_ST_BADPARAM);
            CHECK(cm.UpdateWatchFromWatchers(nullptr) == DCGM_ST_BADPARAM);
            CHECK(cm.UpdateWatchFromWatchers(watchInfo) == DCGM_ST_NOT_WATCHED);
            CHECK(cm.RemoveWatcher(watchInfo, &watcher) == DCGM_ST_NOT_WATCHED);
            CHECK(watchInfo->hasSubscribedWatchers == 0);
        }
    }

    GIVEN("multiple watcher registrations")
    {
        dcgm_watch_watcher_info_t slowWatcher {};
        slowWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 101);
        slowWatcher.monitorIntervalUsec = 5000;
        slowWatcher.maxAgeUsec          = 10000;
        slowWatcher.isSubscribed        = 0;

        dcgm_watch_watcher_info_t fastWatcher {};
        fastWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 102);
        fastWatcher.monitorIntervalUsec = 1000;
        fastWatcher.maxAgeUsec          = 20000;
        fastWatcher.isSubscribed        = 1;

        bool wasAdded = false;

        WHEN("watchers are added and one existing watcher is updated")
        {
            uint64_t const initialGeneration = watchInfo->watchGeneration;
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);
            CHECK(watchInfo->watchGeneration == initialGeneration + 1);
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &fastWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);
            CHECK(watchInfo->watchGeneration == initialGeneration + 2);

            slowWatcher.monitorIntervalUsec = 3000;
            slowWatcher.maxAgeUsec          = 40000;
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            CHECK_FALSE(wasAdded);
            CHECK(watchInfo->watchGeneration == initialGeneration + 3);

            THEN("the aggregate interval, age, and subscription flag are recomputed")
            {
                CHECK(watchInfo->watchers.size() == 2);
                CHECK(watchInfo->monitorIntervalUsec == 1000);
                CHECK(watchInfo->maxAgeUsec == 40000);
                CHECK(watchInfo->hasSubscribedWatchers == 1);
            }
        }

        WHEN("watchers are removed")
        {
            uint64_t const initialGeneration = watchInfo->watchGeneration;
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            CHECK(watchInfo->watchGeneration == initialGeneration + 1);
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &fastWatcher) == DCGM_ST_OK);
            CHECK(watchInfo->watchGeneration == initialGeneration + 2);
            watchInfo->isWatched = 1;

            REQUIRE(cm.RemoveWatcher(watchInfo, &fastWatcher) == DCGM_ST_OK);
            CHECK(watchInfo->watchGeneration == initialGeneration + 3);

            THEN("remaining watcher state is preserved until the last watcher is removed")
            {
                CHECK(watchInfo->watchers.size() == 1);
                CHECK(watchInfo->isWatched == 1);
                CHECK(watchInfo->monitorIntervalUsec == slowWatcher.monitorIntervalUsec);
                CHECK(watchInfo->maxAgeUsec == slowWatcher.maxAgeUsec);
                CHECK(watchInfo->hasSubscribedWatchers == 0);

                REQUIRE(cm.RemoveWatcher(watchInfo, &slowWatcher) == DCGM_ST_OK);
                CHECK(watchInfo->watchers.empty());
                CHECK(watchInfo->isWatched == 0);
                CHECK(watchInfo->watchGeneration == initialGeneration + 4);
            }
        }
    }
}

TEST_CASE("CacheManager: watch precheck and clear helpers")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cm;
    dcgm_entity_key_t key {};
    cm.EntityIdToWatchKey(&key, DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    auto watchInfo = cm.AllocWatchInfo(key);
    REQUIRE(watchInfo != nullptr);
    DcgmNs::Defer freeWatchInfo([&] { cm.FreeWatchInfo(watchInfo); });

    GIVEN("watch info without cached samples")
    {
        THEN("precheck distinguishes missing watches, unwatched fields, driver errors, and empty data")
        {
            CHECK(cm.PrecheckWatchInfoForSamples(nullptr) == DCGM_ST_NOT_WATCHED);
            CHECK(cm.PrecheckWatchInfoForSamples(watchInfo) == DCGM_ST_NOT_WATCHED);

            watchInfo->isWatched  = 1;
            watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
            CHECK(cm.PrecheckWatchInfoForSamples(watchInfo) == DCGM_ST_NOT_SUPPORTED);

            watchInfo->lastStatus = NVML_SUCCESS;
            CHECK(cm.PrecheckWatchInfoForSamples(watchInfo) == DCGM_ST_NO_DATA);
        }
    }

    GIVEN("a populated watch info object")
    {
        dcgm_watch_watcher_info_t watcher {};
        watcher.watcher                  = DcgmWatcher(DcgmWatcherTypeClient, 201);
        watcher.monitorIntervalUsec      = 10;
        watcher.maxAgeUsec               = 20;
        watcher.isSubscribed             = 1;
        bool wasAdded                    = false;
        uint64_t const initialGeneration = watchInfo->watchGeneration;
        REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);
        watchInfo->isWatched       = 1;
        watchInfo->pushedByModule  = true;
        watchInfo->lastStatus      = NVML_ERROR_TIMEOUT;
        watchInfo->lastQueriedUsec = 99;

        WHEN("the watch info is cleared without cache destruction")
        {
            CHECK(watchInfo->watchGeneration == initialGeneration + 1);
            cm.ClearWatchInfo(watchInfo, 0);

            THEN("transient watcher state is reset")
            {
                CHECK(watchInfo->watchers.empty());
                CHECK(watchInfo->isWatched == 0);
                CHECK(watchInfo->hasSubscribedWatchers == 0);
                CHECK_FALSE(watchInfo->pushedByModule);
                CHECK(watchInfo->lastStatus == NVML_SUCCESS);
                CHECK(watchInfo->lastQueriedUsec == 0);
                CHECK(watchInfo->monitorIntervalUsec == 0);
                CHECK(watchInfo->maxAgeUsec == DCGM_MAX_AGE_USEC_DEFAULT);
                CHECK(watchInfo->watchGeneration == initialGeneration + 2);
            }
        }
    }
}

TEST_CASE("CacheManager: field watch lifecycle helpers")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cm;
    DcgmWatcher const watcher(DcgmWatcherTypeClient, 7001);

    SECTION("GIVEN NVML is unavailable WHEN GPU fields are watched THEN the request is rejected early")
    {
        bool wereFirstWatcher = false;

        cm.m_nvmlLoaded.store(false, std::memory_order_release);

        CHECK(cm.AddFieldWatch(
                  DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 60.0, 2, watcher, false, false, wereFirstWatcher)
              == DCGM_ST_NVML_NOT_LOADED);
        CHECK_FALSE(wereFirstWatcher);
    }

    SECTION("GIVEN a global field WHEN watched and removed THEN global watch state is updated")
    {
        bool wereFirstWatcher = false;

        cm.m_nvmlLoaded.store(false, std::memory_order_release);

        REQUIRE(
            cm.AddFieldWatch(
                DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, 1000, 60.0, 2, watcher, true, false, wereFirstWatcher)
            == DCGM_ST_OK);
        CHECK(wereFirstWatcher);
        CHECK(cm.m_haveAnyLiveSubscribers);

        auto watchInfo = cm.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 0);
        REQUIRE(watchInfo != nullptr);
        CHECK(watchInfo->isWatched == 1);
        CHECK(watchInfo->watchers.size() == 1);
        CHECK(watchInfo->watchers[0].isSubscribed == 1);

        CHECK(cm.RemoveFieldWatch(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, 0, watcher) == DCGM_ST_OK);
        CHECK(watchInfo->isWatched == 0);
        CHECK(watchInfo->watchers.empty());
        CHECK(cm.RemoveGlobalFieldWatch(DCGM_FI_MAX_FIELDS, 0, watcher) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN fake GPU fields WHEN watched and removed THEN entity watch state is updated")
    {
        bool wereFirstWatcher     = false;
        unsigned int const gpuId  = cm.AddFakeGpu();
        DcgmWatcher const watcher = DcgmWatcher(DcgmWatcherTypeClient, 7002);

        cm.m_nvmlLoaded.store(true, std::memory_order_release);

        REQUIRE(cm.AddFieldWatch(DCGM_FE_GPU,
                                 gpuId,
                                 DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                 5000,
                                 120.0,
                                 4,
                                 watcher,
                                 true,
                                 false,
                                 wereFirstWatcher)
                == DCGM_ST_OK);
        CHECK(wereFirstWatcher);

        auto watchInfo = cm.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0);
        REQUIRE(watchInfo != nullptr);
        CHECK(watchInfo->isWatched == 1);
        CHECK(watchInfo->watchers.size() == 1);
        CHECK(watchInfo->monitorIntervalUsec == 5000);
        CHECK(watchInfo->hasSubscribedWatchers == 1);

        CHECK(cm.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcher) == DCGM_ST_OK);
        CHECK(watchInfo->isWatched == 0);
        CHECK(watchInfo->watchers.empty());
        CHECK(cm.RemoveEntityFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_MAX_FIELDS, 0, watcher) == DCGM_ST_BADPARAM);
        CHECK(cm.RemoveFieldWatch(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcher)
              == DCGM_ST_BADPARAM);
        CHECK(cm.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcher) == DCGM_ST_OK);
        CHECK(cm.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, 0, watcher)
              == DCGM_ST_NOT_WATCHED);
    }

    SECTION("GIVEN pre and post watch helpers WHEN called for cheap paths THEN status codes are stable")
    {
        unsigned int const gpuId = cm.AddFakeGpu();

        cm.m_nvmlLoaded.store(true, std::memory_order_release);

        CHECK(cm.NvmlPreWatch(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_OK);
        CHECK(cm.NvmlPreWatch(DCGM_MAX_NUM_DEVICES, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_GENERIC_ERROR);
        CHECK(cm.NvmlPreWatch(gpuId, DCGM_FI_SYSTEM_FIELD_UNKNOWN) == DCGM_ST_UNKNOWN_FIELD);

        cm.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK(cm.NvmlPostWatch(gpuId, DCGM_FI_DEV_XID_ERROR) == DCGM_ST_OK);
        CHECK(cm.NvmlPostWatch(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_OK);
    }

    SECTION("GIVEN entity pairs WHEN GPM support is queried THEN local support flags and entity validation are used")
    {
        unsigned int const gpuId = cm.AddFakeGpu();
        dcgmGroupEntityPair_t const gpuPair { DCGM_FE_GPU, gpuId };
        dcgmGroupEntityPair_t const badPair { DCGM_FE_SWITCH, 0 };

        cm.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK_FALSE(cm.EntityPairSupportsGpm(gpuPair));

        cm.m_nvmlLoaded.store(true, std::memory_order_release);
        CHECK_FALSE(cm.EntityPairSupportsGpm(badPair));
        CHECK_FALSE(cm.EntityPairSupportsGpm({ DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES }));

        cm.m_gpus[gpuId].supportGpm = false;
        CHECK_FALSE(cm.EntityPairSupportsGpm(gpuPair));

        cm.m_gpus[gpuId].supportGpm = true;
        CHECK(cm.EntityPairSupportsGpm(gpuPair));

        cm.m_gpus[gpuId].supportGpm     = false;
        cm.m_forceProfMetricsThroughGpm = true;
        dcgm_entity_key_t profilingKey {};
        profilingKey.entityGroupId = DCGM_FE_GPU;
        profilingKey.entityId      = gpuId;
        profilingKey.fieldId       = DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO;
        CHECK(cm.EntityKeySupportsGpm(profilingKey));

        profilingKey.fieldId = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        CHECK_FALSE(cm.EntityKeySupportsGpm(profilingKey));

        dcgm_link_t validLink {};
        validLink.parsed.type  = DCGM_FE_GPU;
        validLink.parsed.gpuId = gpuId;
        validLink.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU - 1;

        dcgm_link_t invalidIndex  = validLink;
        invalidIndex.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU;

        dcgm_link_t invalidGpu  = validLink;
        invalidGpu.parsed.gpuId = DCGM_MAX_NUM_DEVICES;

        dcgm_link_t invalidParent = validLink;
        invalidParent.parsed.type = DCGM_FE_SWITCH;

        CHECK(cm.EntityPairSupportsGpm({ DCGM_FE_LINK, validLink.raw }));
        CHECK_FALSE(cm.EntityPairSupportsGpm({ DCGM_FE_LINK, invalidIndex.raw }));
        CHECK_FALSE(cm.EntityPairSupportsGpm({ DCGM_FE_LINK, invalidGpu.raw }));
        CHECK_FALSE(cm.EntityPairSupportsGpm({ DCGM_FE_LINK, invalidParent.raw }));
    }
}

TEST_CASE("CacheManager: watch key and practical entity helpers")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    SECTION("GIVEN a null watch key destination WHEN building a key THEN no write is attempted")
    {
        cm.EntityIdToWatchKey(nullptr, DCGM_FE_GPU, 7, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    }

    SECTION("GIVEN entity values WHEN building a watch key THEN fields are copied")
    {
        dcgm_entity_key_t key {};

        cm.EntityIdToWatchKey(&key, DCGM_FE_GPU, 7, DCGM_FI_DEV_GPU_TEMP_CELSIUS);

        CHECK(key.entityGroupId == DCGM_FE_GPU);
        CHECK(key.entityId == 7);
        CHECK(key.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    }

    SECTION("GIVEN non-MIG entities WHEN practical entity info is set THEN existing entity mapping is preserved")
    {
        dcgmcm_watch_info_t watchInfo {};
        cm.EntityIdToWatchKey(&watchInfo.watchKey, DCGM_FE_GPU, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        watchInfo.practicalEntityGroupId = DCGM_FE_GPU;
        watchInfo.practicalEntityId      = 2;

        REQUIRE(cm.SetPracticalEntityInfo(watchInfo) == DCGM_ST_OK);
        CHECK(watchInfo.practicalEntityGroupId == DCGM_FE_GPU);
        CHECK(watchInfo.practicalEntityId == 2);
    }

    SECTION("GIVEN invalid practical entity inputs WHEN mapping is requested THEN bad parameters are returned")
    {
        dcgmcm_watch_info_t badGroup {};
        cm.EntityIdToWatchKey(&badGroup.watchKey, DCGM_FE_COUNT, 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        CHECK(cm.SetPracticalEntityInfo(badGroup) == DCGM_ST_BADPARAM);

        dcgmcm_watch_info_t badField {};
        cm.EntityIdToWatchKey(&badField.watchKey, DCGM_FE_GPU_I, 1, DCGM_FI_MAX_FIELDS);
        CHECK(cm.SetPracticalEntityInfo(badField) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN entity keys WHEN checking GPM support THEN non-profiling fields are rejected early")
    {
        dcgm_entity_key_t key {};
        cm.EntityIdToWatchKey(&key, DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS);

        CHECK_FALSE(cm.EntityKeySupportsGpm(key));
    }
}

SCENARIO("CacheManager: canonical NVLink GPU and LINK watches remain independent")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();
    cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
    cacheManager.m_forceProfMetricsThroughGpm = true;

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 17;

    DcgmWatcher watcher(DcgmWatcherTypeClient, 6201);
    bool wereFirstWatcher = false;

    GIVEN("the canonical TX field is watched on a GPU and one of its links")
    {
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_PROF_NVLINK_TX_BYTES,
                                           1000000,
                                           60.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           DCGM_FI_PROF_NVLINK_TX_BYTES,
                                           1000000,
                                           60.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);

        auto *gpuWatch  = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_PROF_NVLINK_TX_BYTES, 0);
        auto *linkWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, DCGM_FI_PROF_NVLINK_TX_BYTES, 0);

        THEN("the cache uses distinct watch records")
        {
            REQUIRE(gpuWatch != nullptr);
            REQUIRE(linkWatch != nullptr);
            CHECK(gpuWatch != linkWatch);
            CHECK(gpuWatch->isWatched);
            CHECK(linkWatch->isWatched);
        }

        WHEN("the GPU watch is removed")
        {
            REQUIRE(cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_PROF_NVLINK_TX_BYTES, 0, watcher)
                    == DCGM_ST_OK);

            THEN("the LINK watch remains active")
            {
                REQUIRE(gpuWatch != nullptr);
                REQUIRE(linkWatch != nullptr);
                CHECK_FALSE(gpuWatch->isWatched);
                CHECK(linkWatch->isWatched);
            }
        }
    }
}

TEST_CASE("CacheManager: sample API edge cases")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;
    unsigned int const gpuId = cm.AddFakeGpu();

    SECTION("GIVEN invalid sample API inputs WHEN called THEN bad parameter errors are returned")
    {
        dcgmcm_sample_t sample {};
        dcgmcm_sample_t samples[2] {};
        int count = 2;

        CHECK(cm.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(
            cm.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples, nullptr, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
            == DCGM_ST_BADPARAM);

        count = 0;
        CHECK(
            cm.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
            == DCGM_ST_BADPARAM);

        count = 2;
        CHECK(
            cm.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
            == DCGM_ST_BADPARAM);

        count = 2;
        CHECK(cm.GetSamples(DCGM_FE_GPU,
                            gpuId,
                            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                            samples,
                            &count,
                            0,
                            0,
                            static_cast<dcgmOrder_t>(99),
                            nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(count == 0);

        count = 2;
        CHECK(
            cm.GetSamples(DCGM_FE_GPU, gpuId, DCGM_FI_MAX_FIELDS, samples, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
            == DCGM_ST_UNKNOWN_FIELD);
        CHECK(count == 0);

        CHECK(cm.InjectSamples(DCGM_FE_GPU, gpuId, 0, &sample, 1) == DCGM_ST_BADPARAM);
        CHECK(cm.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, 1) == DCGM_ST_BADPARAM);
        CHECK(cm.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample, 0) == DCGM_ST_BADPARAM);
        CHECK(cm.FreeSamples(nullptr, 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);
        CHECK(cm.FreeSamples(&sample, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);
        CHECK(cm.FreeSamples(&sample, 1, 0) == DCGM_ST_BADPARAM);
        CHECK(cm.AppendSamples(nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GIVEN unwatched fields WHEN samples are requested THEN not-watched is returned")
    {
        dcgmcm_sample_t sample {};
        dcgmcm_sample_t samples[2] {};
        int count = 2;

        CHECK(cm.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample, nullptr)
              == DCGM_ST_NOT_WATCHED);
        CHECK(
            cm.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr)
            == DCGM_ST_NOT_WATCHED);
        CHECK(count == 0);
    }

    SECTION(
        "GIVEN injected integer samples WHEN queried THEN latest, ascending, descending, and range cases are handled")
    {
        timelib64_t const firstTimestamp  = timelib_usecSince1970() + 1000000;
        timelib64_t const secondTimestamp = firstTimestamp + 1;
        dcgmcm_sample_t injected[2] {};
        injected[0].timestamp = firstTimestamp;
        injected[0].val.i64   = 11;
        injected[1].timestamp = secondTimestamp;
        injected[1].val.i64   = 22;

        REQUIRE(cm.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, injected, 2) == DCGM_ST_OK);

        dcgmcm_sample_t latest {};
        REQUIRE(cm.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, &latest, nullptr) == DCGM_ST_OK);
        CHECK(latest.val.i64 == 22);

        dcgmcm_sample_t samples[4] {};
        int count = 4;
        REQUIRE(cm.GetSamples(DCGM_FE_GPU,
                              gpuId,
                              DCGM_FI_DEV_ECC_MODE,
                              samples,
                              &count,
                              firstTimestamp,
                              secondTimestamp,
                              DCGM_ORDER_ASCENDING,
                              nullptr)
                == DCGM_ST_OK);
        REQUIRE(count == 2);
        CHECK(samples[0].val.i64 == 11);
        CHECK(samples[1].val.i64 == 22);

        count = 4;
        REQUIRE(cm.GetSamples(DCGM_FE_GPU,
                              gpuId,
                              DCGM_FI_DEV_ECC_MODE,
                              samples,
                              &count,
                              firstTimestamp,
                              secondTimestamp,
                              DCGM_ORDER_DESCENDING,
                              nullptr)
                == DCGM_ST_OK);
        REQUIRE(count == 2);
        CHECK(samples[0].val.i64 == 22);
        CHECK(samples[1].val.i64 == 11);

        count = 4;
        CHECK(cm.GetSamples(DCGM_FE_GPU,
                            gpuId,
                            DCGM_FI_DEV_ECC_MODE,
                            samples,
                            &count,
                            secondTimestamp + 1,
                            secondTimestamp + 2,
                            DCGM_ORDER_ASCENDING,
                            nullptr)
              == DCGM_ST_NO_DATA);
        CHECK(count == 0);
    }

    SECTION("GIVEN string samples WHEN injected or freed THEN string paths are covered")
    {
        dcgmcm_sample_t badString {};
        badString.timestamp = 100;
        CHECK(cm.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &badString, 1) == DCGM_ST_BADPARAM);

        dcgmcm_sample_t ownedString {};
        ownedString.val.str = strdup("owned");
        REQUIRE(ownedString.val.str != nullptr);
        CHECK(cm.FreeSamples(&ownedString, 1, DCGM_FI_DEV_GPU_NAME) == DCGM_ST_OK);
        CHECK(ownedString.val.str == nullptr);
    }
}

void callback(unsigned int gpuId, void *userData)
{
    auto gpuIdPtr = (unsigned int *)userData;
    *gpuIdPtr     = gpuId;
}

TEST_CASE("CacheManager: Test Event Register")
{
    TestDcgmCacheManager cm;
    dcgmcmEventSubscription_t sub {};
    sub.type     = DcgmcmEventTypeMigReconfigure;
    sub.fn.migCb = nullptr;

    REQUIRE(cm.SubscribeForEvent(sub) == DCGM_ST_BADPARAM);
    sub.fn.migCb       = callback;
    unsigned int gpuId = 0;
    sub.userData       = &gpuId;

    REQUIRE(cm.SubscribeForEvent(sub) == DCGM_ST_OK);
    unsigned int updated = 4;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);

    updated = 6;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);

    updated = 1;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);
}

TEST_CASE("CacheManager: CUDA_VISIBLE_DEVICES")
{
    TestDcgmCacheManager cm;
    unsigned int gpuId = cm.AddFakeGpu();
    unsigned int instanceIds[2];
    instanceIds[0] = cm.AddFakeInstance(gpuId);
    instanceIds[1] = cm.AddFakeInstance(gpuId);
    unsigned int computeInstanceIds[2];
    computeInstanceIds[0] = cm.AddFakeComputeInstance(instanceIds[0]);
    computeInstanceIds[1] = cm.AddFakeComputeInstance(instanceIds[1]);

    std::stringstream buf;
    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_SWITCH, 0, buf);
    CHECK(buf.str() == "Unsupported");

    cm.GenerateCudaVisibleDevicesValue(100, DCGM_FE_GPU, 100, buf);
    CHECK(buf.str() == "Invalid GPU id: 100");

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU, gpuId, buf);

    std::stringstream tmp;

    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000";
    CHECK(buf.str() == tmp.str());

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU_I, instanceIds[0], buf);
    tmp.str("");
    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000/0"; // Dummy UUID because this is a fake GPU
    CHECK(buf.str() == tmp.str());

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU_CI, computeInstanceIds[0], buf);
    tmp.str("");
    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000/0/0"; // Dummy UUID because this is a fake GPU
    CHECK(buf.str() == tmp.str());

    /* NvSwitch ID is not a CUDA item */
    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_SWITCH, 0, buf);
    tmp.str("");
    tmp << "Unsupported";
    CHECK(buf.str() == tmp.str());

    /* Link ID is not a CUDA item */
    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_LINK, 0, buf);
    tmp.str("");
    tmp << "Unsupported";
    CHECK(buf.str() == tmp.str());
}

typedef struct
{
    unsigned int nvmlFabricState;
    nvmlReturn_t nvmlRet;
    dcgmReturn_t callReturn;
    dcgmFabricManagerStatus_t resultFMStatus;
    bool errorShouldBeBlank;
    dcgmReturn_t resultFMError;
} cacheResultCheck_t;

TEST_CASE("CacheManger::GetFMStatusFromStruct")
{
    nvmlGpuFabricInfoV_t gpuFabricInfo {};
    dcgmFabricManagerStatus_t status;
    uint64_t fmError;

    constexpr unsigned int END_OF_SERIES = 100;

    cacheResultCheck_t const series[] = {
        { NVML_GPU_FABRIC_STATE_NOT_SUPPORTED, NVML_SUCCESS, DCGM_ST_OK, DcgmFMStatusNotSupported, true, DCGM_ST_OK },
        { NVML_GPU_FABRIC_STATE_NOT_STARTED, NVML_SUCCESS, DCGM_ST_OK, DcgmFMStatusNotStarted, true, DCGM_ST_OK },
        { NVML_GPU_FABRIC_STATE_IN_PROGRESS, NVML_SUCCESS, DCGM_ST_OK, DcgmFMStatusInProgress, true, DCGM_ST_OK },
        { NVML_GPU_FABRIC_STATE_COMPLETED, NVML_SUCCESS, DCGM_ST_OK, DcgmFMStatusSuccess, false, DCGM_ST_OK },
        { NVML_GPU_FABRIC_STATE_COMPLETED,
          NVML_ERROR_TIMEOUT,
          DCGM_ST_OK,
          DcgmFMStatusFailure,
          false,
          DCGM_ST_TIMEOUT },
        { 27, NVML_ERROR_TIMEOUT, DCGM_ST_BADPARAM, DcgmFMStatusUnrecognized, true, DCGM_ST_TIMEOUT },
        { END_OF_SERIES, NVML_SUCCESS, DCGM_ST_BADPARAM, DcgmFMStatusSuccess, false, DCGM_ST_OK },
    };

    for (unsigned int i = 0; series[i].nvmlFabricState != END_OF_SERIES; i++)
    {
        gpuFabricInfo.state  = series[i].nvmlFabricState;
        gpuFabricInfo.status = series[i].nvmlRet;
        CHECK(series[i].callReturn == TestDcgmCacheManager::GetFMStatusFromStruct(gpuFabricInfo, status, fmError));

        CHECK(status == series[i].resultFMStatus);
        if (series[i].errorShouldBeBlank)
        {
            CHECK(DCGM_INT64_IS_BLANK(fmError));
        }
        else
        {
            CHECK(fmError == (uint64_t)series[i].resultFMError);
        }
    }
}

TEST_CASE("DcgmCacheManager::UpdateAllFields")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    /* Initialize in manual mode */
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    // field from dedicated API
    unsigned short constexpr fieldIdUUID = DCGM_FI_DEV_GPU_UUID;
    // field from nvmlDeviceGetFieldValues
    unsigned short constexpr fieldIdECCCurrent       = DCGM_FI_DEV_ECC_MODE;
    std::array<unsigned short, 2> constexpr fieldIds = { fieldIdUUID, fieldIdECCCurrent };
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    for (auto fieldId : fieldIds)
    {
        WatchField(cacheManager, gpuId, fieldId, watcher);
    }

    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

    dcgmcm_sample_t sample {};

    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldIdUUID, &sample, nullptr) == DCGM_ST_OK);
    REQUIRE(sample.val.str != nullptr);
    // The UUID of GPU 0 is 26a0ce63-ce32-b34e-acf2-5a0273328ee5 in H200.yaml
    REQUIRE(std::string_view(sample.val.str) == "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5");
    REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldIdUUID) == DCGM_ST_OK);

    std::memset(&sample, 0, sizeof(sample));
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldIdECCCurrent, &sample, nullptr) == DCGM_ST_OK);
    REQUIRE(sample.val.i64 == 1);
    REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldIdECCCurrent) == DCGM_ST_OK);

    for (auto fieldId : fieldIds)
    {
        auto ret = cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, fieldId, 0, watcher);
        REQUIRE(ret == DCGM_ST_OK);
    }
    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);
}

TEST_CASE("DcgmCacheManager: NVLink TX/RX throughput watches with NVML field injection")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto nvmlRet = nvmlInit_v2();
    REQUIRE(nvmlRet == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5577);

    /* All NVLink throughput fields: per-link L0..L17 plus TOTAL for TX and RX (contiguous IDs). */
    std::vector<unsigned short> throughputFields;
    throughputFields.reserve(
        static_cast<size_t>(DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL - DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0 + 1)
        + static_cast<size_t>(DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL - DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0 + 1));
    for (unsigned short fid = DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0; fid <= DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL; ++fid)
    {
        throughputFields.push_back(fid);
    }
    for (unsigned short fid = DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0; fid <= DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL; ++fid)
    {
        throughputFields.push_back(fid);
    }

    std::array<unsigned short, 2> constexpr nvlinkNvmlFields = {
        NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
        NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
    };
    int64_t const ts         = timelib_usecSince1970() + 128 * 1000000;
    uint64_t constexpr value = 0xc8763ull;

    for (auto fieldId : nvlinkNvmlFields)
    {
        nvmlFieldValue_t nvmlFv {};
        nvmlFv.fieldId      = fieldId;
        nvmlFv.scopeId      = 0;
        nvmlFv.nvmlReturn   = NVML_SUCCESS;
        nvmlFv.timestamp    = ts;
        nvmlFv.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        nvmlFv.value.ullVal = value;
        REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &nvmlFv) == NVML_SUCCESS);
    }

    for (auto fieldId : throughputFields)
    {
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(
                    DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                == DCGM_ST_OK);
    }

    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

    for (auto fieldId : throughputFields)
    {
        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE_FALSE(DCGM_INT64_IS_BLANK(sample.val.i64));
        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }

    for (auto fieldId : throughputFields)
    {
        REQUIRE(cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, fieldId, 0, watcher) == DCGM_ST_OK);
    }
}

SCENARIO("CacheManager: memory temperature field honors per-field NVML status")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5578);

        auto injectMemoryTemp = [&](nvmlReturn_t fieldReturn, nvmlValueType_t valueType, unsigned int uiVal) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            nvmlFieldValue_t fv {};
            fv.fieldId     = NVML_FI_DEV_MEMORY_TEMP;
            fv.nvmlReturn  = fieldReturn;
            fv.valueType   = valueType;
            fv.value.uiVal = uiVal;
            fv.timestamp   = timelib_usecSince1970();
            REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &fv) == NVML_SUCCESS);
        };

        auto sampleMemoryTemp = [&]() {
            WatchField(cacheManager, gpuId, DCGM_FI_DEV_MEMORY_TEMP_CELSIUS, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_MEMORY_TEMP_CELSIUS, &sample, nullptr)
                    == DCGM_ST_OK);
            return sample;
        };

        WHEN("NVML reports per-field NOT_SUPPORTED with uiVal 0")
        {
            injectMemoryTemp(NVML_ERROR_NOT_SUPPORTED, NVML_VALUE_TYPE_UNSIGNED_INT, 0);
            auto sample = sampleMemoryTemp();

            THEN("DCGM reports NOT_SUPPORTED instead of passing through zero")
            {
                CHECK(sample.val.i64 == DCGM_INT64_NOT_SUPPORTED);
                CHECK(sample.val.i64 != 0);
                REQUIRE(cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_MEMORY_TEMP_CELSIUS) == DCGM_ST_OK);
            }
        }

        WHEN("NVML reports success with mismatched valueType and valid uiVal")
        {
            injectMemoryTemp(NVML_SUCCESS, NVML_VALUE_TYPE_DOUBLE, 61);
            auto sample = sampleMemoryTemp();

            THEN("DCGM reads uiVal despite the valueType workaround")
            {
                CHECK(sample.val.i64 == 61);
                REQUIRE(cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_MEMORY_TEMP_CELSIUS) == DCGM_ST_OK);
            }
        }

        WHEN("NVML reports success with uiVal above the supported range")
        {
            injectMemoryTemp(NVML_SUCCESS, NVML_VALUE_TYPE_UNSIGNED_INT, 201);
            auto sample = sampleMemoryTemp();

            THEN("DCGM blanks the out-of-range temperature")
            {
                CHECK(DCGM_INT64_IS_BLANK(sample.val.i64));
                REQUIRE(cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_MEMORY_TEMP_CELSIUS) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: board power raw field honors per-field NVML status")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5579);

        auto injectBoardPowerRaw = [&](nvmlReturn_t fieldReturn, unsigned int uiVal) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            nvmlFieldValue_t fv {};
            fv.fieldId     = NVML_FI_DEV_POWER_INSTANT;
            fv.nvmlReturn  = fieldReturn;
            fv.valueType   = NVML_VALUE_TYPE_UNSIGNED_INT;
            fv.value.uiVal = uiVal;
            fv.timestamp   = timelib_usecSince1970();
            REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &fv) == NVML_SUCCESS);
        };

        auto sampleBoardPowerRaw = [&]() {
            WatchField(cacheManager, gpuId, DCGM_FI_DEV_BOARD_POWER_RAW_WATTS, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(
                cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_RAW_WATTS, &sample, nullptr)
                == DCGM_ST_OK);
            return sample;
        };

        WHEN("NVML reports per-field NOT_SUPPORTED with uiVal 0")
        {
            injectBoardPowerRaw(NVML_ERROR_NOT_SUPPORTED, 0);
            auto sample = sampleBoardPowerRaw();

            THEN("DCGM reports NOT_SUPPORTED instead of passing through zero watts")
            {
                CHECK(sample.val.d == DCGM_FP64_NOT_SUPPORTED);
                CHECK(sample.val.d != 0.0);
                REQUIRE(cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_BOARD_POWER_RAW_WATTS) == DCGM_ST_OK);
            }
        }

        WHEN("NVML reports success with uiVal in milliwatts")
        {
            injectBoardPowerRaw(NVML_SUCCESS, 125500);
            auto sample = sampleBoardPowerRaw();

            THEN("DCGM converts milliwatts to watts")
            {
                CHECK(sample.val.d == Catch::Approx(125.5));
                REQUIRE(cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_BOARD_POWER_RAW_WATTS) == DCGM_ST_OK);
            }
        }
    }
}

TEST_CASE("DcgmCacheManager::IsGpuMigEnabled & IsMigEnabledAnywhere")
{
    TestDcgmCacheManager cacheManager;
    unsigned int constexpr gpuCount = 8;

    cacheManager.m_gpus.fill({});
    cacheManager.m_numGpus = gpuCount;

    SECTION("No MIG")
    {
        for (unsigned int i = 0; i < gpuCount; i++)
        {
            REQUIRE(cacheManager.IsGpuMigEnabled(i) == false);
        }
        REQUIRE(cacheManager.IsMigEnabledAnywhere() == false);
    }

    SECTION("MIG Enabled")
    {
        cacheManager.m_gpus[7].migEnabled = true;

        REQUIRE(cacheManager.IsGpuMigEnabled(7) == true);
        for (unsigned int i = 0; i < 7; i++)
        {
            REQUIRE(cacheManager.IsGpuMigEnabled(i) == false);
        }
        REQUIRE(cacheManager.IsMigEnabledAnywhere() == true);
    }
}

TEST_CASE("DcgmCacheManager: pure state helpers")
{
    ScopedDcgmFields dcgmFields;

    SECTION("GIVEN constructor environment WHEN force-GPM is set THEN the flag is enabled")
    {
        ScopedEnvVars envVars("__DCGM_FORCE_PROF_METRICS_THROUGH_GPM", "1");

        TestDcgmCacheManager cacheManager;

        CHECK(cacheManager.m_forceProfMetricsThroughGpm);
    }

    SECTION("GIVEN global watch fields WHEN global watch info is requested THEN create and lookup paths agree")
    {
        TestDcgmCacheManager cacheManager;

        CHECK(cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 0) == nullptr);

        auto watchInfo = cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 1);
        REQUIRE(watchInfo != nullptr);
        CHECK(watchInfo->watchKey.entityGroupId == DCGM_FE_NONE);
        CHECK(watchInfo->watchKey.entityId == 0);
        CHECK(watchInfo->watchKey.fieldId == DCGM_FI_SYSTEM_DRIVER_VERSION);
        CHECK(cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 0) == watchInfo);
    }

    SECTION("GIVEN fake GPU topology WHEN helper is called THEN cached GPU data is returned")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0 = cacheManager.AddFakeGpu(0x20B0, 0x145F);
        unsigned int const gpu1 = cacheManager.AddFakeGpu(0x2330, 0x1626);
        SafeCopyTo(cacheManager.m_gpus[gpu0].pciInfo.busId, "00000000:01:00.0");
        SafeCopyTo(cacheManager.m_gpus[gpu1].pciInfo.busId, "00000000:02:00.0");
        cacheManager.m_gpus[gpu0].numNvLinks = 4;
        cacheManager.m_gpus[gpu1].numNvLinks = 2;
        cacheManager.m_gpus[gpu0].arch       = DCGM_CHIP_ARCH_AMPERE;
        cacheManager.m_gpus[gpu1].arch       = DCGM_CHIP_ARCH_HOPPER;

        auto topology = cacheManager.GetTopologyHelper(false);

        REQUIRE(topology.size() == 2);
        CHECK(topology[0].gpuId == gpu0);
        CHECK(topology[0].status == DcgmEntityStatusFake);
        CHECK(topology[0].numNvLinks == 4);
        CHECK(topology[0].arch == DCGM_CHIP_ARCH_AMPERE);
        CHECK(std::string(topology[0].busId) == "00000000:01:00.0");
        CHECK(topology[1].gpuId == gpu1);
        CHECK(topology[1].numNvLinks == 2);
        CHECK(topology[1].arch == DCGM_CHIP_ARCH_HOPPER);
    }

    SECTION("GIVEN cached MIG hierarchy WHEN it is cleared THEN GPU instance counters are decremented")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId     = cacheManager.AddFakeGpu();
        unsigned int const instance0 = cacheManager.AddFakeInstance(gpuId);
        unsigned int const instance1 = cacheManager.AddFakeInstance(gpuId);
        unsigned int const ci0       = cacheManager.AddFakeComputeInstance(instance0);
        unsigned int const ci1       = cacheManager.AddFakeComputeInstance(instance1);
        auto &gpuInfo                = cacheManager.m_gpus[gpuId];

        REQUIRE(instance0 != instance1);
        REQUIRE(ci0 != ci1);
        REQUIRE(gpuInfo.instances.size() == 2);
        cacheManager.m_numInstances = gpuInfo.instances.size();
        cacheManager.m_numComputeInstances
            = gpuInfo.instances[0].GetComputeInstanceCount() + gpuInfo.instances[1].GetComputeInstanceCount();
        REQUIRE(cacheManager.m_numInstances == 2);
        REQUIRE(cacheManager.m_numComputeInstances == 2);

        cacheManager.ClearGpuMigInfo(gpuInfo);

        CHECK(gpuInfo.instances.empty());
        CHECK(cacheManager.m_numInstances == 0);
        CHECK(cacheManager.m_numComputeInstances == 0);
    }

    SECTION("GIVEN unavailable NVML state WHEN GPU instances are initialized THEN early returns avoid driver calls")
    {
        TestDcgmCacheManager cacheManager;
        dcgmcm_gpu_info_t gpuInfo {};

        gpuInfo.status = DcgmEntityStatusLost;
        CHECK(cacheManager.InitializeGpuInstances(gpuInfo) == DCGM_ST_GPU_IS_LOST);

        gpuInfo.status = DcgmEntityStatusOk;
        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK(cacheManager.InitializeGpuInstances(gpuInfo) == DCGM_ST_NVML_NOT_LOADED);
    }

    SECTION("GIVEN NVML is not loaded WHEN event set is initialized THEN no event set is created")
    {
        TestDcgmCacheManager cacheManager;

        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK(cacheManager.InitializeNvmlEventSet() == DCGM_ST_OK);
        CHECK_FALSE(cacheManager.m_nvmlEventSetInitialized);
    }

    SECTION("GIVEN fake GPUs and MIG entities WHEN entity helpers are queried THEN cached state is reported")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0          = cacheManager.AddFakeGpu(0x20B0, 0x145F);
        unsigned int const gpu1          = cacheManager.AddFakeGpu(0x20B0, 0x145F);
        unsigned int const gpu2          = cacheManager.AddFakeGpu(0x2330, 0x1626);
        unsigned int const instance0     = cacheManager.AddFakeInstance(gpu0);
        unsigned int const ci0           = cacheManager.AddFakeComputeInstance(instance0);
        cacheManager.m_gpus[gpu1].status = DcgmEntityStatusDetached;

        std::vector<unsigned int> gpuIds;
        REQUIRE(cacheManager.GetGpuIds(0, gpuIds) == DCGM_ST_OK);
        CHECK(gpuIds == std::vector<unsigned int> { gpu0, gpu1, gpu2 });

        REQUIRE(cacheManager.GetGpuIds(1, gpuIds) == DCGM_ST_OK);
        CHECK(gpuIds == std::vector<unsigned int> { gpu0, gpu2 });
        CHECK(cacheManager.GetGpuCount(0) == 3);
        CHECK(cacheManager.GetGpuCount(1) == 2);

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU);
        CHECK(entities[0].entityId == gpu0);
        CHECK(entities[1].entityId == gpu2);

        REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_I, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 1);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU_I);
        CHECK(entities[0].entityId == instance0);

        REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_CI, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 1);
        CHECK(entities[0].entityGroupId == DCGM_FE_GPU_CI);
        CHECK(entities[0].entityId == ci0);

        CHECK(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_VGPU, entities) == DCGM_ST_NOT_SUPPORTED);
        CHECK(entities.empty());

        CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, gpu0) == DcgmEntityStatusFake);
        CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, gpu1) == DcgmEntityStatusDetached);
        CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_I, instance0) == DcgmEntityStatusFake);
        CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_CI, ci0) == DcgmEntityStatusFake);
        CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, 99) == DcgmEntityStatusUnknown);
        CHECK(cacheManager.GetEntityStatus(DCGM_FE_NONE, 0) == DcgmEntityStatusUnknown);

        std::vector<unsigned int> matchingSku { gpu0, gpu1 };
        std::vector<unsigned int> mixedSku { gpu0, gpu2 };
        std::vector<unsigned int> invalidSku { gpu0, 99 };
        CHECK(cacheManager.AreAllGpuIdsSameSku(matchingSku) == 1);
        CHECK(cacheManager.AreAllGpuIdsSameSku(mixedSku) == 0);
        CHECK(cacheManager.AreAllGpuIdsSameSku(invalidSku) == 0);
    }

    SECTION("GIVEN field metadata WHEN validation helpers are called THEN scope and id checks are enforced")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();

        CHECK(cacheManager.CheckValidGlobalField(DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_OK);
        CHECK(cacheManager.CheckValidGlobalField(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);
        CHECK(cacheManager.CheckValidGlobalField(DCGM_FI_SYSTEM_FIELD_UNKNOWN) == DCGM_ST_UNKNOWN_FIELD);

        CHECK(cacheManager.CheckValidGpuField(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_OK);
        CHECK(cacheManager.CheckValidGpuField(gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_BADPARAM);
        CHECK(cacheManager.CheckValidGpuField(gpuId, DCGM_FI_SYSTEM_FIELD_UNKNOWN) == DCGM_ST_UNKNOWN_FIELD);
        CHECK(cacheManager.CheckValidGpuField(gpuId + 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS) == DCGM_ST_BADPARAM);

        dcgmcm_runtime_stats_t stats {};
        cacheManager.GetRuntimeStats(nullptr);
        cacheManager.GetRuntimeStats(&stats);
        CHECK(stats.lockCount >= 0);

        std::vector<unsigned short> validFieldIds;
        cacheManager.GetValidFieldIds(validFieldIds, true);
        REQUIRE_FALSE(validFieldIds.empty());
        CHECK(std::ranges::find(validFieldIds, DCGM_FI_DEV_GPU_TEMP_CELSIUS) != validFieldIds.end());

        cacheManager.GetValidFieldIds(validFieldIds, false);
        REQUIRE_FALSE(validFieldIds.empty());
        CHECK(std::ranges::find(validFieldIds, DCGM_FI_DEV_GPU_TEMP_CELSIUS) != validFieldIds.end());
    }

    SECTION("GIVEN watched fields WHEN snapshots are requested THEN watched and unwatched paths are distinct")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        dcgmcm_watch_info_t snapshot {};
        CHECK(cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr)
              == DCGM_ST_BADPARAM);
        CHECK(cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
              == DCGM_ST_NOT_WATCHED);

        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                           1000000,
                                           60.0,
                                           0,
                                           watcher,
                                           true,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
        CHECK(cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
              == DCGM_ST_OK);
        CHECK(snapshot.watchKey.entityGroupId == DCGM_FE_GPU);
        CHECK(snapshot.watchKey.entityId == gpuId);
        CHECK(snapshot.watchKey.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);

        std::vector<dcgmcm_watch_info_p> watchers;
        cacheManager.GetAllWatchObjects(watchers);
        CHECK_FALSE(watchers.empty());

        CHECK(cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1, watcher)
              == DCGM_ST_OK);
    }
}

TEST_CASE("GetMultipleLatestLiveSamples: MIG entity remapping for mapped fields")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("SKU file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto nvmlRet = nvmlInit_v2();
    REQUIRE(nvmlRet == NVML_SUCCESS);
    DcgmNs::Defer nvmlCleanup([&] { nvmlShutdown(); });

    TestDcgmCacheManager cm;
    REQUIRE(cm.Init(1, 14400.0, true) == DCGM_ST_OK);
    cm.Start();

    unsigned int constexpr gpuId      = 0;
    unsigned int instanceId           = cm.AddFakeInstance(gpuId);
    unsigned int computeInstanceId    = cm.AddFakeComputeInstance(instanceId);
    unsigned short constexpr fieldId  = DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION;
    long long constexpr expectedValue = 683729965;

    REQUIRE(instanceId != DCGM_ENTITY_ID_BAD);
    REQUIRE(computeInstanceId != DCGM_ENTITY_ID_BAD);

    std::vector<dcgmGroupEntityPair_t> entities
        = { { DCGM_FE_GPU, gpuId }, { DCGM_FE_GPU_I, instanceId }, { DCGM_FE_GPU_CI, computeInstanceId } };
    std::vector<unsigned short> fieldIds = { fieldId };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(cm.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    dcgmBufferedFvCursor_t cursor = 0;
    for (auto const &entity : entities)
    {
        dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == entity.entityGroupId);
        REQUIRE(fv->entityId == entity.entityId);
        REQUIRE(fv->value.i64 == expectedValue);
    }

    REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);
}

TEST_CASE("NvmlFieldRequiresNvLinkAggregate")
{
    SECTION("Returns true for all 34 NVLink COUNT fields")
    {
        // All NVLink COUNT fields that require aggregate scopeId
        std::vector<unsigned short> nvlinkCountFields = {
            // Basic COUNT fields
            NVML_FI_DEV_NVLINK_COUNT_XMIT_PACKETS,
            NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES,
            NVML_FI_DEV_NVLINK_COUNT_RCV_PACKETS,
            NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES,
            NVML_FI_DEV_NVLINK_COUNT_MALFORMED_PACKET_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_BUFFER_OVERRUN_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_RCV_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_RCV_REMOTE_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_RCV_GENERAL_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_XMIT_DISCARDS,
            NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS,
            NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS,
            NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS,
            NVML_FI_DEV_NVLINK_COUNT_SYMBOL_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER,
            // Effective errors/BER
            NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS,
            NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER,
            // FEC history fields
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_1,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_2,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_3,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_4,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_5,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_6,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_7,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_8,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_9,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_10,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_11,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_12,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_13,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_14,
            NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15,
        };

        // Verify we have all 34 fields
        REQUIRE(nvlinkCountFields.size() == 34);

        for (auto fieldId : nvlinkCountFields)
        {
            CHECK(NvmlFieldRequiresNvLinkAggregate(fieldId) == true);
        }
    }

    SECTION("Returns false for non-NVLink COUNT fields")
    {
        std::vector<unsigned short> nonCountFields = {
            // ECC fields
            NVML_FI_DEV_ECC_CURRENT,
            NVML_FI_DEV_ECC_PENDING,
            NVML_FI_DEV_ECC_SBE_VOL_TOTAL,
            NVML_FI_DEV_ECC_DBE_VOL_TOTAL,
            // NVLink bandwidth fields (not COUNT)
            NVML_FI_DEV_NVLINK_BANDWIDTH_C0_L0,
            NVML_FI_DEV_NVLINK_BANDWIDTH_C1_L0,
            NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL,
            // Memory temperature
            NVML_FI_DEV_MEMORY_TEMP,
            // Retired pages
            NVML_FI_DEV_RETIRED_SBE,
            NVML_FI_DEV_RETIRED_DBE,
        };

        for (auto fieldId : nonCountFields)
        {
            CHECK(NvmlFieldRequiresNvLinkAggregate(fieldId) == false);
        }
    }

    SECTION("Returns false for invalid/unknown field IDs")
    {
        std::vector<unsigned short> invalidFields = { 0, 9999, 65535 };

        for (auto fieldId : invalidFields)
        {
            CHECK(NvmlFieldRequiresNvLinkAggregate(fieldId) == false);
        }
    }
}

TEST_CASE("DcgmCacheManager: vGPU Type Fields with Supported vGPUs")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    for (auto fieldId : VgpuTypeFieldIds)
    {
        WatchField(cacheManager, gpuId, fieldId, watcher);
    }
    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_NAME, [](void *blob) {
        // vGPU.yaml GPU 0 has vGPU type 555 with Name="NVIDIA A40-1B"
        auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(vgpuTypeNames[0]) == "NVIDIA A40-1B");
    });

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_CLASS, [](void *blob) {
        // vGPU.yaml GPU 0 has vGPU type 555 with Class="NVS"
        auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(vgpuTypeClasses[0]) == "NVS");
    });

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_LICENSE, [](void *blob) {
        // vGPU.yaml GPU 0 has vGPU type 555 with License containing "GRID-Virtual-PC"
        auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(blob);
        std::string_view license(vgpuTypeLicenses[0]);
        REQUIRE(license.find("GRID-Virtual-PC") != std::string_view::npos);
    });
}

TEST_CASE("DcgmCacheManager: vGPU Type Fields when vGPU Not Supported")
{
    // H200.yaml returns FunctionReturn: 3 (NVML_ERROR_NOT_SUPPORTED) for SupportedVgpus
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    for (auto fieldId : VgpuTypeFieldIds)
    {
        WatchField(cacheManager, gpuId, fieldId, watcher);
    }

    // Should not crash even when vGPU is not supported.
    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_NAME, [](void *blob) {
        auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(vgpuTypeNames[0]) == std::string_view("Unknown"));
    });

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_CLASS, [](void *blob) {
        auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(vgpuTypeClasses[0]) == std::string_view("Unknown"));
    });

    CheckLatestBlob(cacheManager, gpuId, DCGM_FI_DEV_VGPU_TYPE_LICENSE, [](void *blob) {
        auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(vgpuTypeLicenses[0]) == std::string_view("Unknown"));
    });
}

TEST_CASE("DcgmCacheManager::GetSupportedVgpuTypeNames - DCGM-6550 Memory Safety")
{
    // Test all scenarios for DCGM-6550 fix:
    // - vgpuCount = 0: No stack data leak (arrays initialized)
    // - vgpuCount < 32: Normal operation
    // - vgpuCount = 32: Boundary condition (exactly at limit)
    // - vgpuCount > 32: Buffer overflow protection (capping to 32)

    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    ScopedDcgmFields dcgmFields;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    for (unsigned int gpuId : { 0U, 3U, 4U, 5U })
    {
        for (auto fieldId : VgpuTypeFieldIds)
        {
            WatchField(cacheManager, gpuId, fieldId, watcher);
        }
    }

    REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

    auto requireAllNamesUnknown = [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
        {
            REQUIRE(std::string_view(values[i]) == std::string_view("Unknown"));
        }
    };

    auto requireAllLicensesUnknown = [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(blob);
        for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
        {
            REQUIRE(std::string_view(values[i]) == std::string_view("Unknown"));
        }
    };

    auto requireAllNamesKnown = [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
        {
            REQUIRE(values[i][0] != '\0');
            REQUIRE(std::string_view(values[i]) != std::string_view("Unknown"));
        }
    };

    auto requireAllLicensesKnown = [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(blob);
        for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
        {
            REQUIRE(values[i][0] != '\0');
            REQUIRE(std::string_view(values[i]) != std::string_view("Unknown"));
        }
    };

    // GPU index 3 has 0 vGPU types.
    CheckLatestBlob(cacheManager, 3, DCGM_FI_DEV_VGPU_TYPE_NAME, requireAllNamesUnknown);
    CheckLatestBlob(cacheManager, 3, DCGM_FI_DEV_VGPU_TYPE_CLASS, requireAllNamesUnknown);
    CheckLatestBlob(cacheManager, 3, DCGM_FI_DEV_VGPU_TYPE_LICENSE, requireAllLicensesUnknown);

    // GPU index 0 has 29 vGPU types.
    CheckLatestBlob(cacheManager, 0, DCGM_FI_DEV_VGPU_TYPE_NAME, [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(values[0][0] != '\0');
        REQUIRE(std::string_view(values[0]) != std::string_view("Unknown"));
        REQUIRE(values[28][0] != '\0');
        REQUIRE(std::string_view(values[28]) != std::string_view("Unknown"));
        REQUIRE(std::string_view(values[29]) == std::string_view("Unknown"));
        REQUIRE(std::string_view(values[31]) == std::string_view("Unknown"));
    });
    CheckLatestBlob(cacheManager, 0, DCGM_FI_DEV_VGPU_TYPE_CLASS, [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(blob);
        REQUIRE(std::string_view(values[0]) != std::string_view("Unknown"));
        REQUIRE(std::string_view(values[28]) != std::string_view("Unknown"));
        REQUIRE(std::string_view(values[29]) == std::string_view("Unknown"));
    });
    CheckLatestBlob(cacheManager, 0, DCGM_FI_DEV_VGPU_TYPE_LICENSE, [](void *blob) {
        auto *values = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(blob);
        REQUIRE(values[0][0] != '\0');
        REQUIRE(std::string_view(values[0]) != std::string_view("Unknown"));
        REQUIRE(values[28][0] != '\0');
        REQUIRE(std::string_view(values[28]) != std::string_view("Unknown"));
        REQUIRE(std::string_view(values[29]) == std::string_view("Unknown"));
    });

    // GPU index 4 has exactly 32 vGPU types.
    CheckLatestBlob(cacheManager, 4, DCGM_FI_DEV_VGPU_TYPE_NAME, requireAllNamesKnown);
    CheckLatestBlob(cacheManager, 4, DCGM_FI_DEV_VGPU_TYPE_CLASS, requireAllNamesKnown);
    CheckLatestBlob(cacheManager, 4, DCGM_FI_DEV_VGPU_TYPE_LICENSE, requireAllLicensesKnown);

    // GPU index 5 has 40 vGPU types, but only 32 should be stored.
    CheckLatestBlob(cacheManager, 5, DCGM_FI_DEV_VGPU_TYPE_NAME, requireAllNamesKnown);
    CheckLatestBlob(cacheManager, 5, DCGM_FI_DEV_VGPU_TYPE_CLASS, requireAllNamesKnown);
    CheckLatestBlob(cacheManager, 5, DCGM_FI_DEV_VGPU_TYPE_LICENSE, requireAllLicensesKnown);
}

TEST_CASE("DcgmCacheManager::IsModulePushedFieldId")
{
    ScopedDcgmFields dcgmFields;
    TestDcgmCacheManager cm;

    SECTION("Returns false for GPU device fields below the NvSwitch boundary")
    {
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_BOARD_POWER_WATTS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_FB_USED) == false);
    }

    SECTION("Returns true for NvSwitch fields (700-899)")
    {
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_FIRST_NVSWITCH_FIELD_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_LAST_NVSWITCH_FIELD_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT) == true);
    }

    SECTION("Returns false for NVLink TX/RX throughput fields (IDs overlap NvSwitch range)")
    {
        // Per-link and total throughput use IDs inside 700-899 but are updated by the
        // cache from NVML, not pushed by the NvSwitch module.
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL) == false);
    }

    SECTION("Returns true for profiling fields")
    {
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_PROF_FIRST_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_PROF_LAST_ID) == true);
    }

    SECTION("Returns true for sysmon fields (1100-1141)")
    {
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_SYSMON_FIRST_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_SYSMON_LAST_ID) == true);
    }

    SECTION("Returns true for ConnectX fields (1300-1399)")
    {
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_FIRST_CONNECTX_FIELD_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_LAST_CONNECTX_FIELD_ID) == true);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CONNECTX_HEALTH) == true);
    }

    SECTION("Returns false for clock event reason NS fields (1420-1424)")
    {
        // NVML-polled GPU device fields outside all module-pushed ranges
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP_NS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST_NS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN_NS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN_NS) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN_NS) == false);
    }

    SECTION("Returns false for power smoothing fields (1425-1442)")
    {
        // NVML-polled GPU device fields outside all module-pushed ranges
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_PWR_SMOOTHING_ENABLED) == false);
        CHECK(cm.IsModulePushedFieldId(DCGM_FI_DEV_PWR_SMOOTHING_ADMIN_OVERRIDE_RAMP_DOWN_HYST_VAL) == false);
    }

    SECTION("Returns false by default for unknown high-numbered fields outside module ranges")
    {
        // Any future GPU device field added outside the known module ranges must
        // default to false (polled), not true (skipped).  This guards against the
        // class of bug fixed in DCGM-6546.
        // Pick a field ID in the gap between profiling range 1 and range 2
        // (e.g. 1500) that is not in any known module range.
        unsigned int const hypotheticalFutureField = 1500;
        CHECK(cm.IsModulePushedFieldId(hypotheticalFutureField) == false);
    }
}

TEST_CASE("DCGM_FIELD_ID_IS_PROF_FIELD")
{
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_FIRST_ID));
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_LAST_ID));
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_FP16_CYCLES_ACTIVE_TOTAL));
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK_ID));
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK_ID));

    static_assert(!DCGM_FIELD_ID_IS_PROF_FIELD(1000));
    static_assert(!DCGM_FIELD_ID_IS_PROF_FIELD(1097));
    static_assert(!DCGM_FIELD_ID_IS_PROF_FIELD(1100));
    static_assert(!DCGM_FIELD_ID_IS_PROF_FIELD(1500));
}

TEST_CASE("DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH")
{
    // PCIe throughput
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_PCIE_TX_BYTES));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_PCIE_RX_BYTES));
    // NVLink aggregate throughput
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_TX_BYTES));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_RX_BYTES));
    // NVLink per-link throughput (legacy L0-L17 and dcgm_link_t-keyed)
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_FIRST));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_LAST));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK_ID));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK_ID));
    // C2C throughput
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_C2C_TX_ALL_BYTES));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_C2C_RX_DATA_BYTES));

    // Ratio fields are not bandwidth
    static_assert(!DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO));
    // Cumulative counters are not bandwidth
    static_assert(!DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_SM_CYCLES_ELAPSED_TOTAL));
    static_assert(!DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_PCIE_TX_BYTES_TOTAL));
}

TEST_CASE("DcgmFieldIsNvLinkCountField")
{
    SECTION("Returns true for NVLink COUNT fields")
    {
        std::vector<unsigned short> countFields = {
            DCGM_FI_DEV_NVLINK_TX_PACKET_TOTAL,      DCGM_FI_DEV_NVLINK_TX_BYTES_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_PACKET_TOTAL,      DCGM_FI_DEV_NVLINK_RX_BYTES_TOTAL,
            DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL,       DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW,
            DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO,     DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL,
            DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO,  DCGM_FI_DEV_NVLINK_FEC_HISTORY_0_TOTAL,
            DCGM_FI_DEV_NVLINK_FEC_HISTORY_15_TOTAL,
        };
        for (auto fieldId : countFields)
        {
            CHECK(DcgmFieldIsNvLinkCountField(fieldId) == true);
        }
    }

    SECTION("Returns false for non-COUNT fields")
    {
        std::vector<unsigned short> nonCountFields = {
            DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL,
            DCGM_FI_DEV_NVLINK_THROUGHPUT_L0,
            DCGM_FI_DEV_BOARD_POWER_WATTS,
            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
            0,
            1199,
            1220,
            9999,
        };
        for (auto fieldId : nonCountFields)
        {
            CHECK(DcgmFieldIsNvLinkCountField(fieldId) == false);
        }
    }
}

TEST_CASE("DcgmCacheManager::AddFieldWatch allows module-pushed NvSwitch link fields without NVML")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cm;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    dcgm_link_t switchLink {};
    switchLink.parsed.type     = DCGM_FE_SWITCH;
    switchLink.parsed.switchId = 0;
    switchLink.parsed.index    = 0;

    bool wereFirstWatcher = false;
    REQUIRE(cm.AddFieldWatch(DCGM_FE_LINK,
                             switchLink.raw,
                             DCGM_FI_DEV_NVSWITCH_LINK_ID,
                             1000000,
                             30.0,
                             1,
                             watcher,
                             false,
                             false,
                             wereFirstWatcher)
            == DCGM_ST_OK);

    dcgm_link_t gpuLink {};
    gpuLink.parsed.type  = DCGM_FE_GPU;
    gpuLink.parsed.gpuId = 0;
    gpuLink.parsed.index = 0;

    wereFirstWatcher = false;
    REQUIRE(cm.AddFieldWatch(DCGM_FE_LINK,
                             gpuLink.raw,
                             DCGM_FI_DEV_NVLINK_LINK_STATUS,
                             1000000,
                             30.0,
                             1,
                             watcher,
                             false,
                             false,
                             wereFirstWatcher)
            == DCGM_ST_NVML_NOT_LOADED);
}

TEST_CASE("CacheManager: typed append helpers cache and buffer values")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    auto makeContext = [&](unsigned short fieldId, DcgmFvBuffer &fvBuffer) {
        dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched  = 1;
        watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));

        dcgmcm_update_thread_t threadCtx;
        threadCtx.watchInfo               = watchInfo;
        threadCtx.fvBuffer                = &fvBuffer;
        threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
        threadCtx.entityKey.entityId      = gpuId;
        threadCtx.entityKey.fieldId       = fieldId;
        return threadCtx;
    };

    GIVEN("watched GPU fields for each value type")
    {
        SECTION("double values are written to the time series and field-value buffer")
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_BOARD_POWER_WATTS;
            double constexpr value           = 125.5;
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeContext(fieldId, fvBuffer);

            WHEN("the double append helper is used")
            {
                timelib64_t const now = timelib_usecSince1970();
                REQUIRE(cacheManager.AppendEntityDouble(threadCtx, value, 0.0, now, 0) == DCGM_ST_OK);

                THEN("the buffered and cached values match")
                {
                    dcgmBufferedFvCursor_t cursor = 0;
                    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->fieldType == DCGM_FT_DOUBLE);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->value.dbl == value);
                    CHECK(threadCtx.affectedSubscribers == 0);

                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
                    CHECK(sample.val.d == value);
                }
            }
        }

        SECTION("int64 values are written to the time series and field-value buffer")
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_ECC_MODE;
            long long constexpr value        = 2;
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeContext(fieldId, fvBuffer);

            WHEN("the int64 append helper is used")
            {
                timelib64_t const now = timelib_usecSince1970();
                REQUIRE(cacheManager.AppendEntityInt64(threadCtx, value, 0, now, 0) == DCGM_ST_OK);

                THEN("the buffered and cached values match")
                {
                    dcgmBufferedFvCursor_t cursor = 0;
                    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->fieldType == DCGM_FT_INT64);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->value.i64 == value);

                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
                    CHECK(sample.val.i64 == value);
                }
            }
        }

        SECTION("string values are written to the time series and field-value buffer")
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_GPU_NAME;
            char const *value                = "unit-test-gpu";
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeContext(fieldId, fvBuffer);

            WHEN("the string append helper is used")
            {
                timelib64_t const now = timelib_usecSince1970();
                REQUIRE(cacheManager.AppendEntityString(threadCtx, value, now, 0) == DCGM_ST_OK);

                THEN("the buffered and cached values match")
                {
                    dcgmBufferedFvCursor_t cursor = 0;
                    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->fieldType == DCGM_FT_STRING);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(std::strcmp(fv->value.str, value) == 0);

                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
                    REQUIRE(sample.val.str != nullptr);
                    CHECK(std::strcmp(sample.val.str, value) == 0);
                    REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
                }
            }
        }

        SECTION("blob values are written to the time series and field-value buffer")
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS;
            std::array<unsigned char, 4> value { 0x1, 0x2, 0x3, 0x4 };
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeContext(fieldId, fvBuffer);

            WHEN("the blob append helper is used")
            {
                timelib64_t const now = timelib_usecSince1970();
                REQUIRE(cacheManager.AppendEntityBlob(threadCtx, value.data(), value.size(), now, 0) == DCGM_ST_OK);

                THEN("the buffered and cached values match")
                {
                    dcgmBufferedFvCursor_t cursor = 0;
                    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->fieldType == DCGM_FT_BINARY);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->length == (sizeof(*fv) - sizeof(fv->value)) + value.size());
                    CHECK(std::memcmp(fv->value.blob, value.data(), value.size()) == 0);

                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
                    REQUIRE(sample.val.blob != nullptr);
                    CHECK(sample.val2.ptrSize == value.size());
                    CHECK(std::memcmp(sample.val.blob, value.data(), value.size()) == 0);
                    REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
                }
            }
        }
    }
}

TEST_CASE("CacheManager: entity mapping helpers handle GPUs and fake MIG entities")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId              = cacheManager.AddFakeGpu();
    unsigned int const gpuInstanceId      = cacheManager.AddFakeInstance(gpuId);
    unsigned int const computeInstanceId  = cacheManager.AddFakeComputeInstance(gpuInstanceId);
    unsigned int constexpr missingNvmlId  = 99;
    dcgm_field_eid_t constexpr missingEid = 4242;

    GIVEN("a cache manager with fake GPU, GPU instance, and compute instance entities")
    {
        SECTION("entity-to-GPU lookup recognizes direct and MIG entities")
        {
            auto gpuLookup = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU, gpuId);
            REQUIRE(gpuLookup.has_value());
            CHECK(gpuLookup.value() == gpuId);

            auto instanceLookup = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_I, gpuInstanceId);
            REQUIRE(instanceLookup.has_value());
            CHECK(instanceLookup.value() == gpuId);

            auto computeInstanceLookup = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_CI, computeInstanceId);
            REQUIRE(computeInstanceLookup.has_value());
            CHECK(computeInstanceLookup.value() == gpuId);

            CHECK_FALSE(cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_I, missingEid).has_value());
            CHECK_FALSE(cacheManager.GetGpuIdForEntity(DCGM_FE_NONE, gpuId).has_value());
        }

        SECTION("NVML MIG IDs map to DCGM entity IDs")
        {
            CHECK(cacheManager.GetInstanceEntityId(gpuId, DcgmNs::Mig::Nvml::GpuInstanceId { 0 }) == gpuInstanceId);
            CHECK(cacheManager.GetComputeInstanceEntityId(
                      gpuId, DcgmNs::Mig::Nvml::ComputeInstanceId { 0 }, DcgmNs::Mig::Nvml::GpuInstanceId { 0 })
                  == computeInstanceId);

            CHECK(cacheManager.GetInstanceEntityId(gpuId, DcgmNs::Mig::Nvml::GpuInstanceId { missingNvmlId })
                  == DCGM_BLANK_ENTITY_ID);
            CHECK(cacheManager.GetComputeInstanceEntityId(gpuId,
                                                          DcgmNs::Mig::Nvml::ComputeInstanceId { missingNvmlId },
                                                          DcgmNs::Mig::Nvml::GpuInstanceId { 0 })
                  == DCGM_BLANK_ENTITY_ID);
            CHECK(cacheManager.GetInstanceProfile(gpuId, DcgmNs::Mig::Nvml::GpuInstanceId { missingNvmlId })
                  == DcgmMigProfileNone);
        }

        SECTION("profiling entity filtering honors forced GPM mode")
        {
            std::vector<dcgmGroupEntityPair_t> entities = {
                { DCGM_FE_GPU, gpuId },
                { DCGM_FE_GPU_I, gpuInstanceId },
                { DCGM_FE_GPU_CI, computeInstanceId },
            };

            cacheManager.m_forceProfMetricsThroughGpm = true;
            cacheManager.GetProfModuleServicedEntities(entities);
            CHECK(entities.empty());
        }

        SECTION("active entity filtering keeps valid fake GPU and MIG entities")
        {
            std::vector<dcgmGroupEntityPair_t> entities = {
                { DCGM_FE_GPU, gpuId },        { DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES }, { DCGM_FE_GPU_I, gpuInstanceId },
                { DCGM_FE_GPU_I, missingEid }, { DCGM_FE_GPU_CI, computeInstanceId }, { DCGM_FE_GPU_CI, missingEid },
                { DCGM_FE_SWITCH, 9 },
            };

            auto activeEntities = cacheManager.FilterActiveEntities(entities);

            REQUIRE(activeEntities.size() == 4);
            CHECK(activeEntities[0].entityGroupId == DCGM_FE_GPU);
            CHECK(activeEntities[0].entityId == gpuId);
            CHECK(activeEntities[1].entityGroupId == DCGM_FE_GPU_I);
            CHECK(activeEntities[1].entityId == gpuInstanceId);
            CHECK(activeEntities[2].entityGroupId == DCGM_FE_GPU_CI);
            CHECK(activeEntities[2].entityId == computeInstanceId);
            CHECK(activeEntities[3].entityGroupId == DCGM_FE_SWITCH);
            CHECK(activeEntities[3].entityId == 9);
        }

        SECTION("MIG index lookup validates output pointers and entity types")
        {
            unsigned int foundGpuId = DCGM_MAX_NUM_DEVICES;
            DcgmNs::Mig::GpuInstanceId foundInstanceId {};
            DcgmNs::Mig::ComputeInstanceId foundComputeInstanceId {};

            REQUIRE(cacheManager.GetMigIndicesForEntity(
                        { DCGM_FE_GPU_I, gpuInstanceId }, &foundGpuId, &foundInstanceId, &foundComputeInstanceId)
                    == DCGM_ST_OK);
            CHECK(foundGpuId == gpuId);
            CHECK(foundInstanceId.id == gpuInstanceId);

            CHECK(cacheManager.GetMigIndicesForEntity(
                      { DCGM_FE_GPU_I, gpuInstanceId }, nullptr, &foundInstanceId, &foundComputeInstanceId)
                  == DCGM_ST_BADPARAM);

            REQUIRE(cacheManager.GetMigIndicesForEntity(
                        { DCGM_FE_GPU_CI, computeInstanceId }, &foundGpuId, &foundInstanceId, &foundComputeInstanceId)
                    == DCGM_ST_OK);
            CHECK(foundGpuId == gpuId);
            CHECK(foundInstanceId.id == gpuInstanceId);
            CHECK(foundComputeInstanceId.id == computeInstanceId);

            CHECK(cacheManager.GetMigIndicesForEntity(
                      { DCGM_FE_GPU_CI, computeInstanceId }, &foundGpuId, &foundInstanceId, nullptr)
                  == DCGM_ST_BADPARAM);
            CHECK(cacheManager.GetMigIndicesForEntity(
                      { DCGM_FE_GPU_CI, missingEid }, &foundGpuId, &foundInstanceId, &foundComputeInstanceId)
                  == DCGM_ST_NO_DATA);
        }

        SECTION("NVLink and NVML handle helpers validate simple GPU states")
        {
            dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_GPU] {};

            CHECK(cacheManager.GetEntityNvLinkLinkStatus(DCGM_FE_NONE, gpuId, linkStates) == DCGM_ST_BADPARAM);
            CHECK(cacheManager.GetEntityNvLinkLinkStatus(DCGM_FE_GPU, gpuId, nullptr) == DCGM_ST_BADPARAM);
            CHECK(cacheManager.GetEntityNvLinkLinkStatus(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES, linkStates)
                  == DCGM_ST_BADPARAM);
            CHECK(cacheManager.GetEntityNvLinkLinkStatus(DCGM_FE_GPU, gpuId, linkStates) == DCGM_ST_OK);
            CHECK(cacheManager.UpdateNvLinkLinkState(DCGM_MAX_NUM_DEVICES) == DCGM_ST_BADPARAM);
            CHECK(cacheManager.UpdateNvLinkLinkState(gpuId) == DCGM_ST_OK);

            CHECK(cacheManager.GetNvmlDeviceFromEntityId(DCGM_MAX_NUM_DEVICES) == nullptr);
            CHECK(cacheManager.GetActiveGpuHandles().empty());

            auto safeHandle = cacheManager.GetSafeNvmlHandle(gpuId);
            REQUIRE(safeHandle.has_value());
            CHECK(safeHandle->nvmlDevice == nullptr);

            auto missingHandle = cacheManager.GetSafeNvmlHandle(DCGM_MAX_NUM_DEVICES);
            REQUIRE(missingHandle.is_error());
            CHECK(missingHandle.error() == DCGM_ST_BADPARAM);
        }
    }
}

TEST_CASE("CachePrmField")
{
    ScopedDcgmFields dcgmFields;

    dcgm_link_t link              = {};
    link.parsed.type              = DCGM_FE_GPU;
    link.parsed.gpuId             = 0;
    link.parsed.index             = 1;
    dcgm_field_eid_t linkEntityId = link.raw;

    TestDcgmCacheManager cm;
    dcgmcm_update_thread_t threadCtx;
    DcgmFvBuffer fvBuffer;
    threadCtx.fvBuffer                = &fvBuffer;
    threadCtx.entityKey.entityGroupId = DCGM_FE_LINK;
    threadCtx.entityKey.entityId      = linkEntityId;

    auto watchField = [&](unsigned short fieldId) {
        dcgmcm_watch_info_p watchInfo = cm.GetEntityWatchInfo(DCGM_FE_LINK, linkEntityId, fieldId, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched  = 1;
        watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));
    };

    SECTION("populates time series for requested field in live mode")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL;
        long long constexpr testValue    = 42LL;

        watchField(fieldId);

        threadCtx.entityKey.fieldId = fieldId;

        timelib64_t now = timelib_usecSince1970();

        cm.CachePrmField(threadCtx, linkEntityId, fieldId, fieldId, static_cast<uint64_t>(testValue), now);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == linkEntityId);
        REQUIRE(fv->fieldId == fieldId);
        REQUIRE(fv->value.i64 == testValue);

        dcgmcm_sample_t sample {};
        REQUIRE(cm.GetLatestSample(DCGM_FE_LINK, linkEntityId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.i64 == testValue);
    }

    SECTION("consecutive live mode calls use correct watchInfo per field")
    {
        unsigned short constexpr fieldA = DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL;
        unsigned short constexpr fieldB = DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL;
        long long constexpr valueA      = 11LL;
        long long constexpr valueB      = 22LL;

        watchField(fieldA);
        watchField(fieldB);

        timelib64_t now = timelib_usecSince1970();

        threadCtx.entityKey.fieldId = fieldA;
        cm.CachePrmField(threadCtx, linkEntityId, fieldA, fieldA, static_cast<uint64_t>(valueA), now);

        threadCtx.entityKey.fieldId = fieldB;
        cm.CachePrmField(threadCtx, linkEntityId, fieldB, fieldB, static_cast<uint64_t>(valueB), now + 1);

        dcgmcm_sample_t sampleA {};
        REQUIRE(cm.GetLatestSample(DCGM_FE_LINK, linkEntityId, fieldA, &sampleA, nullptr) == DCGM_ST_OK);
        REQUIRE(sampleA.val.i64 == valueA);

        dcgmcm_sample_t sampleB {};
        REQUIRE(cm.GetLatestSample(DCGM_FE_LINK, linkEntityId, fieldB, &sampleB, nullptr) == DCGM_ST_OK);
        REQUIRE(sampleB.val.i64 == valueB);
    }
}

TEST_CASE("GetMultipleLatestLiveSamples: unsupported PRM ports return one not-supported value")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cm;
    cm.m_nvmlLoaded.store(true, std::memory_order_release);
    unsigned int const gpuId = cm.AddFakeGpu();

    auto makeLinkEntity = [&](unsigned int portIndex) {
        dcgm_link_t link  = {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = gpuId;
        link.parsed.index = portIndex;
        return dcgmGroupEntityPair_t { DCGM_FE_LINK, link.raw };
    };

    auto watchField = [&](dcgm_field_eid_t entityId, unsigned short fieldId) {
        dcgmcm_watch_info_p watchInfo = cm.GetEntityWatchInfo(DCGM_FE_LINK, entityId, fieldId, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched  = 1;
        watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));
    };

    auto requireUnsupportedPort = [&](unsigned short fieldId, unsigned int portIndex) {
        auto entity = makeLinkEntity(portIndex);
        watchField(entity.entityId, fieldId);

        std::vector<dcgmGroupEntityPair_t> entities = { entity };
        std::vector<unsigned short> fieldIds        = { fieldId };
        DcgmFvBuffer fvBuffer(1024);

        REQUIRE(cm.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == entity.entityId);
        REQUIRE(fv->fieldId == fieldId);
        REQUIRE(fv->status == DCGM_ST_OK);
        REQUIRE(fv->value.i64 == DCGM_INT64_NOT_SUPPORTED);
        REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);

        dcgmcm_sample_t sample {};
        REQUIRE(cm.GetLatestSample(DCGM_FE_LINK, entity.entityId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.i64 == DCGM_INT64_NOT_SUPPORTED);
    };

    requireUnsupportedPort(DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY, DCGM_NVLINK_MAX_LINKS_PER_GPU);
    requireUnsupportedPort(DCGM_FI_DEV_NVLINK_PPCNT_IBPC_TX_WAIT_TOTAL, DCGM_NVLINK_MAX_LINKS_PER_GPU + 1);
}

TEST_CASE("GetMultipleLatestLiveSamples: GPU-only legacy NVLink profiling fields reject LINK entities")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };

    for (unsigned short const fieldId :
         { DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_FIRST, DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_LAST })
    {
        CAPTURE(fieldId);
        std::vector<unsigned short> fieldIds = { fieldId };
        DcgmFvBuffer fvBuffer(1024);

        REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        CHECK(fv->entityGroupId == DCGM_FE_LINK);
        CHECK(fv->entityId == link.raw);
        CHECK(fv->fieldId == fieldId);
        CHECK(fv->status == DCGM_ST_NOT_SUPPORTED);
        CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
    }
}

SCENARIO("CacheManager: legacy NVLink profiling dispatch rejects missing watch information")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU and no watch information")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        dcgmcm_update_thread_t threadCtx {};
        threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
        threadCtx.entityKey.entityId      = gpuId;

        WHEN("each legacy NVLink profiling field is dispatched")
        {
            std::vector<std::pair<unsigned short, dcgmReturn_t>> results;
            for (unsigned short fieldId = DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_FIRST;
                 fieldId <= DCGM_FI_PROF_NVLINK_THROUGHPUT_LEGACY_LAST;
                 ++fieldId)
            {
                auto *fieldMeta = DcgmFieldGetById(fieldId);
                REQUIRE(fieldMeta != nullptr);
                threadCtx.entityKey.fieldId = fieldId;
                results.emplace_back(fieldId, cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta));
            }

            THEN("each dispatch reports the missing watch information")
            {
                for (auto const &[fieldId, result] : results)
                {
                    CAPTURE(fieldId);
                    CHECK(result == DCGM_ST_GENERIC_ERROR);
                }
            }
        }
    }
}

SCENARIO("CacheManager: sample APIs return deterministic cached samples, buffers, and summaries")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("integer and floating point samples injected with stable timestamps")
    {
        dcgm_field_meta_p tempMeta  = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        dcgm_field_meta_p powerMeta = DcgmFieldGetById(DCGM_FI_DEV_BOARD_POWER_WATTS);
        REQUIRE(tempMeta != nullptr);
        REQUIRE(powerMeta != nullptr);
        REQUIRE(tempMeta->fieldType == DCGM_FT_INT64);
        REQUIRE(powerMeta->fieldType == DCGM_FT_DOUBLE);

        std::array<dcgmcm_sample_t, 3> tempSamples {};
        tempSamples[0].timestamp = 100;
        tempSamples[0].val.i64   = 10;
        tempSamples[1].timestamp = 200;
        tempSamples[1].val.i64   = 20;
        tempSamples[2].timestamp = 300;
        tempSamples[2].val.i64   = 30;

        std::array<dcgmcm_sample_t, 3> powerSamples {};
        powerSamples[0].timestamp = 100;
        powerSamples[0].val.d     = 1.5;
        powerSamples[1].timestamp = 200;
        powerSamples[1].val.d     = 2.5;
        powerSamples[2].timestamp = 300;
        powerSamples[2].val.d     = 4.0;

        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                           tempSamples.data(),
                                           static_cast<int>(tempSamples.size()))
                == DCGM_ST_OK);
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_BOARD_POWER_WATTS,
                                           powerSamples.data(),
                                           static_cast<int>(powerSamples.size()))
                == DCGM_ST_OK);

        WHEN("unwatched samples are requested in bounded ascending order")
        {
            std::array<dcgmcm_sample_t, 4> returned {};
            int count = static_cast<int>(returned.size());

            auto ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                               gpuId,
                                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                               returned.data(),
                                               &count,
                                               150,
                                               300,
                                               DCGM_ORDER_ASCENDING,
                                               nullptr);

            THEN("the unwatched field is rejected without modifying caller capacity")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
                CHECK(count == 0);
            }
        }

        WHEN("unwatched samples are requested into a field-value buffer")
        {
            DcgmFvBuffer fvBuffer;
            int count = 3;

            auto ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                               gpuId,
                                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                               nullptr,
                                               &count,
                                               0,
                                               250,
                                               DCGM_ORDER_DESCENDING,
                                               &fvBuffer);

            THEN("the unwatched field is rejected and no values are buffered")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
                CHECK(count == 0);
                dcgmBufferedFvCursor_t cursor = 0;
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("multiple latest samples are requested for both cached fields")
        {
            std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId } };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
            DcgmFvBuffer fvBuffer;

            REQUIRE(cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

            THEN("the latest value for each field is returned in the request order")
            {
                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *temp        = fvBuffer.GetNextFv(&cursor);
                REQUIRE(temp != nullptr);
                CHECK(temp->fieldType == DCGM_FT_INT64);
                CHECK(temp->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(temp->timestamp == 0);
                CHECK(temp->value.i64 == DCGM_INT64_BLANK);

                dcgmBufferedFv_t *power = fvBuffer.GetNextFv(&cursor);
                REQUIRE(power != nullptr);
                CHECK(power->fieldType == DCGM_FT_DOUBLE);
                CHECK(power->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(power->timestamp == 0);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("integer summaries are requested with and without a callback filter")
        {
            std::array<DcgmcmSummaryType_t, 7> summaryTypes { DcgmcmSummaryTypeMinimum,   DcgmcmSummaryTypeMaximum,
                                                              DcgmcmSummaryTypeAverage,   DcgmcmSummaryTypeSum,
                                                              DcgmcmSummaryTypeCount,     DcgmcmSummaryTypeIntegral,
                                                              DcgmcmSummaryTypeDifference };
            std::array<long long, 7> summaryValues {};

            auto ret = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                        gpuId,
                                                        DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                        static_cast<int>(summaryTypes.size()),
                                                        summaryTypes.data(),
                                                        summaryValues.data(),
                                                        0,
                                                        0,
                                                        nullptr,
                                                        nullptr);

            THEN("unwatched integer summaries are rejected")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
            }

            AND_THEN("callback filtering changes the summary population")
            {
                timelib64_t cutoff = 200;
                std::array<DcgmcmSummaryType_t, 3> filteredTypes { DcgmcmSummaryTypeMinimum,
                                                                   DcgmcmSummaryTypeCount,
                                                                   DcgmcmSummaryTypeSum };
                std::array<long long, 3> filteredValues {};

                CHECK(cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                       gpuId,
                                                       DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                       static_cast<int>(filteredTypes.size()),
                                                       filteredTypes.data(),
                                                       filteredValues.data(),
                                                       0,
                                                       0,
                                                       UseOnlySamplesAtOrAfterTimestamp,
                                                       &cutoff)
                      == DCGM_ST_NOT_WATCHED);
            }
        }

        WHEN("floating point summaries and mismatched summary types are requested")
        {
            std::array<DcgmcmSummaryType_t, 4> summaryTypes { DcgmcmSummaryTypeMinimum,
                                                              DcgmcmSummaryTypeMaximum,
                                                              DcgmcmSummaryTypeAverage,
                                                              DcgmcmSummaryTypeDifference };
            std::array<double, 4> summaryValues {};

            auto ret = cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                       gpuId,
                                                       DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                       static_cast<int>(summaryTypes.size()),
                                                       summaryTypes.data(),
                                                       summaryValues.data(),
                                                       0,
                                                       0,
                                                       nullptr,
                                                       nullptr);

            THEN("unwatched floating point summaries are rejected")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
            }

            AND_THEN("asking the fp64 helper to summarize an int64 time series is rejected")
            {
                CHECK(cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                      gpuId,
                                                      DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                      static_cast<int>(summaryTypes.size()),
                                                      summaryTypes.data(),
                                                      summaryValues.data(),
                                                      0,
                                                      0,
                                                      nullptr,
                                                      nullptr)
                      == DCGM_ST_GENERIC_ERROR);
            }
        }
    }
}

SCENARIO("CacheManager: append and watch lifecycle helpers maintain observable watch state")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();
    DcgmWatcher watcherA(DcgmWatcherTypeClient, 1001);
    DcgmWatcher watcherB(DcgmWatcherTypeHealthWatch, DCGM_CONNECTION_ID_NONE);

    GIVEN("a cache manager with no live NVML dependency")
    {
        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);

        WHEN("append samples is called with null and unwatched entity inputs")
        {
            DcgmFvBuffer ignoredBuffer;
            ignoredBuffer.AddInt64Value(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 72, 500, DCGM_ST_OK);

            THEN("null input is rejected and unwatched entity values are ignored")
            {
                CHECK(cacheManager.AppendSamples(nullptr) == DCGM_ST_BADPARAM);
                REQUIRE(cacheManager.AppendSamples(&ignoredBuffer) == DCGM_ST_OK);

                dcgmcm_sample_t sample {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }

        WHEN("append samples is called for existing entity and global watch objects")
        {
            dcgmcm_watch_info_p entityWatch
                = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
            dcgmcm_watch_info_p globalWatch = cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 1);
            REQUIRE(entityWatch != nullptr);
            REQUIRE(globalWatch != nullptr);
            entityWatch->isWatched  = 1;
            globalWatch->isWatched  = 1;
            entityWatch->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(60));
            globalWatch->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(60));

            DcgmFvBuffer source;
            REQUIRE(source.AddInt64Value(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 81, 600, DCGM_ST_OK)
                    != nullptr);
            REQUIRE(
                source.AddStringValue(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, "test-driver", 700, DCGM_ST_OK)
                != nullptr);

            REQUIRE(cacheManager.AppendSamples(&source) == DCGM_ST_OK);

            THEN("latest sample reads report no cached data for the appended entity and global values")
            {
                dcgmcm_sample_t tempSample {};
                REQUIRE(
                    cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &tempSample, nullptr)
                    == DCGM_ST_NO_DATA);

                dcgmcm_sample_t driverSample {};
                CHECK(
                    cacheManager.GetLatestSample(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &driverSample, nullptr)
                    == DCGM_ST_NO_DATA);
            }
        }

        WHEN("watchers are added, updated, removed, and cleaned up by connection")
        {
            dcgmcm_watch_info_p watchInfo
                = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
            REQUIRE(watchInfo != nullptr);

            dcgm_watch_watcher_info_t first {};
            first.watcher             = watcherA;
            first.monitorIntervalUsec = 1000;
            first.maxAgeUsec          = 10000;
            first.isSubscribed        = 0;

            dcgm_watch_watcher_info_t second {};
            second.watcher             = watcherB;
            second.monitorIntervalUsec = 500;
            second.maxAgeUsec          = 20000;
            second.isSubscribed        = 1;

            bool wasAdded = false;
            REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &first) == DCGM_ST_OK);
            REQUIRE(wasAdded);
            watchInfo->isWatched = 1;
            REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &second) == DCGM_ST_OK);
            REQUIRE(wasAdded);

            THEN("aggregate watch settings reflect the active watcher set")
            {
                CHECK(watchInfo->watchers.size() == 2);
                CHECK(watchInfo->monitorIntervalUsec == 500);
                CHECK(watchInfo->maxAgeUsec == 20000);
                CHECK(watchInfo->hasSubscribedWatchers == 1);

                first.monitorIntervalUsec = 250;
                first.maxAgeUsec          = 30000;
                first.isSubscribed        = 1;
                REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &first) == DCGM_ST_OK);
                CHECK_FALSE(wasAdded);
                CHECK(watchInfo->watchers.size() == 2);
                CHECK(watchInfo->monitorIntervalUsec == 250);
                CHECK(watchInfo->maxAgeUsec == 30000);

                REQUIRE(cacheManager.RemoveWatcher(watchInfo, &second) == DCGM_ST_OK);
                REQUIRE(watchInfo->watchers.size() == 1);
                CHECK(watchInfo->isWatched == 1);
                CHECK(watchInfo->monitorIntervalUsec == 250);

                REQUIRE(cacheManager.RemoveWatcher(watchInfo, &first) == DCGM_ST_OK);
                CHECK(watchInfo->watchers.empty());
                CHECK(watchInfo->isWatched == 0);
                CHECK(watchInfo->hasSubscribedWatchers == 0);
                CHECK(cacheManager.RemoveWatcher(watchInfo, &first) == DCGM_ST_NOT_WATCHED);
            }
        }

        WHEN("connection-owned watches and meta-group watches are registered")
        {
            bool wereFirstWatcher = false;
            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
            REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                               gpuId,
                                               DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                               1000,
                                               60.0,
                                               5,
                                               watcherA,
                                               false,
                                               false,
                                               wereFirstWatcher)
                    == DCGM_ST_OK);
            REQUIRE(wereFirstWatcher);

            cacheManager.AddMetaGroupWatchedField(7, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 60.0, 5, watcherA);
            CHECK(cacheManager.m_metaGroupWatchedFields[7].empty());

            THEN("explicit unwatch calls clear matching client watches and meta watches")
            {
                cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
                CHECK(cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcherA)
                      == DCGM_ST_OK);
                cacheManager.RemoveMetaGroupWatchedField(7, DCGM_FI_DEV_GPU_TEMP_CELSIUS, watcherA);

                dcgmcm_watch_info_t snapshot {};
                CHECK(
                    cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
                    == DCGM_ST_OK);
                CHECK(snapshot.watchers.empty());
                CHECK(snapshot.isWatched == 0);
                CHECK(cacheManager.m_metaGroupWatchedFields[7].empty());
            }
        }
    }
}

SCENARIO("CacheManager: cached GPU, NvLink, and MIG hierarchy helpers use fake state")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpu0              = cacheManager.AddFakeGpu(0x20B0, 0x145F);
    unsigned int const gpu1              = cacheManager.AddFakeGpu(0x2330, 0x1626);
    unsigned int const gpuInstanceId     = cacheManager.AddFakeInstance(gpu0);
    unsigned int const computeInstanceId = cacheManager.AddFakeComputeInstance(gpuInstanceId);

    GIVEN("two fake GPUs with deterministic cached properties")
    {
        SafeCopyTo(cacheManager.m_gpus[gpu0].uuid, "GPU-cache-0");
        SafeCopyTo(cacheManager.m_gpus[gpu0].serial, "serial-0");
        SafeCopyTo(cacheManager.m_gpus[gpu0].deviceName, "fake-zero");
        cacheManager.m_gpus[gpu0].brand              = DCGM_GPU_BRAND_GRID;
        cacheManager.m_gpus[gpu0].arch               = DCGM_CHIP_ARCH_HOPPER;
        cacheManager.m_gpus[gpu0].virtualizationMode = DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU;
        cacheManager.m_gpus[gpu1].status             = DcgmEntityStatusDetached;

        WHEN("GPU info and identity helpers are queried")
        {
            std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
            REQUIRE(cacheManager.GetAllGpuInfo(gpuInfo) == DCGM_ST_OK);

            THEN("cached GPU properties and lookup boundaries are reported")
            {
                REQUIRE(gpuInfo.size() == 2);
                CHECK(gpuInfo[0].gpuId == gpu0);
                CHECK(gpuInfo[0].status == DcgmEntityStatusFake);
                CHECK(gpuInfo[0].brand == DCGM_GPU_BRAND_GRID);
                CHECK(gpuInfo[0].nvmlIndex == gpu0);
                CHECK(gpuInfo[0].arch == DCGM_CHIP_ARCH_HOPPER);
                CHECK(gpuInfo[0].virtualizationMode == DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU);
                CHECK(std::strcmp(gpuInfo[0].uuid, "GPU-cache-0") == 0);
                CHECK(std::strcmp(gpuInfo[0].serial, "serial-0") == 0);
                CHECK(std::strcmp(gpuInfo[0].deviceName, "fake-zero") == 0);
                CHECK(gpuInfo[1].status == DcgmEntityStatusDetached);

                CHECK(cacheManager.AreAnyGpusInHostVGPUMode());
                CHECK(cacheManager.GetGpuBrand(gpu0) == DCGM_GPU_BRAND_GRID);
                CHECK(cacheManager.GetGpuBrand(DCGM_MAX_NUM_DEVICES) == DCGM_GPU_BRAND_UNKNOWN);

                dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_UNKNOWN;
                REQUIRE(cacheManager.GetGpuArch(gpu0, arch) == DCGM_ST_OK);
                CHECK(arch == DCGM_CHIP_ARCH_HOPPER);
                CHECK(cacheManager.GetGpuArch(DCGM_MAX_NUM_DEVICES, arch) == DCGM_ST_BADPARAM);

                REQUIRE(cacheManager.GpuIdToNvmlIndex(gpu0).has_value());
                CHECK(cacheManager.GpuIdToNvmlIndex(gpu0).value() == static_cast<int>(gpu0));
                REQUIRE(cacheManager.NvmlIndexToGpuId(gpu0).has_value());
                CHECK(cacheManager.NvmlIndexToGpuId(gpu0).value() == gpu0);
                CHECK_FALSE(cacheManager.GpuIdToNvmlIndex(DCGM_MAX_NUM_DEVICES).has_value());
                CHECK_FALSE(cacheManager.NvmlIndexToGpuId(DCGM_MAX_NUM_DEVICES).has_value());
            }
        }

        WHEN("entity validation and NvLink state helpers are queried")
        {
            dcgm_link_t gpuLink {};
            gpuLink.parsed.type  = DCGM_FE_GPU;
            gpuLink.parsed.gpuId = gpu0;
            gpuLink.parsed.index = 1;

            dcgm_link_t badGpuLink {};
            badGpuLink.parsed.type  = DCGM_FE_GPU;
            badGpuLink.parsed.gpuId = DCGM_MAX_NUM_DEVICES;
            badGpuLink.parsed.index = 1;

            THEN("fake GPU, MIG, CPU, and link entity boundaries are deterministic")
            {
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_GPU, gpu0));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_GPU_I, gpuInstanceId));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_GPU_CI, computeInstanceId));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_CPU, 99));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_NONE, 0));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, gpuLink.raw));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, badGpuLink.raw));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_SWITCH, 0));

                CHECK(cacheManager.SetGpuNvLinkLinkState(gpu0, 0, DcgmNvLinkLinkStateUp) == DCGM_ST_OK);
                CHECK(cacheManager.SetGpuNvLinkLinkState(gpu0, DCGM_NVLINK_MAX_LINKS_PER_GPU, DcgmNvLinkLinkStateUp)
                      == DCGM_ST_BADPARAM);
                CHECK(cacheManager.SetEntityNvLinkLinkState(DCGM_FE_GPU, gpu0, 1, DcgmNvLinkLinkStateDisabled)
                      == DCGM_ST_OK);
                CHECK(cacheManager.SetEntityNvLinkLinkState(DCGM_FE_SWITCH, 0, 0, DcgmNvLinkLinkStateUp)
                      == DCGM_ST_NOT_SUPPORTED);

                dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_GPU] {};
                REQUIRE(cacheManager.GetEntityNvLinkLinkStatus(DCGM_FE_GPU, gpu0, linkStates) == DCGM_ST_OK);
                CHECK(linkStates[0] == DcgmNvLinkLinkStateUp);
                CHECK(linkStates[1] == DcgmNvLinkLinkStateDisabled);
            }
        }

        WHEN("top-level NvLink and MIG hierarchy structures are populated")
        {
            cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
            dcgmMigHierarchy_v2 migHierarchy {};
            CHECK(cacheManager.PopulateMigHierarchy(migHierarchy) == DCGM_ST_NVML_NOT_LOADED);
            CHECK(migHierarchy.count == 0);

            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
            REQUIRE(cacheManager.PopulateMigHierarchy(migHierarchy) == DCGM_ST_OK);

            dcgmNvLinkStatus_v4 nvLinkStatus {};
            REQUIRE(cacheManager.PopulateNvLinkLinkStatus(nvLinkStatus) == DCGM_ST_OK);

            THEN("detached GPUs are skipped and fake MIG parentage is preserved")
            {
                REQUIRE(migHierarchy.count == 2);
                CHECK(migHierarchy.entityList[0].entity.entityGroupId == DCGM_FE_GPU_I);
                CHECK(migHierarchy.entityList[0].entity.entityId == gpuInstanceId);
                CHECK(migHierarchy.entityList[0].parent.entityGroupId == DCGM_FE_GPU);
                CHECK(migHierarchy.entityList[0].parent.entityId == gpu0);
                CHECK(std::strcmp(migHierarchy.entityList[0].info.gpuUuid, "GPU-cache-0") == 0);
                CHECK(migHierarchy.entityList[1].entity.entityGroupId == DCGM_FE_GPU_CI);
                CHECK(migHierarchy.entityList[1].entity.entityId == computeInstanceId);
                CHECK(migHierarchy.entityList[1].parent.entityGroupId == DCGM_FE_GPU_I);
                CHECK(migHierarchy.entityList[1].parent.entityId == gpuInstanceId);

                CHECK(nvLinkStatus.version == dcgmNvLinkStatus_version4);
                REQUIRE(nvLinkStatus.numGpus == 1);
                CHECK(nvLinkStatus.gpus[0].entityId == gpu0);
            }
        }
    }
}


SCENARIO("CacheManager: entity lists and metagroup watches handle active filters and replacements")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const activeGpu               = cacheManager.AddFakeGpu(0x20B0, 0x145F);
    unsigned int const detachedGpu             = cacheManager.AddFakeGpu(0x2330, 0x1626);
    unsigned int const activeInstance          = cacheManager.AddFakeInstance(activeGpu);
    unsigned int const detachedInstance        = cacheManager.AddFakeInstance(detachedGpu);
    unsigned int const activeComputeInstance   = cacheManager.AddFakeComputeInstance(activeInstance);
    unsigned int const detachedComputeInstance = cacheManager.AddFakeComputeInstance(detachedInstance);
    cacheManager.m_gpus[detachedGpu].status    = DcgmEntityStatusDetached;

    GIVEN("one active fake GPU and one detached fake GPU with MIG children")
    {
        WHEN("entity lists are queried with activeOnly set")
        {
            std::vector<dcgmGroupEntityPair_t> gpus;
            std::vector<dcgmGroupEntityPair_t> instances;
            std::vector<dcgmGroupEntityPair_t> computeInstances;

            auto gpuRet      = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, gpus);
            auto instanceRet = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_I, instances);
            auto computeRet  = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_CI, computeInstances);

            THEN("detached GPU hierarchy entries are filtered out")
            {
                REQUIRE(gpuRet == DCGM_ST_OK);
                REQUIRE(instanceRet == DCGM_ST_OK);
                REQUIRE(computeRet == DCGM_ST_OK);
                REQUIRE(gpus.size() == 1);
                CHECK(gpus[0].entityId == activeGpu);
                REQUIRE(instances.size() == 1);
                CHECK(instances[0].entityId == activeInstance);
                REQUIRE(computeInstances.size() == 1);
                CHECK(computeInstances[0].entityId == activeComputeInstance);
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, detachedGpu) == DcgmEntityStatusDetached);
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_I, detachedInstance) == DcgmEntityStatusDetached);
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_CI, detachedComputeInstance)
                      == DcgmEntityStatusDetached);
            }
        }

        WHEN("GPU entity lists are queried without active filtering")
        {
            std::vector<dcgmGroupEntityPair_t> gpus;
            auto ret = cacheManager.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, gpus);

            THEN("both active and detached GPUs are reported in cached order")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(gpus.size() == 2);
                CHECK(gpus[0].entityId == activeGpu);
                CHECK(gpus[1].entityId == detachedGpu);
            }
        }
    }

    GIVEN("all-GPU metagroup watch state for two client watchers")
    {
        DcgmWatcher clientA(DcgmWatcherTypeClient, 11);
        DcgmWatcher clientB(DcgmWatcherTypeClient, 22);
        cacheManager.AddMetaGroupWatchedField(
            DCGM_GROUP_ALL_GPUS, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 30.0, 4, clientA);
        cacheManager.AddMetaGroupWatchedField(
            DCGM_GROUP_ALL_GPUS, DCGM_FI_DEV_BOARD_POWER_WATTS, 2000, 40.0, 5, clientB);

        WHEN("an existing watcher adds the same field with new parameters")
        {
            cacheManager.AddMetaGroupWatchedField(
                DCGM_GROUP_ALL_GPUS, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 3000, 90.0, 9, clientA);

            THEN("the entry is replaced rather than duplicated")
            {
                auto const &watchedFields = cacheManager.m_metaGroupWatchedFields[DCGM_GROUP_ALL_GPUS];
                REQUIRE(watchedFields.size() == 2);
                auto tempIt = std::find_if(watchedFields.begin(), watchedFields.end(), [clientA](auto const &field) {
                    return field.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS && field.watcher == clientA;
                });
                REQUIRE(tempIt != watchedFields.end());
                CHECK(tempIt->monitorIntervalUsec == 3000);
                CHECK(tempIt->maxSampleAge == 90.0);
                CHECK(tempIt->maxKeepSamples == 9);
            }
        }

        WHEN("one metagroup watcher is removed and unsupported metagroups are modified")
        {
            cacheManager.RemoveMetaGroupWatchedField(DCGM_GROUP_ALL_GPUS, DCGM_FI_DEV_GPU_TEMP_CELSIUS, clientA);
            cacheManager.AddMetaGroupWatchedField(
                DCGM_GROUP_ALL_INSTANCES, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1, 1.0, 1, clientA);
            cacheManager.RemoveMetaGroupWatchedField(DCGM_GROUP_ALL_INSTANCES, DCGM_FI_DEV_GPU_TEMP_CELSIUS, clientA);

            THEN("only the matching all-GPU watch is removed")
            {
                auto const &allGpuFields = cacheManager.m_metaGroupWatchedFields[DCGM_GROUP_ALL_GPUS];
                REQUIRE(allGpuFields.size() == 1);
                CHECK(allGpuFields.front().fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(allGpuFields.front().watcher == clientB);
                CHECK(cacheManager.m_metaGroupWatchedFields[DCGM_GROUP_ALL_INSTANCES].empty());
            }
        }
    }
}


SCENARIO("CacheManager: clear helpers reset matching fake entity watches")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpu0 = cacheManager.AddFakeGpu();
    unsigned int const gpu1 = cacheManager.AddFakeGpu();
    cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

    DcgmWatcher watcher0(DcgmWatcherTypeClient, 610);
    DcgmWatcher watcher1(DcgmWatcherTypeClient, 611);
    bool wereFirstWatcher = false;

    REQUIRE(
        cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpu0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 30.0, 2, watcher0, false, false, wereFirstWatcher)
        == DCGM_ST_OK);
    REQUIRE(wereFirstWatcher);

    wereFirstWatcher = false;
    REQUIRE(
        cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpu1, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 2000, 40.0, 3, watcher1, false, false, wereFirstWatcher)
        == DCGM_ST_OK);
    REQUIRE(wereFirstWatcher);

    GIVEN("two watched fake GPU entities")
    {
        WHEN("one entity is cleared")
        {
            auto const ret = cacheManager.ClearEntity(DCGM_FE_GPU, gpu0, 0);

            THEN("only the matching entity watch state is reset")
            {
                CHECK(ret == DCGM_ST_OK);

                dcgmcm_watch_info_t gpu0Snapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu0Snapshot)
                        == DCGM_ST_OK);
                CHECK(gpu0Snapshot.isWatched == 0);
                CHECK(gpu0Snapshot.watchers.empty());
                CHECK(gpu0Snapshot.lastQueriedUsec == 0);
                CHECK(gpu0Snapshot.monitorIntervalUsec == 0);

                dcgmcm_watch_info_t gpu1Snapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu1, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu1Snapshot)
                        == DCGM_ST_OK);
                CHECK(gpu1Snapshot.isWatched == 1);
                REQUIRE(gpu1Snapshot.watchers.size() == 1);
                CHECK(gpu1Snapshot.watchers[0].watcher == watcher1);
                CHECK(gpu1Snapshot.monitorIntervalUsec == 2000);
            }
        }

        WHEN("a missing entity is cleared")
        {
            auto const ret = cacheManager.ClearEntity(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES, 1);

            THEN("existing watched entities are left intact")
            {
                CHECK(ret == DCGM_ST_OK);

                dcgmcm_watch_info_t gpu0Snapshot {};
                dcgmcm_watch_info_t gpu1Snapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu0Snapshot)
                        == DCGM_ST_OK);
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu1, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu1Snapshot)
                        == DCGM_ST_OK);
                CHECK(gpu0Snapshot.isWatched == 1);
                CHECK(gpu1Snapshot.isWatched == 1);
                REQUIRE(gpu0Snapshot.watchers.size() == 1);
                REQUIRE(gpu1Snapshot.watchers.size() == 1);
                CHECK(gpu0Snapshot.watchers[0].watcher == watcher0);
                CHECK(gpu1Snapshot.watchers[0].watcher == watcher1);
            }
        }

        WHEN("all entity watches are cleared")
        {
            auto const ret = cacheManager.ClearAllEntities(1);

            THEN("each existing watch object remains queryable but inactive")
            {
                CHECK(ret == DCGM_ST_OK);

                dcgmcm_watch_info_t gpu0Snapshot {};
                dcgmcm_watch_info_t gpu1Snapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu0Snapshot)
                        == DCGM_ST_OK);
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, gpu1, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpu1Snapshot)
                        == DCGM_ST_OK);
                CHECK(gpu0Snapshot.isWatched == 0);
                CHECK(gpu1Snapshot.isWatched == 0);
                CHECK(gpu0Snapshot.watchers.empty());
                CHECK(gpu1Snapshot.watchers.empty());
                CHECK(gpu0Snapshot.lastStatus == NVML_SUCCESS);
                CHECK(gpu1Snapshot.lastStatus == NVML_SUCCESS);
            }
        }
    }
}

SCENARIO("CacheManager: fake entity creation rejects invalid parents without mutating active lists")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const activeGpu            = cacheManager.AddFakeGpu();
    unsigned int const detachedGpu          = cacheManager.AddFakeGpu();
    cacheManager.m_gpus[detachedGpu].status = DcgmEntityStatusDetached;

    GIVEN("one active fake GPU and one detached fake GPU")
    {
        WHEN("MIG children are created with valid and invalid parents")
        {
            auto const validInstance        = cacheManager.AddFakeInstance(activeGpu);
            auto const invalidInstance      = cacheManager.AddFakeInstance(DCGM_MAX_NUM_DEVICES);
            auto const validComputeInstance = cacheManager.AddFakeComputeInstance(validInstance);
            auto const invalidCompute       = cacheManager.AddFakeComputeInstance(DCGM_ENTITY_ID_BAD);

            THEN("invalid parents are rejected and active entity lists contain only valid descendants")
            {
                REQUIRE(validInstance != DCGM_ENTITY_ID_BAD);
                REQUIRE(validComputeInstance != DCGM_ENTITY_ID_BAD);
                CHECK(invalidInstance == DCGM_ENTITY_ID_BAD);
                CHECK(invalidCompute == DCGM_ENTITY_ID_BAD);

                std::vector<dcgmGroupEntityPair_t> gpus;
                std::vector<dcgmGroupEntityPair_t> instances;
                std::vector<dcgmGroupEntityPair_t> computeInstances;
                REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, gpus) == DCGM_ST_OK);
                REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_I, instances) == DCGM_ST_OK);
                REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_CI, computeInstances) == DCGM_ST_OK);

                REQUIRE(gpus.size() == 1);
                CHECK(gpus[0].entityId == activeGpu);
                REQUIRE(instances.size() == 1);
                CHECK(instances[0].entityId == validInstance);
                REQUIRE(computeInstances.size() == 1);
                CHECK(computeInstances[0].entityId == validComputeInstance);

                auto const instanceOwner = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_I, validInstance);
                auto const computeOwner  = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_CI, validComputeInstance);
                REQUIRE(instanceOwner.has_value());
                REQUIRE(computeOwner.has_value());
                CHECK(instanceOwner.value() == activeGpu);
                CHECK(computeOwner.value() == activeGpu);
                CHECK_FALSE(cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_I, invalidInstance).has_value());
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, detachedGpu) == DcgmEntityStatusDetached);
            }
        }
    }
}


SCENARIO("CacheManager: field info reports validation, watcher, and timeseries state")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with a fake GPU")
    {
        TestDcgmCacheManager cm;
        cm.m_nvmlLoaded.store(false, std::memory_order_release);
        unsigned int const gpuId = cm.AddFakeGpu();

        WHEN("field info input is invalid or unwatched")
        {
            dcgmCacheManagerFieldInfo_v4_t info {};
            info.version       = dcgmCacheManagerFieldInfo_version4;
            info.entityGroupId = DCGM_FE_GPU;
            info.entityId      = gpuId;
            info.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

            THEN("the helper returns explicit local validation statuses")
            {
                CHECK(cm.GetCacheManagerFieldInfo(nullptr) == DCGM_ST_BADPARAM);

                info.version = 0;
                CHECK(cm.GetCacheManagerFieldInfo(&info) == DCGM_ST_VER_MISMATCH);

                info.version = dcgmCacheManagerFieldInfo_version4;
                info.fieldId = DCGM_FI_MAX_FIELDS;
                CHECK(cm.GetCacheManagerFieldInfo(&info) == DCGM_ST_BADPARAM);

                info.fieldId = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
                CHECK(cm.GetCacheManagerFieldInfo(&info) == DCGM_ST_NOT_WATCHED);
            }
        }

        WHEN("a watched field has multiple watchers and injected samples")
        {
            auto *watchInfo = cm.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
            REQUIRE(watchInfo != nullptr);

            dcgm_watch_watcher_info_t clientWatcher {};
            clientWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 301);
            clientWatcher.monitorIntervalUsec = 1000;
            clientWatcher.maxAgeUsec          = 9000;
            clientWatcher.isSubscribed        = 1;

            dcgm_watch_watcher_info_t healthWatcher {};
            healthWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeHealthWatch, 0);
            healthWatcher.monitorIntervalUsec = 2000;
            healthWatcher.maxAgeUsec          = 12000;
            healthWatcher.isSubscribed        = 0;

            bool wasAdded = false;
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &clientWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &healthWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);
            watchInfo->isWatched       = 1;
            watchInfo->lastStatus      = NVML_ERROR_TIMEOUT;
            watchInfo->fetchCount      = 7;
            watchInfo->execTimeUsec    = 55;
            watchInfo->lastQueriedUsec = 88;

            timelib64_t const firstTimestamp  = timelib_usecSince1970() + 10000;
            timelib64_t const secondTimestamp = firstTimestamp + 50;
            dcgmcm_sample_t samples[2] {};
            samples[0].timestamp = firstTimestamp;
            samples[0].val.i64   = 41;
            samples[1].timestamp = secondTimestamp;
            samples[1].val.i64   = 42;
            REQUIRE(cm.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples, 2) == DCGM_ST_OK);

            THEN("field info exposes flags, aggregate watch parameters, watcher entries, and timestamps")
            {
                dcgmCacheManagerFieldInfo_v4_t info {};
                info.version       = dcgmCacheManagerFieldInfo_version4;
                info.entityGroupId = DCGM_FE_GPU;
                info.entityId      = gpuId;
                info.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

                REQUIRE(cm.GetCacheManagerFieldInfo(&info) == DCGM_ST_OK);
                CHECK(info.lastStatus == NVML_ERROR_TIMEOUT);
                CHECK(info.monitorIntervalUsec == 1000);
                CHECK(info.maxAgeUsec == 12000);
                CHECK(info.fetchCount == 7);
                CHECK(info.execTimeUsec == 55);
                REQUIRE(info.numWatchers == 2);
                CHECK(info.watchers[0].watcherType == DcgmWatcherTypeClient);
                CHECK(info.watchers[0].connectionId == 301);
                CHECK(info.watchers[0].monitorIntervalUsec == 1000);
                CHECK(info.watchers[1].watcherType == DcgmWatcherTypeHealthWatch);
                CHECK(info.watchers[1].maxAgeUsec == 12000);
                CHECK(info.numSamples == 2);
                CHECK(info.oldestTimestamp == firstTimestamp);
                CHECK(info.newestTimestamp == secondTimestamp);
            }
        }
    }

    GIVEN("a global field watch")
    {
        TestDcgmCacheManager cm;
        auto *watchInfo = cm.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched           = 1;
        watchInfo->monitorIntervalUsec = 3333;
        watchInfo->maxAgeUsec          = 4444;
        watchInfo->lastStatus          = NVML_SUCCESS;

        WHEN("field info is requested without an entity scope")
        {
            dcgmCacheManagerFieldInfo_v4_t info {};
            info.version = dcgmCacheManagerFieldInfo_version4;
            info.fieldId = DCGM_FI_SYSTEM_DRIVER_VERSION;

            THEN("global watch metadata is returned without cached samples")
            {
                REQUIRE(cm.GetCacheManagerFieldInfo(&info) == DCGM_ST_OK);
                CHECK(info.monitorIntervalUsec == 3333);
                CHECK(info.maxAgeUsec == 4444);
                CHECK(info.numSamples == 0);
                CHECK(info.oldestTimestamp == 0);
                CHECK(info.newestTimestamp == 0);
            }
        }
    }
}


SCENARIO("CacheManager: time-series allocation, quota, and SetValue local errors stay hardware-free")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a watched int64 field with explicit time-series storage")
    {
        dcgmcm_watch_info_p watchInfo
            = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched = 1;

        WHEN("the time-series is allocated twice and old samples are pruned by timestamp")
        {
            auto const allocRet      = cacheManager.AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_INT64);
            auto const allocAgainRet = cacheManager.AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_INT64);

            dcgmcm_update_thread_t threadCtx;
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
            REQUIRE(cacheManager.AppendEntityInt64(threadCtx, 10, 0, 100, 0) == DCGM_ST_OK);
            REQUIRE(cacheManager.AppendEntityInt64(threadCtx, 20, 0, 200, 0) == DCGM_ST_OK);
            REQUIRE(cacheManager.AppendEntityInt64(threadCtx, 30, 0, 300, 0) == DCGM_ST_OK);
            auto const quotaRet = cacheManager.EnforceWatchInfoQuota(watchInfo, 300, 250);

            THEN("allocation is idempotent and only samples at or after the quota cutoff remain")
            {
                CHECK(allocRet == DCGM_ST_OK);
                CHECK(allocAgainRet == DCGM_ST_OK);
                CHECK(quotaRet == DCGM_ST_OK);
                CHECK(cacheManager.PrecheckWatchInfoForSamples(watchInfo) == DCGM_ST_OK);

                dcgmcm_sample_t returned[3] {};
                int count = 3;
                REQUIRE(cacheManager.GetSamples(DCGM_FE_GPU,
                                                gpuId,
                                                DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                returned,
                                                &count,
                                                0,
                                                0,
                                                DCGM_ORDER_ASCENDING,
                                                nullptr)
                        == DCGM_ST_OK);
                REQUIRE(count == 1);
                CHECK(returned[0].timestamp == 300);
                CHECK(returned[0].val.i64 == 30);
            }
        }

        WHEN("cached data exists on an inactive watch with a stale NVML status")
        {
            dcgmcm_update_thread_t threadCtx;
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
            REQUIRE(cacheManager.AppendEntityInt64(threadCtx, 44, 0, 400, 0) == DCGM_ST_OK);
            watchInfo->isWatched  = 0;
            watchInfo->lastStatus = NVML_ERROR_TIMEOUT;

            THEN("precheck allows the cached value to be read")
            {
                CHECK(cacheManager.PrecheckWatchInfoForSamples(watchInfo) == DCGM_ST_OK);

                dcgmcm_sample_t latest {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                        == DCGM_ST_OK);
                CHECK(latest.val.i64 == 44);
            }
        }
    }

    GIVEN("SetValue inputs that fail before reaching NVML")
    {
        dcgmcm_sample_t value {};
        value.timestamp = 123;
        value.val.i64   = 1;

        WHEN("local validation handles nulls, unknown fields, unloaded NVML, bad GPUs, and lost GPUs")
        {
            auto const nullRet         = cacheManager.SetValueImpl(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr);
            auto const unknownFieldRet = cacheManager.SetValueImpl(gpuId, DCGM_FI_MAX_FIELDS, &value);
            cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
            auto const nvmlNotLoadedRet = cacheManager.SetValueImpl(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &value);
            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
            auto const badGpuRet
                = cacheManager.SetValueImpl(DCGM_MAX_NUM_DEVICES, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &value);
            cacheManager.m_gpus[gpuId].status = DcgmEntityStatusDetached;
            auto const lostGpuRet             = cacheManager.SetValueImpl(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &value);

            THEN("precise status codes are returned without issuing driver calls")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(unknownFieldRet == DCGM_ST_UNKNOWN_FIELD);
                CHECK(nvmlNotLoadedRet == DCGM_ST_NVML_NOT_LOADED);
                CHECK(badGpuRet == DCGM_ST_BADPARAM);
                CHECK(lostGpuRet == DCGM_ST_GPU_IS_LOST);
            }
        }
    }

    GIVEN("null quota and allocation inputs")
    {
        WHEN("helper preconditions are exercised")
        {
            THEN("no-op quota paths succeed and null allocation reports not watched")
            {
                CHECK(cacheManager.EnforceWatchInfoQuota(nullptr, 0, 0) == DCGM_ST_OK);
                CHECK(cacheManager.AllocWatchInfoTimeSeries(nullptr, TS_TYPE_INT64) == DCGM_ST_NOT_WATCHED);
            }
        }
    }
}


SCENARIO("CacheManager: XID recording and live-sample helpers use cached fake entities")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a fake GPU with a watched XID field")
    {
        dcgmcm_update_thread_t threadCtx {};
        timelib64_t const now = 5000;

        WHEN("a GPU-level XID is recorded")
        {
            cacheManager.RecordXidForGpu(gpuId, threadCtx, 79, NVML_SUCCESS, now);

            THEN("the XID is available through the public sample API")
            {
                dcgmcm_sample_t latest {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_XID_ERROR, &latest, nullptr)
                        == DCGM_ST_OK);
                CHECK(latest.timestamp == now);
                CHECK(latest.val.i64 == 79);
                CHECK(threadCtx.entityKey.entityGroupId == DCGM_FE_GPU);
                CHECK(threadCtx.entityKey.entityId == gpuId);
                CHECK(threadCtx.entityKey.fieldId == DCGM_FI_DEV_XID_ERROR);
            }
        }

        WHEN("the XID is requested through the multi-entity latest-sample helper")
        {
            cacheManager.RecordXidForGpu(gpuId, threadCtx, 31, NVML_SUCCESS, now);
            std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId }, { DCGM_FE_GPU, gpuId + 1 } };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_XID_ERROR };
            DcgmFvBuffer fvBuffer;

            auto const ret = cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);

            THEN("cached entities are returned and invalid entities are represented as errors")
            {
                CHECK(ret == DCGM_ST_OK);
                size_t bufferSize   = 0;
                size_t elementCount = 0;
                REQUIRE(fvBuffer.GetSize(&bufferSize, &elementCount) == DCGM_ST_OK);
                CHECK(bufferSize > 0);
                CHECK(elementCount == 2);

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_DEV_XID_ERROR);
                CHECK(fv->status == DCGM_ST_OK);
                CHECK(fv->value.i64 == 31);

                fv = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId + 1);
                CHECK(fv->fieldId == DCGM_FI_DEV_XID_ERROR);
                CHECK(fv->status != DCGM_ST_OK);
            }
        }
    }

    GIVEN("event data that references missing MIG entities")
    {
        nvmlEventData_t eventData {};
        eventData.gpuInstanceId     = 111;
        eventData.computeInstanceId = 222;
        eventData.eventData         = 45;

        WHEN("instance and compute-instance XIDs are recorded for unknown MIG children")
        {
            dcgmcm_update_thread_t instanceCtx {};
            dcgmcm_update_thread_t computeCtx {};
            cacheManager.RecordXidForGpuInstance(gpuId, instanceCtx, eventData, NVML_SUCCESS, 6000);
            cacheManager.RecordXidForComputeInstance(gpuId, computeCtx, eventData, NVML_SUCCESS, 7000);

            THEN("no synthetic child XID samples are created")
            {
                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU_I, 111, DCGM_FI_DEV_XID_ERROR, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU_CI, 222, DCGM_FI_DEV_XID_ERROR, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }
    }
}

SCENARIO("CacheManager: driver and watch lifecycle helpers expose deterministic local state")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("driver version strings and skip-driver flags")
    {
        WHEN("driver helpers are evaluated without attaching NVML")
        {
            cacheManager.m_driverVersion = "53510405";
            auto const newerThanR535     = cacheManager.DriverVersionIsAtLeast("535.104.05");
            auto const newerThanR550     = cacheManager.DriverVersionIsAtLeast("550.0.0");
            auto const skipInitially     = cacheManager.ShouldSkipDriverCalls();
            cacheManager.m_skipDriverCalls.store(true, std::memory_order_release);
            auto const skipWhenFlagged = cacheManager.ShouldSkipDriverCalls();
            cacheManager.m_skipDriverCalls.store(false, std::memory_order_release);
            auto const skipWhenCleared = cacheManager.ShouldSkipDriverCalls();

            THEN("version comparisons and skip decisions are based only on cached state")
            {
                CHECK(newerThanR535);
                CHECK_FALSE(newerThanR550);
                CHECK_FALSE(skipInitially);
                CHECK(skipWhenFlagged);
                CHECK_FALSE(skipWhenCleared);
            }
        }
    }

    GIVEN("a client watcher and a connection id")
    {
        dcgmcm_watch_info_p watchInfo
            = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        dcgm_watch_watcher_info_t watcher {};
        watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 123);
        watcher.monitorIntervalUsec = 1000;
        watcher.maxAgeUsec          = 2000000;
        bool wasAdded               = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);
        REQUIRE(wasAdded);
        REQUIRE(watchInfo->watchers.size() == 1);

        WHEN("the connection removal path is applied directly")
        {
            cacheManager.OnConnectionRemoveImpl(123);

            THEN("the matching watcher is removed without touching unrelated state")
            {
                CHECK(watchInfo->watchers.empty());
                CHECK_FALSE(watchInfo->isWatched);
            }
        }
    }

    GIVEN("a mixture of GPU, instance, compute-instance, and unsupported entities")
    {
        unsigned int const instanceId        = cacheManager.AddFakeInstance(gpuId);
        unsigned int const computeInstanceId = cacheManager.AddFakeComputeInstance(instanceId);
        std::vector<dcgmGroupEntityPair_t> entities {
            { DCGM_FE_GPU, gpuId },        { DCGM_FE_GPU, gpuId + 99 },
            { DCGM_FE_GPU_I, instanceId }, { DCGM_FE_GPU_CI, computeInstanceId },
            { DCGM_FE_SWITCH, 0 },
        };

        WHEN("the active-entity filter is applied")
        {
            auto const activeEntities = cacheManager.FilterActiveEntities(entities);

            THEN("known fake GPU and MIG entities remain while the invalid GPU is removed")
            {
                REQUIRE(activeEntities.size() == 4);
                CHECK(activeEntities[0].entityGroupId == DCGM_FE_GPU);
                CHECK(activeEntities[0].entityId == gpuId);
                CHECK(activeEntities[1].entityGroupId == DCGM_FE_GPU_I);
                CHECK(activeEntities[1].entityId == instanceId);
                CHECK(activeEntities[2].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(activeEntities[2].entityId == computeInstanceId);
                CHECK(activeEntities[3].entityGroupId == DCGM_FE_SWITCH);
                CHECK(activeEntities[3].entityId == 0);
            }
        }
    }
}


SCENARIO("CacheManager: watcher updates aggregate intervals subscriptions and removals")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("an entity watch with two watcher owners")
    {
        dcgmcm_watch_info_p watchInfo
            = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);

        dcgm_watch_watcher_info_t slowWatcher {};
        slowWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 501);
        slowWatcher.monitorIntervalUsec = 10000;
        slowWatcher.maxAgeUsec          = 90000;
        slowWatcher.isSubscribed        = 0;

        dcgm_watch_watcher_info_t fastSubscriber {};
        fastSubscriber.watcher             = DcgmWatcher(DcgmWatcherTypeHealthWatch, 0);
        fastSubscriber.monitorIntervalUsec = 1000;
        fastSubscriber.maxAgeUsec          = 5000;
        fastSubscriber.isSubscribed        = 1;

        bool wasAdded = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
        REQUIRE(wasAdded);
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &fastSubscriber) == DCGM_ST_OK);
        REQUIRE(wasAdded);

        WHEN("watcher-derived state is recomputed")
        {
            auto const updateRet = cacheManager.UpdateWatchFromWatchers(watchInfo);

            THEN("the watch uses the most aggressive interval and records live subscribers")
            {
                CHECK(updateRet == DCGM_ST_OK);
                REQUIRE(watchInfo->watchers.size() == 2);
                CHECK(watchInfo->monitorIntervalUsec == 1000);
                CHECK(watchInfo->maxAgeUsec == 90000);
                CHECK(watchInfo->hasSubscribedWatchers == 1);

                DcgmFvBuffer fvBuffer;
                dcgmcm_update_thread_t threadCtx {};
                threadCtx.watchInfo               = watchInfo;
                threadCtx.fvBuffer                = &fvBuffer;
                threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
                threadCtx.entityKey.entityId      = gpuId;
                threadCtx.entityKey.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
                REQUIRE(cacheManager.AppendEntityInt64(threadCtx, 75, 0, 1000, 0) == DCGM_ST_OK);
                CHECK((threadCtx.affectedSubscribers & (1 << DcgmWatcherTypeHealthWatch)) != 0);
            }
        }

        WHEN("an existing watcher is updated and then both watchers are removed")
        {
            slowWatcher.monitorIntervalUsec = 250;
            slowWatcher.maxAgeUsec          = 120000;
            REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);

            dcgm_watch_watcher_info_t missingWatcher {};
            missingWatcher.watcher = DcgmWatcher(DcgmWatcherTypeClient, 999);

            auto const missingRet    = cacheManager.RemoveWatcher(watchInfo, &missingWatcher);
            auto const removeSlowRet = cacheManager.RemoveWatcher(watchInfo, &slowWatcher);
            auto const removeFastRet = cacheManager.RemoveWatcher(watchInfo, &fastSubscriber);
            auto const emptyRet      = cacheManager.UpdateWatchFromWatchers(watchInfo);

            THEN("updates do not duplicate owners and the inactive watch reports not watched")
            {
                CHECK_FALSE(wasAdded);
                CHECK(missingRet == DCGM_ST_NOT_WATCHED);
                CHECK(removeSlowRet == DCGM_ST_OK);
                CHECK(removeFastRet == DCGM_ST_OK);
                CHECK(emptyRet == DCGM_ST_NOT_WATCHED);
                CHECK(watchInfo->watchers.empty());
                CHECK(watchInfo->isWatched == 0);
                CHECK(watchInfo->hasSubscribedWatchers == 0);
            }
        }
    }
}

SCENARIO("CacheManager: sample retrieval honors caller buffers and global field normalization")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId         = cacheManager.AddFakeGpu();
    timelib64_t const firstTimestamp = timelib_usecSince1970() + 100000;

    GIVEN("three cached integer samples")
    {
        dcgmcm_sample_t samples[3] {};
        for (int i = 0; i < 3; i++)
        {
            samples[i].timestamp = firstTimestamp + i;
            samples[i].val.i64   = 10 + i;
        }
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, samples, 3)
                == DCGM_ST_OK);

        WHEN("the caller asks for fewer samples than are cached")
        {
            dcgmcm_sample_t returned[2] {};
            int count      = 2;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     returned,
                                                     &count,
                                                     0,
                                                     0,
                                                     DCGM_ORDER_ASCENDING,
                                                     nullptr);

            THEN("only the requested count is copied in timestamp order")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == 2);
                CHECK(returned[0].timestamp == firstTimestamp);
                CHECK(returned[0].val.i64 == 10);
                CHECK(returned[1].timestamp == firstTimestamp + 1);
                CHECK(returned[1].val.i64 == 11);
            }
        }

        WHEN("samples are requested through a field-value buffer only")
        {
            DcgmFvBuffer fvBuffer;
            int count      = 3;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     nullptr,
                                                     &count,
                                                     firstTimestamp,
                                                     firstTimestamp + 2,
                                                     DCGM_ORDER_DESCENDING,
                                                     &fvBuffer);

            THEN("buffered values preserve descending order and sample count")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == 3);

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->value.i64 == 12);
                CHECK(fv->timestamp == firstTimestamp + 2);
                fv = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->value.i64 == 11);
                fv = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->value.i64 == 10);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }

    GIVEN("a global driver-version sample injected with a GPU entity")
    {
        char driverVersion[] = "535.104.05";
        dcgmcm_sample_t sample {};
        sample.timestamp = firstTimestamp;
        sample.val.str   = driverVersion;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, &sample, 1)
                == DCGM_ST_OK);

        WHEN("the same global field is queried with GPU and global entity scopes")
        {
            dcgmcm_sample_t gpuScoped {};
            dcgmcm_sample_t globalScoped {};
            auto const gpuRet
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, &gpuScoped, nullptr);
            auto const globalRet
                = cacheManager.GetLatestSample(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &globalScoped, nullptr);

            THEN("both queries resolve to the global watch object")
            {
                REQUIRE(gpuRet == DCGM_ST_OK);
                REQUIRE(globalRet == DCGM_ST_OK);
                REQUIRE(gpuScoped.val.str != nullptr);
                REQUIRE(globalScoped.val.str != nullptr);
                CHECK(std::strcmp(gpuScoped.val.str, driverVersion) == 0);
                CHECK(std::strcmp(globalScoped.val.str, driverVersion) == 0);
                REQUIRE(cacheManager.FreeSamples(&gpuScoped, 1, DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&globalScoped, 1, DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: clear-cache and helper snapshots keep sample lifetime explicit")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId    = cacheManager.AddFakeGpu();
    timelib64_t const timestamp = timelib_usecSince1970() + 200000;

    GIVEN("a watched fake GPU field with one cached sample")
    {
        dcgmcm_sample_t sample {};
        sample.timestamp = timestamp;
        sample.val.i64   = 55;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample, 1) == DCGM_ST_OK);

        WHEN("the watch is cleared but cache storage is retained")
        {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0);
            REQUIRE(watchInfo != nullptr);
            cacheManager.ClearWatchInfo(watchInfo, 0);

            THEN("sample retrieval still returns the cached value and snapshot shows inactive watch state")
            {
                dcgmcm_sample_t latest {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                        == DCGM_ST_OK);
                CHECK(latest.val.i64 == 55);

                dcgmcm_watch_info_t snapshot {};
                REQUIRE(
                    cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot)
                    == DCGM_ST_OK);
                CHECK(snapshot.isWatched == 0);
                CHECK(snapshot.timeSeries != nullptr);
            }
        }

        WHEN("the same watch is cleared with cache deletion enabled")
        {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0);
            REQUIRE(watchInfo != nullptr);
            cacheManager.ClearWatchInfo(watchInfo, 1);

            THEN("sample retrieval reports the field is no longer watched")
            {
                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);

                std::vector<dcgmcm_watch_info_p> watchers;
                cacheManager.GetAllWatchObjects(watchers);
                CHECK(std::find(watchers.begin(), watchers.end(), watchInfo) != watchers.end());
                CHECK(watchInfo->timeSeries == nullptr);
            }
        }
    }
}

SCENARIO("CacheManager: valid-field and runtime helpers expose local state")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with fake GPU and MIG state")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId              = cacheManager.AddFakeGpu();
        unsigned int const instanceId         = cacheManager.AddFakeInstance(gpuId);
        unsigned int const computeId          = cacheManager.AddFakeComputeInstance(instanceId);
        cacheManager.m_gpus[gpuId].migEnabled = true;

        WHEN("valid fields are requested with and without module-published ids")
        {
            std::vector<unsigned short> withModuleFields;
            std::vector<unsigned short> withoutModuleFields;
            cacheManager.GetValidFieldIds(withModuleFields, true);
            cacheManager.GetValidFieldIds(withoutModuleFields, false);

            THEN("the filtered list is a subset and MIG removes process-accounting stats")
            {
                REQUIRE_FALSE(withModuleFields.empty());
                REQUIRE_FALSE(withoutModuleFields.empty());
                CHECK(withoutModuleFields.size() <= withModuleFields.size());
                CHECK(std::find(
                          withoutModuleFields.begin(), withoutModuleFields.end(), DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS)
                      == withoutModuleFields.end());
                CHECK(std::find(withModuleFields.begin(), withModuleFields.end(), DCGM_FI_DEV_GPU_TEMP_CELSIUS)
                      != withModuleFields.end());
            }
        }

        WHEN("runtime stats and MIG index helpers are queried")
        {
            dcgmcm_runtime_stats_t stats {};
            cacheManager.GetRuntimeStats(&stats);
            auto const initialLockCount = stats.lockCount;
            cacheManager.GetRuntimeStats(nullptr);

            std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
            auto const gpuInfoRet   = cacheManager.GetAllGpuInfo(gpuInfo);
            unsigned int foundGpuId = DCGM_MAX_NUM_DEVICES;
            DcgmNs::Mig::GpuInstanceId foundInstanceId {};
            DcgmNs::Mig::ComputeInstanceId foundComputeInstanceId {};
            auto const computeIndexRet = cacheManager.GetMigIndicesForEntity(
                { DCGM_FE_GPU_CI, computeId }, &foundGpuId, &foundInstanceId, &foundComputeInstanceId);

            THEN("helpers return stable local counters and fake MIG ids")
            {
                CHECK(stats.lockCount >= initialLockCount);
                CHECK(gpuInfoRet == DCGM_ST_OK);
                REQUIRE_FALSE(gpuInfo.empty());
                CHECK(gpuInfo[0].gpuId == gpuId);
                CHECK(computeIndexRet == DCGM_ST_OK);
                CHECK(foundGpuId == gpuId);
                CHECK(foundInstanceId.id != DCGM_ENTITY_ID_BAD);
                CHECK(foundComputeInstanceId.id != DCGM_ENTITY_ID_BAD);
            }
        }
    }
}


SCENARIO("CacheManager: multi-latest aggregation preserves per-field statuses")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpu0 = cacheManager.AddFakeGpu();
    unsigned int const gpu1 = cacheManager.AddFakeGpu();

    GIVEN("one watched integer field and neighboring unwatched requests")
    {
        dcgmcm_sample_t sample {};
        sample.timestamp = timelib_usecSince1970() + 1000;
        sample.val.i64   = 12345;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_ECC_MODE, &sample, 1) == DCGM_ST_OK);

        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpu0 }, { DCGM_FE_GPU, gpu1 } };
        std::vector<unsigned short> fieldIds { DCGM_FI_DEV_ECC_MODE, DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        DcgmFvBuffer fvBuffer;

        WHEN("the aggregate latest-sample helper is called")
        {
            auto const ret = cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);

            THEN("each entity-field pair produces an independent buffered result")
            {
                CHECK(ret == DCGM_ST_OK);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *first                   = fvBuffer.GetNextFv(&cursor);
                REQUIRE(first != nullptr);
                CHECK(first->entityGroupId == DCGM_FE_GPU);
                CHECK(first->entityId == gpu0);
                CHECK(first->fieldId == DCGM_FI_DEV_ECC_MODE);
                CHECK(first->status == DCGM_ST_OK);
                CHECK(first->value.i64 == 12345);

                auto *second = fvBuffer.GetNextFv(&cursor);
                REQUIRE(second != nullptr);
                CHECK(second->entityId == gpu0);
                CHECK(second->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(second->status == DCGM_ST_NOT_WATCHED);
                CHECK(second->value.i64 == DCGM_INT64_BLANK);

                auto *third = fvBuffer.GetNextFv(&cursor);
                REQUIRE(third != nullptr);
                CHECK(third->entityId == gpu1);
                CHECK(third->fieldId == DCGM_FI_DEV_ECC_MODE);
                CHECK(third->status == DCGM_ST_OK);
                CHECK(third->value.i64 == 1);

                auto *fourth = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fourth != nullptr);
                CHECK(fourth->entityId == gpu1);
                CHECK(fourth->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(fourth->status == DCGM_ST_NOT_WATCHED);
                CHECK(fourth->value.i64 == DCGM_INT64_BLANK);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}


#ifdef INJECTION_LIBRARY_AVAILABLE
#endif

SCENARIO("CacheManager: sample range retrieval is ordered, bounded, and buffer-safe")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a watched integer field with deterministic timestamps")
    {
        timelib64_t const firstTimestamp = timelib_usecSince1970() + 100000;
        dcgmcm_sample_t samples[4] {};
        for (int index = 0; index < 4; ++index)
        {
            samples[index].timestamp = firstTimestamp + (index * 100);
            samples[index].val.i64   = 10 + (index * 10);
        }
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, samples, 4)
                == DCGM_ST_OK);

        WHEN("samples are requested in ascending order within a closed time range")
        {
            dcgmcm_sample_t out[4] {};
            int count      = 4;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     out,
                                                     &count,
                                                     firstTimestamp + 50,
                                                     firstTimestamp + 250,
                                                     DCGM_ORDER_ASCENDING,
                                                     nullptr);

            THEN("only matching samples are copied oldest to newest")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == 2);
                CHECK(out[0].timestamp == firstTimestamp + 100);
                CHECK(out[0].val.i64 == 20);
                CHECK(out[1].timestamp == firstTimestamp + 200);
                CHECK(out[1].val.i64 == 30);
            }
        }

        WHEN("samples are requested in descending order with a smaller caller buffer")
        {
            dcgmcm_sample_t out[2] {};
            int count      = 2;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     out,
                                                     &count,
                                                     firstTimestamp,
                                                     firstTimestamp + 300,
                                                     DCGM_ORDER_DESCENDING,
                                                     nullptr);

            THEN("the newest matching samples are returned up to caller capacity")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == 2);
                CHECK(out[0].timestamp == firstTimestamp + 300);
                CHECK(out[0].val.i64 == 40);
                CHECK(out[1].timestamp == firstTimestamp + 200);
                CHECK(out[1].val.i64 == 30);
            }
        }

        WHEN("the caller supplies only a field-value buffer")
        {
            DcgmFvBuffer fvBuffer;
            int count      = 4;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     nullptr,
                                                     &count,
                                                     firstTimestamp,
                                                     firstTimestamp + 150,
                                                     DCGM_ORDER_ASCENDING,
                                                     &fvBuffer);

            THEN("matching samples are written as buffered field values")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == 2);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *first                   = fvBuffer.GetNextFv(&cursor);
                REQUIRE(first != nullptr);
                CHECK(first->entityGroupId == DCGM_FE_GPU);
                CHECK(first->entityId == gpuId);
                CHECK(first->fieldId == DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
                CHECK(first->status == DCGM_ST_OK);
                CHECK(first->value.i64 == 10);
                CHECK(first->timestamp == firstTimestamp);

                auto *second = fvBuffer.GetNextFv(&cursor);
                REQUIRE(second != nullptr);
                CHECK(second->value.i64 == 20);
                CHECK(second->timestamp == firstTimestamp + 100);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("the requested range has no samples")
        {
            dcgmcm_sample_t out[2] {};
            int count      = 2;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                     out,
                                                     &count,
                                                     firstTimestamp + 500,
                                                     firstTimestamp + 600,
                                                     DCGM_ORDER_ASCENDING,
                                                     nullptr);

            THEN("the helper reports no data and clears the returned count")
            {
                CHECK(ret == DCGM_ST_NO_DATA);
                CHECK(count == 0);
                CHECK(out[0].val.i64 == 0);
            }
        }

        WHEN("local preconditions are violated")
        {
            dcgmcm_sample_t out[1] {};
            int zeroCount         = 0;
            int invalidOrderCnt   = 1;
            auto const noCountRet = cacheManager.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, out, nullptr, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
            auto const zeroCountRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                              gpuId,
                                                              DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              out,
                                                              &zeroCount,
                                                              0,
                                                              0,
                                                              DCGM_ORDER_ASCENDING,
                                                              nullptr);
            auto const noOutputRet  = cacheManager.GetSamples(DCGM_FE_GPU,
                                                              gpuId,
                                                              DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              nullptr,
                                                              &invalidOrderCnt,
                                                              0,
                                                              0,
                                                              DCGM_ORDER_ASCENDING,
                                                              nullptr);
            invalidOrderCnt         = 1;
            auto const badOrderRet  = cacheManager.GetSamples(DCGM_FE_GPU,
                                                              gpuId,
                                                              DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                              out,
                                                              &invalidOrderCnt,
                                                              0,
                                                              0,
                                                              static_cast<dcgmOrder_t>(99),
                                                              nullptr);

            THEN("each invalid request is rejected with a stable local status")
            {
                CHECK(noCountRet == DCGM_ST_BADPARAM);
                CHECK(zeroCountRet == DCGM_ST_BADPARAM);
                CHECK(noOutputRet == DCGM_ST_BADPARAM);
                CHECK(badOrderRet == DCGM_ST_BADPARAM);
                CHECK(invalidOrderCnt == 0);
            }
        }
    }
}

SCENARIO("CacheManager: string and blob samples are copied and released through public helpers")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId       = cacheManager.AddFakeGpu();
    timelib64_t const currentStamp = timelib_usecSince1970() + 100000;

    GIVEN("a string field injected from caller-owned storage")
    {
        char value[] = "deterministic-gpu-name";
        dcgmcm_sample_t sample {};
        sample.timestamp    = currentStamp;
        sample.val.str      = value;
        sample.val2.ptrSize = sizeof(value);
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &sample, 1) == DCGM_ST_OK);

        WHEN("the latest sample is read into a sample structure")
        {
            dcgmcm_sample_t latest {};
            auto const ret = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &latest, nullptr);

            THEN("the returned string is an owned copy that can be freed by the helper")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(latest.val.str != nullptr);
                CHECK(std::string(latest.val.str) == "deterministic-gpu-name");
                CHECK(latest.val.str != value);
                CHECK(latest.val2.ptrSize == sizeof(value));
                CHECK(cacheManager.FreeSamples(&latest, 1, DCGM_FI_DEV_GPU_NAME) == DCGM_ST_OK);
            }
        }
    }

    GIVEN("a binary field injected from caller-owned storage")
    {
        std::array<unsigned char, 5> payload { 1, 3, 5, 7, 9 };
        dcgmcm_sample_t sample {};
        sample.timestamp    = currentStamp + 1;
        sample.val.blob     = payload.data();
        sample.val2.ptrSize = payload.size();
        REQUIRE(
            cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK, &sample, 1)
            == DCGM_ST_OK);

        WHEN("the latest sample is read into a sample structure")
        {
            dcgmcm_sample_t latest {};
            auto const ret = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK, &latest, nullptr);

            THEN("the returned blob is an owned byte-for-byte copy")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(latest.val.blob != nullptr);
                REQUIRE(latest.val2.ptrSize == payload.size());
                CHECK(latest.val.blob != payload.data());
                CHECK(std::memcmp(latest.val.blob, payload.data(), payload.size()) == 0);
                CHECK(cacheManager.FreeSamples(&latest, 1, DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK)
                      == DCGM_ST_OK);
            }
        }
    }

    GIVEN("invalid free requests")
    {
        dcgmcm_sample_t sample {};

        WHEN("null arrays and unknown fields are passed to FreeSamples")
        {
            auto const nullRet         = cacheManager.FreeSamples(nullptr, 1, DCGM_FI_DEV_GPU_NAME);
            auto const unknownFieldRet = cacheManager.FreeSamples(&sample, 1, DCGM_FI_MAX_FIELDS);
            auto const intFieldRet     = cacheManager.FreeSamples(&sample, 1, DCGM_FI_DEV_ECC_MODE);

            THEN("null cleanup is rejected, unknown fields fail, and non-owned field types are accepted")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(unknownFieldRet == DCGM_ST_GENERIC_ERROR);
                CHECK(intFieldRet == DCGM_ST_OK);
            }
        }
    }
}


SCENARIO("CacheManager: injected samples publish field-value updates to subscribed watchers")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a subscribed field watch and a field-value update callback")
    {
        FvSubscriptionCallbackState state {};
        dcgmcmEventSubscription_t sub {};
        sub.type     = DcgmcmEventTypeFvUpdate;
        sub.fn.fvCb  = FvSubscriptionCallback;
        sub.userData = &state;
        REQUIRE(cacheManager.SubscribeForEvent(sub) == DCGM_ST_OK);

        DcgmWatcher watcher(DcgmWatcherTypeClient, 8801);
        bool wereFirstWatcher = false;
        REQUIRE(
            cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 60.0, 2, watcher, true, false, wereFirstWatcher)
            == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);

        WHEN("an int64 sample is injected into the watched field")
        {
            dcgmcm_sample_t sample {};
            sample.timestamp = timelib_usecSince1970() + 1000;
            sample.val.i64   = 73;
            auto const injectRet
                = cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample, 1);

            THEN("the callback receives exactly the subscribed watcher type and buffered value count")
            {
                REQUIRE(injectRet == DCGM_ST_OK);
                CHECK(state.calls == 1);
                REQUIRE(state.watcherCount == 1);
                CHECK(state.watchers[0] == DcgmWatcherTypeClient);
                CHECK(state.elementCount == 1);

                dcgmcm_sample_t latest {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                        == DCGM_ST_OK);
                CHECK(latest.val.i64 == 73);
            }
        }
    }
}


SCENARIO("CacheManager: fake entity limits and validation boundaries stay hardware-free")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager filled to the fake GPU limit")
    {
        TestDcgmCacheManager cacheManager;
        std::vector<unsigned int> createdGpuIds;
        createdGpuIds.reserve(DCGM_MAX_NUM_DEVICES);

        WHEN("fake GPUs are added through the inclusive capacity boundary")
        {
            for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
            {
                auto const gpuId = cacheManager.AddFakeGpu(0x2000 + i, 0x3000 + i);
                CAPTURE(i, gpuId);
                REQUIRE(gpuId == i);
                createdGpuIds.push_back(gpuId);
            }
            auto const overflowGpuId = cacheManager.AddFakeGpu(0xFFFF, 0xEEEE);

            THEN("the final valid id is accepted and the first overflow request is rejected")
            {
                CHECK(overflowGpuId == DCGM_GPU_ID_BAD);
                CHECK(cacheManager.GetGpuCount(0) == static_cast<int>(DCGM_MAX_NUM_DEVICES));
                CHECK(cacheManager.GetGpuCount(1) == static_cast<int>(DCGM_MAX_NUM_DEVICES));

                std::vector<unsigned int> allGpuIds;
                REQUIRE(cacheManager.GetGpuIds(0, allGpuIds) == DCGM_ST_OK);
                CHECK(allGpuIds == createdGpuIds);

                std::vector<dcgmGroupEntityPair_t> gpuEntities;
                REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, gpuEntities) == DCGM_ST_OK);
                REQUIRE(gpuEntities.size() == static_cast<decltype(gpuEntities.size())>(DCGM_MAX_NUM_DEVICES));
                CHECK(gpuEntities.front().entityId == 0);
                CHECK(gpuEntities.back().entityId == DCGM_MAX_NUM_DEVICES - 1);
            }
        }
    }

    GIVEN("unsupported entity groups and packed link ids")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId } };

        dcgm_link_t lastGpuLink {};
        lastGpuLink.parsed.type  = DCGM_FE_GPU;
        lastGpuLink.parsed.gpuId = gpuId;
        lastGpuLink.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU * 2 - 1;

        dcgm_link_t firstInvalidGpuLink  = lastGpuLink;
        firstInvalidGpuLink.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU * 2;

        dcgm_link_t lastSwitchLink {};
        lastSwitchLink.parsed.type  = DCGM_FE_SWITCH;
        lastSwitchLink.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH * 2 - 1;

        dcgm_link_t firstInvalidSwitchLink  = lastSwitchLink;
        firstInvalidSwitchLink.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH * 2;

        dcgm_link_t unsupportedLink {};
        unsupportedLink.parsed.type  = DCGM_FE_CONNECTX;
        unsupportedLink.parsed.index = 0;

        WHEN("entity-list and link validators evaluate boundary inputs")
        {
            auto const unsupportedRet = cacheManager.GetAllEntitiesOfEntityGroup(0, DCGM_FE_NONE, entities);

            THEN("unsupported groups clear caller lists and link index limits are enforced")
            {
                CHECK(unsupportedRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(entities.empty());
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, lastGpuLink.raw));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, firstInvalidGpuLink.raw));
                CHECK(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, lastSwitchLink.raw));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, firstInvalidSwitchLink.raw));
                CHECK_FALSE(cacheManager.GetIsValidEntityId(DCGM_FE_LINK, unsupportedLink.raw));
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_NONE, 0) == DcgmEntityStatusUnknown);
            }
        }
    }

    GIVEN("fake MIG parents that fail late validation")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId     = cacheManager.AddFakeGpu();
        auto const gpuInstanceId     = cacheManager.AddFakeInstance(gpuId);
        auto const computeInstanceId = cacheManager.AddFakeComputeInstance(gpuInstanceId);
        REQUIRE(gpuInstanceId != DCGM_ENTITY_ID_BAD);
        REQUIRE(computeInstanceId != DCGM_ENTITY_ID_BAD);

        WHEN("the parent GPU no longer has available GPC or compute-instance capacity")
        {
            cacheManager.m_gpus[gpuId].maxGpcs = 0;
            auto const noGpcComputeInstance    = cacheManager.AddFakeComputeInstance(gpuInstanceId);

            cacheManager.m_gpus[gpuId].maxGpcs = DCGM_MAX_INSTANCES_PER_GPU;
            cacheManager.m_gpus[gpuId].ciCount = DCGM_MAX_COMPUTE_INSTANCES_PER_GPU;
            auto const fullComputeInstance     = cacheManager.AddFakeComputeInstance(gpuInstanceId);

            THEN("both failure modes reject creation without disturbing existing mappings")
            {
                CHECK(noGpcComputeInstance == DCGM_ENTITY_ID_BAD);
                CHECK(fullComputeInstance == DCGM_ENTITY_ID_BAD);

                auto const owner = cacheManager.GetGpuIdForEntity(DCGM_FE_GPU_CI, computeInstanceId);
                REQUIRE(owner.has_value());
                CHECK(owner.value() == gpuId);

                std::vector<dcgmGroupEntityPair_t> computeEntities;
                REQUIRE(cacheManager.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU_CI, computeEntities) == DCGM_ST_OK);
                REQUIRE(computeEntities.size() == 1);
                CHECK(computeEntities[0].entityId == computeInstanceId);
            }
        }
    }
}


SCENARIO("CacheManager: local helpers cover field validation SKU and CUDA visible contracts")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with fake GPU state")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0 = cacheManager.AddFakeGpu(0x20B0, 0x145F);
        unsigned int const gpu1 = cacheManager.AddFakeGpu(0x20B0, 0x145F);
        unsigned int const gpu2 = cacheManager.AddFakeGpu(0x2330, 0x1626);

        WHEN("field validation helpers evaluate global device and invalid fields")
        {
            auto const validGlobalRet   = cacheManager.CheckValidGlobalField(DCGM_FI_SYSTEM_DRIVER_VERSION);
            auto const deviceGlobalRet  = cacheManager.CheckValidGlobalField(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            auto const unknownGlobalRet = cacheManager.CheckValidGlobalField(DCGM_FI_MAX_FIELDS);
            auto const validGpuRet      = cacheManager.CheckValidGpuField(gpu0, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            auto const globalGpuRet     = cacheManager.CheckValidGpuField(gpu0, DCGM_FI_SYSTEM_DRIVER_VERSION);
            auto const unknownGpuRet    = cacheManager.CheckValidGpuField(gpu0, DCGM_FI_MAX_FIELDS);
            auto const badGpuRet = cacheManager.CheckValidGpuField(DCGM_MAX_NUM_DEVICES, DCGM_FI_DEV_GPU_TEMP_CELSIUS);

            THEN("each invalid domain reports the specific local status")
            {
                CHECK(validGlobalRet == DCGM_ST_OK);
                CHECK(deviceGlobalRet == DCGM_ST_BADPARAM);
                CHECK(unknownGlobalRet == DCGM_ST_UNKNOWN_FIELD);
                CHECK(validGpuRet == DCGM_ST_OK);
                CHECK(globalGpuRet == DCGM_ST_BADPARAM);
                CHECK(unknownGpuRet == DCGM_ST_UNKNOWN_FIELD);
                CHECK(badGpuRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("SKU comparisons include empty single matching mismatching and invalid sets")
        {
            std::vector<unsigned int> empty;
            std::vector<unsigned int> single { gpu0 };
            std::vector<unsigned int> matching { gpu0, gpu1 };
            std::vector<unsigned int> mismatching { gpu0, gpu2 };
            std::vector<unsigned int> invalid { gpu0, DCGM_MAX_NUM_DEVICES };

            THEN("only non-empty sets with all known matching PCI ids are treated as the same SKU")
            {
                CHECK(cacheManager.AreAllGpuIdsSameSku(empty) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(single) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(matching) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(mismatching) == 0);
                CHECK(cacheManager.AreAllGpuIdsSameSku(invalid) == 0);
            }
        }
    }

    GIVEN("fake GPU and MIG entities with deterministic UUIDs")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        SafeCopyTo(cacheManager.m_gpus[gpuId].uuid, "GPU-11111111-2222-3333-4444-555555555555");
        auto const instanceId        = cacheManager.AddFakeInstance(gpuId);
        auto const computeInstanceId = cacheManager.AddFakeComputeInstance(instanceId);
        REQUIRE(instanceId != DCGM_ENTITY_ID_BAD);
        REQUIRE(computeInstanceId != DCGM_ENTITY_ID_BAD);

        WHEN("CUDA_VISIBLE_DEVICES values are generated for supported and unsupported entity groups")
        {
            std::stringstream gpuValue;
            std::stringstream instanceValue;
            std::stringstream computeInstanceValue;
            std::stringstream invalidGpuValue;
            std::stringstream unsupportedValue;

            cacheManager.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU, gpuId, gpuValue);
            cacheManager.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU_I, instanceId, instanceValue);
            cacheManager.GenerateCudaVisibleDevicesValue(
                gpuId, DCGM_FE_GPU_CI, computeInstanceId, computeInstanceValue);
            cacheManager.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES, invalidGpuValue);
            cacheManager.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_SWITCH, 0, unsupportedValue);

            THEN("GPU and MIG paths are formatted from cached fake state while invalid domains are explicit")
            {
                CHECK(gpuValue.str() == "MIG-GPU-11111111-2222-3333-4444-555555555555");
                CHECK(instanceValue.str() == "MIG-GPU-11111111-2222-3333-4444-555555555555/0");
                CHECK(computeInstanceValue.str() == "MIG-GPU-11111111-2222-3333-4444-555555555555/0/0");
                CHECK(invalidGpuValue.str() == "Invalid GPU id: " + std::to_string(DCGM_MAX_NUM_DEVICES));
                CHECK(unsupportedValue.str() == "Unsupported");
            }
        }
    }
}


SCENARIO("CacheManager: watcher metadata helper boundaries stay observable through snapshots")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a field watch with subscribed and unsubscribed owners")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        auto *watchInfo          = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);

        dcgm_watch_watcher_info_t subscribed {};
        subscribed.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 7001);
        subscribed.monitorIntervalUsec = 1000;
        subscribed.maxAgeUsec          = 10000;
        subscribed.isSubscribed        = 1;

        dcgm_watch_watcher_info_t unsubscribed {};
        unsubscribed.watcher             = DcgmWatcher(DcgmWatcherTypeHealthWatch, 0);
        unsubscribed.monitorIntervalUsec = 2000;
        unsubscribed.maxAgeUsec          = 20000;
        unsubscribed.isSubscribed        = 0;

        bool wasAdded = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &subscribed) == DCGM_ST_OK);
        REQUIRE(wasAdded);
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &unsubscribed) == DCGM_ST_OK);
        REQUIRE(wasAdded);

        WHEN("subscriber state is projected into an update-thread context")
        {
            dcgmcm_update_thread_t threadCtx {};
            cacheManager.MarkSubscribersInThreadCtx(threadCtx, nullptr);
            auto const noWatchersMask = threadCtx.affectedSubscribers;
            cacheManager.MarkSubscribersInThreadCtx(threadCtx, watchInfo);

            THEN("only subscribed watcher types are marked")
            {
                CHECK(noWatchersMask == 0);
                CHECK((threadCtx.affectedSubscribers & (1 << static_cast<unsigned int>(DcgmWatcherTypeClient))) != 0);
                CHECK((threadCtx.affectedSubscribers & (1 << static_cast<unsigned int>(DcgmWatcherTypeHealthWatch)))
                      == 0);
            }
        }

        WHEN("a snapshot is requested for present missing and null watch destinations")
        {
            dcgmcm_watch_info_t snapshot {};
            auto const presentRet
                = cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot);
            auto const missingRet
                = cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, &snapshot);
            auto const nullRet
                = cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr);

            THEN("the copied snapshot preserves watcher state and bad destinations are rejected")
            {
                CHECK(presentRet == DCGM_ST_OK);
                CHECK(snapshot.watchKey.entityGroupId == DCGM_FE_GPU);
                CHECK(snapshot.watchKey.entityId == gpuId);
                CHECK(snapshot.watchKey.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                REQUIRE(snapshot.watchers.size() == 2);
                CHECK(snapshot.watchers[0].watcher.connectionId == 7001);
                CHECK(missingRet == DCGM_ST_NOT_WATCHED);
                CHECK(nullRet == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("more field-info watchers than the public info structure can hold")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        auto *watchInfo          = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);

        for (unsigned int index = 0; index < DCGM_CM_FIELD_INFO_NUM_WATCHERS + 3; ++index)
        {
            dcgm_watch_watcher_info_t watcher {};
            watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 8000 + index);
            watcher.monitorIntervalUsec = 1000 + index;
            watcher.maxAgeUsec          = 2000 + index;
            bool wasAdded               = false;
            REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);
            REQUIRE(wasAdded);
        }

        WHEN("field info is copied into the fixed public watcher array")
        {
            dcgmCacheManagerFieldInfo_v4_t info {};
            info.version       = dcgmCacheManagerFieldInfo_version4;
            info.entityGroupId = DCGM_FE_GPU;
            info.entityId      = gpuId;
            info.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
            auto const ret     = cacheManager.GetCacheManagerFieldInfo(&info);

            THEN("the public structure is capped without losing aggregate watch state")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(info.numWatchers == DCGM_CM_FIELD_INFO_NUM_WATCHERS);
                CHECK(info.watchers[0].connectionId == 8000);
                CHECK(info.watchers[DCGM_CM_FIELD_INFO_NUM_WATCHERS - 1].connectionId
                      == 8000 + DCGM_CM_FIELD_INFO_NUM_WATCHERS - 1);
                CHECK(info.monitorIntervalUsec == 1000);
                CHECK(info.maxAgeUsec == 2000 + DCGM_CM_FIELD_INFO_NUM_WATCHERS + 2);
            }
        }
    }
}

SCENARIO("CacheManager: cached handle helpers expose active and detached GPU state")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const activeGpu   = cacheManager.AddFakeGpu(0x20B0, 0x145F);
    unsigned int const fakeGpu     = cacheManager.AddFakeGpu(0x2330, 0x1626);
    unsigned int const detachedGpu = cacheManager.AddFakeGpu(0x26B5, 0x18A4);

    GIVEN("fake GPUs with cached safe-handle state")
    {
        auto *activeHandle                    = reinterpret_cast<nvmlDevice_t>(0x1234);
        cacheManager.m_gpus[activeGpu].status = DcgmEntityStatusOk;
        cacheManager.m_gpus[activeGpu].safeNvmlHandle
            = SafeNvmlHandle(activeHandle, 41, cacheManager.m_nvmlDriver->GetGeneration());
        cacheManager.m_gpus[detachedGpu].status = DcgmEntityStatusDetached;

        WHEN("active handles and individual safe handles are requested")
        {
            auto activeHandles      = cacheManager.GetActiveGpuHandles();
            auto activeSafeHandle   = cacheManager.GetSafeNvmlHandle(activeGpu);
            auto fakeSafeHandle     = cacheManager.GetSafeNvmlHandle(fakeGpu);
            auto detachedSafeHandle = cacheManager.GetSafeNvmlHandle(detachedGpu);
            auto invalidSafeHandle  = cacheManager.GetSafeNvmlHandle(DCGM_MAX_NUM_DEVICES);

            THEN("only OK GPUs are reported as active while fake and detached statuses remain observable")
            {
                REQUIRE(activeHandles.size() == 1);
                CHECK(activeHandles[0].gpuId == activeGpu);
                CHECK(activeHandles[0].safeNvmlHandle.nvmlDevice == activeHandle);
                CHECK(activeHandles[0].safeNvmlHandle.nvmlIndex == 41);

                REQUIRE_FALSE(activeSafeHandle.is_error());
                CHECK(activeSafeHandle.value().nvmlDevice == activeHandle);
                CHECK(activeSafeHandle.value().nvmlIndex == 41);

                REQUIRE_FALSE(fakeSafeHandle.is_error());
                CHECK(fakeSafeHandle.value().nvmlDevice == nullptr);
                CHECK(fakeSafeHandle.value().nvmlIndex == fakeGpu);

                REQUIRE(detachedSafeHandle.is_error());
                CHECK(detachedSafeHandle.error() == DCGM_ST_GPU_IS_LOST);

                REQUIRE(invalidSafeHandle.is_error());
                CHECK(invalidSafeHandle.error() == DCGM_ST_BADPARAM);
            }
        }

        WHEN("raw NVML handles are looked up by entity id")
        {
            auto const activeRawHandle  = cacheManager.GetNvmlDeviceFromEntityId(activeGpu);
            auto const fakeRawHandle    = cacheManager.GetNvmlDeviceFromEntityId(fakeGpu);
            auto const invalidRawHandle = cacheManager.GetNvmlDeviceFromEntityId(DCGM_MAX_NUM_DEVICES);

            THEN("the helper returns cached handles and null for invalid ids")
            {
                CHECK(activeRawHandle == activeHandle);
                CHECK(fakeRawHandle == nullptr);
                CHECK(invalidRawHandle == nullptr);
            }
        }

        WHEN("active entity filtering is applied to valid detached invalid and unsupported entities")
        {
            auto const activeInstance          = cacheManager.AddFakeInstance(activeGpu);
            auto const detachedInstance        = cacheManager.AddFakeInstance(detachedGpu);
            auto const activeComputeInstance   = cacheManager.AddFakeComputeInstance(activeInstance);
            auto const detachedComputeInstance = cacheManager.AddFakeComputeInstance(detachedInstance);
            std::vector<dcgmGroupEntityPair_t> entities {
                { DCGM_FE_GPU, activeGpu },
                { DCGM_FE_GPU, fakeGpu },
                { DCGM_FE_GPU, detachedGpu },
                { DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES },
                { DCGM_FE_GPU_I, activeInstance },
                { DCGM_FE_GPU_I, detachedInstance },
                { DCGM_FE_GPU_CI, activeComputeInstance },
                { DCGM_FE_GPU_CI, detachedComputeInstance },
                { DCGM_FE_SWITCH, 99 },
            };

            auto const activeEntities = cacheManager.FilterActiveEntities(entities);

            THEN("active OK fake and non-GPU entities remain while detached hierarchy entries are skipped")
            {
                REQUIRE(activeEntities.size() == 5);
                CHECK(activeEntities[0].entityGroupId == DCGM_FE_GPU);
                CHECK(activeEntities[0].entityId == activeGpu);
                CHECK(activeEntities[1].entityGroupId == DCGM_FE_GPU);
                CHECK(activeEntities[1].entityId == fakeGpu);
                CHECK(activeEntities[2].entityGroupId == DCGM_FE_GPU_I);
                CHECK(activeEntities[2].entityId == activeInstance);
                CHECK(activeEntities[3].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(activeEntities[3].entityId == activeComputeInstance);
                CHECK(activeEntities[4].entityGroupId == DCGM_FE_SWITCH);
                CHECK(activeEntities[4].entityId == 99);
            }
        }
    }
}

SCENARIO("CacheManager: append helpers cache and buffer double string and blob values")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a watched fp64 field and a subscribed client watcher")
    {
        auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched = 1;

        dcgm_watch_watcher_info_t watcher {};
        watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 9001);
        watcher.monitorIntervalUsec = 1000;
        watcher.maxAgeUsec          = 1000000;
        watcher.isSubscribed        = 1;
        bool wasAdded               = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);
        REQUIRE(wasAdded);

        WHEN("a double value is appended with both cache and field-value-buffer outputs")
        {
            DcgmFvBuffer fvBuffer;
            dcgmcm_update_thread_t threadCtx {};
            threadCtx.watchInfo               = watchInfo;
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = DCGM_FI_DEV_BOARD_POWER_WATTS;

            auto const ret = cacheManager.AppendEntityDouble(threadCtx, 237.5, 0.0, 1000, 0);

            THEN("the cached sample and buffered field value contain the appended double")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK((threadCtx.affectedSubscribers & (1 << static_cast<unsigned int>(DcgmWatcherTypeClient))) != 0);

                dcgmcm_sample_t latest {};
                REQUIRE(
                    cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, &latest, nullptr)
                    == DCGM_ST_OK);
                CHECK(latest.timestamp == 1000);
                CHECK_THAT(latest.val.d, Catch::Matchers::WithinAbs(237.5, 1.0e-12));

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(fv->fieldType == DCGM_FT_DOUBLE);
                CHECK(fv->status == DCGM_ST_OK);
                CHECK_THAT(fv->value.dbl, Catch::Matchers::WithinAbs(237.5, 1.0e-12));
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("cached samples are requested directly into a field-value buffer")
        {
            dcgmcm_update_thread_t threadCtx {};
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = DCGM_FI_DEV_BOARD_POWER_WATTS;
            REQUIRE(cacheManager.AppendEntityDouble(threadCtx, 10.0, 0.0, 100, 0) == DCGM_ST_OK);
            REQUIRE(cacheManager.AppendEntityDouble(threadCtx, 20.0, 0.0, 200, 0) == DCGM_ST_OK);
            REQUIRE(cacheManager.AppendEntityDouble(threadCtx, 30.0, 0.0, 300, 0) == DCGM_ST_OK);

            DcgmFvBuffer fvBuffer;
            int count      = 2;
            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                     nullptr,
                                                     &count,
                                                     0,
                                                     0,
                                                     DCGM_ORDER_DESCENDING,
                                                     &fvBuffer);

            THEN("the requested count limits buffered descending samples")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(count == 2);

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *first       = fvBuffer.GetNextFv(&cursor);
                dcgmBufferedFv_t *second      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(first != nullptr);
                REQUIRE(second != nullptr);
                CHECK(first->timestamp == 300);
                CHECK(second->timestamp == 200);
                CHECK_THAT(first->value.dbl, Catch::Matchers::WithinAbs(30.0, 1.0e-12));
                CHECK_THAT(second->value.dbl, Catch::Matchers::WithinAbs(20.0, 1.0e-12));
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }

    GIVEN("watched string and blob fields")
    {
        auto *stringWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, 1);
        auto *blobWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, 1);
        REQUIRE(stringWatch != nullptr);
        REQUIRE(blobWatch != nullptr);
        stringWatch->isWatched = 1;
        blobWatch->isWatched   = 1;

        WHEN("string and blob values are appended through the helper APIs")
        {
            dcgmcm_update_thread_t stringCtx {};
            stringCtx.watchInfo               = stringWatch;
            stringCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            stringCtx.entityKey.entityId      = gpuId;
            stringCtx.entityKey.fieldId       = DCGM_FI_DEV_GPU_NAME;

            dcgmcm_update_thread_t blobCtx {};
            blobCtx.watchInfo               = blobWatch;
            blobCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            blobCtx.entityKey.entityId      = gpuId;
            blobCtx.entityKey.fieldId       = DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS;

            std::array<unsigned char, 5> blob { 9, 8, 7, 6, 5 };
            auto const stringRet = cacheManager.AppendEntityString(stringCtx, "cache-helper-gpu", 4000, 0);
            auto const blobRet
                = cacheManager.AppendEntityBlob(blobCtx, blob.data(), static_cast<int>(blob.size()), 5000, 0);

            THEN("latest sample conversion owns returned string and blob memory")
            {
                REQUIRE(stringRet == DCGM_ST_OK);
                REQUIRE(blobRet == DCGM_ST_OK);

                dcgmcm_sample_t stringSample {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &stringSample, nullptr)
                        == DCGM_ST_OK);
                REQUIRE(stringSample.val.str != nullptr);
                CHECK(stringSample.timestamp == 4000);
                CHECK(std::string_view(stringSample.val.str) == "cache-helper-gpu");
                CHECK(cacheManager.FreeSamples(&stringSample, 1, DCGM_FI_DEV_GPU_NAME) == DCGM_ST_OK);
                CHECK(stringSample.val.str == nullptr);

                dcgmcm_sample_t blobSample {};
                REQUIRE(cacheManager.GetLatestSample(
                            DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, &blobSample, nullptr)
                        == DCGM_ST_OK);
                REQUIRE(blobSample.val.blob != nullptr);
                CHECK(blobSample.timestamp == 5000);
                CHECK(blobSample.val2.ptrSize == static_cast<long long>(blob.size()));
                CHECK(std::memcmp(blobSample.val.blob, blob.data(), blob.size()) == 0);
                CHECK(cacheManager.FreeSamples(&blobSample, 1, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS) == DCGM_ST_OK);
                CHECK(blobSample.val.blob == nullptr);
            }
        }
    }
}


SCENARIO("CacheManager: entity status field-info and error-buffer paths are local and deterministic")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("a GPU hierarchy whose parent status changes")
    {
        unsigned int const instanceId        = cacheManager.AddFakeInstance(gpuId);
        unsigned int const computeInstanceId = cacheManager.AddFakeComputeInstance(instanceId);
        cacheManager.m_gpus[gpuId].status    = DcgmEntityStatusOk;

        WHEN("entity status is queried for valid invalid and unsupported entities")
        {
            auto const gpuStatus             = cacheManager.GetEntityStatus(DCGM_FE_GPU, gpuId);
            auto const instanceStatus        = cacheManager.GetEntityStatus(DCGM_FE_GPU_I, instanceId);
            auto const computeInstanceStatus = cacheManager.GetEntityStatus(DCGM_FE_GPU_CI, computeInstanceId);
            auto const badGpuStatus          = cacheManager.GetEntityStatus(DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES);
            auto const badInstanceStatus     = cacheManager.GetEntityStatus(DCGM_FE_GPU_I, DCGM_MAX_NUM_DEVICES);
            auto const unsupportedStatus     = cacheManager.GetEntityStatus(DCGM_FE_SWITCH, 0);

            THEN("GPU instance and compute instance statuses all resolve through the parent GPU")
            {
                CHECK(gpuStatus == DcgmEntityStatusOk);
                CHECK(instanceStatus == DcgmEntityStatusOk);
                CHECK(computeInstanceStatus == DcgmEntityStatusOk);
                CHECK(badGpuStatus == DcgmEntityStatusUnknown);
                CHECK(badInstanceStatus == DcgmEntityStatusUnknown);
                CHECK(unsupportedStatus == DcgmEntityStatusUnknown);
            }
        }

        WHEN("the parent GPU is marked detached")
        {
            cacheManager.m_gpus[gpuId].status = DcgmEntityStatusDetached;

            THEN("child entities report the detached parent status")
            {
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU, gpuId) == DcgmEntityStatusDetached);
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_I, instanceId) == DcgmEntityStatusDetached);
                CHECK(cacheManager.GetEntityStatus(DCGM_FE_GPU_CI, computeInstanceId) == DcgmEntityStatusDetached);
            }
        }
    }

    GIVEN("cache-manager field-info requests")
    {
        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        fieldInfo.version       = dcgmCacheManagerFieldInfo_version4;
        fieldInfo.entityGroupId = DCGM_FE_GPU;
        fieldInfo.entityId      = gpuId;
        fieldInfo.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

        WHEN("required envelopes are null stale invalid or not watched")
        {
            auto const nullRet = cacheManager.GetCacheManagerFieldInfo(nullptr);

            fieldInfo.version     = 0;
            auto const versionRet = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);

            fieldInfo.version   = dcgmCacheManagerFieldInfo_version4;
            fieldInfo.fieldId   = std::numeric_limits<unsigned short>::max();
            auto const fieldRet = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);

            fieldInfo.fieldId        = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
            auto const notWatchedRet = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);

            THEN("each local validation branch reports its specific status")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(versionRet == DCGM_ST_VER_MISMATCH);
                CHECK(fieldRet == DCGM_ST_BADPARAM);
                CHECK(notWatchedRet == DCGM_ST_NOT_WATCHED);
            }
        }

        WHEN("a global field is watched and queried through a GPU-scoped envelope")
        {
            auto *watchInfo = cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched           = 1;
            watchInfo->lastStatus          = NVML_ERROR_NOT_SUPPORTED;
            watchInfo->monitorIntervalUsec = 1234;
            watchInfo->maxAgeUsec          = 5678;
            watchInfo->fetchCount          = 9;
            watchInfo->execTimeUsec        = 10;

            dcgm_watch_watcher_info_t watcher {};
            watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 77);
            watcher.monitorIntervalUsec = 111;
            watcher.maxAgeUsec          = 222;
            bool wasAdded               = false;
            REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);

            dcgmCacheManagerFieldInfo_v4_t globalInfo {};
            globalInfo.version       = dcgmCacheManagerFieldInfo_version4;
            globalInfo.entityGroupId = DCGM_FE_GPU;
            globalInfo.entityId      = gpuId;
            globalInfo.fieldId       = DCGM_FI_SYSTEM_DRIVER_VERSION;
            auto const ret           = cacheManager.GetCacheManagerFieldInfo(&globalInfo);

            THEN("global scope is normalized and watcher metadata is copied into the public snapshot")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(globalInfo.version == dcgmCacheManagerFieldInfo_version4);
                CHECK((globalInfo.flags & DCGM_CMI_F_WATCHED) != 0);
                CHECK(globalInfo.lastStatus == NVML_ERROR_NOT_SUPPORTED);
                CHECK(globalInfo.monitorIntervalUsec == 111);
                CHECK(globalInfo.maxAgeUsec == 222);
                CHECK(globalInfo.fetchCount == 9);
                CHECK(globalInfo.execTimeUsec == 10);
                CHECK(globalInfo.numSamples == 0);
                REQUIRE(globalInfo.numWatchers == 1);
                CHECK(globalInfo.watchers[0].watcherType == DcgmWatcherTypeClient);
                CHECK(globalInfo.watchers[0].connectionId == 77);
                CHECK(globalInfo.watchers[0].monitorIntervalUsec == 111);
                CHECK(globalInfo.watchers[0].maxAgeUsec == 222);
            }
        }
    }

    GIVEN("latest-sample requests that fail before data conversion")
    {
        WHEN("the caller omits both output forms")
        {
            auto const ret
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, nullptr);

            THEN("the request is rejected as a bad parameter")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("an unknown field is requested with a field-value buffer")
        {
            DcgmFvBuffer fvBuffer;
            auto const ret = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, std::numeric_limits<unsigned short>::max(), nullptr, &fvBuffer);

            THEN("the API returns unknown field before appending a field value")
            {
                CHECK(ret == DCGM_ST_UNKNOWN_FIELD);
                dcgmBufferedFvCursor_t cursor = 0;
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("a valid but unwatched field is requested with a field-value buffer")
        {
            DcgmFvBuffer fvBuffer;
            auto const ret
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, &fvBuffer);

            THEN("the aggregate return and buffered field value both report not watched")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(fv->status == DCGM_ST_NOT_WATCHED);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}


SCENARIO("CacheManager: process accounting helper validation stays local and deterministic")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("PID list helper inputs that fail before accounting aggregation")
    {
        std::array<dcgmProcessUtilInfo_t, 2> pidInfo {};
        std::array<unsigned int, 2> pids {};
        unsigned int numPids = static_cast<unsigned int>(pidInfo.size());

        WHEN("null pointers invalid fields and unwatched fields are requested")
        {
            auto const nullInfoRet  = cacheManager.GetUniquePidLists(DCGM_FE_GPU,
                                                                     gpuId,
                                                                     DCGM_FI_DEV_GRAPHICS_PIDS,
                                                                     0,
                                                                     static_cast<dcgmProcessUtilInfo_t *>(nullptr),
                                                                     &numPids,
                                                                     0,
                                                                     0);
            auto const nullCountRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, 0, pidInfo.data(), nullptr, 0, 0);
            auto const invalidFieldRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, pidInfo.data(), &numPids, 0, 0);
            auto const unwatchedStructRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, 0, pidInfo.data(), &numPids, 0, 0);
            auto const nullPidArrayRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_COMPUTE_PIDS, 0, static_cast<unsigned int *>(nullptr), &numPids, 0, 0);
            auto const unwatchedPidArrayRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_COMPUTE_PIDS, 0, pids.data(), &numPids, 0, 0);

            THEN("the overloads return precise validation statuses without modifying caller buffers")
            {
                CHECK(nullInfoRet == DCGM_ST_BADPARAM);
                CHECK(nullCountRet == DCGM_ST_BADPARAM);
                CHECK(invalidFieldRet == DCGM_ST_BADPARAM);
                CHECK(unwatchedStructRet == DCGM_ST_NOT_WATCHED);
                CHECK(nullPidArrayRet == DCGM_ST_BADPARAM);
                CHECK(unwatchedPidArrayRet == DCGM_ST_NOT_WATCHED);
                CHECK(pidInfo[0].pid == 0);
                CHECK(pids[0] == 0);
            }
        }
    }

    GIVEN("PID utilization and latest accounting helpers")
    {
        std::array<dcgmProcessUtilSample_t, 2> utilSamples {};
        unsigned int numSamples = static_cast<unsigned int>(utilSamples.size());
        dcgmDevicePidAccountingStats_t accountingStats {};

        WHEN("validation fails before accounting data is read")
        {
            auto const nullUtilRet = cacheManager.GetUniquePidUtilLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, 0, nullptr, &numSamples, 0, 0);
            auto const nullUtilCountRet = cacheManager.GetUniquePidUtilLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, 0, utilSamples.data(), nullptr, 0, 0);
            auto const invalidUtilFieldRet = cacheManager.GetUniquePidUtilLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, 0, utilSamples.data(), &numSamples, 0, 0);
            auto const unwatchedUtilRet = cacheManager.GetUniquePidUtilLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES, 0, utilSamples.data(), &numSamples, 0, 0);
            auto const zeroPidRet        = cacheManager.GetLatestProcessInfo(gpuId, 0, &accountingStats);
            auto const nullStatsRet      = cacheManager.GetLatestProcessInfo(gpuId, 1234, nullptr);
            auto const unwatchedStatsRet = cacheManager.GetLatestProcessInfo(gpuId, 1234, &accountingStats);

            THEN("the helpers report local validation or watch-state errors")
            {
                CHECK(nullUtilRet == DCGM_ST_BADPARAM);
                CHECK(nullUtilCountRet == DCGM_ST_BADPARAM);
                CHECK(invalidUtilFieldRet == DCGM_ST_BADPARAM);
                CHECK(unwatchedUtilRet == DCGM_ST_NOT_WATCHED);
                CHECK(zeroPidRet == DCGM_ST_BADPARAM);
                CHECK(nullStatsRet == DCGM_ST_BADPARAM);
                CHECK(unwatchedStatsRet == DCGM_ST_NOT_WATCHED);
                CHECK(utilSamples[0].pid == 0);
            }
        }

        WHEN("watched accounting fields carry stale NVML statuses but no samples")
        {
            auto *graphicsWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, 1);
            auto *utilWatch     = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, 1);
            auto *accountingWatch
                = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, 1);
            REQUIRE(graphicsWatch != nullptr);
            REQUIRE(utilWatch != nullptr);
            REQUIRE(accountingWatch != nullptr);
            graphicsWatch->isWatched    = 1;
            utilWatch->isWatched        = 1;
            accountingWatch->isWatched  = 1;
            graphicsWatch->lastStatus   = NVML_ERROR_TIMEOUT;
            utilWatch->lastStatus       = NVML_ERROR_NOT_SUPPORTED;
            accountingWatch->lastStatus = NVML_ERROR_GPU_IS_LOST;

            std::array<unsigned int, 2> pids {};
            unsigned int numPids  = static_cast<unsigned int>(pids.size());
            auto const pidListRet = cacheManager.GetUniquePidLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GRAPHICS_PIDS, 0, pids.data(), &numPids, 0, 0);
            auto const utilRet = cacheManager.GetUniquePidUtilLists(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_UTIL_SAMPLES, 0, utilSamples.data(), &numSamples, 0, 0);
            auto const accountingRet = cacheManager.GetLatestProcessInfo(gpuId, 1234, &accountingStats);

            THEN("precheck translates the stored NVML status before accounting helpers run")
            {
                CHECK(pidListRet == DCGM_ST_TIMEOUT);
                CHECK(utilRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(accountingRet == DCGM_ST_GPU_IS_LOST);
            }
        }
    }
}


SCENARIO("CacheManager: fatal XID environment helper applies deterministic overrides")
{
    char const *envKey = "__DCGM_FATAL_XIDS__";
    ScopedEnvVars envVars(envKey, std::nullopt);

    GIVEN("the fatal XID override is absent")
    {
        WHEN("the cache-manager helper reads its default set")
        {
            auto const fatalXids = ReadEnvForFatalXids();

            THEN("the built-in fatal XID remains enabled")
            {
                CHECK(fatalXids == std::unordered_set<uint32_t> { 119 });
            }
        }
    }

    GIVEN("the fatal XID override adds and removes comma-separated entries")
    {
        envVars.Add(envKey, "-119,31,48");

        WHEN("the cache-manager helper reads the override")
        {
            auto const fatalXids = ReadEnvForFatalXids();

            THEN("the resulting set reflects removals and additions")
            {
                CHECK(fatalXids == std::unordered_set<uint32_t> { 31, 48 });
            }
        }
    }

    GIVEN("the fatal XID override contains an invalid token after a valid edit")
    {
        envVars.Add(envKey, "31,invalid,48");

        WHEN("the cache-manager helper reads the override")
        {
            auto const fatalXids = ReadEnvForFatalXids();

            THEN("parsing stops after preserving the default and completed edits")
            {
                CHECK(fatalXids == std::unordered_set<uint32_t> { 31, 119 });
            }
        }
    }
}


SCENARIO("CacheManager: local public helpers preserve caller-visible state across edge paths")
{
    GIVEN("a cache manager with fake GPUs that differ in attachment and SKU metadata")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0          = cacheManager.AddFakeGpu(0x20B0, 0x12BA);
        unsigned int const gpu1          = cacheManager.AddFakeGpu(0x20B0, 0x12BA);
        unsigned int const gpu2          = cacheManager.AddFakeGpu(0x2330, 0x1459);
        cacheManager.m_gpus[gpu1].status = DcgmEntityStatusDetached;

        WHEN("GPU ids and counts are requested with active-only filtering")
        {
            std::vector<unsigned int> allGpuIds { 99 };
            std::vector<unsigned int> activeGpuIds { 88 };

            auto const allRet    = cacheManager.GetGpuIds(0, allGpuIds);
            auto const activeRet = cacheManager.GetGpuIds(1, activeGpuIds);

            THEN("the output vectors are replaced and detached GPUs are filtered only for active queries")
            {
                CHECK(allRet == DCGM_ST_OK);
                CHECK(activeRet == DCGM_ST_OK);
                CHECK(allGpuIds == std::vector<unsigned int> { gpu0, gpu1, gpu2 });
                CHECK(activeGpuIds == std::vector<unsigned int> { gpu0, gpu2 });
                CHECK(cacheManager.GetGpuCount(0) == 3);
                CHECK(cacheManager.GetGpuCount(1) == 2);
            }
        }

        WHEN("SKU comparison receives empty single matching mixed and invalid lists")
        {
            std::vector<unsigned int> empty;
            std::vector<unsigned int> single { gpu0 };
            std::vector<unsigned int> matching { gpu0, gpu1 };
            std::vector<unsigned int> mixed { gpu0, gpu2 };
            std::vector<unsigned int> invalid { gpu0, DCGM_MAX_NUM_DEVICES };

            THEN("only lists with fewer than two or matching PCI ids are accepted")
            {
                CHECK(cacheManager.AreAllGpuIdsSameSku(empty) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(single) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(matching) == 1);
                CHECK(cacheManager.AreAllGpuIdsSameSku(mixed) == 0);
                CHECK(cacheManager.AreAllGpuIdsSameSku(invalid) == 0);
            }
        }

        WHEN("driver version strings are set through the test-visible cache state")
        {
            cacheManager.m_driverVersion     = "535.104.05";
            cacheManager.m_driverFullVersion = "535.104.05-1.el8";

            THEN("the public helpers report the cached version and lexicographic threshold results")
            {
                CHECK(cacheManager.GetDriverVersion() == "535.104.05-1.el8");
                CHECK(cacheManager.DriverVersionIsAtLeast("535.104.04"));
                CHECK(cacheManager.DriverVersionIsAtLeast("535.104.05"));
                CHECK_FALSE(cacheManager.DriverVersionIsAtLeast("535.104.06"));
            }
        }
    }

    GIVEN("a cache manager with no watch state")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();

        WHEN("runtime and field-info helpers receive null or unwatched requests")
        {
            dcgmcm_runtime_stats_t stats {};
            dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
            fieldInfo.version       = dcgmCacheManagerFieldInfo_version4;
            fieldInfo.entityGroupId = DCGM_FE_GPU;
            fieldInfo.entityId      = gpuId;
            fieldInfo.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

            cacheManager.GetRuntimeStats(nullptr);
            cacheManager.GetRuntimeStats(&stats);
            auto const nullInfoRet      = cacheManager.GetCacheManagerFieldInfo(nullptr);
            auto const unwatchedInfoRet = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);

            THEN("null stats are ignored and unwatched field queries fail without mutating identifiers")
            {
                CHECK(stats.lockCount >= 0);
                CHECK(nullInfoRet == DCGM_ST_BADPARAM);
                CHECK(unwatchedInfoRet == DCGM_ST_BADPARAM);
                CHECK(fieldInfo.entityGroupId == DCGM_FE_GPU);
                CHECK(fieldInfo.entityId == gpuId);
                CHECK(fieldInfo.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            }
        }
    }
}

SCENARIO("CacheManager: MIG create delete and handle helpers validate local state before NVML calls")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with fake GPU and MIG hierarchy")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId                  = cacheManager.AddFakeGpu();
        unsigned int const detachedGpuId          = cacheManager.AddFakeGpu();
        auto const gpuInstanceId                  = cacheManager.AddFakeInstance(gpuId);
        auto const computeInstanceId              = cacheManager.AddFakeComputeInstance(gpuInstanceId);
        cacheManager.m_gpus[detachedGpuId].status = DcgmEntityStatusDetached;

        WHEN("MIG create requests fail validation before any NVML operation is needed")
        {
            dcgmCreateMigEntity_v1 badProfile {};
            badProfile.version      = dcgmCreateMigEntity_version1;
            badProfile.parentId     = gpuId;
            badProfile.profile      = DcgmMigProfileNone;
            badProfile.createOption = DcgmMigCreateGpuInstance;

            dcgmCreateMigEntity_v1 missingParent {};
            missingParent.version      = dcgmCreateMigEntity_version1;
            missingParent.parentId     = DCGM_MAX_NUM_DEVICES;
            missingParent.profile      = DcgmMigProfileGpuInstanceSlice1;
            missingParent.createOption = DcgmMigCreateGpuInstance;
            missingParent.flags        = DCGM_MIG_RECONFIG_DELAY_PROCESSING;

            dcgmCreateMigEntity_v1 detachedParent {};
            detachedParent.version      = dcgmCreateMigEntity_version1;
            detachedParent.parentId     = detachedGpuId;
            detachedParent.profile      = DcgmMigProfileGpuInstanceSlice1;
            detachedParent.createOption = DcgmMigCreateGpuInstance;

            dcgmCreateMigEntity_v1 missingInstance {};
            missingInstance.version      = dcgmCreateMigEntity_version1;
            missingInstance.parentId     = 424242;
            missingInstance.profile      = DcgmMigProfileComputeInstanceSlice1;
            missingInstance.createOption = DcgmMigCreateComputeInstance;

            auto const badProfileRet      = cacheManager.CreateMigEntity(badProfile);
            auto const missingParentRet   = cacheManager.CreateMigEntity(missingParent);
            auto const delayedTimestamp   = cacheManager.m_delayedMigReconfigProcessingTimestamp;
            auto const detachedParentRet  = cacheManager.CreateMigEntity(detachedParent);
            auto const missingInstanceRet = cacheManager.CreateMigEntity(missingInstance);

            THEN("the public return codes identify invalid profile parent and detached-parent cases")
            {
                CHECK(badProfileRet == DCGM_ST_BADPARAM);
                CHECK(missingParentRet == DCGM_ST_BADPARAM);
                CHECK(delayedTimestamp > 0);
                CHECK(detachedParentRet == DCGM_ST_GPU_NOT_SUPPORTED);
                CHECK(cacheManager.m_delayedMigReconfigProcessingTimestamp == 0);
                CHECK(missingInstanceRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("MIG delete requests fail validation before any NVML operation is needed")
        {
            dcgmDeleteMigEntity_v1 badGroup {};
            badGroup.version       = dcgmDeleteMigEntity_version1;
            badGroup.entityGroupId = DCGM_FE_GPU;
            badGroup.entityId      = gpuId;
            badGroup.flags         = DCGM_MIG_RECONFIG_DELAY_PROCESSING;

            dcgmDeleteMigEntity_v1 missingInstance {};
            missingInstance.version       = dcgmDeleteMigEntity_version1;
            missingInstance.entityGroupId = DCGM_FE_GPU_I;
            missingInstance.entityId      = 424242;

            dcgmDeleteMigEntity_v1 missingComputeInstance {};
            missingComputeInstance.version       = dcgmDeleteMigEntity_version1;
            missingComputeInstance.entityGroupId = DCGM_FE_GPU_CI;
            missingComputeInstance.entityId      = 515151;

            auto const badGroupRet               = cacheManager.DeleteMigEntity(badGroup);
            auto const delayedTimestamp          = cacheManager.m_delayedMigReconfigProcessingTimestamp;
            auto const missingInstanceRet        = cacheManager.DeleteMigEntity(missingInstance);
            auto const missingComputeInstanceRet = cacheManager.DeleteMigEntity(missingComputeInstance);

            THEN("invalid entity groups and missing MIG ids are rejected with explicit errors")
            {
                CHECK(badGroupRet == DCGM_ST_BADPARAM);
                CHECK(delayedTimestamp > 0);
                CHECK(missingInstanceRet == DCGM_ST_INSTANCE_NOT_FOUND);
                CHECK(cacheManager.m_delayedMigReconfigProcessingTimestamp == 0);
                CHECK(missingComputeInstanceRet == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
            }
        }

        WHEN("compute-instance NVML handles are requested for valid and invalid MIG entities")
        {
            auto const invalidGpuHandle
                = cacheManager.GetComputeInstanceNvmlDevice(DCGM_MAX_NUM_DEVICES, DCGM_FE_GPU_CI, computeInstanceId);
            auto const missingInstanceHandle = cacheManager.GetComputeInstanceNvmlDevice(gpuId, DCGM_FE_GPU_I, 424242);
            auto const gpuInstanceHandle
                = cacheManager.GetComputeInstanceNvmlDevice(gpuId, DCGM_FE_GPU_I, gpuInstanceId);
            auto const computeInstanceHandle
                = cacheManager.GetComputeInstanceNvmlDevice(gpuId, DCGM_FE_GPU_CI, computeInstanceId);

            THEN("fake hierarchy lookups stay deterministic and never require a real NVML device")
            {
                CHECK(invalidGpuHandle.nvmlDevice == nullptr);
                CHECK(invalidGpuHandle.nvmlIndex == 0);
                CHECK(missingInstanceHandle.nvmlDevice == nullptr);
                CHECK(gpuInstanceHandle.nvmlDevice == nullptr);
                CHECK(gpuInstanceHandle.nvmlIndex == gpuId);
                CHECK(computeInstanceHandle.nvmlDevice == nullptr);
                CHECK(computeInstanceHandle.nvmlIndex == gpuId);
            }
        }
    }
}

SCENARIO("CacheManager: MIG entity enumeration and formatting helpers expose fake hierarchy state")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with active and detached fake GPUs")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const activeGpuId              = cacheManager.AddFakeGpu();
        unsigned int const detachedGpuId            = cacheManager.AddFakeGpu();
        auto const gpuInstanceId                    = cacheManager.AddFakeInstance(activeGpuId);
        auto const computeInstanceId                = cacheManager.AddFakeComputeInstance(gpuInstanceId);
        cacheManager.m_gpus[activeGpuId].migEnabled = true;
        cacheManager.m_gpus[detachedGpuId].status   = DcgmEntityStatusDetached;
        SafeCopyTo(cacheManager.m_gpus[activeGpuId].uuid, "GPU-cache-test-uuid");
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        WHEN("entity groups and active entity filters are requested")
        {
            std::vector<unsigned int> activeGpuIds;
            std::vector<unsigned int> allGpuIds;
            std::vector<dcgmGroupEntityPair_t> gpuEntities;
            std::vector<dcgmGroupEntityPair_t> instanceEntities;
            std::vector<dcgmGroupEntityPair_t> computeInstanceEntities;
            std::vector<dcgmGroupEntityPair_t> unsupportedEntities;
            auto const activeGpuRet = cacheManager.GetGpuIds(1, activeGpuIds);
            auto const allGpuRet    = cacheManager.GetGpuIds(0, allGpuIds);
            auto const gpuRet       = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, gpuEntities);
            auto const instanceRet  = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_I, instanceEntities);
            auto const ciRet = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU_CI, computeInstanceEntities);
            auto const unsupportedRet = cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_VGPU, unsupportedEntities);

            std::vector<dcgmGroupEntityPair_t> mixedEntities {
                { DCGM_FE_GPU, activeGpuId },     { DCGM_FE_GPU, detachedGpuId }, { DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES },
                { DCGM_FE_GPU_I, gpuInstanceId }, { DCGM_FE_GPU_I, 999999 },      { DCGM_FE_GPU_CI, computeInstanceId },
                { DCGM_FE_GPU_CI, 999998 },       { DCGM_FE_SWITCH, 7 },
            };
            auto const filteredEntities = cacheManager.FilterActiveEntities(mixedEntities);

            THEN("only active GPU and MIG entities are returned while unsupported groups fail explicitly")
            {
                CHECK(activeGpuRet == DCGM_ST_OK);
                CHECK(allGpuRet == DCGM_ST_OK);
                CHECK(gpuRet == DCGM_ST_OK);
                CHECK(instanceRet == DCGM_ST_OK);
                CHECK(ciRet == DCGM_ST_OK);
                CHECK(unsupportedRet == DCGM_ST_NOT_SUPPORTED);
                CHECK(activeGpuIds == std::vector<unsigned int> { activeGpuId });
                CHECK(allGpuIds == std::vector<unsigned int> { activeGpuId, detachedGpuId });
                REQUIRE(gpuEntities.size() == 1);
                CHECK(gpuEntities[0].entityGroupId == DCGM_FE_GPU);
                CHECK(gpuEntities[0].entityId == activeGpuId);
                REQUIRE(instanceEntities.size() == 1);
                CHECK(instanceEntities[0].entityGroupId == DCGM_FE_GPU_I);
                CHECK(instanceEntities[0].entityId == gpuInstanceId);
                REQUIRE(computeInstanceEntities.size() == 1);
                CHECK(computeInstanceEntities[0].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(computeInstanceEntities[0].entityId == computeInstanceId);
                CHECK(unsupportedEntities.empty());

                std::vector<dcgmGroupEntityPair_t> expectedFiltered {
                    { DCGM_FE_GPU, activeGpuId },
                    { DCGM_FE_GPU_I, gpuInstanceId },
                    { DCGM_FE_GPU_CI, computeInstanceId },
                    { DCGM_FE_SWITCH, 7 },
                };
                REQUIRE(filteredEntities.size() == expectedFiltered.size());
                for (size_t i = 0; i < expectedFiltered.size(); ++i)
                {
                    CHECK(filteredEntities[i].entityGroupId == expectedFiltered[i].entityGroupId);
                    CHECK(filteredEntities[i].entityId == expectedFiltered[i].entityId);
                }
            }
        }

        WHEN("MIG hierarchy and entity index helpers are queried")
        {
            dcgmMigHierarchy_v2 hierarchy {};
            auto const hierarchyRet = cacheManager.PopulateMigHierarchy(hierarchy);

            unsigned int mappedGpuId {};
            DcgmNs::Mig::GpuInstanceId mappedInstanceId {};
            DcgmNs::Mig::ComputeInstanceId mappedComputeInstanceId {};
            auto const instanceIndexRet = cacheManager.GetMigIndicesForEntity(
                { DCGM_FE_GPU_I, gpuInstanceId }, &mappedGpuId, &mappedInstanceId, nullptr);
            auto const ciIndexRet = cacheManager.GetMigIndicesForEntity(
                { DCGM_FE_GPU_CI, computeInstanceId }, &mappedGpuId, &mappedInstanceId, &mappedComputeInstanceId);
            auto const nullIndexRet
                = cacheManager.GetMigIndicesForEntity({ DCGM_FE_GPU_CI, computeInstanceId }, nullptr, nullptr, nullptr);
            auto const missingIndexRet = cacheManager.GetMigIndicesForEntity(
                { DCGM_FE_GPU_CI, 999998 }, &mappedGpuId, &mappedInstanceId, &mappedComputeInstanceId);

            THEN("the hierarchy and index helpers expose the fake MIG parent-child relationships")
            {
                CHECK(hierarchyRet == DCGM_ST_OK);
                REQUIRE(hierarchy.count == 2);
                CHECK(hierarchy.entityList[0].entity.entityGroupId == DCGM_FE_GPU_I);
                CHECK(hierarchy.entityList[0].entity.entityId == gpuInstanceId);
                CHECK(hierarchy.entityList[0].parent.entityGroupId == DCGM_FE_GPU);
                CHECK(hierarchy.entityList[0].parent.entityId == activeGpuId);
                CHECK(hierarchy.entityList[1].entity.entityGroupId == DCGM_FE_GPU_CI);
                CHECK(hierarchy.entityList[1].entity.entityId == computeInstanceId);
                CHECK(hierarchy.entityList[1].parent.entityGroupId == DCGM_FE_GPU_I);
                CHECK(hierarchy.entityList[1].parent.entityId == gpuInstanceId);

                CHECK(instanceIndexRet == DCGM_ST_OK);
                CHECK(ciIndexRet == DCGM_ST_OK);
                CHECK(mappedGpuId == activeGpuId);
                CHECK(mappedInstanceId.id == gpuInstanceId);
                CHECK(mappedComputeInstanceId.id == computeInstanceId);
                CHECK(nullIndexRet == DCGM_ST_BADPARAM);
                CHECK(missingIndexRet == DCGM_ST_NO_DATA);
            }
        }

        WHEN("CUDA_VISIBLE_DEVICES values are generated for valid and invalid entities")
        {
            std::stringstream gpuValue;
            std::stringstream instanceValue;
            std::stringstream computeInstanceValue;
            std::stringstream invalidGpuValue;
            std::stringstream missingComputeInstanceValue;
            std::stringstream unsupportedValue;
            cacheManager.GenerateCudaVisibleDevicesValue(activeGpuId, DCGM_FE_GPU, activeGpuId, gpuValue);
            cacheManager.GenerateCudaVisibleDevicesValue(activeGpuId, DCGM_FE_GPU_I, gpuInstanceId, instanceValue);
            cacheManager.GenerateCudaVisibleDevicesValue(
                activeGpuId, DCGM_FE_GPU_CI, computeInstanceId, computeInstanceValue);
            cacheManager.GenerateCudaVisibleDevicesValue(
                activeGpuId, DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES, invalidGpuValue);
            cacheManager.GenerateCudaVisibleDevicesValue(
                activeGpuId, DCGM_FE_GPU_CI, 999998, missingComputeInstanceValue);
            cacheManager.GenerateCudaVisibleDevicesValue(activeGpuId, DCGM_FE_SWITCH, 7, unsupportedValue);

            THEN("valid entities format GPU and MIG identifiers and invalid entities produce diagnostics")
            {
                CHECK(gpuValue.str() == "MIG-GPU-cache-test-uuid");
                CHECK(instanceValue.str().find("MIG-GPU-cache-test-uuid/") == 0);
                CHECK(computeInstanceValue.str().find("MIG-GPU-cache-test-uuid/") == 0);
                CHECK(invalidGpuValue.str() == "Invalid GPU id: 32");
                CHECK_FALSE(missingComputeInstanceValue.str().empty());
                CHECK(unsupportedValue.str() == "Unsupported");
            }
        }
    }
}


#ifdef INJECTION_LIBRARY_AVAILABLE
#endif

SCENARIO("CacheManager: GPU field reader buffers NVML errors with a blocked task runner")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU and a blocked NVML task runner")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlDriver->BlockNewTasks();

        auto runField = [&](unsigned short fieldId) {
            DcgmFvBuffer fvBuffer;
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched           = 1;
            watchInfo->monitorIntervalUsec = 1234;
            watchInfo->maxAgeUsec          = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));

            dcgmcm_update_thread_t threadCtx {};
            threadCtx.watchInfo               = watchInfo;
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = fieldId;

            auto *fieldMeta = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);

            auto ret = cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
            size_t bufferSize {};
            size_t elementCount {};
            REQUIRE(fvBuffer.GetSize(&bufferSize, &elementCount) == DCGM_ST_OK);
            return std::tuple { ret, watchInfo->lastStatus, elementCount, fieldMeta->fieldType };
        };

        WHEN("identity and PCIe fields are sampled")
        {
            struct ExpectedBlockedField
            {
                unsigned short fieldId;
                dcgmReturn_t ret;
                nvmlReturn_t lastStatus;
                char fieldType;
            };

            std::array<ExpectedBlockedField, 9> const fields {
                ExpectedBlockedField {
                    DCGM_FI_DEV_GPU_BRAND, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_STRING },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCI_BUS_ID, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_STRING },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCI_COMBINED_ID, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCI_SUBSYS_ID, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCIE_MAX_LINK_GEN, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCIE_LINK_GEN, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_PCIE_LINK_WIDTH, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_GPU_PSTATE, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64 },
            };

            THEN("each field records a deterministic error payload without real NVML")
            {
                for (auto const &field : fields)
                {
                    CAPTURE(field.fieldId);
                    auto const [ret, lastStatus, elementCount, fieldType] = runField(field.fieldId);
                    CHECK(ret == field.ret);
                    CHECK(lastStatus == field.lastStatus);
                    CHECK(elementCount > 0);
                    CHECK(fieldType == field.fieldType);
                }
            }
        }

        WHEN("thermal memory and power profile fields are sampled")
        {
            struct ExpectedBlockedField
            {
                unsigned short fieldId;
                dcgmReturn_t ret;
                nvmlReturn_t lastStatus;
                char fieldType;
                size_t elementCount;
            };

            std::array<ExpectedBlockedField, 12> const fields {
                ExpectedBlockedField {
                    DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64, 1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_MEMORY_TEMP_CELSIUS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64, 1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_MEMORY_UNREPAIRABLE, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_INT64, 1 },
                ExpectedBlockedField { DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_INT64,
                                       1 },
                ExpectedBlockedField { DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_INT64,
                                       1 },
                ExpectedBlockedField { DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_INT64,
                                       1 },
                ExpectedBlockedField { DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_INT64,
                                       1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK, DCGM_ST_BADPARAM, NVML_SUCCESS, DCGM_FT_BINARY, 1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_BOARD_POWER_PROFILE_SUPPORTED_MASK, DCGM_ST_BADPARAM, NVML_SUCCESS, DCGM_FT_BINARY, 1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_BOARD_POWER_PROFILE_ENFORCED_MASK, DCGM_ST_BADPARAM, NVML_SUCCESS, DCGM_FT_BINARY, 1 },
                ExpectedBlockedField { DCGM_FI_DEV_XID_ERROR, DCGM_ST_OK, NVML_SUCCESS, DCGM_FT_INT64, 0 },
                ExpectedBlockedField { DCGM_FI_DEV_NVLINK_ERROR, DCGM_ST_OK, NVML_SUCCESS, DCGM_FT_INT64, 0 },
            };

            THEN("each field exits through an observable cached or buffered value")
            {
                for (auto const &field : fields)
                {
                    CAPTURE(field.fieldId);
                    auto const [ret, lastStatus, elementCount, fieldType] = runField(field.fieldId);
                    CHECK(ret == field.ret);
                    CHECK(lastStatus == field.lastStatus);
                    CHECK(elementCount == field.elementCount);
                    CHECK(fieldType == field.fieldType);
                }
            }
        }

        WHEN("process accounting fields are sampled")
        {
            struct ExpectedBlockedField
            {
                unsigned short fieldId;
                dcgmReturn_t ret;
                nvmlReturn_t lastStatus;
                char fieldType;
                size_t elementCount;
            };

            std::array<ExpectedBlockedField, 8> const fields {
                ExpectedBlockedField {
                    DCGM_FI_DEV_COMPUTE_PIDS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_BINARY, 0 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_GRAPHICS_PIDS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_BINARY, 0 },
                ExpectedBlockedField { DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_BINARY,
                                       0 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_GPU_UTIL_SAMPLES, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_DOUBLE, 1 },
                ExpectedBlockedField { DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                       DCGM_ST_BADPARAM,
                                       NVML_ERROR_INVALID_ARGUMENT,
                                       DCGM_FT_DOUBLE,
                                       1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_ENC_STATS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_BINARY, 1 },
                ExpectedBlockedField {
                    DCGM_FI_DEV_FBC_STATS, DCGM_ST_BADPARAM, NVML_ERROR_INVALID_ARGUMENT, DCGM_FT_BINARY, 1 },
                ExpectedBlockedField { DCGM_FI_DEV_FBC_SESSIONS_INFO,
                                       DCGM_ST_NVML_NOT_LOADED,
                                       NVML_ERROR_UNINITIALIZED,
                                       DCGM_FT_BINARY,
                                       1 },
            };

            THEN("each path records a deterministic return code and field-value payload")
            {
                for (auto const &field : fields)
                {
                    CAPTURE(field.fieldId);
                    auto const [ret, lastStatus, elementCount, fieldType] = runField(field.fieldId);
                    CAPTURE(ret);
                    CAPTURE(lastStatus);
                    CAPTURE(elementCount);
                    CAPTURE(fieldType);
                    CHECK(ret == field.ret);
                    CHECK(lastStatus == field.lastStatus);
                    CHECK(elementCount == field.elementCount);
                    CHECK(fieldType == field.fieldType);
                }
            }
        }
    }
}
SCENARIO("CacheManager: live GPU sampling covers injected NVML success fields")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    GIVEN("an initialized cache manager backed by NVML injection")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, 0 } };

        WHEN("identity, thermal, power, PCIe, and utilization fields are sampled live")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_NAME,
                                                   DCGM_FI_DEV_GPU_UUID,
                                                   DCGM_FI_DEV_GPU_BRAND,
                                                   DCGM_FI_DEV_NVML_INDEX,
                                                   DCGM_FI_DEV_BOARD_SERIAL,
                                                   DCGM_FI_DEV_GPU_UUID,
                                                   DCGM_FI_DEV_GPU_MINOR_NUMBER,
                                                   DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY,
                                                   DCGM_FI_DEV_CPU_AFFINITY_0,
                                                   DCGM_FI_DEV_CPU_AFFINITY_1,
                                                   DCGM_FI_DEV_CPU_AFFINITY_2,
                                                   DCGM_FI_DEV_CPU_AFFINITY_3,
                                                   DCGM_FI_DEV_MEMORY_AFFINITY_0,
                                                   DCGM_FI_DEV_MEMORY_AFFINITY_1,
                                                   DCGM_FI_DEV_MEMORY_AFFINITY_2,
                                                   DCGM_FI_DEV_MEMORY_AFFINITY_3,
                                                   DCGM_FI_DEV_INFOROM_OEM_VERSION,
                                                   DCGM_FI_DEV_INFOROM_ECC_VERSION,
                                                   DCGM_FI_DEV_INFOROM_POWER_VERSION,
                                                   DCGM_FI_DEV_INFOROM_IMAGE_VERSION,
                                                   DCGM_FI_DEV_INFOROM_CHECKSUM,
                                                   DCGM_FI_DEV_INFOROM_VALID,
                                                   DCGM_FI_DEV_VBIOS_VERSION,
                                                   DCGM_FI_DEV_PCI_BUS_ID,
                                                   DCGM_FI_DEV_PCI_COMBINED_ID,
                                                   DCGM_FI_DEV_PCI_SUBSYS_ID,
                                                   DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_GPU_TEMP_MARGIN_CELSIUS,
                                                   DCGM_FI_DEV_MEMORY_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_MEMORY_MAX_OP_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_GPU_MAX_OP_TEMP_CELSIUS,
                                                   DCGM_FI_DEV_GPU_TEMP_SLOWDOWN_CELSIUS,
                                                   DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS,
                                                   DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                   DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                                   DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_RAW_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_LIMIT_MIN_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_LIMIT_MAX_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_LIMIT_DEFAULT_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_LIMIT_ENFORCED_WATTS,
                                                   DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK,
                                                   DCGM_FI_DEV_BOARD_POWER_PROFILE_ENFORCED_MASK,
                                                   DCGM_FI_DEV_BOARD_POWER_PROFILE_SUPPORTED_MASK,
                                                   DCGM_FI_DEV_GPU_UTIL,
                                                   DCGM_FI_DEV_MEM_COPY_UTIL,
                                                   DCGM_FI_DEV_ENC_UTIL,
                                                   DCGM_FI_DEV_DEC_UTIL,
                                                   DCGM_FI_DEV_FB_TOTAL,
                                                   DCGM_FI_DEV_FB_FREE,
                                                   DCGM_FI_DEV_FB_USED,
                                                   DCGM_FI_DEV_FB_RESERVED,
                                                   DCGM_FI_DEV_FB_USED_RATIO,
                                                   DCGM_FI_DEV_BAR1_TOTAL,
                                                   DCGM_FI_DEV_BAR1_USED,
                                                   DCGM_FI_DEV_BAR1_FREE,
                                                   DCGM_FI_DEV_ECC_MODE,
                                                   DCGM_FI_DEV_ECC_PENDING,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_TOTAL,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_TOTAL,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_L1,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_L1,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_L2,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_L2,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_DEV,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_REG,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_REG,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_TEX,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_TEX,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_SHM,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_SHM,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_CBU,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_CBU,
                                                   DCGM_FI_DEV_ECC_SBE_VOL_SRM,
                                                   DCGM_FI_DEV_ECC_DBE_VOL_SRM,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_L1,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_L1,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_L2,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_L2,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_DEV,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_DEV,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_REG,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_REG,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_TEX,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_TEX,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_SHM,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_SHM,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_CBU,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_CBU,
                                                   DCGM_FI_DEV_ECC_SBE_AGG_SRM,
                                                   DCGM_FI_DEV_ECC_DBE_AGG_SRM,
                                                   DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL,
                                                   DCGM_FI_DEV_PAGE_RETIRED_DBE_TOTAL,
                                                   DCGM_FI_DEV_PAGE_RETIRED_PENDING,
                                                   DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL,
                                                   DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL,
                                                   DCGM_FI_DEV_ROW_REMAP_FAILED,
                                                   DCGM_FI_DEV_ROW_REMAP_PENDING,
                                                   DCGM_FI_DEV_PCIE_MAX_LINK_GEN,
                                                   DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH,
                                                   DCGM_FI_DEV_PCIE_LINK_GEN,
                                                   DCGM_FI_DEV_PCIE_LINK_WIDTH,
                                                   DCGM_FI_DEV_PCIE_REPLAY_TOTAL,
                                                   DCGM_FI_DEV_GPU_PSTATE,
                                                   DCGM_FI_DEV_SM_CLOCK,
                                                   DCGM_FI_DEV_MEM_CLOCK,
                                                   DCGM_FI_DEV_VIDEO_CLOCK,
                                                   DCGM_FI_DEV_CLOCKS_SUPPORTED,
                                                   DCGM_FI_DEV_CLOCKS_EVENT_REASONS,
                                                   DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE,
                                                   DCGM_FI_DEV_CLOCKS_SUPPORTED,
                                                   DCGM_FI_DEV_POWER_VIOLATION,
                                                   DCGM_FI_DEV_THERMAL_VIOLATION,
                                                   DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                                   DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                                   DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                                   DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_CRC_ERROR_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_RECOVERY_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_REPLAY_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL,
                                                   DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL,
                                                   DCGM_FI_DEV_MAX_SM_CLOCK,
                                                   DCGM_FI_DEV_MAX_MEM_CLOCK,
                                                   DCGM_FI_DEV_MAX_VIDEO_CLOCK,
                                                   DCGM_FI_DEV_FAN_SPEED,
                                                   DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                                   DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                                   DCGM_FI_DEV_APP_SM_CLOCK,
                                                   DCGM_FI_DEV_APP_MEM_CLOCK,
                                                   DCGM_FI_DEV_GPU_COMPUTE_MODE,
                                                   DCGM_FI_DEV_GPU_PERSISTENCE_MODE,
                                                   DCGM_FI_DEV_MIG_MODE,
                                                   DCGM_FI_DEV_CC_MODE,
                                                   DCGM_FI_DEV_GPM_SUPPORT,
                                                   DCGM_FI_DEV_MIG_MAX_SLICES,
                                                   DCGM_FI_DEV_FABRIC_MANAGER_STATUS,
                                                   DCGM_FI_DEV_FABRIC_MANAGER_ERROR,
                                                   DCGM_FI_DEV_FABRIC_CLUSTER_UUID,
                                                   DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                                                   DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY,
                                                   DCGM_FI_DEV_FABRIC_CLIQUE_ID,
                                                   DCGM_FI_DEV_GPU_VIRTUAL_MODE,
                                                   DCGM_FI_DEV_ENC_STATS,
                                                   DCGM_FI_DEV_FBC_STATS,
                                                   DCGM_FI_DEV_FBC_SESSIONS_INFO,
                                                   DCGM_FI_DEV_SRAM_EXCEEDED,
                                                   DCGM_FI_DEV_PAGE_RETIRED_SBE_TOTAL,
                                                   DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX,
                                                   DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH,
                                                   DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL,
                                                   DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW,
                                                   DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE };
            for (unsigned short fieldId = DCGM_FI_DEV_NVLINK_THROUGHPUT_L0;
                 fieldId <= DCGM_FI_DEV_NVLINK_THROUGHPUT_TOTAL;
                 ++fieldId)
            {
                fieldIds.push_back(fieldId);
            }
            for (unsigned short fieldId = DCGM_FI_DEV_NVLINK_THROUGHPUT_L0_BYTES_PER_SECOND;
                 fieldId <= DCGM_FI_DEV_NVLINK_THROUGHPUT_BYTES_PER_SECOND;
                 ++fieldId)
            {
                fieldIds.push_back(fieldId);
            }
            for (unsigned short fieldId = DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_L0;
                 fieldId <= DCGM_FI_DEV_NVLINK_TX_THROUGHPUT_TOTAL;
                 ++fieldId)
            {
                fieldIds.push_back(fieldId);
            }
            for (unsigned short fieldId = DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_L0;
                 fieldId <= DCGM_FI_DEV_NVLINK_RX_THROUGHPUT_TOTAL;
                 ++fieldId)
            {
                fieldIds.push_back(fieldId);
            }
            for (auto fieldId :
                 { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL, DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL, DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL, DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL, DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL,
                   DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL,    DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL,    DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL,    DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL,    DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL,    DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL,   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL,   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL,   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL,
                   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL,   DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL,  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL,  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL,  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL,  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL,  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL, DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL, DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL, DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL,
                   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL, DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL })
            {
                fieldIds.push_back(fieldId);
            }
            DcgmFvBuffer fvBuffer(256 * 1024);

            auto const ret = cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);

            THEN("one observable field value is returned for each requested field")
            {
                REQUIRE(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                std::set<unsigned short> observedFields;
                while (auto *fv = fvBuffer.GetNextFv(&cursor))
                {
                    CHECK(fv->entityGroupId == DCGM_FE_GPU);
                    CHECK(fv->entityId == 0);
                    observedFields.insert(fv->fieldId);
                }

                for (auto fieldId : fieldIds)
                {
                    CAPTURE(fieldId);
                    CHECK(observedFields.contains(fieldId));
                }
            }
        }
    }
}


SCENARIO("CacheManager: lifecycle helpers preserve observable cache state")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const activeGpu            = cacheManager.AddFakeGpu();
    unsigned int const disabledGpu          = cacheManager.AddFakeGpu();
    unsigned int const detachedGpu          = cacheManager.AddFakeGpu();
    cacheManager.m_gpus[activeGpu].status   = DcgmEntityStatusOk;
    cacheManager.m_gpus[disabledGpu].status = DcgmEntityStatusDisabled;
    cacheManager.m_gpus[detachedGpu].status = DcgmEntityStatusDetached;

    GIVEN("an active GPU, a disabled GPU, and a detached GPU")
    {
        WHEN("the cache manager is paused through the public API")
        {
            auto const pauseRet = cacheManager.Pause();

            THEN("only the active GPU transitions to disabled and detached state is preserved")
            {
                REQUIRE(pauseRet == DCGM_ST_OK);
                CHECK(cacheManager.GetGpuStatus(activeGpu) == DcgmEntityStatusDisabled);
                CHECK(cacheManager.GetGpuStatus(disabledGpu) == DcgmEntityStatusDisabled);
                CHECK(cacheManager.GetGpuStatus(detachedGpu) == DcgmEntityStatusDetached);
            }
        }

        WHEN("the cache manager is resumed through the public API")
        {
            auto const resumeRet = cacheManager.Resume();

            THEN("disabled GPUs return to OK while detached GPUs remain detached")
            {
                REQUIRE(resumeRet == DCGM_ST_OK);
                CHECK(cacheManager.GetGpuStatus(activeGpu) == DcgmEntityStatusOk);
                CHECK(cacheManager.GetGpuStatus(disabledGpu) == DcgmEntityStatusOk);
                CHECK(cacheManager.GetGpuStatus(detachedGpu) == DcgmEntityStatusDetached);
            }
        }
    }

    GIVEN("watched entity and global fields with cached samples")
    {
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9201);
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                           activeGpu,
                                           DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                           1000,
                                           60.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);

        dcgmcm_sample_t gpuSample {};
        gpuSample.timestamp = 100;
        gpuSample.val.i64   = 62;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, activeGpu, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpuSample, 1)
                == DCGM_ST_OK);

        dcgmcm_sample_t globalSample {};
        globalSample.timestamp = 200;
        globalSample.val.str   = const_cast<char *>("driver-from-test");
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &globalSample, 1)
                == DCGM_ST_OK);

        WHEN("the cache is emptied")
        {
            auto const emptyRet = cacheManager.EmptyCache();

            THEN("watch metadata and cached values are cleared for both scopes")
            {
                REQUIRE(emptyRet == DCGM_ST_OK);

                dcgmcm_sample_t latest {};
                CHECK(
                    cacheManager.GetLatestSample(DCGM_FE_GPU, activeGpu, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                    == DCGM_ST_NOT_WATCHED);
                CHECK(cacheManager.GetLatestSample(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);

                dcgmcm_watch_info_t gpuSnapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(
                            DCGM_FE_GPU, activeGpu, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &gpuSnapshot)
                        == DCGM_ST_OK);
                CHECK(gpuSnapshot.isWatched == 0);
                CHECK(gpuSnapshot.watchers.empty());

                auto *globalWatch = cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 0);
                REQUIRE(globalWatch != nullptr);
                CHECK(globalWatch->isWatched == 0);
            }
        }
    }
}


SCENARIO("CacheManager: watch metadata update helper validates inputs and updates retention")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager with a fake GPU field watch record")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        auto *watchInfo          = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9401);

        WHEN("the helper is called with a null watch pointer")
        {
            auto const ret = cacheManager.UpdateFieldWatch(nullptr, 1000, 60.0, 2, watcher);

            THEN("the request is rejected before any watch state is needed")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the helper is called for an unwatched field")
        {
            watchInfo->isWatched           = 0;
            watchInfo->monitorIntervalUsec = 7;
            watchInfo->maxAgeUsec          = 9;

            auto const ret = cacheManager.UpdateFieldWatch(watchInfo, 2000, 30.0, 4, watcher);

            THEN("the request reports not watched and preserves the existing timing metadata")
            {
                CHECK(ret == DCGM_ST_NOT_WATCHED);
                CHECK(watchInfo->monitorIntervalUsec == 7);
                CHECK(watchInfo->maxAgeUsec == 9);
            }
        }

        WHEN("the helper is called for a watched field")
        {
            watchInfo->isWatched           = 1;
            watchInfo->monitorIntervalUsec = 7;
            watchInfo->maxAgeUsec          = 9;

            auto const ret = cacheManager.UpdateFieldWatch(watchInfo, 3000, 45.0, 5, watcher);

            THEN("the monitor interval and retention age are updated consistently")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(watchInfo->monitorIntervalUsec == 3000);
                CHECK(watchInfo->maxAgeUsec > 0);
                CHECK(watchInfo->maxAgeUsec != 9);
            }
        }
    }
}

SCENARIO("CacheManager: injected and buffered samples cover boundary field types")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();

    GIVEN("string and blob sample requests with invalid payload pointers")
    {
        WHEN("invalid string and blob samples are injected")
        {
            dcgmcm_sample_t badString {};
            badString.timestamp = 1000;
            badString.val.str   = nullptr;

            dcgmcm_sample_t badBlob {};
            badBlob.timestamp    = 2000;
            badBlob.val.blob     = nullptr;
            badBlob.val2.ptrSize = 8;

            auto const stringRet = cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &badString, 1);
            auto const blobRet
                = cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, &badBlob, 1);

            THEN("bad parameters are reported without creating readable samples")
            {
                CHECK(stringRet == DCGM_ST_BADPARAM);
                CHECK(blobRet == DCGM_ST_BADPARAM);

                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
                CHECK(cacheManager.GetLatestSample(
                          DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }
    }

    GIVEN("a field-value buffer with unsupported and unwatched entries")
    {
        DcgmFvBuffer fvBuffer;
        REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 73, 3000, DCGM_ST_OK)
                != nullptr);

        WHEN("module-pushed samples are appended")
        {
            auto const appendRet = cacheManager.AppendSamples(&fvBuffer);

            THEN("unwatched entity entries are ignored and unknown fields do not poison the call")
            {
                REQUIRE(appendRet == DCGM_ST_OK);
                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }
    }
}

SCENARIO("CacheManager: NvLink wrappers validate cheap boundary paths")
{
    ScopedDcgmFields dcgmFields;

    TestDcgmCacheManager cacheManager;
    unsigned int const activeGpu            = cacheManager.AddFakeGpu();
    unsigned int const detachedGpu          = cacheManager.AddFakeGpu();
    cacheManager.m_gpus[activeGpu].status   = DcgmEntityStatusOk;
    cacheManager.m_gpus[detachedGpu].status = DcgmEntityStatusDetached;

    GIVEN("a P2P NvLink status request that names only detached GPUs")
    {
        dcgmNvLinkP2PStatus_t status {};
        status.numGpus          = 1;
        status.gpus[0].entityId = detachedGpu;

        WHEN("the P2P status is populated")
        {
            auto const ret = cacheManager.CreateAllNvlinksP2PStatus(status);

            THEN("the detached GPU is reported as not supported without calling NVML")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(status.numGpus == 1);
                CHECK(status.gpus[0].entityId == detachedGpu);
                for (unsigned int gpuIndex = 0; gpuIndex < DCGM_MAX_NUM_DEVICES; ++gpuIndex)
                {
                    CAPTURE(gpuIndex);
                    CHECK(status.gpus[0].linkStatus[gpuIndex] == DcgmNvLinkP2pStatusNotSupported);
                }
            }
        }
    }
}

SCENARIO("CacheManager: uncovered local branches remain hardware-free and observable")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("watched double and blob fields with module-pushed field values")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();

        auto *powerWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, 1);
        auto *blobWatch  = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, 1);
        REQUIRE(powerWatch != nullptr);
        REQUIRE(blobWatch != nullptr);
        powerWatch->isWatched  = 1;
        powerWatch->lastStatus = NVML_SUCCESS;
        blobWatch->isWatched   = 1;
        blobWatch->lastStatus  = NVML_SUCCESS;

        dcgm_watch_watcher_info_t powerWatcher {};
        powerWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 9101);
        powerWatcher.monitorIntervalUsec = 1000;
        powerWatcher.maxAgeUsec          = 1000000;
        powerWatcher.isSubscribed        = 1;
        bool powerWatcherAdded           = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(powerWatch, &powerWatcherAdded, &powerWatcher) == DCGM_ST_OK);

        dcgm_watch_watcher_info_t blobWatcher {};
        blobWatcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 9102);
        blobWatcher.monitorIntervalUsec = 1000;
        blobWatcher.maxAgeUsec          = 1000000;
        blobWatcher.isSubscribed        = 1;
        bool blobWatcherAdded           = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(blobWatch, &blobWatcherAdded, &blobWatcher) == DCGM_ST_OK);

        dcgmDevicePidAccountingStats_t accountingStats {};
        accountingStats.version           = dcgmDevicePidAccountingStats_version;
        accountingStats.pid               = 4242;
        accountingStats.gpuUtilization    = 91;
        accountingStats.memoryUtilization = 27;

        DcgmFvBuffer fvBuffer;
        REQUIRE(fvBuffer.AddDoubleValue(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, 225.5, 1100, DCGM_ST_OK)
                != nullptr);
        REQUIRE(fvBuffer.AddBlobValue(DCGM_FE_GPU,
                                      gpuId,
                                      DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS,
                                      &accountingStats,
                                      sizeof(accountingStats),
                                      1200,
                                      DCGM_ST_OK)
                != nullptr);

        WHEN("the field values are appended")
        {
            auto const appendRet = cacheManager.AppendSamples(&fvBuffer);

            THEN("module-pushed values are accepted without requiring hardware access")
            {
                REQUIRE(appendRet == DCGM_ST_OK);

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *power       = fvBuffer.GetNextFv(&cursor);
                REQUIRE(power != nullptr);
                CHECK(power->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(power->fieldType == DCGM_FT_DOUBLE);
                CHECK_THAT(power->value.dbl, Catch::Matchers::WithinAbs(225.5, 0.0001));

                dcgmBufferedFv_t *blob = fvBuffer.GetNextFv(&cursor);
                REQUIRE(blob != nullptr);
                CHECK(blob->fieldId == DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS);
                CHECK(blob->fieldType == DCGM_FT_BINARY);
                REQUIRE(blob->length >= sizeof(accountingStats));
                auto const *cachedStats = reinterpret_cast<dcgmDevicePidAccountingStats_t const *>(blob->value.blob);
                CHECK(cachedStats->pid == accountingStats.pid);
                CHECK(cachedStats->gpuUtilization == accountingStats.gpuUtilization);
                CHECK(cachedStats->memoryUtilization == accountingStats.memoryUtilization);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }

    GIVEN("a fake GPU and SetValueImpl inputs that pass local validation")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        dcgmcm_sample_t sample {};
        sample.val.i64 = 76;

        WHEN("an unsupported settable field is requested")
        {
            auto const ret = cacheManager.SetValueImpl(gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample);

            THEN("the request fails as unimplemented without mutating a readable sample")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);

                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }
    }

    GIVEN("detached GPUs and local helper requests")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0          = cacheManager.AddFakeGpu();
        unsigned int const gpu1          = cacheManager.AddFakeGpu();
        cacheManager.m_gpus[gpu0].status = DcgmEntityStatusDetached;
        cacheManager.m_gpus[gpu1].status = DcgmEntityStatusDetached;

        WHEN("P2P status is requested for all GPUs")
        {
            dcgmNvLinkP2PStatus_t status {};
            status.numGpus = 0;

            auto const ret = cacheManager.CreateAllNvlinksP2PStatus(status);

            THEN("all detached GPUs are returned as not supported without NVML")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(status.numGpus == 2);
                CHECK(status.gpus[0].entityId == gpu0);
                CHECK(status.gpus[1].entityId == gpu1);

                for (unsigned int statusIndex = 0; statusIndex < status.numGpus; ++statusIndex)
                {
                    for (unsigned int peerIndex = 0; peerIndex < DCGM_MAX_NUM_DEVICES; ++peerIndex)
                    {
                        CAPTURE(statusIndex, peerIndex);
                        CHECK(status.gpus[statusIndex].linkStatus[peerIndex] == DcgmNvLinkP2pStatusNotSupported);
                    }
                }
            }
        }

        WHEN("workload power profile info is requested for a detached GPU")
        {
            dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo {};
            dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus {};

            auto const ret = cacheManager.GetWorkloadPowerProfilesInfo(gpu0, &profilesInfo, &profilesStatus);

            THEN("the local status check reports the GPU as lost")
            {
                CHECK(ret == DCGM_ST_GPU_IS_LOST);
                CHECK(profilesInfo.version == 0);
                CHECK(profilesStatus.version == 0);
            }
        }
    }
}


SCENARIO("CacheManager: broad sample and watch contracts stay deterministic for fake entities")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with watched int, double, string, and global fields")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9301);
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        auto watchField = [&](dcgm_field_entity_group_t entityGroupId,
                              dcgm_field_eid_t entityId,
                              unsigned short fieldId,
                              bool subscribe) {
            bool wereFirstWatcher = false;
            REQUIRE(cacheManager.AddFieldWatch(
                        entityGroupId, entityId, fieldId, 1000, 30.0, 8, watcher, subscribe, false, wereFirstWatcher)
                    == DCGM_ST_OK);
            REQUIRE(wereFirstWatcher);

            auto *watchInfo = entityGroupId == DCGM_FE_NONE
                                  ? cacheManager.GetGlobalWatchInfo(fieldId, 0)
                                  : cacheManager.GetEntityWatchInfo(entityGroupId, entityId, fieldId, 0);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched = 1;
        };

        watchField(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, true);
        watchField(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, false);
        watchField(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, false);
        watchField(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, false);

        WHEN("multiple sample types are injected and read back by range")
        {
            timelib64_t const baseTimestamp = timelib_usecSince1970() + 1000000;

            std::array<dcgmcm_sample_t, 4> tempSamples {};
            tempSamples[0].timestamp = baseTimestamp + 100;
            tempSamples[0].val.i64   = 60;
            tempSamples[1].timestamp = baseTimestamp + 200;
            tempSamples[1].val.i64   = 61;
            tempSamples[2].timestamp = baseTimestamp + 300;
            tempSamples[2].val.i64   = 62;
            tempSamples[3].timestamp = baseTimestamp + 400;
            tempSamples[3].val.i64   = 63;

            std::array<dcgmcm_sample_t, 2> powerSamples {};
            powerSamples[0].timestamp = baseTimestamp + 125;
            powerSamples[0].val.d     = 250.5;
            powerSamples[1].timestamp = baseTimestamp + 325;
            powerSamples[1].val.d     = 275.25;

            dcgmcm_sample_t nameSample {};
            nameSample.timestamp = baseTimestamp + 150;
            nameSample.val.str   = const_cast<char *>("fake-cache-gpu");

            dcgmcm_sample_t driverSample {};
            driverSample.timestamp = baseTimestamp + 175;
            driverSample.val.str   = const_cast<char *>("535.104.05");

            REQUIRE(cacheManager.InjectSamples(
                        DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, tempSamples.data(), tempSamples.size())
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.InjectSamples(
                        DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, powerSamples.data(), powerSamples.size())
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &nameSample, 1) == DCGM_ST_OK);
            REQUIRE(cacheManager.InjectSamples(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &driverSample, 1)
                    == DCGM_ST_OK);

            THEN("ascending and descending requests honor time bounds and caller capacity")
            {
                std::array<dcgmcm_sample_t, 3> ascSamples {};
                int ascCount = static_cast<int>(ascSamples.size());
                REQUIRE(cacheManager.GetSamples(DCGM_FE_GPU,
                                                gpuId,
                                                DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                ascSamples.data(),
                                                &ascCount,
                                                baseTimestamp + 150,
                                                baseTimestamp + 350,
                                                DCGM_ORDER_ASCENDING,
                                                nullptr)
                        == DCGM_ST_OK);
                REQUIRE(ascCount == 2);
                CHECK(ascSamples[0].timestamp == baseTimestamp + 200);
                CHECK(ascSamples[0].val.i64 == 61);
                CHECK(ascSamples[1].timestamp == baseTimestamp + 300);
                CHECK(ascSamples[1].val.i64 == 62);

                std::array<dcgmcm_sample_t, 1> descSamples {};
                int descCount = static_cast<int>(descSamples.size());
                REQUIRE(cacheManager.GetSamples(DCGM_FE_GPU,
                                                gpuId,
                                                DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                descSamples.data(),
                                                &descCount,
                                                0,
                                                baseTimestamp + 400,
                                                DCGM_ORDER_DESCENDING,
                                                nullptr)
                        == DCGM_ST_OK);
                REQUIRE(descCount == 1);
                CHECK(descSamples[0].timestamp == baseTimestamp + 400);
                CHECK(descSamples[0].val.i64 == 63);

                std::array<dcgmcm_sample_t, 2> emptyRange {};
                int emptyCount = static_cast<int>(emptyRange.size());
                CHECK(cacheManager.GetSamples(DCGM_FE_GPU,
                                              gpuId,
                                              DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                              emptyRange.data(),
                                              &emptyCount,
                                              baseTimestamp + 450,
                                              baseTimestamp + 500,
                                              DCGM_ORDER_ASCENDING,
                                              nullptr)
                      == DCGM_ST_NO_DATA);
                CHECK(emptyCount == 0);
            }

            THEN("field-value buffers preserve entity, type, status, and latest values")
            {
                DcgmFvBuffer fvBuffer;
                int fvCount = 8;
                REQUIRE(cacheManager.GetSamples(DCGM_FE_GPU,
                                                gpuId,
                                                DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                nullptr,
                                                &fvCount,
                                                0,
                                                0,
                                                DCGM_ORDER_ASCENDING,
                                                &fvBuffer)
                        == DCGM_ST_OK);
                REQUIRE(fvCount == 2);

                dcgmBufferedFvCursor_t cursor = 0;
                dcgmBufferedFv_t *first       = fvBuffer.GetNextFv(&cursor);
                REQUIRE(first != nullptr);
                CHECK(first->entityGroupId == DCGM_FE_GPU);
                CHECK(first->entityId == gpuId);
                CHECK(first->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(first->fieldType == DCGM_FT_DOUBLE);
                CHECK(first->status == DCGM_ST_OK);
                CHECK_THAT(first->value.dbl, Catch::Matchers::WithinAbs(250.5, 0.0001));

                dcgmBufferedFv_t *second = fvBuffer.GetNextFv(&cursor);
                REQUIRE(second != nullptr);
                CHECK(second->timestamp == baseTimestamp + 325);
                CHECK_THAT(second->value.dbl, Catch::Matchers::WithinAbs(275.25, 0.0001));
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);

                dcgmcm_sample_t latestName {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &latestName, nullptr)
                        == DCGM_ST_OK);
                REQUIRE(latestName.val.str != nullptr);
                CHECK(std::string_view(latestName.val.str) == "fake-cache-gpu");
                REQUIRE(cacheManager.FreeSamples(&latestName, 1, DCGM_FI_DEV_GPU_NAME) == DCGM_ST_OK);

                dcgmcm_sample_t latestDriver {};
                REQUIRE(cacheManager.GetLatestSample(
                            DCGM_FE_GPU, gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, &latestDriver, nullptr)
                        == DCGM_ST_OK);
                REQUIRE(latestDriver.val.str != nullptr);
                CHECK(std::string_view(latestDriver.val.str) == "535.104.05");
                REQUIRE(cacheManager.FreeSamples(&latestDriver, 1, DCGM_FI_SYSTEM_DRIVER_VERSION) == DCGM_ST_OK);
            }
        }

        WHEN("watch metadata is queried and removed through public APIs")
        {
            dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
            fieldInfo.version       = dcgmCacheManagerFieldInfo_version4;
            fieldInfo.entityGroupId = DCGM_FE_GPU;
            fieldInfo.entityId      = gpuId;
            fieldInfo.fieldId       = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

            auto const infoRet = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);
            auto const removeRet
                = cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcher);
            dcgmcm_watch_info_t snapshot {};
            auto const snapshotRet
                = cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &snapshot);

            THEN("watcher details are observable and removal clears watched state while preserving the watch object")
            {
                REQUIRE(infoRet == DCGM_ST_OK);
                CHECK((fieldInfo.flags & DCGM_CMI_F_WATCHED) != 0);
                REQUIRE(fieldInfo.numWatchers == 1);
                CHECK(fieldInfo.watchers[0].watcherType == DcgmWatcherTypeClient);
                CHECK(fieldInfo.watchers[0].connectionId == 9301);
                CHECK(fieldInfo.watchers[0].monitorIntervalUsec == 1000);
                CHECK(fieldInfo.watchers[0].maxAgeUsec == 30000000);

                REQUIRE(removeRet == DCGM_ST_OK);
                REQUIRE(snapshotRet == DCGM_ST_OK);
                CHECK(snapshot.isWatched == 0);
                CHECK(snapshot.watchers.empty());
                CHECK(snapshot.hasSubscribedWatchers == 0);
            }
        }
    }

    GIVEN("invalid sample requests against an otherwise empty cache manager")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        dcgmcm_sample_t sample {};
        int count = 1;

        WHEN("caller parameters and field ids are invalid")
        {
            auto const nullCountRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                              gpuId,
                                                              DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                              &sample,
                                                              nullptr,
                                                              0,
                                                              0,
                                                              DCGM_ORDER_ASCENDING,
                                                              nullptr);
            auto const noOutputRet  = cacheManager.GetSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr, &count, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
            count                  = 1;
            auto const badOrderRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                             gpuId,
                                                             DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                             &sample,
                                                             &count,
                                                             0,
                                                             0,
                                                             static_cast<dcgmOrder_t>(99),
                                                             nullptr);
            auto const unknownFieldRet
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_MAX_FIELDS, &sample, nullptr);
            auto const badInjectRet = cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, 0, &sample, 1);

            THEN("the public APIs reject each single-fault invalid request")
            {
                CHECK(nullCountRet == DCGM_ST_BADPARAM);
                CHECK(noOutputRet == DCGM_ST_BADPARAM);
                CHECK(badOrderRet == DCGM_ST_BADPARAM);
                CHECK(unknownFieldRet == DCGM_ST_UNKNOWN_FIELD);
                CHECK(badInjectRet == DCGM_ST_BADPARAM);
            }
        }
    }
}


SCENARIO("CacheManager: global sample paths normalize entity metadata and skip invalid rows")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a global field watched through a GPU-scoped request with more watchers than field info can report")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        for (int watcherIndex = 0; watcherIndex < DCGM_CM_FIELD_INFO_NUM_WATCHERS + 2; ++watcherIndex)
        {
            bool wereFirstWatcher = false;
            DcgmWatcher const watcher(DcgmWatcherTypeClient, 9900 + watcherIndex);
            REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                               gpuId,
                                               DCGM_FI_SYSTEM_DRIVER_VERSION,
                                               1000 + watcherIndex,
                                               30.0,
                                               4,
                                               watcher,
                                               watcherIndex % 2 == 0,
                                               false,
                                               wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        DcgmFvBuffer fvBuffer;
        REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, gpuId, DCGM_FI_MAX_FIELDS, 111, 4000, DCGM_ST_OK) != nullptr);
        REQUIRE(fvBuffer.AddStringValue(
                    DCGM_FE_GPU, gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, "global-driver-row", 4010, DCGM_ST_OK)
                != nullptr);

        WHEN("module-pushed values are appended and metadata is queried through the original GPU scope")
        {
            auto const appendRet = cacheManager.AppendSamples(&fvBuffer);

            dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
            fieldInfo.version       = dcgmCacheManagerFieldInfo_version4;
            fieldInfo.entityGroupId = DCGM_FE_GPU;
            fieldInfo.entityId      = gpuId;
            fieldInfo.fieldId       = DCGM_FI_SYSTEM_DRIVER_VERSION;
            auto const infoRet      = cacheManager.GetCacheManagerFieldInfo(&fieldInfo);

            THEN("invalid rows are ignored and field info reports the bounded watcher list")
            {
                REQUIRE(appendRet == DCGM_ST_OK);
                REQUIRE(infoRet == DCGM_ST_OK);
                CHECK((fieldInfo.flags & DCGM_CMI_F_WATCHED) != 0);
                CHECK(fieldInfo.numWatchers == DCGM_CM_FIELD_INFO_NUM_WATCHERS);
                CHECK(fieldInfo.numSamples == 0);
                CHECK(fieldInfo.oldestTimestamp == 0);
                CHECK(fieldInfo.newestTimestamp == 0);
                CHECK(fieldInfo.watchers[0].connectionId == 9900);
                CHECK(fieldInfo.watchers[DCGM_CM_FIELD_INFO_NUM_WATCHERS - 1].connectionId
                      == 9900 + DCGM_CM_FIELD_INFO_NUM_WATCHERS - 1);
            }
        }
    }

    GIVEN("an otherwise valid fake GPU sample injection with an unknown nonzero field id")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        dcgmcm_sample_t sample {};
        sample.timestamp = 5000;
        sample.val.i64   = 1;

        WHEN("the sample is injected")
        {
            auto const injectRet = cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_MAX_FIELDS, &sample, 1);

            THEN("the missing field metadata is reported as a generic injection failure")
            {
                CHECK(injectRet == DCGM_ST_GENERIC_ERROR);
            }
        }
    }
}


SCENARIO("CacheManager: latest-sample reads preserve cached values after watch removal")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with a watched temperature field and injected samples")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9501);
        bool wereFirstWatcher = false;
        REQUIRE(
            cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 30.0, 4, watcher, true, false, wereFirstWatcher)
            == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);

        timelib64_t const firstTimestamp = timelib_usecSince1970() + 10000;
        std::array<dcgmcm_sample_t, 3> samples {};
        samples[0].timestamp = firstTimestamp;
        samples[0].val.i64   = 61;
        samples[1].timestamp = firstTimestamp + 100;
        samples[1].val.i64   = 62;
        samples[2].timestamp = firstTimestamp + 200;
        samples[2].val.i64   = 63;
        REQUIRE(
            cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples.data(), samples.size())
            == DCGM_ST_OK);

        WHEN("the latest value is read before and after removing the only watcher")
        {
            dcgmcm_sample_t latest {};
            DcgmFvBuffer latestBuffer;
            auto const latestRet = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, &latestBuffer);
            auto const removeRet
                = cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, watcher);
            dcgmcm_sample_t afterRemove {};
            DcgmFvBuffer afterRemoveBuffer;
            auto const afterRemoveRet = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &afterRemove, &afterRemoveBuffer);

            THEN("the cached latest sample remains visible after the watcher is removed")
            {
                REQUIRE(latestRet == DCGM_ST_OK);
                CHECK(latest.timestamp == firstTimestamp + 200);
                CHECK(latest.val.i64 == 63);
                dcgmBufferedFvCursor_t latestCursor = 0;
                auto *latestFv                      = latestBuffer.GetNextFv(&latestCursor);
                REQUIRE(latestFv != nullptr);
                CHECK(latestFv->entityGroupId == DCGM_FE_GPU);
                CHECK(latestFv->entityId == gpuId);
                CHECK(latestFv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(latestFv->status == DCGM_ST_OK);
                CHECK(latestFv->value.i64 == 63);

                REQUIRE(removeRet == DCGM_ST_OK);
                REQUIRE(afterRemoveRet == DCGM_ST_OK);
                CHECK(afterRemove.timestamp == firstTimestamp + 200);
                CHECK(afterRemove.val.i64 == 63);
                dcgmBufferedFvCursor_t removedCursor = 0;
                auto *removedFv                      = afterRemoveBuffer.GetNextFv(&removedCursor);
                REQUIRE(removedFv != nullptr);
                CHECK(removedFv->entityGroupId == DCGM_FE_GPU);
                CHECK(removedFv->entityId == gpuId);
                CHECK(removedFv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(removedFv->status == DCGM_ST_OK);
                CHECK(removedFv->value.i64 == 63);
            }
        }
    }
}


SCENARIO("CacheManager: live value buffering covers detached skip-driver and NvLink edge paths")
{
    ScopedDcgmFields dcgmFields;

    auto makeThreadContext = [](DcgmFvBuffer &fvBuffer,
                                dcgmcm_watch_info_p watchInfo,
                                dcgm_field_entity_group_t entityGroupId,
                                dcgm_field_eid_t entityId,
                                unsigned short fieldId) {
        dcgmcm_update_thread_t threadCtx {};
        threadCtx.fvBuffer                = &fvBuffer;
        threadCtx.watchInfo               = watchInfo;
        threadCtx.entityKey.entityGroupId = entityGroupId;
        threadCtx.entityKey.entityId      = entityId;
        threadCtx.entityKey.fieldId       = fieldId;
        return threadCtx;
    };

    GIVEN("a fake GPU with cached identity strings and detached state")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        cacheManager.m_gpus[gpuId].status = DcgmEntityStatusDetached;
        SafeCopyTo(cacheManager.m_gpus[gpuId].uuid, "GPU-detached-cache");
        SafeCopyTo(cacheManager.m_gpus[gpuId].deviceName, "detached-cache-name");
        SafeCopyTo(cacheManager.m_gpus[gpuId].pciInfo.busId, "00000000:FE:00.0");

        auto readField = [&](unsigned short fieldId, DcgmFvBuffer &fvBuffer) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));
            auto *fieldMeta       = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);
            auto threadCtx = makeThreadContext(fvBuffer, watchInfo, DCGM_FE_GPU, gpuId, fieldId);

            return cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        };

        WHEN("cached identity fields and an uncached scalar field are read")
        {
            DcgmFvBuffer uuidBuffer;
            DcgmFvBuffer nameBuffer;
            DcgmFvBuffer pciBuffer;
            DcgmFvBuffer tempBuffer;
            auto const uuidRet = readField(DCGM_FI_DEV_GPU_UUID, uuidBuffer);
            auto const nameRet = readField(DCGM_FI_DEV_GPU_NAME, nameBuffer);
            auto const pciRet  = readField(DCGM_FI_DEV_PCI_BUS_ID, pciBuffer);
            auto const tempRet = readField(DCGM_FI_DEV_GPU_TEMP_CELSIUS, tempBuffer);

            THEN("detached GPUs return cached identity strings and blank scalar values without NVML calls")
            {
                auto checkString = [](DcgmFvBuffer &buffer, char const *expected) {
                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = buffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->status == DCGM_ST_OK);
                    CHECK(fv->fieldType == DCGM_FT_STRING);
                    REQUIRE(fv->value.str != nullptr);
                    CHECK(std::string_view(fv->value.str) == expected);
                    CHECK(buffer.GetNextFv(&cursor) == nullptr);
                };

                CHECK(uuidRet == DCGM_ST_OK);
                CHECK(nameRet == DCGM_ST_OK);
                CHECK(pciRet == DCGM_ST_OK);
                checkString(uuidBuffer, "GPU-detached-cache");
                checkString(nameBuffer, "detached-cache-name");
                checkString(pciBuffer, "00000000:FE:00.0");

                CHECK(tempRet == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                auto *tempFv                  = tempBuffer.GetNextFv(&cursor);
                REQUIRE(tempFv != nullptr);
                CHECK(tempFv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(tempFv->fieldType == DCGM_FT_INT64);
                CHECK(tempFv->value.i64 == DCGM_INT64_BLANK);
                CHECK(tempBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }

    GIVEN("a fake GPU with driver calls skipped")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_skipDriverCalls.store(true, std::memory_order_release);
        DcgmNs::Defer skipGuard(
            [&cacheManager] { cacheManager.m_skipDriverCalls.store(false, std::memory_order_release); });

        WHEN("non-mapped GPU and global fields are read through the live value helper")
        {
            std::array<unsigned short, 3> fieldIds {
                DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                DCGM_FI_DEV_GPU_NAME,
                DCGM_FI_SYSTEM_DRIVER_VERSION,
            };

            THEN("each field receives a deterministic blank/error payload")
            {
                for (auto fieldId : fieldIds)
                {
                    CAPTURE(fieldId);
                    DcgmFvBuffer fvBuffer;
                    auto *fieldMeta = DcgmFieldGetById(fieldId);
                    REQUIRE(fieldMeta != nullptr);
                    dcgm_field_entity_group_t const entityGroupId
                        = fieldMeta->scope == DCGM_FS_GLOBAL ? DCGM_FE_NONE : DCGM_FE_GPU;
                    dcgm_field_eid_t const entityId = entityGroupId == DCGM_FE_NONE ? 0 : gpuId;
                    auto *watchInfo = entityGroupId == DCGM_FE_NONE
                                          ? cacheManager.GetGlobalWatchInfo(fieldId, 1)
                                          : cacheManager.GetEntityWatchInfo(entityGroupId, entityId, fieldId, 1);
                    REQUIRE(watchInfo != nullptr);
                    watchInfo->isWatched  = 1;
                    watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));
                    auto threadCtx        = makeThreadContext(fvBuffer, watchInfo, entityGroupId, entityId, fieldId);

                    auto const ret = cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);

                    CHECK(ret == DCGM_ST_NVML_DRIVER_TIMEOUT);
                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->entityGroupId == entityGroupId);
                    CHECK(fv->entityId == entityId);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
                }
            }
        }
    }

    GIVEN("packed NvLink entity ids for unsupported local paths")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        cacheManager.m_gpus[gpuId].arch = DCGM_CHIP_ARCH_AMPERE;

        auto readNvLinkBer = [&](dcgm_link_t link, unsigned short fieldId, DcgmFvBuffer &fvBuffer) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(30));
            auto *fieldMeta       = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);
            auto threadCtx = makeThreadContext(fvBuffer, watchInfo, DCGM_FE_LINK, link.raw, fieldId);

            return cacheManager.ReadAndCacheNvLinkCountField(threadCtx, fieldMeta);
        };

        WHEN("the link type is not GPU")
        {
            dcgm_link_t switchLink {};
            switchLink.parsed.type     = DCGM_FE_SWITCH;
            switchLink.parsed.switchId = 7;
            switchLink.parsed.index    = 1;

            DcgmFvBuffer fvBuffer;
            auto const ret = readNvLinkBer(switchLink, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, fvBuffer);

            THEN("the helper writes a not-supported float payload and returns success")
            {
                CHECK(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_LINK);
                CHECK(fv->entityId == switchLink.raw);
                CHECK(fv->fieldId == DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO);
                CHECK(fv->fieldType == DCGM_FT_DOUBLE);
                CHECK(fv->value.dbl == DCGM_FP64_NOT_SUPPORTED);
            }
        }

        WHEN("a GPU link requests NVLink5-only BER on an older architecture")
        {
            dcgm_link_t gpuLink {};
            gpuLink.parsed.type  = DCGM_FE_GPU;
            gpuLink.parsed.gpuId = gpuId;
            gpuLink.parsed.index = 0;

            DcgmFvBuffer fvBuffer;
            auto const ret = readNvLinkBer(gpuLink, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO, fvBuffer);

            THEN("the helper reports not-supported without querying NVML")
            {
                CHECK(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_LINK);
                CHECK(fv->entityId == gpuLink.raw);
                CHECK(fv->fieldId == DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO);
                CHECK(fv->fieldType == DCGM_FT_DOUBLE);
                CHECK(fv->value.dbl == DCGM_FP64_NOT_SUPPORTED);
            }
        }
    }
}

SCENARIO("CacheManager: watcher removal and live sample error rows are explicit")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a watched fake GPU field with two independent watchers")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        auto *watchInfo          = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched = 1;

        dcgm_watch_watcher_info_t first {};
        first.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 6101);
        first.monitorIntervalUsec = 1000;
        first.maxAgeUsec          = 1000000;
        dcgm_watch_watcher_info_t second {};
        second.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 6102);
        second.monitorIntervalUsec = 2000;
        second.maxAgeUsec          = 2000000;
        bool added                 = false;
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &added, &first) == DCGM_ST_OK);
        REQUIRE(added);
        REQUIRE(cacheManager.AddOrUpdateWatcher(watchInfo, &added, &second) == DCGM_ST_OK);
        REQUIRE(added);

        WHEN("watchers are removed one at a time and an unknown watcher is requested")
        {
            dcgm_watch_watcher_info_t missing {};
            missing.watcher       = DcgmWatcher(DcgmWatcherTypeClient, 6199);
            auto const missingRet = cacheManager.RemoveWatcher(watchInfo, &missing);
            auto const firstRet   = cacheManager.RemoveWatcher(watchInfo, &first);
            auto const secondRet  = cacheManager.RemoveWatcher(watchInfo, &second);

            THEN("the watch remains active until the last watcher is removed")
            {
                CHECK(missingRet == DCGM_ST_NOT_WATCHED);
                CHECK(firstRet == DCGM_ST_OK);
                CHECK(secondRet == DCGM_ST_OK);
                CHECK(watchInfo->watchers.empty());
                CHECK(watchInfo->isWatched == 0);
            }
        }
    }

    GIVEN("live sample requests with invalid entity and unsupported field combinations")
    {
        TestDcgmCacheManager cacheManager;
        auto const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        std::vector<dcgmGroupEntityPair_t> entities {
            { DCGM_FE_GPU, DCGM_MAX_NUM_DEVICES },
            { DCGM_FE_VGPU, 99 },
        };
        std::vector<unsigned short> fieldIds {
            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
        };
        DcgmFvBuffer fvBuffer;

        WHEN("the batch live-sample helper evaluates each entity-field pair")
        {
            auto const ret = cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);

            THEN("bad entities and unsupported entity groups are represented as field-value status rows")
            {
                CHECK(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                std::vector<dcgmBufferedFv_t const *> rows;
                while (auto *fv = fvBuffer.GetNextFv(&cursor))
                {
                    rows.push_back(fv);
                }
                REQUIRE(rows.size() == entities.size() * fieldIds.size());
                CHECK(rows[0]->entityGroupId == DCGM_FE_GPU);
                CHECK(rows[0]->entityId == DCGM_MAX_NUM_DEVICES);
                CHECK(rows[0]->status == DCGM_ST_BADPARAM);
                CHECK(rows[1]->entityGroupId == DCGM_FE_GPU);
                CHECK(rows[1]->status == DCGM_ST_BADPARAM);
                CHECK(rows[2]->entityGroupId == DCGM_FE_VGPU);
                CHECK(rows[2]->status == DCGM_ST_FIELD_UNSUPPORTED_BY_API);
                CHECK(rows[3]->entityGroupId == DCGM_FE_VGPU);
                CHECK(rows[3]->status == DCGM_ST_FIELD_UNSUPPORTED_BY_API);
            }
        }
    }
}

SCENARIO("CacheManager: cached multi-latest buffers global and unwatched-field results")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with cached entity and global samples")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        timelib64_t const baseTimestamp = timelib_usecSince1970() + 50000;
        dcgmcm_sample_t tempSample {};
        tempSample.timestamp = baseTimestamp;
        tempSample.val.i64   = 71;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &tempSample, 1)
                == DCGM_ST_OK);

        char driverVersion[] = "535.104.05-cache";
        dcgmcm_sample_t driverSample {};
        driverSample.timestamp = baseTimestamp + 10;
        driverSample.val.str   = driverVersion;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &driverSample, 1)
                == DCGM_ST_OK);

        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId } };
        std::vector<unsigned short> fieldIds {
            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
            DCGM_FI_SYSTEM_DRIVER_VERSION,
            DCGM_FI_DEV_BOARD_POWER_WATTS,
        };
        DcgmFvBuffer fvBuffer;

        WHEN("cached latest values are requested for entity global and unwatched fields together")
        {
            auto const ret = cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);

            THEN("the buffer preserves the successful rows and the per-field error row")
            {
                CHECK(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;

                auto *temp = fvBuffer.GetNextFv(&cursor);
                REQUIRE(temp != nullptr);
                CHECK(temp->entityGroupId == DCGM_FE_GPU);
                CHECK(temp->entityId == gpuId);
                CHECK(temp->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(temp->status == DCGM_ST_OK);
                CHECK(temp->value.i64 == 71);

                auto *driver = fvBuffer.GetNextFv(&cursor);
                REQUIRE(driver != nullptr);
                CHECK(driver->entityGroupId == DCGM_FE_GPU);
                CHECK(driver->entityId == gpuId);
                CHECK(driver->fieldId == DCGM_FI_SYSTEM_DRIVER_VERSION);
                CHECK(driver->status == DCGM_ST_OK);
                REQUIRE(driver->value.str != nullptr);
                CHECK(std::string_view(driver->value.str) == driverVersion);

                auto *unwatched = fvBuffer.GetNextFv(&cursor);
                REQUIRE(unwatched != nullptr);
                CHECK(unwatched->entityGroupId == DCGM_FE_GPU);
                CHECK(unwatched->entityId == gpuId);
                CHECK(unwatched->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(unwatched->status == DCGM_ST_NOT_WATCHED);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}

SCENARIO("CacheManager: subscribed injections summaries and sample cleanup stay observable")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("a subscribed fake GPU watch and a registered field-value callback")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        FvSubscriptionCallbackState state {};
        dcgmcmEventSubscription_t sub {};
        sub.type     = DcgmcmEventTypeFvUpdate;
        sub.fn.fvCb  = FvSubscriptionCallback;
        sub.userData = &state;
        REQUIRE(cacheManager.SubscribeForEvent(sub) == DCGM_ST_OK);

        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9701);
        bool wereFirstWatcher = false;
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        REQUIRE(
            cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 30.0, 4, watcher, true, false, wereFirstWatcher)
            == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);

        std::array<dcgmcm_sample_t, 2> samples {};
        samples[0].timestamp = 10000;
        samples[0].val.i64   = 70;
        samples[1].timestamp = 10010;
        samples[1].val.i64   = 71;

        WHEN("multiple samples are injected into the subscribed watch")
        {
            auto const ret = cacheManager.InjectSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples.data(), static_cast<int>(samples.size()));

            THEN("the callback receives one batched notification with the subscribed watcher type")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(state.calls == 1);
                REQUIRE(state.watcherCount == 1);
                CHECK(state.watchers[0] == DcgmWatcherTypeClient);
                CHECK(state.elementCount == samples.size());

                dcgmcm_sample_t latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr)
                      == DCGM_ST_NO_DATA);
            }
        }
    }

    GIVEN("int64 and fp64 samples containing blank values")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        timelib64_t const t0 = timelib_usecSince1970() + 70000;

        std::array<dcgmcm_sample_t, 3> intSamples {};
        intSamples[0].timestamp = t0;
        intSamples[0].val.i64   = DCGM_INT64_BLANK;
        intSamples[1].timestamp = t0 + 10;
        intSamples[1].val.i64   = 5;
        intSamples[2].timestamp = t0 + 20;
        intSamples[2].val.i64   = 9;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
                                           intSamples.data(),
                                           static_cast<int>(intSamples.size()))
                == DCGM_ST_OK);

        std::array<dcgmcm_sample_t, 3> fpSamples {};
        fpSamples[0].timestamp = t0;
        fpSamples[0].val.d     = DCGM_FP64_BLANK;
        fpSamples[1].timestamp = t0 + 10;
        fpSamples[1].val.d     = 1.25;
        fpSamples[2].timestamp = t0 + 20;
        fpSamples[2].val.d     = 3.75;
        REQUIRE(
            cacheManager.InjectSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, fpSamples.data(), static_cast<int>(fpSamples.size()))
            == DCGM_ST_OK);

        WHEN("summaries are requested across the full sample range")
        {
            std::array<DcgmcmSummaryType_t, 4> summaryTypes {
                DcgmcmSummaryTypeMinimum,
                DcgmcmSummaryTypeMaximum,
                DcgmcmSummaryTypeSum,
                DcgmcmSummaryTypeCount,
            };
            std::array<long long, summaryTypes.size()> intValues {};
            std::array<double, summaryTypes.size()> fpValues {};

            auto const intRet = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                                 gpuId,
                                                                 DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
                                                                 static_cast<int>(summaryTypes.size()),
                                                                 summaryTypes.data(),
                                                                 intValues.data(),
                                                                 t0,
                                                                 t0 + 20,
                                                                 nullptr,
                                                                 nullptr);
            auto const fpRet  = cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                                gpuId,
                                                                DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                                static_cast<int>(summaryTypes.size()),
                                                                summaryTypes.data(),
                                                                fpValues.data(),
                                                                t0,
                                                                t0 + 20,
                                                                nullptr,
                                                                nullptr);

            THEN("blank values are skipped for extrema and sum while scanned count remains observable")
            {
                REQUIRE(intRet == DCGM_ST_OK);
                CHECK(intValues[0] == 5);
                CHECK(intValues[1] == 9);
                CHECK(intValues[2] == 14);
                CHECK(intValues[3] == 3);

                REQUIRE(fpRet == DCGM_ST_OK);
                CHECK_THAT(fpValues[0], Catch::Matchers::WithinAbs(1.25, 1.0e-12));
                CHECK_THAT(fpValues[1], Catch::Matchers::WithinAbs(3.75, 1.0e-12));
                CHECK_THAT(fpValues[2], Catch::Matchers::WithinAbs(5.0, 1.0e-12));
                CHECK_THAT(fpValues[3], Catch::Matchers::WithinAbs(3.0, 1.0e-12));
            }
        }

        WHEN("the summary callback excludes every entry")
        {
            DcgmcmSummaryType_t summaryType = DcgmcmSummaryTypeMinimum;
            long long intValue              = 123;
            double fpValue                  = 123.0;
            timelib64_t cutoff              = t0 + 1000;

            auto const intRet = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                                 gpuId,
                                                                 DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,
                                                                 1,
                                                                 &summaryType,
                                                                 &intValue,
                                                                 t0,
                                                                 t0 + 20,
                                                                 UseOnlySamplesAtOrAfterTimestamp,
                                                                 &cutoff);
            auto const fpRet  = cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                                gpuId,
                                                                DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                                1,
                                                                &summaryType,
                                                                &fpValue,
                                                                t0,
                                                                t0 + 20,
                                                                UseOnlySamplesAtOrAfterTimestamp,
                                                                &cutoff);

            THEN("no-data is reported and output values are reset to type-specific blanks")
            {
                CHECK(intRet == DCGM_ST_NOT_WATCHED);
                CHECK(intValue == DCGM_INT64_BLANK);
                CHECK(fpRet == DCGM_ST_NOT_WATCHED);
                CHECK(fpValue == DCGM_FP64_BLANK);
            }
        }
    }

    GIVEN("retrieved string and blob samples owned by the caller")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        char name[] = "cleanup-gpu";
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9702);
        bool wereFirstNameWatcher = false;
        REQUIRE(
            cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, 0, 14400.0, 4, watcher, false, false, wereFirstNameWatcher)
            == DCGM_ST_OK);

        bool wereFirstBlobWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS,
                                           0,
                                           14400.0,
                                           4,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstBlobWatcher)
                == DCGM_ST_OK);
        dcgmcm_sample_t nameSample {};
        nameSample.timestamp = 20000;
        nameSample.val.str   = name;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &nameSample, 1) == DCGM_ST_OK);

        std::array<unsigned char, 4> blobPayload { 0x1, 0x2, 0x3, 0x4 };
        dcgmcm_sample_t blobSample {};
        blobSample.timestamp    = 20010;
        blobSample.val.blob     = blobPayload.data();
        blobSample.val2.ptrSize = blobPayload.size();
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, &blobSample, 1)
                == DCGM_ST_OK);

        WHEN("owned string and blob samples are freed")
        {
            dcgmcm_sample_t copiedName {};
            copiedName.val.str      = strdup("owned-cleanup-gpu");
            copiedName.val2.ptrSize = std::strlen(copiedName.val.str) + 1;
            REQUIRE(copiedName.val.str != nullptr);

            dcgmcm_sample_t copiedBlob {};
            copiedBlob.val2.ptrSize = blobPayload.size();
            copiedBlob.val.blob     = static_cast<unsigned char *>(std::malloc(copiedBlob.val2.ptrSize));
            REQUIRE(copiedBlob.val.blob != nullptr);
            std::memcpy(copiedBlob.val.blob, blobPayload.data(), copiedBlob.val2.ptrSize);

            auto const freeNameRet   = cacheManager.FreeSamples(&copiedName, 1, DCGM_FI_DEV_GPU_NAME);
            auto const freeBlobRet   = cacheManager.FreeSamples(&copiedBlob, 1, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS);
            auto const freeScalarRet = cacheManager.FreeSamples(&blobSample, 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS);

            THEN("allocated copy pointers are cleared and scalar frees are harmless")
            {
                CHECK(freeNameRet == DCGM_ST_OK);
                CHECK(copiedName.val.str == nullptr);
                CHECK(copiedName.val2.ptrSize == 0);
                CHECK(freeBlobRet == DCGM_ST_OK);
                CHECK(copiedBlob.val.blob == nullptr);
                CHECK(copiedBlob.val2.ptrSize == 0);
                CHECK(freeScalarRet == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: summary contracts cover integral difference and type errors")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with int64 and fp64 time-series samples")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        timelib64_t const t0 = timelib_usecSince1970() + 1000000;

        std::array<dcgmcm_sample_t, 3> intSamples {};
        intSamples[0].timestamp = t0;
        intSamples[0].val.i64   = 10;
        intSamples[1].timestamp = t0 + 10;
        intSamples[1].val.i64   = 20;
        intSamples[2].timestamp = t0 + 30;
        intSamples[2].val.i64   = 50;
        REQUIRE(cacheManager.InjectSamples(
                    DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, intSamples.data(), static_cast<int>(intSamples.size()))
                == DCGM_ST_OK);

        std::array<dcgmcm_sample_t, 3> fpSamples {};
        fpSamples[0].timestamp = t0;
        fpSamples[0].val.d     = 1.0;
        fpSamples[1].timestamp = t0 + 10;
        fpSamples[1].val.d     = 3.0;
        fpSamples[2].timestamp = t0 + 30;
        fpSamples[2].val.d     = 7.0;
        REQUIRE(
            cacheManager.InjectSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, fpSamples.data(), static_cast<int>(fpSamples.size()))
            == DCGM_ST_OK);

        WHEN("all non-extrema summary types are requested")
        {
            std::array<DcgmcmSummaryType_t, 4> summaryTypes {
                DcgmcmSummaryTypeAverage,
                DcgmcmSummaryTypeIntegral,
                DcgmcmSummaryTypeDifference,
                DcgmcmSummaryTypeCount,
            };
            std::array<long long, summaryTypes.size()> intValues {};
            std::array<double, summaryTypes.size()> fpValues {};

            auto const intRet = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                                 gpuId,
                                                                 DCGM_FI_DEV_ECC_MODE,
                                                                 static_cast<int>(summaryTypes.size()),
                                                                 summaryTypes.data(),
                                                                 intValues.data(),
                                                                 t0,
                                                                 t0 + 30,
                                                                 nullptr,
                                                                 nullptr);
            auto const fpRet  = cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                                gpuId,
                                                                DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                                static_cast<int>(summaryTypes.size()),
                                                                summaryTypes.data(),
                                                                fpValues.data(),
                                                                t0,
                                                                t0 + 30,
                                                                nullptr,
                                                                nullptr);

            THEN("averages integrals differences and counts are computed from cached samples")
            {
                REQUIRE(intRet == DCGM_ST_OK);
                CHECK(intValues[0] == 26);
                CHECK(intValues[1] == 850);
                CHECK(intValues[2] == 40);
                CHECK(intValues[3] == 3);

                REQUIRE(fpRet == DCGM_ST_OK);
                CHECK_THAT(fpValues[0], Catch::Matchers::WithinAbs(11.0 / 3.0, 1.0e-12));
                CHECK_THAT(fpValues[1], Catch::Matchers::WithinAbs(120.0, 1.0e-12));
                CHECK_THAT(fpValues[2], Catch::Matchers::WithinAbs(6.0, 1.0e-12));
                CHECK_THAT(fpValues[3], Catch::Matchers::WithinAbs(3.0, 1.0e-12));
            }
        }

        WHEN("summary APIs are called with the wrong sample type and an unknown summary type")
        {
            DcgmcmSummaryType_t averageType = DcgmcmSummaryTypeAverage;
            long long intValue              = 777;
            double fpValue                  = 777.0;
            auto const intWrongTypeRet      = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                                               gpuId,
                                                                               DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                                               1,
                                                                               &averageType,
                                                                               &intValue,
                                                                               t0,
                                                                               t0 + 30,
                                                                               nullptr,
                                                                               nullptr);
            auto const fpWrongTypeRet       = cacheManager.GetFp64SummaryData(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, 1, &averageType, &fpValue, t0, t0 + 30, nullptr, nullptr);

            auto badType = static_cast<DcgmcmSummaryType_t>(9999);
            long long badTypeValue {};
            auto const badTypeRet = cacheManager.GetInt64SummaryData(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, 1, &badType, &badTypeValue, t0, t0 + 30, nullptr, nullptr);

            THEN("the APIs reject incompatible cached types and unknown summary requests")
            {
                CHECK(intWrongTypeRet == DCGM_ST_GENERIC_ERROR);
                CHECK(intValue == DCGM_INT64_BLANK);
                CHECK(fpWrongTypeRet == DCGM_ST_GENERIC_ERROR);
                CHECK(fpValue == DCGM_FP64_BLANK);
                CHECK(badTypeRet == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO("CacheManager: sample retrieval buffers preserve bounded and filtered results")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with cached integer samples")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        timelib64_t const t0 = timelib_usecSince1970() + 2000000;
        std::array<dcgmcm_sample_t, 4> samples {};
        for (size_t i = 0; i < samples.size(); ++i)
        {
            samples[i].timestamp = t0 + static_cast<timelib64_t>(i * 10);
            samples[i].val.i64   = static_cast<long long>(70 + i);
        }
        REQUIRE(cacheManager.InjectSamples(
                    DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, samples.data(), static_cast<int>(samples.size()))
                == DCGM_ST_OK);

        WHEN("ascending samples are copied into both caller storage and a field-value buffer")
        {
            std::array<dcgmcm_sample_t, 3> copied {};
            int copiedCount = static_cast<int>(copied.size());
            DcgmFvBuffer fvBuffer;

            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_MODE,
                                                     copied.data(),
                                                     &copiedCount,
                                                     t0 + 5,
                                                     t0 + 25,
                                                     DCGM_ORDER_ASCENDING,
                                                     &fvBuffer);

            THEN("both outputs contain the same bounded sample subset")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(copiedCount == 2);
                CHECK(copied[0].timestamp == t0 + 10);
                CHECK(copied[0].val.i64 == 71);
                CHECK(copied[1].timestamp == t0 + 20);
                CHECK(copied[1].val.i64 == 72);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *first                   = fvBuffer.GetNextFv(&cursor);
                REQUIRE(first != nullptr);
                CHECK(first->entityGroupId == DCGM_FE_GPU);
                CHECK(first->entityId == gpuId);
                CHECK(first->fieldId == DCGM_FI_DEV_ECC_MODE);
                CHECK(first->status == DCGM_ST_OK);
                CHECK(first->value.i64 == 71);

                auto *second = fvBuffer.GetNextFv(&cursor);
                REQUIRE(second != nullptr);
                CHECK(second->value.i64 == 72);
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("a descending query starts before every sample")
        {
            std::array<dcgmcm_sample_t, 2> copied {};
            int copiedCount = static_cast<int>(copied.size());

            auto const ret = cacheManager.GetSamples(DCGM_FE_GPU,
                                                     gpuId,
                                                     DCGM_FI_DEV_ECC_MODE,
                                                     copied.data(),
                                                     &copiedCount,
                                                     t0 + 100,
                                                     t0 + 200,
                                                     DCGM_ORDER_DESCENDING,
                                                     nullptr);

            THEN("no-data is reported because the cache has records outside the requested range")
            {
                CHECK(ret == DCGM_ST_NO_DATA);
                CHECK(copiedCount == 0);
            }
        }
    }
}

SCENARIO("CacheManager: entity cleanup distinguishes clearing watches from clearing cached samples")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("two fake GPUs with watched and cached fields")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpu0 = cacheManager.AddFakeGpu();
        unsigned int const gpu1 = cacheManager.AddFakeGpu();
        REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
        REQUIRE(gpu1 != DCGM_GPU_ID_BAD);

        DcgmWatcher const watcher(DcgmWatcherTypeClient, 9801);
        auto watchField = [&](unsigned int gpuId, unsigned short fieldId) {
            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
            bool wereFirstWatcher = false;
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_GPU, gpuId, fieldId, 1000, 60.0, 4, watcher, true, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        };
        watchField(gpu0, DCGM_FI_DEV_ECC_MODE);
        watchField(gpu0, DCGM_FI_DEV_BOARD_POWER_WATTS);
        watchField(gpu1, DCGM_FI_DEV_ECC_MODE);

        dcgmcm_sample_t gpu0Mode {};
        gpu0Mode.timestamp = timelib_usecSince1970() + 4000000;
        gpu0Mode.val.i64   = 80;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_ECC_MODE, &gpu0Mode, 1) == DCGM_ST_OK);

        dcgmcm_sample_t gpu0Power {};
        gpu0Power.timestamp = gpu0Mode.timestamp + 10;
        gpu0Power.val.d     = 290.5;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_BOARD_POWER_WATTS, &gpu0Power, 1)
                == DCGM_ST_OK);

        dcgmcm_sample_t gpu1Mode {};
        gpu1Mode.timestamp = gpu0Mode.timestamp + 20;
        gpu1Mode.val.i64   = 81;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpu1, DCGM_FI_DEV_ECC_MODE, &gpu1Mode, 1) == DCGM_ST_OK);

        WHEN("one entity is cleared without clearing cached samples")
        {
            auto const clearRet = cacheManager.ClearEntity(DCGM_FE_GPU, gpu0, 0);

            THEN("its watches are removed but its cached samples remain readable")
            {
                CHECK(clearRet == DCGM_ST_OK);

                dcgmcm_watch_info_t snapshot {};
                REQUIRE(cacheManager.GetEntityWatchInfoSnapshot(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_ECC_MODE, &snapshot)
                        == DCGM_ST_OK);
                CHECK(snapshot.isWatched == 0);
                CHECK(snapshot.hasSubscribedWatchers == 0);

                dcgmcm_sample_t gpu0Latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_ECC_MODE, &gpu0Latest, nullptr)
                      == DCGM_ST_OK);
                CHECK(gpu0Latest.val.i64 == gpu0Mode.val.i64);

                dcgmcm_sample_t gpu1Latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpu1, DCGM_FI_DEV_ECC_MODE, &gpu1Latest, nullptr)
                      == DCGM_ST_OK);
                CHECK(gpu1Latest.val.i64 == gpu1Mode.val.i64);
            }
        }

        WHEN("all entity watches are cleared with cached samples")
        {
            auto const clearRet = cacheManager.ClearAllEntities(1);

            THEN("all entity sample lookups report no data through their cleared watch records")
            {
                CHECK(clearRet == DCGM_ST_OK);

                dcgmcm_sample_t gpu0Latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpu0, DCGM_FI_DEV_ECC_MODE, &gpu0Latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);

                dcgmcm_sample_t gpu1Latest {};
                CHECK(cacheManager.GetLatestSample(DCGM_FE_GPU, gpu1, DCGM_FI_DEV_ECC_MODE, &gpu1Latest, nullptr)
                      == DCGM_ST_NOT_WATCHED);
            }
        }
    }
}

SCENARIO("CacheManager: multi-latest rows expose known field status failures")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with one cached field and one watch that stores a driver failure")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        dcgmcm_sample_t sample {};
        sample.timestamp = timelib_usecSince1970() + 5000000;
        sample.val.i64   = 85;
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, &sample, 1) == DCGM_ST_OK);

        auto *powerWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, 1);
        REQUIRE(powerWatch != nullptr);
        powerWatch->isWatched  = 1;
        powerWatch->lastStatus = NVML_ERROR_TIMEOUT;

        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId } };
        std::vector<unsigned short> fieldIds {
            DCGM_FI_DEV_ECC_MODE,
            DCGM_FI_DEV_BOARD_POWER_WATTS,
        };
        DcgmFvBuffer fvBuffer;

        WHEN("cached latest values are requested for good and bad-status combinations")
        {
            auto const ret = cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);

            THEN("each known-field combination is represented by an explicit field-value row")
            {
                REQUIRE(ret == DCGM_ST_OK);

                dcgmBufferedFvCursor_t cursor = 0;
                std::vector<dcgmBufferedFv_t const *> rows;
                while (auto *fv = fvBuffer.GetNextFv(&cursor))
                {
                    rows.push_back(fv);
                }
                REQUIRE(rows.size() == fieldIds.size());

                CHECK(rows[0]->entityGroupId == DCGM_FE_GPU);
                CHECK(rows[0]->entityId == gpuId);
                CHECK(rows[0]->fieldId == DCGM_FI_DEV_ECC_MODE);
                CHECK(rows[0]->status == DCGM_ST_OK);
                CHECK(rows[0]->value.i64 == sample.val.i64);

                CHECK(rows[1]->entityId == gpuId);
                CHECK(rows[1]->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(rows[1]->status == DCGM_ST_TIMEOUT);
            }
        }
    }
}

SCENARIO("CacheManager: live buffering reports lost and inaccessible fake GPU states")

{
    ScopedDcgmFields dcgmFields;

    auto makeThreadContext = [](DcgmFvBuffer &fvBuffer,
                                dcgmcm_watch_info_p watchInfo,
                                dcgm_field_entity_group_t entityGroupId,
                                dcgm_field_eid_t entityId,
                                unsigned short fieldId) {
        dcgmcm_update_thread_t threadCtx {};
        threadCtx.fvBuffer                = &fvBuffer;
        threadCtx.watchInfo               = watchInfo;
        threadCtx.entityKey.entityGroupId = entityGroupId;
        threadCtx.entityKey.entityId      = entityId;
        threadCtx.entityKey.fieldId       = fieldId;
        return threadCtx;
    };

    GIVEN("a fake GPU that becomes lost after watches are created")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        auto readLostField = [&](unsigned short fieldId, DcgmFvBuffer &fvBuffer) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(10));

            cacheManager.m_gpus[gpuId].status = DcgmEntityStatusLost;
            auto *fieldMeta                   = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);
            auto threadCtx = makeThreadContext(fvBuffer, watchInfo, DCGM_FE_GPU, gpuId, fieldId);

            return cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        };

        WHEN("representative field types are read")
        {
            DcgmFvBuffer intBuffer;
            DcgmFvBuffer stringBuffer;
            DcgmFvBuffer fpBuffer;
            DcgmFvBuffer blobBuffer;

            auto const intRet    = readLostField(DCGM_FI_DEV_GPU_TEMP_CELSIUS, intBuffer);
            auto const stringRet = readLostField(DCGM_FI_DEV_GPU_NAME, stringBuffer);
            auto const fpRet     = readLostField(DCGM_FI_DEV_BOARD_POWER_WATTS, fpBuffer);
            auto const blobRet   = readLostField(DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, blobBuffer);

            THEN("each field receives an explicit buffered error row")
            {
                auto checkOneRow = [gpuId](DcgmFvBuffer &fvBuffer, unsigned short fieldId, int type) {
                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->entityGroupId == DCGM_FE_GPU);
                    CHECK(fv->entityId == gpuId);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->fieldType == type);
                    CHECK(fv->status == DCGM_ST_OK);
                    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
                };

                CHECK(intRet == DCGM_ST_OK);
                CHECK(stringRet == DCGM_ST_OK);
                CHECK(fpRet == DCGM_ST_OK);
                CHECK(blobRet == DCGM_ST_OK);
                checkOneRow(intBuffer, DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FT_INT64);
                checkOneRow(stringBuffer, DCGM_FI_DEV_GPU_NAME, DCGM_FT_STRING);
                checkOneRow(fpBuffer, DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_FT_DOUBLE);

                dcgmBufferedFvCursor_t blobCursor = 0;
                CHECK(blobBuffer.GetNextFv(&blobCursor) == nullptr);
            }
        }
    }

    GIVEN("a fake GPU that is inaccessible but still has cached identity")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        cacheManager.m_gpus[gpuId].status = DcgmEntityStatusInaccessible;
        SafeCopyTo(cacheManager.m_gpus[gpuId].uuid, "GPU-inaccessible-cache");
        SafeCopyTo(cacheManager.m_gpus[gpuId].deviceName, "inaccessible-cache-name");
        SafeCopyTo(cacheManager.m_gpus[gpuId].pciInfo.busId, "00000000:AA:00.0");

        auto readField = [&](unsigned short fieldId, DcgmFvBuffer &fvBuffer) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched = 1;

            auto *fieldMeta = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);
            auto threadCtx = makeThreadContext(fvBuffer, watchInfo, DCGM_FE_GPU, gpuId, fieldId);

            return cacheManager.BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        };

        WHEN("identity fields and non-identity fields are read")
        {
            DcgmFvBuffer uuidBuffer;
            DcgmFvBuffer nameBuffer;
            DcgmFvBuffer pciBuffer;
            DcgmFvBuffer tempBuffer;
            DcgmFvBuffer powerBuffer;

            auto const uuidRet  = readField(DCGM_FI_DEV_GPU_UUID, uuidBuffer);
            auto const nameRet  = readField(DCGM_FI_DEV_GPU_NAME, nameBuffer);
            auto const pciRet   = readField(DCGM_FI_DEV_PCI_BUS_ID, pciBuffer);
            auto const tempRet  = readField(DCGM_FI_DEV_GPU_TEMP_CELSIUS, tempBuffer);
            auto const powerRet = readField(DCGM_FI_DEV_BOARD_POWER_WATTS, powerBuffer);

            THEN("identity reads use cached strings and other fields receive typed blanks")
            {
                auto requireStringRow = [gpuId](DcgmFvBuffer &fvBuffer, unsigned short fieldId, char const *expected) {
                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->entityGroupId == DCGM_FE_GPU);
                    CHECK(fv->entityId == gpuId);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->fieldType == DCGM_FT_STRING);
                    REQUIRE(fv->value.str != nullptr);
                    CHECK(std::string_view(fv->value.str) == expected);
                    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
                };

                CHECK(uuidRet == DCGM_ST_OK);
                CHECK(nameRet == DCGM_ST_OK);
                CHECK(pciRet == DCGM_ST_OK);
                requireStringRow(uuidBuffer, DCGM_FI_DEV_GPU_UUID, "GPU-inaccessible-cache");
                requireStringRow(nameBuffer, DCGM_FI_DEV_GPU_NAME, "inaccessible-cache-name");
                requireStringRow(pciBuffer, DCGM_FI_DEV_PCI_BUS_ID, "00000000:AA:00.0");

                CHECK(tempRet == DCGM_ST_OK);
                dcgmBufferedFvCursor_t tempCursor = 0;
                auto *tempFv                      = tempBuffer.GetNextFv(&tempCursor);
                REQUIRE(tempFv != nullptr);
                CHECK(tempFv->fieldType == DCGM_FT_INT64);
                CHECK(tempFv->value.i64 == DCGM_INT64_BLANK);

                CHECK(powerRet == DCGM_ST_OK);
                dcgmBufferedFvCursor_t powerCursor = 0;
                auto *powerFv                      = powerBuffer.GetNextFv(&powerCursor);
                REQUIRE(powerFv != nullptr);
                CHECK(powerFv->fieldType == DCGM_FT_DOUBLE);
                CHECK(powerFv->value.dbl == DCGM_FP64_BLANK);
            }
        }
    }
}

SCENARIO("CacheManager: SetValueImpl validates settable field inputs without hardware")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with NVML marked available")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        WHEN("single-fault validation failures are requested")
        {
            auto const nullRet = cacheManager.SetValueImpl(gpuId, DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE, nullptr);

            dcgmcm_sample_t validValue {};
            validValue.val.i64       = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
            auto const invalidGpuRet = cacheManager.SetValueImpl(
                static_cast<int>(DCGM_MAX_NUM_DEVICES), DCGM_FI_DEV_GPU_COMPUTE_MODE, &validValue);

            validValue.val.i64 = 9999;
            auto const invalidComputeModeRet
                = cacheManager.SetValueImpl(static_cast<int>(gpuId), DCGM_FI_DEV_GPU_COMPUTE_MODE, &validValue);

            THEN("the contract returns precise DCGM status codes")
            {
                CHECK(nullRet == DCGM_ST_BADPARAM);
                CHECK(invalidGpuRet == DCGM_ST_BADPARAM);
                CHECK(invalidComputeModeRet == DCGM_ST_BADPARAM);
            }
        }

        WHEN("a valid field is requested while NVML is unavailable")
        {
            cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
            dcgmcm_sample_t sample {};
            sample.val.i64 = DCGM_CONFIG_COMPUTEMODE_DEFAULT;

            auto const ret = cacheManager.SetValueImpl(static_cast<int>(gpuId), DCGM_FI_DEV_GPU_COMPUTE_MODE, &sample);

            THEN("the request is rejected before any driver-facing work is attempted")
            {
                CHECK(ret == DCGM_ST_NVML_NOT_LOADED);
            }
        }
    }
}

SCENARIO("CacheManager: workload power profile and NvLink BER helper branches stay deterministic")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("a cache manager and a workload power profile blob")
    {
        TestDcgmCacheManager cacheManager;
        SafeNvmlHandle safeNvmlHandle {};
        dcgmcmWorkloadPowerProfile_t profile {};
        dcgmcm_sample_t sample {};
        sample.val.blob     = &profile;
        sample.val2.ptrSize = sizeof(profile);

        WHEN("the workload action is a local no-op")
        {
            profile.action         = DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_NONE;
            profile.profileMask[0] = 0x5U;

            auto const ret = cacheManager.SetWorkloadPowerProfile(safeNvmlHandle, sample);

            THEN("the helper accepts the request without calling NVML")
            {
                CHECK(ret == DCGM_ST_OK);
            }
        }

        WHEN("the workload payload is too small")
        {
            sample.val2.ptrSize = sizeof(profile) - 1;

            auto const ret = cacheManager.SetWorkloadPowerProfile(safeNvmlHandle, sample);

            THEN("the helper rejects it before interpreting the action")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the workload action is unknown")
        {
            profile.action = static_cast<dcgmcmWorkloadPowerProfileAction_t>(99);

            auto const ret = cacheManager.SetWorkloadPowerProfile(safeNvmlHandle, sample);

            THEN("the helper reports a caller parameter error")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a fake GPU and NvLink BER watch context")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        auto makeThreadContext = [&](DcgmFvBuffer &fvBuffer, unsigned short fieldId) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(10));

            dcgmcm_update_thread_t threadCtx {};
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = fieldId;
            return threadCtx;
        };

        WHEN("NVML is not loaded")
        {
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeThreadContext(fvBuffer, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO);
            cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);

            auto const ret = cacheManager.ReadAndCacheNvLinkBer(
                threadCtx, SafeNvmlHandle {}, DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO, 0, gpuId);

            THEN("the helper returns the NVML-not-loaded status before buffering a value")
            {
                CHECK(ret == DCGM_ST_NVML_NOT_LOADED);
                dcgmBufferedFvCursor_t cursor = 0;
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }

        WHEN("the field id is not a BER field")
        {
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeThreadContext(fvBuffer, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

            auto const ret = cacheManager.ReadAndCacheNvLinkBer(
                threadCtx, SafeNvmlHandle {}, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0, gpuId);

            THEN("the helper rejects the invalid field id")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                dcgmBufferedFvCursor_t cursor = 0;
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }


        WHEN("NVML is loaded but the safe handle is empty")
        {
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeThreadContext(fvBuffer, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO);
            cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
            cacheManager.m_gpus[gpuId].arch = DCGM_CHIP_ARCH_BLACKWELL;

            auto const ret = cacheManager.ReadAndCacheNvLinkBer(
                threadCtx, SafeNvmlHandle {}, DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO, 1234, gpuId);

            THEN("the helper buffers an NVML error value and updates the watch status")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                REQUIRE(threadCtx.watchInfo != nullptr);
                CHECK(threadCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO);
                CHECK(fv->fieldType == DCGM_FT_DOUBLE);
                CHECK(fv->value.dbl == NvmlErrorToDoubleValue(NVML_ERROR_INVALID_ARGUMENT));
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}

SCENARIO("CacheManager: cached topology and memory helper branches stay deterministic")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("no cached topology data and NVML is unavailable")
    {
        TestDcgmCacheManager cacheManager;
        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);

        WHEN("the public topology helpers are requested")
        {
            dcgmAffinity_t affinity {};
            auto const affinityRet   = cacheManager.PopulateCpuAffinity(affinity);
            dcgmTopology_t *topology = cacheManager.GetNvLinkTopologyInformation();

            THEN("the helpers report that live topology sources are unavailable")
            {
                CHECK(affinityRet == DCGM_ST_NVML_NOT_LOADED);
                CHECK(topology == nullptr);
            }
        }
    }

    GIVEN("fake GPUs with local NvLink and watch state")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const activeGpu   = cacheManager.AddFakeGpu();
        unsigned int const detachedGpu = cacheManager.AddFakeGpu();
        REQUIRE(activeGpu != DCGM_GPU_ID_BAD);
        REQUIRE(detachedGpu != DCGM_GPU_ID_BAD);
        cacheManager.m_gpus[detachedGpu].status = DcgmEntityStatusDetached;
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        REQUIRE(cacheManager.SetGpuNvLinkLinkState(activeGpu, 0, DcgmNvLinkLinkStateUp) == DCGM_ST_OK);
        REQUIRE(cacheManager.SetGpuNvLinkLinkState(activeGpu, 1, DcgmNvLinkLinkStateDisabled) == DCGM_ST_OK);

        WHEN("NvLink status and local prewatch validation run")
        {
            dcgmNvLinkStatus_v4 nvLinkStatus {};
            auto const statusRet = cacheManager.PopulateNvLinkLinkStatus(nvLinkStatus);
            auto const detachedAccountingRet
                = cacheManager.NvmlPreWatch(detachedGpu, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS);

            bool wereFirstWatcher = false;
            DcgmWatcher const watcher(DcgmWatcherTypeClient, 10201);
            auto const inforomWatchRet = cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                                                    activeGpu,
                                                                    DCGM_FI_DEV_INFOROM_VALID,
                                                                    1000,
                                                                    1.0,
                                                                    1,
                                                                    watcher,
                                                                    false,
                                                                    false,
                                                                    wereFirstWatcher);
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, activeGpu, DCGM_FI_DEV_INFOROM_VALID, 0);

            THEN("detached GPUs are skipped and inforom fields get the enforced polling interval")
            {
                REQUIRE(statusRet == DCGM_ST_OK);
                CHECK(nvLinkStatus.version == dcgmNvLinkStatus_version4);
                REQUIRE(nvLinkStatus.numGpus == 1);
                CHECK(nvLinkStatus.gpus[0].entityId == activeGpu);
                CHECK(nvLinkStatus.gpus[0].linkState[0] == DcgmNvLinkLinkStateUp);
                CHECK(nvLinkStatus.gpus[0].linkState[1] == DcgmNvLinkLinkStateDisabled);

                CHECK(detachedAccountingRet == DCGM_ST_GPU_IS_LOST);
                CHECK(inforomWatchRet == DCGM_ST_OK);
                CHECK(wereFirstWatcher);
                REQUIRE(watchInfo != nullptr);
                CHECK(watchInfo->monitorIntervalUsec == 30000000);
                CHECK(watchInfo->watchers.size() == 1);
                CHECK(watchInfo->watchers[0].monitorIntervalUsec == 30000000);
            }
        }
    }

    GIVEN("FB memory watch contexts without a real NVML handle")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        cacheManager.m_gpus[gpuId].status = DcgmEntityStatusOk;
        cacheManager.m_driverMajorVersion = 510;

        auto makeThreadContext = [&](DcgmFvBuffer &fvBuffer, unsigned short fieldId) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(5));

            dcgmcm_update_thread_t threadCtx {};
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = fieldId;
            return threadCtx;
        };

        WHEN("FB memory helpers are called with invalid safe handles")
        {
            DcgmFvBuffer totalBuffer;
            DcgmFvBuffer usedBuffer;
            DcgmFvBuffer reservedBuffer;
            auto totalCtx    = makeThreadContext(totalBuffer, DCGM_FI_DEV_FB_TOTAL);
            auto usedCtx     = makeThreadContext(usedBuffer, DCGM_FI_DEV_FB_USED);
            auto reservedCtx = makeThreadContext(reservedBuffer, DCGM_FI_DEV_FB_RESERVED);

            cacheManager.ReadAndCacheFBMemoryInfo(
                gpuId, SafeNvmlHandle {}, totalCtx, totalCtx.watchInfo, 0, DCGM_FI_DEV_FB_TOTAL);
            cacheManager.ReadAndCacheFBMemoryInfo(
                gpuId, SafeNvmlHandle {}, usedCtx, usedCtx.watchInfo, 0, DCGM_FI_DEV_FB_USED);
            cacheManager.ReadAndCacheFBMemoryInfo(
                gpuId, SafeNvmlHandle {}, reservedCtx, reservedCtx.watchInfo, 0, DCGM_FI_DEV_FB_RESERVED);

            THEN("each field buffers an explicit NVML-invalid-argument value and records the driver status")
            {
                auto requireInvalidArgRow = [gpuId](DcgmFvBuffer &fvBuffer, unsigned short fieldId) {
                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->entityGroupId == DCGM_FE_GPU);
                    CHECK(fv->entityId == gpuId);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->fieldType == DCGM_FT_INT64);
                    CHECK(fv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_INVALID_ARGUMENT));
                    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
                };

                CHECK(totalCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);
                CHECK(usedCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);
                CHECK(reservedCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);
                requireInvalidArgRow(totalBuffer, DCGM_FI_DEV_FB_TOTAL);
                requireInvalidArgRow(usedBuffer, DCGM_FI_DEV_FB_USED);
                requireInvalidArgRow(reservedBuffer, DCGM_FI_DEV_FB_RESERVED);
            }
        }
    }
}

SCENARIO("CacheManager: cached sample edge contracts preserve public outputs")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU with integer and floating-point cached samples")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        timelib64_t const baseTimestamp = timelib_usecSince1970() + 6000000;

        std::array<dcgmcm_sample_t, 3> intSamples {};
        intSamples[0].timestamp = baseTimestamp;
        intSamples[0].val.i64   = 10;
        intSamples[1].timestamp = baseTimestamp + 10;
        intSamples[1].val.i64   = 30;
        intSamples[2].timestamp = baseTimestamp + 20;
        intSamples[2].val.i64   = 50;
        REQUIRE(cacheManager.InjectSamples(
                    DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_MODE, intSamples.data(), static_cast<int>(intSamples.size()))
                == DCGM_ST_OK);

        std::array<dcgmcm_sample_t, 3> fpSamples {};
        fpSamples[0].timestamp = baseTimestamp;
        fpSamples[0].val.d     = 1.5;
        fpSamples[1].timestamp = baseTimestamp + 10;
        fpSamples[1].val.d     = 3.5;
        fpSamples[2].timestamp = baseTimestamp + 20;
        fpSamples[2].val.d     = 5.5;
        REQUIRE(
            cacheManager.InjectSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_BOARD_POWER_WATTS, fpSamples.data(), static_cast<int>(fpSamples.size()))
            == DCGM_ST_OK);

        WHEN("summary requests use every extrema and sum type with a timestamp filter")
        {
            timelib64_t cutoff = baseTimestamp + 10;
            std::array<DcgmcmSummaryType_t, 5> summaryTypes {
                DcgmcmSummaryTypeMinimum, DcgmcmSummaryTypeMaximum, DcgmcmSummaryTypeAverage,
                DcgmcmSummaryTypeSum,     DcgmcmSummaryTypeCount,
            };
            std::array<long long, summaryTypes.size()> intValues {};
            std::array<double, summaryTypes.size()> fpValues {};

            auto const intRet = cacheManager.GetInt64SummaryData(DCGM_FE_GPU,
                                                                 gpuId,
                                                                 DCGM_FI_DEV_ECC_MODE,
                                                                 static_cast<int>(summaryTypes.size()),
                                                                 summaryTypes.data(),
                                                                 intValues.data(),
                                                                 baseTimestamp,
                                                                 baseTimestamp + 20,
                                                                 UseOnlySamplesAtOrAfterTimestamp,
                                                                 &cutoff);
            auto const fpRet  = cacheManager.GetFp64SummaryData(DCGM_FE_GPU,
                                                                gpuId,
                                                                DCGM_FI_DEV_BOARD_POWER_WATTS,
                                                                static_cast<int>(summaryTypes.size()),
                                                                summaryTypes.data(),
                                                                fpValues.data(),
                                                                baseTimestamp,
                                                                baseTimestamp + 20,
                                                                UseOnlySamplesAtOrAfterTimestamp,
                                                                &cutoff);

            THEN("filtered extrema sums averages and counts are computed from the retained rows")
            {
                REQUIRE(intRet == DCGM_ST_OK);
                CHECK(intValues[0] == 30);
                CHECK(intValues[1] == 50);
                CHECK(intValues[2] == 40);
                CHECK(intValues[3] == 80);
                CHECK(intValues[4] == 2);

                REQUIRE(fpRet == DCGM_ST_OK);
                CHECK_THAT(fpValues[0], Catch::Matchers::WithinAbs(3.5, 1.0e-12));
                CHECK_THAT(fpValues[1], Catch::Matchers::WithinAbs(5.5, 1.0e-12));
                CHECK_THAT(fpValues[2], Catch::Matchers::WithinAbs(4.5, 1.0e-12));
                CHECK_THAT(fpValues[3], Catch::Matchers::WithinAbs(9.0, 1.0e-12));
                CHECK_THAT(fpValues[4], Catch::Matchers::WithinAbs(2.0, 1.0e-12));
            }
        }
    }

    GIVEN("a global string field injected through the cache manager")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        dcgmcm_sample_t driverSample {};
        driverSample.timestamp = timelib_usecSince1970() + 7000000;
        driverSample.val.str   = const_cast<char *>("global-driver-version");
        REQUIRE(cacheManager.InjectSamples(DCGM_FE_NONE, 0, DCGM_FI_SYSTEM_DRIVER_VERSION, &driverSample, 1)
                == DCGM_ST_OK);

        WHEN("callers request the global field with a GPU entity scope")
        {
            std::array<dcgmcm_sample_t, 2> copiedSamples {};
            int sampleCount = static_cast<int>(copiedSamples.size());
            DcgmFvBuffer fvBuffer;
            auto const samplesRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                            gpuId,
                                                            DCGM_FI_SYSTEM_DRIVER_VERSION,
                                                            copiedSamples.data(),
                                                            &sampleCount,
                                                            0,
                                                            0,
                                                            DCGM_ORDER_ASCENDING,
                                                            &fvBuffer);
            DcgmNs::Defer freeCopiedSamples(
                [&] { cacheManager.FreeSamples(copiedSamples.data(), sampleCount, DCGM_FI_SYSTEM_DRIVER_VERSION); });

            dcgmcm_sample_t latest {};
            auto const latestRet
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_SYSTEM_DRIVER_VERSION, &latest, nullptr);
            DcgmNs::Defer freeLatest([&] { cacheManager.FreeSamples(&latest, 1, DCGM_FI_SYSTEM_DRIVER_VERSION); });

            THEN("global watch lookup succeeds while preserving the caller requested entity metadata")
            {
                REQUIRE(samplesRet == DCGM_ST_OK);
                REQUIRE(sampleCount == 1);
                REQUIRE(copiedSamples[0].val.str != nullptr);
                CHECK(std::string_view(copiedSamples[0].val.str) == "global-driver-version");

                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_SYSTEM_DRIVER_VERSION);
                CHECK(fv->status == DCGM_ST_OK);
                CHECK(std::string_view(fv->value.str) == "global-driver-version");
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);

                REQUIRE(latestRet == DCGM_ST_OK);
                REQUIRE(latest.val.str != nullptr);
                CHECK(std::string_view(latest.val.str) == "global-driver-version");
            }
        }
    }

    GIVEN("string and blob injections that fail after one valid row")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        std::array<dcgmcm_sample_t, 2> nameSamples {};
        nameSamples[0].timestamp = timelib_usecSince1970() + 8000000;
        nameSamples[0].val.str   = const_cast<char *>("first-name-row");
        nameSamples[1].timestamp = nameSamples[0].timestamp + 10;
        nameSamples[1].val.str   = nullptr;

        std::array<unsigned char, 4> payload { 3, 1, 4, 1 };
        std::array<dcgmcm_sample_t, 2> blobSamples {};
        blobSamples[0].timestamp    = nameSamples[0].timestamp + 20;
        blobSamples[0].val.blob     = payload.data();
        blobSamples[0].val2.ptrSize = payload.size();
        blobSamples[1].timestamp    = nameSamples[0].timestamp + 30;
        blobSamples[1].val.blob     = nullptr;
        blobSamples[1].val2.ptrSize = payload.size();

        WHEN("the second row is invalid for each field type")
        {
            auto const nameRet = cacheManager.InjectSamples(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, nameSamples.data(), static_cast<int>(nameSamples.size()));
            auto const blobRet = cacheManager.InjectSamples(DCGM_FE_GPU,
                                                            gpuId,
                                                            DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS,
                                                            blobSamples.data(),
                                                            static_cast<int>(blobSamples.size()));

            dcgmcm_sample_t latestName {};
            auto const latestNameRet
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_NAME, &latestName, nullptr);
            DcgmNs::Defer freeName([&] { cacheManager.FreeSamples(&latestName, 1, DCGM_FI_DEV_GPU_NAME); });

            dcgmcm_sample_t latestBlob {};
            auto const latestBlobRet = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS, &latestBlob, nullptr);
            DcgmNs::Defer freeBlob(
                [&] { cacheManager.FreeSamples(&latestBlob, 1, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS); });

            THEN("the bad row is rejected and the earlier valid row remains observable")
            {
                CHECK(nameRet == DCGM_ST_BADPARAM);
                REQUIRE(latestNameRet == DCGM_ST_OK);
                REQUIRE(latestName.val.str != nullptr);
                CHECK(std::string_view(latestName.val.str) == "first-name-row");

                CHECK(blobRet == DCGM_ST_BADPARAM);
                REQUIRE(latestBlobRet == DCGM_ST_OK);
                REQUIRE(latestBlob.val.blob != nullptr);
                REQUIRE(latestBlob.val2.ptrSize == static_cast<long long>(payload.size()));
                auto const *blobBytes = static_cast<unsigned char const *>(latestBlob.val.blob);
                CHECK(std::equal(payload.begin(), payload.end(), blobBytes));
            }
        }
    }

    GIVEN("multi-latest requests with empty dimensions and unknown fields")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, gpuId } };
        std::vector<unsigned short> noFields;
        std::vector<unsigned short> badFields { DCGM_FI_MAX_FIELDS };

        WHEN("the request is empty, missing output, or names an unknown field")
        {
            DcgmFvBuffer emptyBuffer;
            auto const emptyRet = cacheManager.GetMultipleLatestSamples(entities, noFields, &emptyBuffer);
            auto const nullRet  = cacheManager.GetMultipleLatestSamples(entities, badFields, nullptr);

            DcgmFvBuffer badFieldBuffer;
            auto const badFieldRet = cacheManager.GetMultipleLatestSamples(entities, badFields, &badFieldBuffer);

            THEN("the API reports boundary status and buffers per-field failures")
            {
                CHECK(emptyRet == DCGM_ST_OK);
                dcgmBufferedFvCursor_t emptyCursor = 0;
                CHECK(emptyBuffer.GetNextFv(&emptyCursor) == nullptr);
                CHECK(nullRet == DCGM_ST_BADPARAM);

                CHECK(badFieldRet == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                CHECK(badFieldBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}

SCENARIO("CacheManager: empty watched sample status branches are observable")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a watched fake GPU field with no cached rows and a remembered NVML failure")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched  = 1;
        watchInfo->lastStatus = NVML_ERROR_TIMEOUT;

        WHEN("latest and ranged sample callers query the empty series")
        {
            dcgmcm_sample_t latest {};
            DcgmFvBuffer latestBuffer;
            auto const latestRet = cacheManager.GetLatestSample(
                DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, &latestBuffer);

            std::array<dcgmcm_sample_t, 2> samples {};
            int sampleCount       = static_cast<int>(samples.size());
            auto const samplesRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                            gpuId,
                                                            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                            samples.data(),
                                                            &sampleCount,
                                                            0,
                                                            0,
                                                            DCGM_ORDER_ASCENDING,
                                                            nullptr);

            THEN("the previous NVML failure is surfaced and buffered as a blank latest value")
            {
                CHECK(latestRet == DCGM_ST_TIMEOUT);
                CHECK(samplesRet == DCGM_ST_TIMEOUT);
                CHECK(sampleCount == 0);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = latestBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(fv->status == DCGM_ST_TIMEOUT);
                CHECK(latestBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }

    GIVEN("a watched fake GPU field with no cached rows and no prior NVML failure")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1);
        REQUIRE(watchInfo != nullptr);
        watchInfo->isWatched  = 1;
        watchInfo->lastStatus = NVML_SUCCESS;

        WHEN("callers query the empty watched series")
        {
            dcgmcm_sample_t latest {};
            auto const latestRet
                = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &latest, nullptr);

            std::array<dcgmcm_sample_t, 1> samples {};
            int sampleCount       = static_cast<int>(samples.size());
            auto const samplesRet = cacheManager.GetSamples(DCGM_FE_GPU,
                                                            gpuId,
                                                            DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                            samples.data(),
                                                            &sampleCount,
                                                            0,
                                                            0,
                                                            DCGM_ORDER_DESCENDING,
                                                            nullptr);

            THEN("the APIs distinguish watched-but-empty from unwatched")
            {
                CHECK(latestRet == DCGM_ST_NO_DATA);
                CHECK(samplesRet == DCGM_ST_NO_DATA);
                CHECK(sampleCount == 0);
            }
        }
    }
}

SCENARIO("CacheManager: public cleanup and fabric error helpers expose deterministic cache behavior")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("client-owned global and GPU watches plus a metagroup watch")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        DcgmWatcher const clientWatcher(DcgmWatcherTypeClient, 10601);
        bool wereFirstWatcher = false;

        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                           gpuId,
                                           DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                           1000,
                                           60.0,
                                           2,
                                           clientWatcher,
                                           true,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);

        wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_NONE,
                                           0,
                                           DCGM_FI_SYSTEM_DRIVER_VERSION,
                                           2000,
                                           120.0,
                                           3,
                                           clientWatcher,
                                           true,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
        REQUIRE(wereFirstWatcher);
        cacheManager.AddMetaGroupWatchedField(
            DCGM_GROUP_ALL_GPUS, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1000, 60.0, 2, clientWatcher);
        REQUIRE_FALSE(cacheManager.m_metaGroupWatchedFields[DCGM_GROUP_ALL_GPUS].empty());

        WHEN("the synchronous connection cleanup helper runs for that client")
        {
            cacheManager.OnConnectionRemoveImpl(clientWatcher.connectionId);

            THEN("all watches owned by the connection are removed")
            {
                auto *gpuWatch = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 0);
                REQUIRE(gpuWatch != nullptr);
                CHECK(gpuWatch->watchers.empty());
                CHECK(gpuWatch->isWatched == 0);
                CHECK(gpuWatch->hasSubscribedWatchers == 0);

                auto *globalWatch = cacheManager.GetGlobalWatchInfo(DCGM_FI_SYSTEM_DRIVER_VERSION, 0);
                REQUIRE(globalWatch != nullptr);
                CHECK(globalWatch->watchers.empty());
                CHECK(globalWatch->isWatched == 0);
                CHECK(globalWatch->hasSubscribedWatchers == 0);
                CHECK(cacheManager.m_metaGroupWatchedFields[DCGM_GROUP_ALL_GPUS].empty());
            }
        }
    }

    GIVEN("fabric-manager field watch contexts without a real NVML handle")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);
        cacheManager.m_nvmlLoaded.store(true, std::memory_order_release);

        auto makeThreadContext = [&](DcgmFvBuffer &fvBuffer, unsigned short fieldId) {
            auto *watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldId, 1);
            REQUIRE(watchInfo != nullptr);
            watchInfo->isWatched  = 1;
            watchInfo->maxAgeUsec = DcgmNs::Timelib::ToLegacyTimestamp(std::chrono::seconds(5));

            dcgmcm_update_thread_t threadCtx {};
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.watchInfo               = watchInfo;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = fieldId;
            return threadCtx;
        };

        WHEN("fabric fields are read with an invalid safe handle")
        {
            std::array<unsigned short, 4> const int64Fields {
                DCGM_FI_DEV_FABRIC_MANAGER_STATUS,
                DCGM_FI_DEV_FABRIC_HEALTH_MASK,
                DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY,
                DCGM_FI_DEV_FABRIC_CLIQUE_ID,
            };

            THEN("integer fabric fields buffer the NVML error sentinel and update watch status")
            {
                for (auto const fieldId : int64Fields)
                {
                    CAPTURE(fieldId);
                    auto *fieldMeta = DcgmFieldGetById(fieldId);
                    REQUIRE(fieldMeta != nullptr);
                    DcgmFvBuffer fvBuffer;
                    auto threadCtx = makeThreadContext(fvBuffer, fieldId);

                    CHECK(cacheManager.ReadFabricManagerStatusField(
                              threadCtx, SafeNvmlHandle {}, fieldMeta, timelib_usecSince1970(), 0)
                          == DCGM_ST_BADPARAM);
                    REQUIRE(threadCtx.watchInfo != nullptr);
                    CHECK(threadCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);

                    dcgmBufferedFvCursor_t cursor = 0;
                    auto *fv                      = fvBuffer.GetNextFv(&cursor);
                    REQUIRE(fv != nullptr);
                    CHECK(fv->entityGroupId == DCGM_FE_GPU);
                    CHECK(fv->entityId == gpuId);
                    CHECK(fv->fieldId == fieldId);
                    CHECK(fv->fieldType == DCGM_FT_INT64);
                    CHECK(fv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_INVALID_ARGUMENT));
                    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
                }
            }
        }

        WHEN("the fabric UUID field is read with an invalid safe handle")
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_FABRIC_CLUSTER_UUID;
            auto *fieldMeta                  = DcgmFieldGetById(fieldId);
            REQUIRE(fieldMeta != nullptr);
            DcgmFvBuffer fvBuffer;
            auto threadCtx = makeThreadContext(fvBuffer, fieldId);

            auto const ret = cacheManager.ReadFabricManagerStatusField(
                threadCtx, SafeNvmlHandle {}, fieldMeta, timelib_usecSince1970(), 0);

            THEN("the string field buffers the NVML error string")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                REQUIRE(threadCtx.watchInfo != nullptr);
                CHECK(threadCtx.watchInfo->lastStatus == NVML_ERROR_INVALID_ARGUMENT);

                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = fvBuffer.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == gpuId);
                CHECK(fv->fieldId == fieldId);
                CHECK(fv->fieldType == DCGM_FT_STRING);
                CHECK(std::string_view(fv->value.str) == NvmlErrorToStringValue(NVML_ERROR_INVALID_ARGUMENT));
                CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}

SCENARIO("CacheManager: blank and NVML error insertion helpers buffer typed sentinel values")

{
    ScopedDcgmFields dcgmFields;

    GIVEN("a fake GPU update context with only an output buffer")
    {
        TestDcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        REQUIRE(gpuId != DCGM_GPU_ID_BAD);

        auto makeThreadContext = [gpuId](DcgmFvBuffer &fvBuffer, unsigned short fieldId) {
            dcgmcm_update_thread_t threadCtx {};
            threadCtx.fvBuffer                = &fvBuffer;
            threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
            threadCtx.entityKey.entityId      = gpuId;
            threadCtx.entityKey.fieldId       = fieldId;
            return threadCtx;
        };

        WHEN("blank values are inserted for supported scalar and string field types")
        {
            DcgmFvBuffer intBuffer;
            DcgmFvBuffer doubleBuffer;
            DcgmFvBuffer stringBuffer;
            auto intCtx    = makeThreadContext(intBuffer, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            auto doubleCtx = makeThreadContext(doubleBuffer, DCGM_FI_DEV_BOARD_POWER_WATTS);
            auto stringCtx = makeThreadContext(stringBuffer, DCGM_FI_DEV_GPU_NAME);

            cacheManager.InsertBlankValue(intCtx, DCGM_FT_INT64, 0);
            cacheManager.InsertBlankValue(doubleCtx, DCGM_FT_DOUBLE, 0);
            cacheManager.InsertBlankValue(stringCtx, DCGM_FT_STRING, 0);

            THEN("the buffer receives the matching DCGM blank sentinel for each type")
            {
                dcgmBufferedFvCursor_t intCursor = 0;
                auto *intFv                      = intBuffer.GetNextFv(&intCursor);
                REQUIRE(intFv != nullptr);
                CHECK(intFv->entityGroupId == DCGM_FE_GPU);
                CHECK(intFv->entityId == gpuId);
                CHECK(intFv->fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(intFv->fieldType == DCGM_FT_INT64);
                CHECK(intFv->value.i64 == DCGM_INT64_BLANK);
                CHECK(intBuffer.GetNextFv(&intCursor) == nullptr);

                dcgmBufferedFvCursor_t doubleCursor = 0;
                auto *doubleFv                      = doubleBuffer.GetNextFv(&doubleCursor);
                REQUIRE(doubleFv != nullptr);
                CHECK(doubleFv->entityGroupId == DCGM_FE_GPU);
                CHECK(doubleFv->entityId == gpuId);
                CHECK(doubleFv->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(doubleFv->fieldType == DCGM_FT_DOUBLE);
                CHECK(DCGM_FP64_IS_BLANK(doubleFv->value.dbl));
                CHECK(doubleBuffer.GetNextFv(&doubleCursor) == nullptr);

                dcgmBufferedFvCursor_t stringCursor = 0;
                auto *stringFv                      = stringBuffer.GetNextFv(&stringCursor);
                REQUIRE(stringFv != nullptr);
                CHECK(stringFv->entityGroupId == DCGM_FE_GPU);
                CHECK(stringFv->entityId == gpuId);
                CHECK(stringFv->fieldId == DCGM_FI_DEV_GPU_NAME);
                CHECK(stringFv->fieldType == DCGM_FT_STRING);
                CHECK(std::string_view(stringFv->value.str) == DCGM_STR_BLANK);
                CHECK(stringBuffer.GetNextFv(&stringCursor) == nullptr);
            }
        }

        WHEN("NVML errors are inserted for supported scalar and string field types")
        {
            DcgmFvBuffer intBuffer;
            DcgmFvBuffer doubleBuffer;
            DcgmFvBuffer stringBuffer;
            auto intCtx    = makeThreadContext(intBuffer, DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            auto doubleCtx = makeThreadContext(doubleBuffer, DCGM_FI_DEV_BOARD_POWER_WATTS);
            auto stringCtx = makeThreadContext(stringBuffer, DCGM_FI_DEV_GPU_NAME);

            cacheManager.InsertNvmlErrorValue(intCtx, DCGM_FT_INT64, NVML_ERROR_NOT_SUPPORTED, 0);
            cacheManager.InsertNvmlErrorValue(doubleCtx, DCGM_FT_DOUBLE, NVML_ERROR_TIMEOUT, 0);
            cacheManager.InsertNvmlErrorValue(stringCtx, DCGM_FT_STRING, NVML_ERROR_INVALID_ARGUMENT, 0);

            THEN("the buffer receives the field-type-specific translated NVML sentinels")
            {
                dcgmBufferedFvCursor_t intCursor = 0;
                auto *intFv                      = intBuffer.GetNextFv(&intCursor);
                REQUIRE(intFv != nullptr);
                CHECK(intFv->fieldType == DCGM_FT_INT64);
                CHECK(intFv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(intBuffer.GetNextFv(&intCursor) == nullptr);

                dcgmBufferedFvCursor_t doubleCursor = 0;
                auto *doubleFv                      = doubleBuffer.GetNextFv(&doubleCursor);
                REQUIRE(doubleFv != nullptr);
                CHECK(doubleFv->fieldType == DCGM_FT_DOUBLE);
                CHECK(doubleFv->value.dbl == NvmlErrorToDoubleValue(NVML_ERROR_TIMEOUT));
                CHECK(doubleBuffer.GetNextFv(&doubleCursor) == nullptr);

                dcgmBufferedFvCursor_t stringCursor = 0;
                auto *stringFv                      = stringBuffer.GetNextFv(&stringCursor);
                REQUIRE(stringFv != nullptr);
                CHECK(stringFv->fieldType == DCGM_FT_STRING);
                CHECK(std::string_view(stringFv->value.str) == NvmlErrorToStringValue(NVML_ERROR_INVALID_ARGUMENT));
                CHECK(stringBuffer.GetNextFv(&stringCursor) == nullptr);
            }
        }

        WHEN("an unsupported field type is requested")
        {
            DcgmFvBuffer blankBuffer;
            DcgmFvBuffer errorBuffer;
            auto blankCtx = makeThreadContext(blankBuffer, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS);
            auto errorCtx = makeThreadContext(errorBuffer, DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS);

            cacheManager.InsertBlankValue(blankCtx, DCGM_FT_BINARY, 0);
            cacheManager.InsertNvmlErrorValue(errorCtx, DCGM_FT_BINARY, NVML_ERROR_UNKNOWN, 0);

            THEN("no typed value is buffered because binary conversion is intentionally unsupported")
            {
                dcgmBufferedFvCursor_t blankCursor = 0;
                dcgmBufferedFvCursor_t errorCursor = 0;
                CHECK(blankBuffer.GetNextFv(&blankCursor) == nullptr);
                CHECK(errorBuffer.GetNextFv(&errorCursor) == nullptr);
            }
        }
    }
}


TEST_CASE("DcgmCacheManager::UpdateAllFields coalesces cached same-key PPCNT PLR watches")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    std::array<nvmlPRMCounter_v1_t, 2> counters {};
    counters[0].inData.localPort                = PRM_LOCAL_PORT_MIN;
    counters[0].counterId                       = NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES;
    counters[0].counterValue.status             = NVML_SUCCESS;
    counters[0].counterValue.outputType         = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    counters[0].counterValue.outputValue.ullVal = 0x100000002ULL;
    counters[1].inData.localPort                = PRM_LOCAL_PORT_MIN;
    counters[1].counterId                       = NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_EVENTS;
    counters[1].counterValue.status             = NVML_SUCCESS;
    counters[1].counterValue.outputType         = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    counters[1].counterValue.outputValue.ullVal = 0x500000006ULL;
    nvmlPRMCounterList_v1_t counterList { .numCounters = static_cast<unsigned int>(counters.size()),
                                          .counters    = counters.data() };
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::array<unsigned short, 2> constexpr watchedFields = {
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL,
    };
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5578);
    timelib64_t constexpr monitorIntervalUsec = 1000000;
    for (auto fieldId : watchedFields)
    {
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           fieldId,
                                           monitorIntervalUsec,
                                           14400.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
    }

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    for (auto fieldId : watchedFields)
    {
        dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, fieldId, 0);
        REQUIRE(watchInfo != nullptr);
        REQUIRE(watchInfo->fetchCount == 1);
        REQUIRE(watchInfo->lastQueriedUsec > 0);
    }
    REQUIRE(cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL, 0)
            == nullptr);

    dcgmcm_sample_t rxCodeSample {};
    REQUIRE(cacheManager.GetLatestSample(
                DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL, &rxCodeSample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(rxCodeSample.val.i64 == 0x100000002LL);

    dcgmcm_sample_t retryEventSample {};
    REQUIRE(cacheManager.GetLatestSample(
                DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL, &retryEventSample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(retryEventSample.val.i64 == 0x500000006LL);

    dcgmcm_sample_t unwatchedSiblingSample {};
    REQUIRE(
        cacheManager.GetLatestSample(
            DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL, &unwatchedSiblingSample, nullptr)
        != DCGM_ST_OK);
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples coalesces same-key PPCNT PLR requests")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    std::array<nvmlPRMCounter_v1_t, 2> counters {};
    counters[0].inData.localPort                = PRM_LOCAL_PORT_MIN;
    counters[0].counterId                       = NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES;
    counters[0].counterValue.status             = NVML_SUCCESS;
    counters[0].counterValue.outputType         = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    counters[0].counterValue.outputValue.ullVal = 0x100000002ULL;
    counters[1].inData.localPort                = PRM_LOCAL_PORT_MIN;
    counters[1].counterId                       = NVML_PRM_COUNTER_ID_PPCNT_PLR_XMIT_RETRY_EVENTS;
    counters[1].counterValue.status             = NVML_SUCCESS;
    counters[1].counterValue.outputType         = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    counters[1].counterValue.outputValue.ullVal = 0x500000006ULL;
    nvmlPRMCounterList_v1_t counterList { .numCounters = static_cast<unsigned int>(counters.size()),
                                          .counters    = counters.data() };
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL,
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    std::map<unsigned short, std::vector<long long>> valuesByFieldId;
    dcgmBufferedFvCursor_t cursor = 0;
    while (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor))
    {
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        REQUIRE(fv->status == DCGM_ST_OK);
        valuesByFieldId[fv->fieldId].push_back(fv->value.i64);
    }

    REQUIRE(valuesByFieldId.size() == 2);
    REQUIRE(valuesByFieldId.at(DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL)
            == std::vector<long long> { 0x100000002LL, 0x100000002LL });
    REQUIRE(valuesByFieldId.at(DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL)
            == std::vector<long long> { 0x500000006LL });
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples coalesces same-key PRM counter requests")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    nvmlPRMCounter_v1_t counter {};
    auto counterList = MakeInjectedPrmCounterResponse(counter, 42);
    std::array<injectNvmlRet_t, 2> injectedReturns {};
    for (auto &injectedReturn : injectedReturns)
    {
        injectedReturn.nvmlRet                           = NVML_SUCCESS;
        injectedReturn.valueCount                        = 1;
        injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
        injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    }
    REQUIRE(nvmlDeviceInjectForFollowingCalls(
                nvmlDevice, "ReadPRMCounters", nullptr, 0, injectedReturns.data(), injectedReturns.size())
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmBufferedFvCursor_t cursor = 0;
    for (int i = 0; i < 2; ++i)
    {
        dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY);
        REQUIRE(fv->status == DCGM_ST_OK);
        REQUIRE(fv->value.i64 == 42);
    }
    REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);
}

TEST_CASE("DcgmCacheManager::UpdateAllFields caches deferred PRM counter watches")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    nvmlPRMCounter_v1_t counter {};
    auto counterList = MakeInjectedPrmCounterResponse(counter, 42);
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5581);
    bool wereFirstWatcher                          = false;
    timelib64_t constexpr monitorIntervalUsec      = 1000000;
    unsigned short constexpr watchedCounterFieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
    REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                       link.raw,
                                       watchedCounterFieldId,
                                       monitorIntervalUsec,
                                       14400.0,
                                       2,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher)
            == DCGM_ST_OK);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmcm_sample_t sample {};
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, watchedCounterFieldId, &sample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(sample.val.i64 == 42);
}

TEST_CASE("DcgmCacheManager::UpdateAllFields caches deferred PRM counter handle failures")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    REQUIRE(cacheManager.GetGpuCount(0) > 0);
    unsigned int const gpuId = cacheManager.GetGpuCount(0);
    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5584);
    bool wereFirstWatcher                          = false;
    timelib64_t constexpr monitorIntervalUsec      = 1000000;
    unsigned short constexpr watchedCounterFieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
    REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                       link.raw,
                                       watchedCounterFieldId,
                                       monitorIntervalUsec,
                                       14400.0,
                                       2,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher)
            == DCGM_ST_OK);

    dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, watchedCounterFieldId, 0);
    REQUIRE(watchInfo != nullptr);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 0);
    CHECK(watchInfo->fetchCount == 1);
    CHECK(watchInfo->lastQueriedUsec > 0);

    dcgmcm_sample_t sample {};
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, watchedCounterFieldId, &sample, nullptr)
            == DCGM_ST_OK);
    CHECK(sample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
}

SCENARIO("DcgmCacheManager::ActuallyUpdateAllFields keeps m_mutex held when no GPU field values are buffered")
{
    GIVEN("one due NVLink PRM counter watch and no buffered GPU mapped fields")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);

        nvmlPRMCounter_v1_t counter {};
        auto counterList = MakeInjectedPrmCounterResponse(counter, 42);
        injectNvmlRet_t injectedReturn {};
        injectedReturn.nvmlRet                           = NVML_SUCCESS;
        injectedReturn.valueCount                        = 1;
        injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
        injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
        REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7751);
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
                                           1000000,
                                           14400.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);

        WHEN("the deferred PRM counter batch update runs")
        {
            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            auto const [ret, mutexPoll] = cacheManager.RunOneUpdateCycleAndPollMutex();

            THEN("the callee returns success and leaves the caller lock guard's mutex held")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(mutexPoll == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }
    }
}

SCENARIO("DcgmCacheManager::UpdateAllFields batches due PRM counter watches per GPU")
{
    GIVEN("two due PRM counter watches on one GPU")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);

        std::array<nvmlPRMCounter_v1_t, 2> counters {};
        counters[0].inData.localPort               = PRM_LOCAL_PORT_MIN;
        counters[0].counterId                      = NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_SINCE_LAST_RECOVERY;
        counters[0].counterValue.status            = NVML_SUCCESS;
        counters[0].counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
        counters[0].counterValue.outputValue.uiVal = 84;
        counters[1].inData.localPort               = PRM_LOCAL_PORT_MIN;
        counters[1].counterId                      = NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY;
        counters[1].counterValue.status            = NVML_SUCCESS;
        counters[1].counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
        counters[1].counterValue.outputValue.uiVal = 42;
        nvmlPRMCounterList_v1_t counterList {};
        counterList.numCounters = counters.size();
        counterList.counters    = counters.data();

        injectNvmlRet_t injectedReturn {};
        injectedReturn.nvmlRet                           = NVML_SUCCESS;
        injectedReturn.valueCount                        = 1;
        injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
        injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
        REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7589);
        bool wereFirstWatcher = false;
        for (unsigned short const fieldId :
             { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS })
        {
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        WHEN("the counter list returns the requested entries in reverse order")
        {
            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            cacheManager.RunOneUpdateCycleForTest();

            THEN("one read updates each watch by returned identity")
            {
                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
                for (auto const &[fieldId, expectedValue] :
                     std::array { std::pair { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY, 42LL },
                                  std::pair { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS, 84LL } })
                {
                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, fieldId, &sample, nullptr)
                            == DCGM_ST_OK);
                    CHECK(sample.val.i64 == expectedValue);
                }
            }
        }
    }
}

SCENARIO("DcgmCacheManager: PPCNT recovery last gap is published in seconds")
{
    GIVEN("NVML returns milliseconds between recoveries")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7590);
        bool wereFirstWatcher = false;
        for (unsigned short const fieldId : { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
                                              DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS })
        {
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        WHEN("both the legacy-millisecond and seconds fields are updated from one NVML identity")
        {
            std::array const conversionCases { std::pair { 0u, 0.0 },        std::pair { 1u, 0.001 },
                                               std::pair { 999u, 0.999 },    std::pair { 1000u, 1.0 },
                                               std::pair { 1999u, 1.999 },   std::pair { 2500u, 2.5 },
                                               std::pair { 65534u, 65.534 }, std::pair { 65535u, 65.535 } };

            for (auto const &[msValue, expectedSeconds] : conversionCases)
            {
                CAPTURE(msValue, expectedSeconds);

                for (unsigned short const fieldId : { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
                                                      DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS })
                {
                    dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, fieldId, 0);
                    REQUIRE(watchInfo != nullptr);
                    watchInfo->lastQueriedUsec = 0;
                }

                nvmlPRMCounter_v1_t counter {};
                counter.inData.localPort    = PRM_LOCAL_PORT_MIN;
                counter.counterId           = NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES;
                counter.counterValue.status = NVML_SUCCESS;
                counter.counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
                counter.counterValue.outputValue.uiVal = msValue;
                nvmlPRMCounterList_v1_t counterList {};
                counterList.numCounters = 1;
                counterList.counters    = &counter;

                injectNvmlRet_t injectedReturn {};
                injectedReturn.nvmlRet                           = NVML_SUCCESS;
                injectedReturn.valueCount                        = 1;
                injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
                injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
                REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                        == NVML_SUCCESS);

                REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
                cacheManager.RunOneUpdateCycleForTest();

                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
                dcgmcm_sample_t msSample {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK,
                                                     link.raw,
                                                     DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
                                                     &msSample,
                                                     nullptr)
                        == DCGM_ST_OK);
                CHECK(msSample.val.i64 == static_cast<long long>(msValue));

                dcgmcm_sample_t secondsSample {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK,
                                                     link.raw,
                                                     DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS,
                                                     &secondsSample,
                                                     nullptr)
                        == DCGM_ST_OK);
                CHECK_THAT(secondsSample.val.d, Catch::Matchers::WithinAbs(expectedSeconds, 1.0e-12));
            }
        }
    }

    GIVEN("NVML returns a blank sentinel for the recovery-interval counter")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7591);
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS,
                                           1000000,
                                           14400.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);

        WHEN("the recovery-interval-seconds field is updated")
        {
            std::array const blankCases {
                std::pair { NVML_ERROR_UNKNOWN, DCGM_FP64_BLANK },
                std::pair { NVML_ERROR_NOT_SUPPORTED, DCGM_FP64_NOT_SUPPORTED },
                std::pair { NVML_ERROR_NOT_FOUND, DCGM_FP64_NOT_FOUND },
                std::pair { NVML_ERROR_NO_PERMISSION, DCGM_FP64_NOT_PERMISSIONED },
            };

            for (auto const &[nvmlStatus, expectedBlank] : blankCases)
            {
                CAPTURE(nvmlStatus, expectedBlank);

                dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(
                    DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS, 0);
                REQUIRE(watchInfo != nullptr);
                watchInfo->lastQueriedUsec = 0;

                nvmlPRMCounter_v1_t counter {};
                counter.inData.localPort    = PRM_LOCAL_PORT_MIN;
                counter.counterId           = NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES;
                counter.counterValue.status = nvmlStatus;
                counter.counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
                counter.counterValue.outputValue.uiVal = 2500;
                nvmlPRMCounterList_v1_t counterList {};
                counterList.numCounters = 1;
                counterList.counters    = &counter;

                injectNvmlRet_t injectedReturn {};
                injectedReturn.nvmlRet                           = NVML_SUCCESS;
                injectedReturn.valueCount                        = 1;
                injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
                injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
                REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                        == NVML_SUCCESS);

                REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
                cacheManager.RunOneUpdateCycleForTest();

                dcgmcm_sample_t sample {};
                REQUIRE(
                    cacheManager.GetLatestSample(
                        DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS, &sample, nullptr)
                    == DCGM_ST_OK);
                CHECK(sample.val.d == expectedBlank);
            }
        }
    }
}

SCENARIO("DcgmCacheManager: unwatching the millisecond sibling still publishes the seconds field")
{
    GIVEN("both dual-publish recovery fields are watched")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7593);
        bool wereFirstWatcher = false;
        for (unsigned short const fieldId : { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
                                              DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS })
        {
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        WHEN("the millisecond field is unwatched before the update")
        {
            REQUIRE(cacheManager.RemoveFieldWatch(
                        DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO, false, watcher)
                    == DCGM_ST_OK);

            dcgmcm_watch_info_p gapWatch = cacheManager.GetEntityWatchInfo(
                DCGM_FE_LINK, link.raw, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS, 0);
            REQUIRE(gapWatch != nullptr);
            gapWatch->lastQueriedUsec = 0;

            nvmlPRMCounter_v1_t counter {};
            counter.inData.localPort        = PRM_LOCAL_PORT_MIN;
            counter.counterId               = NVML_PRM_COUNTER_ID_PPCNT_RECOVERY_CTRS_TIME_BETWEEN_LAST_TWO_RECOVERIES;
            counter.counterValue.status     = NVML_SUCCESS;
            counter.counterValue.outputType = NVML_VALUE_TYPE_UNSIGNED_INT;
            counter.counterValue.outputValue.uiVal = 2500;
            nvmlPRMCounterList_v1_t counterList {};
            counterList.numCounters = 1;
            counterList.counters    = &counter;

            injectNvmlRet_t injectedReturn {};
            injectedReturn.nvmlRet                           = NVML_SUCCESS;
            injectedReturn.valueCount                        = 1;
            injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
            injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
            REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                    == NVML_SUCCESS);

            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            cacheManager.RunOneUpdateCycleForTest();

            THEN("the seconds field is still read and scaled")
            {
                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
                dcgmcm_sample_t secondsSample {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK,
                                                     link.raw,
                                                     DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS,
                                                     &secondsSample,
                                                     nullptr)
                        == DCGM_ST_OK);
                CHECK_THAT(secondsSample.val.d, Catch::Matchers::WithinAbs(2.5, 1.0e-12));
            }
        }
    }
}

SCENARIO("DeferredPrmCounterApply: rejects stale deferred results")
{
    GIVEN("a snapshot generation")
    {
        uint64_t constexpr snapshotGeneration = 42;

        WHEN("the current watch is inactive or has changed generation")
        {
            THEN("the deferred result cannot apply")
            {
                CHECK_FALSE(DcgmNs::Cache::CanApplyDeferredPrmCounterResult(
                    { .isWatched = false, .generation = snapshotGeneration }, snapshotGeneration));
                CHECK_FALSE(DcgmNs::Cache::CanApplyDeferredPrmCounterResult(
                    { .isWatched = true, .generation = snapshotGeneration + 1 }, snapshotGeneration));
            }
        }

        WHEN("the current watch retains the snapshot generation")
        {
            THEN("the deferred result can apply")
            {
                CHECK(DcgmNs::Cache::CanApplyDeferredPrmCounterResult(
                    { .isWatched = true, .generation = snapshotGeneration }, snapshotGeneration));
            }
        }
    }
}

SCENARIO("DeferredPrmCounterApply: omits dual-publish siblings using the still-queued set")
{
    struct Key
    {
        dcgm_field_eid_t linkEntityId {};
        unsigned short fieldId {};
        auto operator<=>(Key const &) const = default;
    };

    dcgm_field_eid_t constexpr linkEntityId = 1;
    unsigned short const msFieldId          = DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO;
    unsigned short const gapFieldId         = DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_LAST_GAP_SECONDS;
    Key const msKey { .linkEntityId = linkEntityId, .fieldId = msFieldId };
    Key const gapKey { .linkEntityId = linkEntityId, .fieldId = gapFieldId };

    GIVEN("an unpaired field ID")
    {
        std::set<Key> const queuedKeys { Key { .linkEntityId = linkEntityId,
                                               .fieldId      = DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS } };

        THEN("the field is not omitted")
        {
            CHECK_FALSE(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(
                queuedKeys,
                Key { .linkEntityId = linkEntityId, .fieldId = DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS }));
        }
    }

    GIVEN("an empty queued set")
    {
        std::set<Key> const queuedKeys {};

        THEN("the higher dual-publish field is not omitted")
        {
            CHECK_FALSE(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(queuedKeys, gapKey));
        }
    }

    GIVEN("both dual-publish fields remain in the snapshot for one link")
    {
        std::set<Key> const snapshotKeys { msKey, gapKey };

        THEN("the higher field ID is omitted from the NVML batch")
        {
            CHECK_FALSE(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(snapshotKeys, msKey));
            CHECK(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(snapshotKeys, gapKey));
        }
    }

    GIVEN("the millisecond sibling is absent from the snapshot")
    {
        std::set<Key> const snapshotKeys { gapKey };

        THEN("the seconds field is requested from NVML")
        {
            CHECK_FALSE(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(snapshotKeys, gapKey));
        }
    }

    GIVEN("the millisecond sibling is queued on a different link")
    {
        std::set<Key> const snapshotKeys { Key { .linkEntityId = 2, .fieldId = msFieldId }, gapKey };

        THEN("the seconds field is requested from NVML")
        {
            CHECK_FALSE(DcgmNs::Cache::ShouldOmitDualPublishSiblingFromBatch(snapshotKeys, gapKey));
        }
    }
}

SCENARIO("DcgmCacheManager::UpdateAllFields leaves an omitted PRM counter result blank")
{
    GIVEN("two watched counters with one injected result")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);
        nvmlPRMCounter_v1_t counter {};
        auto counterList = MakeInjectedPrmCounterResponse(counter, 42);
        injectNvmlRet_t injectedReturn {};
        injectedReturn.nvmlRet                           = NVML_SUCCESS;
        injectedReturn.valueCount                        = 1;
        injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
        injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
        REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7592);
        bool wereFirstWatcher = false;
        for (unsigned short const fieldId :
             { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY, DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS })
        {
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        WHEN("the update receives the incomplete list")
        {
            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            cacheManager.RunOneUpdateCycleForTest();

            THEN("the missing identity is cached as blank")
            {
                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
                std::array<std::pair<unsigned short, int64_t>, 2> const expectedValues {
                    std::pair { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY, int64_t { 42 } },
                    std::pair { DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS, int64_t { DCGM_INT64_BLANK } },
                };
                for (auto const &[fieldId, expectedValue] : expectedValues)
                {
                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, fieldId, &sample, nullptr)
                            == DCGM_ST_OK);
                    CHECK(sample.val.i64 == expectedValue);
                }
            }
        }
    }
}

SCENARIO("DcgmCacheManager::UpdateAllFields leaves duplicate PRM counter results blank")
{
    GIVEN("one watched counter with duplicate injected identities")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(0, &nvmlDevice) == NVML_SUCCESS);
        std::array<nvmlPRMCounter_v1_t, 2> counters {};
        auto counterList                           = MakeInjectedPrmCounterResponse(counters[0], 42);
        counters[1]                                = counters[0];
        counters[1].counterValue.outputValue.uiVal = 84;
        counterList.numCounters                    = counters.size();
        counterList.counters                       = counters.data();
        injectNvmlRet_t injectedReturn {};
        injectedReturn.nvmlRet                           = NVML_SUCCESS;
        injectedReturn.valueCount                        = 1;
        injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
        injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
        REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                == NVML_SUCCESS);

        dcgm_link_t link {};
        link.parsed.type                 = DCGM_FE_GPU;
        link.parsed.gpuId                = 0;
        link.parsed.index                = 0;
        unsigned short constexpr fieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7593);
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(
                    DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                == DCGM_ST_OK);

        WHEN("the update receives the duplicate list")
        {
            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            cacheManager.RunOneUpdateCycleForTest();

            THEN("the duplicate identity is cached as blank")
            {
                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
                dcgmcm_sample_t sample {};
                REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, fieldId, &sample, nullptr) == DCGM_ST_OK);
                CHECK(sample.val.i64 == DCGM_INT64_BLANK);
            }
        }
    }
}

SCENARIO("DcgmCacheManager::UpdateAllFields partitions PRM counter batches by GPU")
{
    GIVEN("one watched counter on each of two injected GPUs")
    {
        auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
        if (!restoreEnv)
        {
            SKIP("Sku file not found: B200.yaml");
        }

        ScopedDcgmFields dcgmFields;
        REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
        DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

        std::array<nvmlPRMCounter_v1_t, 2> counters {};
        std::array<nvmlPRMCounterList_v1_t, 2> counterLists {};
        for (unsigned int gpuId = 0; gpuId < counters.size(); ++gpuId)
        {
            nvmlDevice_t nvmlDevice {};
            REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
            counterLists[gpuId] = MakeInjectedPrmCounterResponse(counters[gpuId], 42 + 42 * gpuId);
            injectNvmlRet_t injectedReturn {};
            injectedReturn.nvmlRet                           = NVML_SUCCESS;
            injectedReturn.valueCount                        = 1;
            injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
            injectedReturn.values[0].value.PRMCounterList_v1 = counterLists[gpuId];
            REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
                    == NVML_SUCCESS);
        }

        unsigned short constexpr fieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 7594);
        bool wereFirstWatcher = false;
        for (unsigned int const gpuId : { 0U, 1U })
        {
            dcgm_link_t link {};
            link.parsed.type  = DCGM_FE_GPU;
            link.parsed.gpuId = gpuId;
            link.parsed.index = 0;
            REQUIRE(cacheManager.AddFieldWatch(
                        DCGM_FE_LINK, link.raw, fieldId, 1000000, 14400.0, 2, watcher, false, false, wereFirstWatcher)
                    == DCGM_ST_OK);
        }

        WHEN("the update reads both GPUs")
        {
            REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
            cacheManager.RunOneUpdateCycleForTest();

            THEN("each GPU receives its own counter-list read and value")
            {
                CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 2);
                for (auto const &[gpuId, expectedValue] : std::array { std::pair { 0U, 42LL }, std::pair { 1U, 84LL } })
                {
                    dcgm_link_t link {};
                    link.parsed.type  = DCGM_FE_GPU;
                    link.parsed.gpuId = gpuId;
                    link.parsed.index = 0;
                    dcgmcm_sample_t sample {};
                    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, fieldId, &sample, nullptr)
                            == DCGM_ST_OK);
                    CHECK(sample.val.i64 == expectedValue);
                }
            }
        }
    }
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples returns deferred PRM counter errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet = NVML_ERROR_UNKNOWN;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmBufferedFvCursor_t cursor = 0;
    for (int i = 0; i < 2; ++i)
    {
        dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY);
        REQUIRE(fv->status == DCGM_ST_NVML_ERROR);
        REQUIRE(fv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
    }
    REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples returns per-counter PRM counter errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    nvmlPRMCounter_v1_t counter {};
    auto counterList = MakeInjectedPrmCounterResponse(counter, 42, NVML_ERROR_UNKNOWN);
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmBufferedFvCursor_t cursor = 0;
    for (int i = 0; i < 2; ++i)
    {
        dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY);
        REQUIRE(fv->status == DCGM_ST_NVML_ERROR);
        REQUIRE(fv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
    }
    REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples returns unresolved PRM counter errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    nvmlPRMCounter_v1_t counter {};
    auto counterList = MakeInjectedPrmCounterResponse(counter, 42);
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);

    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmBufferedFvCursor_t cursor = 0;
    bool sawResolved              = false;
    bool sawUnresolved            = false;
    unsigned int count            = 0;
    while (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor))
    {
        ++count;
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY:
                REQUIRE(fv->status == DCGM_ST_OK);
                REQUIRE(fv->value.i64 == 42);
                sawResolved = true;
                break;
            case DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS:
                REQUIRE(fv->status == DCGM_ST_NVML_ERROR);
                REQUIRE(fv->value.i64 == DCGM_INT64_BLANK);
                sawUnresolved = true;
                break;
            default:
                FAIL("Unexpected field returned from live PRM counter batch");
        }
    }

    REQUIRE(count == 2);
    REQUIRE(sawResolved);
    REQUIRE(sawUnresolved);
}

TEST_CASE("DcgmCacheManager::GetMultipleLatestLiveSamples returns PRM counter handle lookup errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    REQUIRE(cacheManager.GetGpuCount(0) > 0);
    unsigned int const gpuId = cacheManager.GetGpuCount(0);
    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::vector<dcgmGroupEntityPair_t> entities = { { DCGM_FE_LINK, link.raw } };
    std::vector<unsigned short> fieldIds        = {
        DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS,
    };
    DcgmFvBuffer fvBuffer(1024);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    REQUIRE(cacheManager.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer) == DCGM_ST_OK);
    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 0);

    dcgmBufferedFvCursor_t cursor = 0;
    bool sawOperRecovery          = false;
    bool sawRecoveryTime          = false;
    unsigned int count            = 0;
    while (dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor))
    {
        ++count;
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == link.raw);
        REQUIRE(fv->status == DCGM_ST_BADPARAM);
        REQUIRE(fv->value.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY:
                sawOperRecovery = true;
                break;
            case DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SINCE_LAST_SECONDS:
                sawRecoveryTime = true;
                break;
            default:
                FAIL("Unexpected field returned from live PRM counter handle lookup failure");
        }
    }

    REQUIRE(count == 2);
    REQUIRE(sawOperRecovery);
    REQUIRE(sawRecoveryTime);
}

TEST_CASE("DcgmCacheManager::UpdateAllFields retries deferred PRM counter errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet = NVML_ERROR_UNKNOWN;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5582);
    bool wereFirstWatcher                          = false;
    timelib64_t constexpr monitorIntervalUsec      = 1000000;
    unsigned short constexpr watchedCounterFieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
    REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                       link.raw,
                                       watchedCounterFieldId,
                                       monitorIntervalUsec,
                                       14400.0,
                                       2,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher)
            == DCGM_ST_OK);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmcm_watch_info_p watchInfo = cacheManager.GetEntityWatchInfo(DCGM_FE_LINK, link.raw, watchedCounterFieldId, 0);
    REQUIRE(watchInfo != nullptr);
    REQUIRE(watchInfo->fetchCount == 1);
    REQUIRE(watchInfo->lastQueriedUsec > 0);

    dcgmcm_sample_t sample {};
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, watchedCounterFieldId, &sample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(sample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));

    nvmlPRMCounter_v1_t counter {};
    auto counterList                                 = MakeInjectedPrmCounterResponse(counter, 42);
    injectedReturn                                   = {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    watchInfo->lastQueriedUsec = 0;
    REQUIRE(cacheManager.RunOneUpdateCycle() == DCGM_ST_OK);
    REQUIRE(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 2);

    REQUIRE(watchInfo->fetchCount == 2);
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, watchedCounterFieldId, &sample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(sample.val.i64 == 42);
}

TEST_CASE("DcgmCacheManager::UpdateAllFields caches blank deferred PRM counter result errors")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    nvmlPRMCounter_v1_t counter {};
    auto counterList = MakeInjectedPrmCounterResponse(counter, 42, NVML_ERROR_UNKNOWN);
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5583);
    bool wereFirstWatcher                          = false;
    timelib64_t constexpr monitorIntervalUsec      = 1000000;
    unsigned short constexpr watchedCounterFieldId = DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY;
    REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                       link.raw,
                                       watchedCounterFieldId,
                                       monitorIntervalUsec,
                                       14400.0,
                                       2,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher)
            == DCGM_ST_OK);

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    dcgmcm_sample_t sample {};
    REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, watchedCounterFieldId, &sample, nullptr)
            == DCGM_ST_OK);
    REQUIRE(sample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
}

TEST_CASE("DcgmCacheManager::UpdateAllFields keeps PPCNT PLR keys separated by link entity")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    std::array<nvmlPRMCounter_v1_t, 2> counters {};
    for (unsigned int i = 0; i < counters.size(); ++i)
    {
        counters[i].inData.localPort               = i + PRM_LOCAL_PORT_MIN;
        counters[i].counterId                      = NVML_PRM_COUNTER_ID_PPCNT_PLR_RCV_CODES;
        counters[i].counterValue.status            = NVML_SUCCESS;
        counters[i].counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
        counters[i].counterValue.outputValue.uiVal = i + 1;
    }
    nvmlPRMCounterList_v1_t counterList { .numCounters = static_cast<unsigned int>(counters.size()),
                                          .counters    = counters.data() };
    injectNvmlRet_t injectedReturn {};
    injectedReturn.nvmlRet                           = NVML_SUCCESS;
    injectedReturn.valueCount                        = 1;
    injectedReturn.values[0].type                    = INJECTION_PRMCOUNTERLIST_V1;
    injectedReturn.values[0].value.PRMCounterList_v1 = counterList;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(nvmlDevice, "ReadPRMCounters", nullptr, 0, &injectedReturn, 1)
            == NVML_SUCCESS);

    std::array<dcgm_link_t, 2> links {};
    for (unsigned int i = 0; i < links.size(); ++i)
    {
        links[i].parsed.type  = DCGM_FE_GPU;
        links[i].parsed.gpuId = gpuId;
        links[i].parsed.index = i;
    }

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5580);
    timelib64_t constexpr monitorIntervalUsec = 1000000;
    for (auto const &link : links)
    {
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
                                           monitorIntervalUsec,
                                           14400.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
    }

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);
}

TEST_CASE("DcgmCacheManager::UpdateAllFields fans out cached same-key PPCNT PLR read failures")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("B200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: B200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    TestDcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);

    unsigned int constexpr gpuId = 0;
    nvmlDevice_t nvmlDevice {};
    REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);

    std::array<injectNvmlRet_t, 1> injectedReturns {};
    injectedReturns[0].nvmlRet = NVML_ERROR_UNKNOWN;
    REQUIRE(nvmlDeviceInjectForFollowingCalls(
                nvmlDevice, "ReadPRMCounters", nullptr, 0, injectedReturns.data(), injectedReturns.size())
            == NVML_SUCCESS);

    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.gpuId = gpuId;
    link.parsed.index = 0;

    std::array<unsigned short, 2> constexpr watchedFields = {
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL,
    };
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5579);
    timelib64_t constexpr monitorIntervalUsec = 1000000;
    for (auto fieldId : watchedFields)
    {
        bool wereFirstWatcher = false;
        REQUIRE(cacheManager.AddFieldWatch(DCGM_FE_LINK,
                                           link.raw,
                                           fieldId,
                                           monitorIntervalUsec,
                                           14400.0,
                                           2,
                                           watcher,
                                           false,
                                           false,
                                           wereFirstWatcher)
                == DCGM_ST_OK);
    }

    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
    cacheManager.RunOneUpdateCycleForTest();

    CHECK(GetNvmlFuncCallCount("nvmlDeviceReadPRMCounters_v1") == 1);

    for (auto fieldId : watchedFields)
    {
        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_LINK, link.raw, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_UNKNOWN));
    }
}

SCENARIO("CacheManager: MiB fields convert to bytes")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    enum class Bar1Member
    {
        Total,
        Used,
        Free,
    };

    struct MibToBytesField
    {
        unsigned short legacyFieldId; // MiB
        unsigned short bytesFieldId;  // bytes
        Bar1Member member;
    };

    // Add future MiB→bytes migrations that share BAR1MemoryInfo injection here.
    constexpr std::array fields {
        MibToBytesField { DCGM_FI_DEV_BAR1_TOTAL, DCGM_FI_DEV_BAR1_CAPACITY_BYTES, Bar1Member::Total },
        MibToBytesField { DCGM_FI_DEV_BAR1_USED, DCGM_FI_DEV_BAR1_USED_BYTES, Bar1Member::Used },
        MibToBytesField { DCGM_FI_DEV_BAR1_FREE, DCGM_FI_DEV_BAR1_FREE_BYTES, Bar1Member::Free },
    };

    constexpr unsigned long long mib = 1024ull * 1024ull;

    struct ConversionCase
    {
        unsigned long long totalBytes;
        unsigned long long usedBytes;
        unsigned long long freeBytes;
    };

    constexpr std::array conversionCases {
        ConversionCase { 12345ull * mib, 45ull * mib, 12300ull * mib },
        ConversionCase { 0ull, 0ull, 0ull },
        ConversionCase { 1ull * mib, 1ull * mib, 0ull },
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5581);

        auto injectBar1 = [&](nvmlReturn_t nvmlRet, ConversionCase const &values) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            injectNvmlRet_t injectedRet {};
            injectedRet.nvmlRet                              = nvmlRet;
            injectedRet.valueCount                           = 1;
            injectedRet.values[0].type                       = INJECTION_BAR1MEMORY;
            injectedRet.values[0].value.BAR1Memory.bar1Total = values.totalBytes;
            injectedRet.values[0].value.BAR1Memory.bar1Used  = values.usedBytes;
            injectedRet.values[0].value.BAR1Memory.bar1Free  = values.freeBytes;
            REQUIRE(nvmlDeviceInject(nvmlDevice, "BAR1MemoryInfo", nullptr, 0, &injectedRet) == NVML_SUCCESS);
        };

        auto bytesFor = [](MibToBytesField const &field, ConversionCase const &values) -> unsigned long long {
            switch (field.member)
            {
                case Bar1Member::Total:
                    return values.totalBytes;
                case Bar1Member::Used:
                    return values.usedBytes;
                case Bar1Member::Free:
                    return values.freeBytes;
            }
            return 0;
        };

        auto samplePair = [&](MibToBytesField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.bytesFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t mibSample {};
            dcgmcm_sample_t bytesSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &mibSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.bytesFieldId, &bytesSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { mibSample, bytesSample };
        };

        THEN("each bytes field is registered as INT64")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.bytesFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_INT64);
            }
        }

        WHEN("NVML reports successful MiB-source values")
        {
            for (auto const &values : conversionCases)
            {
                injectBar1(NVML_SUCCESS, values);

                for (auto const &field : fields)
                {
                    auto const rawBytes           = bytesFor(field, values);
                    auto [mibSample, bytesSample] = samplePair(field);
                    long long const expectedMib   = static_cast<long long>(rawBytes / mib);
                    long long const expectedBytes = static_cast<long long>(rawBytes);

                    CHECK(mibSample.val.i64 == expectedMib);
                    CHECK(bytesSample.val.i64 == expectedBytes);
                    REQUIRE(cacheManager.FreeSamples(&mibSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&bytesSample, 1, field.bytesFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            injectBar1(NVML_ERROR_NOT_SUPPORTED, ConversionCase {});

            for (auto const &field : fields)
            {
                auto [mibSample, bytesSample] = samplePair(field);

                CHECK(mibSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(bytesSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&mibSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&bytesSample, 1, field.bytesFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: FB MiB fields convert to bytes")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    enum class FbMember
    {
        Capacity,
        Used,
        Free,
        Reserved,
    };

    struct MibToBytesField
    {
        unsigned short legacyFieldId; // MiB
        unsigned short bytesFieldId;  // bytes
        FbMember member;
    };

    constexpr std::array fields {
        MibToBytesField { DCGM_FI_DEV_FB_TOTAL, DCGM_FI_DEV_FB_CAPACITY_BYTES, FbMember::Capacity },
        MibToBytesField { DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_FB_USED_BYTES, FbMember::Used },
        MibToBytesField { DCGM_FI_DEV_FB_FREE, DCGM_FI_DEV_FB_FREE_BYTES, FbMember::Free },
        MibToBytesField { DCGM_FI_DEV_FB_RESERVED, DCGM_FI_DEV_FB_RESERVED_BYTES, FbMember::Reserved },
    };

    constexpr unsigned long long mib = 1024ull * 1024ull;

    struct ConversionCase
    {
        unsigned long long totalBytes;
        unsigned long long usedBytes;
        unsigned long long freeBytes;
        unsigned long long reservedBytes;
    };

    constexpr std::array conversionCases {
        ConversionCase { 12345ull * mib, 45ull * mib, 12200ull * mib, 100ull * mib },
        ConversionCase { 0ull, 0ull, 0ull, 0ull },
        ConversionCase { 1ull * mib, 1ull * mib, 0ull, 0ull },
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5583);

        auto injectMemory = [&](nvmlReturn_t nvmlRet, ConversionCase const &values) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            injectNvmlRet_t injectedRet {};
            injectedRet.nvmlRet                            = nvmlRet;
            injectedRet.valueCount                         = 1;
            injectedRet.values[0].type                     = INJECTION_MEMORY_V2;
            injectedRet.values[0].value.Memory_v2.version  = nvmlMemory_v2;
            injectedRet.values[0].value.Memory_v2.total    = values.totalBytes;
            injectedRet.values[0].value.Memory_v2.used     = values.usedBytes;
            injectedRet.values[0].value.Memory_v2.free     = values.freeBytes;
            injectedRet.values[0].value.Memory_v2.reserved = values.reservedBytes;
            REQUIRE(nvmlDeviceInject(nvmlDevice, "MemoryInfo", nullptr, 0, &injectedRet) == NVML_SUCCESS);
        };

        auto bytesFor = [](MibToBytesField const &field, ConversionCase const &values) -> unsigned long long {
            switch (field.member)
            {
                case FbMember::Capacity:
                    return values.totalBytes;
                case FbMember::Used:
                    return values.usedBytes;
                case FbMember::Free:
                    return values.freeBytes;
                case FbMember::Reserved:
                    return values.reservedBytes;
            }
            return 0;
        };

        auto samplePair = [&](MibToBytesField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.bytesFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t mibSample {};
            dcgmcm_sample_t bytesSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &mibSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.bytesFieldId, &bytesSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { mibSample, bytesSample };
        };

        THEN("each bytes field is registered as INT64")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.bytesFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_INT64);
            }
        }

        WHEN("NVML reports successful MiB-source values")
        {
            for (auto const &values : conversionCases)
            {
                injectMemory(NVML_SUCCESS, values);

                for (auto const &field : fields)
                {
                    auto const rawBytes           = bytesFor(field, values);
                    auto [mibSample, bytesSample] = samplePair(field);
                    long long const expectedMib   = static_cast<long long>(rawBytes / mib);
                    long long const expectedBytes = static_cast<long long>(rawBytes);

                    CHECK(mibSample.val.i64 == expectedMib);
                    CHECK(bytesSample.val.i64 == expectedBytes);
                    REQUIRE(cacheManager.FreeSamples(&mibSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&bytesSample, 1, field.bytesFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            injectMemory(NVML_ERROR_NOT_SUPPORTED, ConversionCase {});

            for (auto const &field : fields)
            {
                auto [mibSample, bytesSample] = samplePair(field);

                CHECK(mibSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(bytesSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&mibSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&bytesSample, 1, field.bytesFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: MHz fields convert to hertz")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    struct MhzToHertzField
    {
        unsigned short legacyFieldId; // MHz
        unsigned short hertzFieldId;  // Hz
        char const *nvmlInjectKey;
        nvmlClockType_t clockType; // used when the NVML API takes a clock domain
    };

    // Add future MHz→Hz migrations that share ClockType + UINT injection here.
    constexpr std::array fields {
        MhzToHertzField { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_SM_CLOCK_HERTZ, "ClockInfo", NVML_CLOCK_SM },
        MhzToHertzField { DCGM_FI_DEV_MEM_CLOCK, DCGM_FI_DEV_MEMORY_CLOCK_HERTZ, "ClockInfo", NVML_CLOCK_MEM },
        MhzToHertzField { DCGM_FI_DEV_VIDEO_CLOCK, DCGM_FI_DEV_VIDEO_CLOCK_HERTZ, "ClockInfo", NVML_CLOCK_VIDEO },
        MhzToHertzField {
            DCGM_FI_DEV_APP_SM_CLOCK, DCGM_FI_DEV_SM_APP_CLOCK_HERTZ, "ApplicationsClock", NVML_CLOCK_GRAPHICS },
        MhzToHertzField {
            DCGM_FI_DEV_APP_MEM_CLOCK, DCGM_FI_DEV_MEMORY_APP_CLOCK_HERTZ, "ApplicationsClock", NVML_CLOCK_MEM },
        MhzToHertzField { DCGM_FI_DEV_MAX_SM_CLOCK, DCGM_FI_DEV_SM_MAX_CLOCK_HERTZ, "MaxClockInfo", NVML_CLOCK_SM },
        MhzToHertzField {
            DCGM_FI_DEV_MAX_MEM_CLOCK, DCGM_FI_DEV_MEMORY_MAX_CLOCK_HERTZ, "MaxClockInfo", NVML_CLOCK_MEM },
        MhzToHertzField {
            DCGM_FI_DEV_MAX_VIDEO_CLOCK, DCGM_FI_DEV_VIDEO_MAX_CLOCK_HERTZ, "MaxClockInfo", NVML_CLOCK_VIDEO },
    };

    constexpr long long mhzToHz = 1000000LL;

    constexpr std::array conversionCases {
        0u,
        1u,
        1410u,
        1980u,
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5582);

        auto injectMhz = [&](MhzToHertzField const &field, nvmlReturn_t nvmlRet, unsigned int mhz) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            injectNvmlVal_t extraKey {};
            extraKey.type            = INJECTION_CLOCKTYPE;
            extraKey.value.ClockType = field.clockType;

            injectNvmlRet_t injectedRet {};
            injectedRet.nvmlRet              = nvmlRet;
            injectedRet.valueCount           = 1;
            injectedRet.values[0].type       = INJECTION_UINT;
            injectedRet.values[0].value.UInt = mhz;
            REQUIRE(nvmlDeviceInject(nvmlDevice, field.nvmlInjectKey, &extraKey, 1, &injectedRet) == NVML_SUCCESS);
        };

        auto samplePair = [&](MhzToHertzField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.hertzFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t mhzSample {};
            dcgmcm_sample_t hzSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &mhzSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.hertzFieldId, &hzSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { mhzSample, hzSample };
        };

        THEN("each hertz field is registered as INT64")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.hertzFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_INT64);
            }
        }

        WHEN("NVML reports successful MHz values")
        {
            for (auto const &field : fields)
            {
                for (auto mhz : conversionCases)
                {
                    injectMhz(field, NVML_SUCCESS, mhz);
                    auto [mhzSample, hzSample] = samplePair(field);

                    CHECK(mhzSample.val.i64 == static_cast<long long>(mhz));
                    CHECK(hzSample.val.i64 == static_cast<long long>(mhz) * mhzToHz);
                    REQUIRE(cacheManager.FreeSamples(&mhzSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&hzSample, 1, field.hertzFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            for (auto const &field : fields)
            {
                injectMhz(field, NVML_ERROR_NOT_SUPPORTED, 0);
                auto [mhzSample, hzSample] = samplePair(field);

                CHECK(mhzSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(hzSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&mhzSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&hzSample, 1, field.hertzFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: millijoule fields convert to joules")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    struct MilliJouleToJouleField
    {
        unsigned short legacyFieldId; // mJ
        unsigned short joulesFieldId; // J
        unsigned int nvmlFieldId;
    };

    // Add future mJ→J migrations that share nvmlDeviceInjectFieldValue here.
    constexpr std::array fields {
        MilliJouleToJouleField { DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                 DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL,
                                 NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION },
    };

    constexpr long long milliJouleToJoule = 1000LL;

    constexpr std::array conversionCases {
        0ull,
        1000ull,
        123456789ull,
        999ull, // truncates toward zero when converting to joules
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5583);

        auto injectMilliJoules
            = [&](MilliJouleToJouleField const &field, nvmlReturn_t nvmlRet, unsigned long long milliJoules) {
                  REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

                  nvmlFieldValue_t fv {};
                  fv.fieldId      = field.nvmlFieldId;
                  fv.nvmlReturn   = nvmlRet;
                  fv.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                  fv.value.ullVal = milliJoules;
                  fv.timestamp    = timelib_usecSince1970();
                  REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &fv) == NVML_SUCCESS);
              };

        auto samplePair = [&](MilliJouleToJouleField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.joulesFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t milliJouleSample {};
            dcgmcm_sample_t jouleSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &milliJouleSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.joulesFieldId, &jouleSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { milliJouleSample, jouleSample };
        };

        THEN("each joules field is registered as INT64")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.joulesFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_INT64);
            }
        }

        WHEN("NVML reports successful millijoule values")
        {
            for (auto const &field : fields)
            {
                for (auto milliJoules : conversionCases)
                {
                    injectMilliJoules(field, NVML_SUCCESS, milliJoules);
                    auto [milliJouleSample, jouleSample] = samplePair(field);

                    CHECK(milliJouleSample.val.i64 == static_cast<long long>(milliJoules));
                    CHECK(jouleSample.val.i64 == static_cast<long long>(milliJoules) / milliJouleToJoule);
                    REQUIRE(cacheManager.FreeSamples(&milliJouleSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&jouleSample, 1, field.joulesFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            for (auto const &field : fields)
            {
                injectMilliJoules(field, NVML_ERROR_NOT_SUPPORTED, 0);
                auto [milliJouleSample, jouleSample] = samplePair(field);

                CHECK(milliJouleSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(jouleSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&milliJouleSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&jouleSample, 1, field.joulesFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: MB/s fields convert to bytes per second")
{
    // GB200 records C2C max BW as supported. Skip nvmlDeviceReset: a post-wait UpdateAllFields
    // cycle can otherwise race a reset and cache the SKU default instead of the injection.
    auto restoreEnv = WithNvmlInjectionSkuFile("GB200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: GB200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    struct MbpsToBytesPerSecondField
    {
        unsigned short legacyFieldId; // MB/s
        unsigned short bytesPerSecondFieldId;
        unsigned int nvmlFieldId;
    };

    constexpr std::array fields {
        MbpsToBytesPerSecondField { DCGM_FI_DEV_C2C_MAX_BANDWIDTH,
                                    DCGM_FI_DEV_C2C_MAX_BANDWIDTH_BYTES_PER_SECOND,
                                    NVML_FI_DEV_C2C_LINK_GET_MAX_BW },
    };

    constexpr long long mbToBytes = static_cast<long long>(1e6);

    constexpr std::array conversionCases {
        0u,
        1u,
        450u,
        900u,
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5585);

        auto injectMbps = [&](MbpsToBytesPerSecondField const &field, nvmlReturn_t nvmlRet, unsigned int mbps) {
            nvmlFieldValue_t fv {};
            fv.fieldId     = field.nvmlFieldId;
            fv.nvmlReturn  = nvmlRet;
            fv.valueType   = NVML_VALUE_TYPE_UNSIGNED_INT;
            fv.value.uiVal = mbps;
            fv.timestamp   = timelib_usecSince1970();
            REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &fv) == NVML_SUCCESS);
        };

        auto samplePair = [&](MbpsToBytesPerSecondField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.bytesPerSecondFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t mbpsSample {};
            dcgmcm_sample_t bytesPerSecondSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &mbpsSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(
                        DCGM_FE_GPU, gpuId, field.bytesPerSecondFieldId, &bytesPerSecondSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { mbpsSample, bytesPerSecondSample };
        };

        THEN("each bytes-per-second field is registered as INT64")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.bytesPerSecondFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_INT64);
            }
        }

        WHEN("NVML reports successful MB/s values")
        {
            for (auto const &field : fields)
            {
                for (auto mbps : conversionCases)
                {
                    injectMbps(field, NVML_SUCCESS, mbps);
                    auto [mbpsSample, bytesPerSecondSample] = samplePair(field);

                    CHECK(mbpsSample.val.i64 == static_cast<long long>(mbps));
                    CHECK(bytesPerSecondSample.val.i64 == static_cast<long long>(mbps) * mbToBytes);
                    REQUIRE(cacheManager.FreeSamples(&mbpsSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&bytesPerSecondSample, 1, field.bytesPerSecondFieldId)
                            == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            for (auto const &field : fields)
            {
                injectMbps(field, NVML_ERROR_NOT_SUPPORTED, 0);
                auto [mbpsSample, bytesPerSecondSample] = samplePair(field);

                CHECK(mbpsSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(bytesPerSecondSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&mbpsSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&bytesPerSecondSample, 1, field.bytesPerSecondFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: percent fields convert to ratio")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    enum class PercentInjectKind
    {
        UIntKey,            // nvmlDeviceInject(key, INJECTION_UINT)
        UtilizationGpu,     // UtilizationRates.gpu
        UtilizationMemory,  // UtilizationRates.memory
        EncoderUtilization, // EncoderUtilization (util + samplingPeriodUs)
        DecoderUtilization, // DecoderUtilization (util + samplingPeriodUs)
    };

    struct PercentToRatioField
    {
        unsigned short legacyFieldId;
        unsigned short ratioFieldId;
        char const *nvmlInjectKey;
        PercentInjectKind injectKind;
    };

    // Add future percent→ratio migrations here.
    constexpr std::array fields {
        PercentToRatioField {
            DCGM_FI_DEV_FAN_SPEED, DCGM_FI_DEV_FAN_SPEED_RATIO, "FanSpeed", PercentInjectKind::UIntKey },
        PercentToRatioField {
            DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_GPU_UTIL_RATIO, "UtilizationRates", PercentInjectKind::UtilizationGpu },
        PercentToRatioField { DCGM_FI_DEV_MEM_COPY_UTIL,
                              DCGM_FI_DEV_MEMORY_UTIL_RATIO,
                              "UtilizationRates",
                              PercentInjectKind::UtilizationMemory },
        PercentToRatioField { DCGM_FI_DEV_ENC_UTIL,
                              DCGM_FI_DEV_ENC_UTIL_RATIO,
                              "EncoderUtilization",
                              PercentInjectKind::EncoderUtilization },
        PercentToRatioField { DCGM_FI_DEV_DEC_UTIL,
                              DCGM_FI_DEV_DEC_UTIL_RATIO,
                              "DecoderUtilization",
                              PercentInjectKind::DecoderUtilization },
    };

    constexpr std::array conversionCases {
        std::pair { 0u, 0.0 },
        std::pair { 100u, 1.0 },
        std::pair { 110u, 1.1 },
        std::pair { 50u, 0.5 },
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5580);

        auto injectPercent = [&](PercentToRatioField const &field, nvmlReturn_t nvmlRet, unsigned int percent) {
            REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

            injectNvmlRet_t injectedRet {};
            injectedRet.nvmlRet = nvmlRet;

            if (field.injectKind == PercentInjectKind::UtilizationGpu
                || field.injectKind == PercentInjectKind::UtilizationMemory)
            {
                injectedRet.valueCount     = 1;
                injectedRet.values[0].type = INJECTION_UTILIZATION;
                injectedRet.values[0].value.Utilization.gpu
                    = (field.injectKind == PercentInjectKind::UtilizationGpu) ? percent : 0;
                injectedRet.values[0].value.Utilization.memory
                    = (field.injectKind == PercentInjectKind::UtilizationMemory) ? percent : 0;
            }
            else if (field.injectKind == PercentInjectKind::EncoderUtilization
                     || field.injectKind == PercentInjectKind::DecoderUtilization)
            {
                injectedRet.valueCount           = 2;
                injectedRet.values[0].type       = INJECTION_UINT;
                injectedRet.values[0].value.UInt = percent;
                injectedRet.values[1].type       = INJECTION_UINT;
                injectedRet.values[1].value.UInt = 0; // samplingPeriodUs unused by cache manager
            }
            else
            {
                injectedRet.valueCount           = 1;
                injectedRet.values[0].type       = INJECTION_UINT;
                injectedRet.values[0].value.UInt = percent;
            }
            REQUIRE(nvmlDeviceInject(nvmlDevice, field.nvmlInjectKey, nullptr, 0, &injectedRet) == NVML_SUCCESS);
        };

        auto samplePair = [&](PercentToRatioField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.ratioFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t percentSample {};
            dcgmcm_sample_t ratioSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &percentSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.ratioFieldId, &ratioSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { percentSample, ratioSample };
        };

        THEN("each ratio field is registered as DOUBLE")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.ratioFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_DOUBLE);
            }
        }

        WHEN("NVML reports successful percent values")
        {
            for (auto const &field : fields)
            {
                for (auto const &[percent, expectedRatio] : conversionCases)
                {
                    injectPercent(field, NVML_SUCCESS, percent);
                    auto [percentSample, ratioSample] = samplePair(field);

                    CHECK(percentSample.val.i64 == static_cast<long long>(percent));
                    CHECK(ratioSample.val.d == Catch::Approx(expectedRatio));
                    REQUIRE(cacheManager.FreeSamples(&percentSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&ratioSample, 1, field.ratioFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            for (auto const &field : fields)
            {
                injectPercent(field, NVML_ERROR_NOT_SUPPORTED, 0);
                auto [percentSample, ratioSample] = samplePair(field);

                CHECK(percentSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(ratioSample.val.d == NvmlErrorToDoubleValue(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&percentSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&ratioSample, 1, field.ratioFieldId) == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("CacheManager: nanosecond violation fields convert to seconds")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    ScopedDcgmFields dcgmFields;

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    struct NanoToSecondsField
    {
        unsigned short legacyFieldId;  // ns INT64
        unsigned short secondsFieldId; // seconds DOUBLE
        unsigned int nvmlFieldId;
    };

    // Add future ns→seconds migrations that share nvmlDeviceInjectFieldValue here.
    constexpr std::array fields {
        NanoToSecondsField {
            DCGM_FI_DEV_POWER_VIOLATION, DCGM_FI_DEV_GPU_POWER_THROTTLE_SECONDS_TOTAL, NVML_FI_DEV_PERF_POLICY_POWER },
        NanoToSecondsField { DCGM_FI_DEV_THERMAL_VIOLATION,
                             DCGM_FI_DEV_GPU_THERMAL_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_THERMAL },
        NanoToSecondsField { DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                             DCGM_FI_DEV_GPU_SYNC_BOOST_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_SYNC_BOOST },
        NanoToSecondsField { DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                             DCGM_FI_DEV_GPU_LIMIT_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT },
        NanoToSecondsField { DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                             DCGM_FI_DEV_GPU_LOW_UTIL_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION },
        NanoToSecondsField { DCGM_FI_DEV_RELIABILITY_VIOLATION,
                             DCGM_FI_DEV_GPU_RELIABILITY_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_RELIABILITY },
        NanoToSecondsField { DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION,
                             DCGM_FI_DEV_GPU_APP_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS },
        NanoToSecondsField { DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION,
                             DCGM_FI_DEV_GPU_BASE_THROTTLE_SECONDS_TOTAL,
                             NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS },
    };

    constexpr std::array conversionCases {
        std::pair { 0ull, 0.0 },
        std::pair { 1000000000ull, 1.0 },
        std::pair { 500000000ull, 0.5 },
        std::pair { 1500000000ull, 1.5 },
    };

    GIVEN("an initialized cache manager and injected GPU")
    {
        TestDcgmCacheManager cacheManager;
        REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
        cacheManager.Start();

        unsigned int constexpr gpuId = 0;
        nvmlDevice_t nvmlDevice {};
        REQUIRE(nvmlDeviceGetHandleByIndex(gpuId, &nvmlDevice) == NVML_SUCCESS);
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5584);

        auto injectNanoseconds
            = [&](NanoToSecondsField const &field, nvmlReturn_t nvmlRet, unsigned long long nanoseconds) {
                  REQUIRE(nvmlDeviceReset(nvmlDevice) == NVML_SUCCESS);

                  nvmlFieldValue_t fv {};
                  fv.fieldId      = field.nvmlFieldId;
                  fv.nvmlReturn   = nvmlRet;
                  fv.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                  fv.value.ullVal = nanoseconds;
                  fv.timestamp    = timelib_usecSince1970();
                  REQUIRE(nvmlDeviceInjectFieldValue(nvmlDevice, &fv) == NVML_SUCCESS);
              };

        auto samplePair = [&](NanoToSecondsField const &field) {
            WatchField(cacheManager, gpuId, field.legacyFieldId, watcher);
            WatchField(cacheManager, gpuId, field.secondsFieldId, watcher);
            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t nanoSample {};
            dcgmcm_sample_t secondsSample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.legacyFieldId, &nanoSample, nullptr)
                    == DCGM_ST_OK);
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, field.secondsFieldId, &secondsSample, nullptr)
                    == DCGM_ST_OK);
            return std::pair { nanoSample, secondsSample };
        };

        THEN("each seconds field is registered as DOUBLE")
        {
            for (auto const &field : fields)
            {
                auto *fieldMeta = DcgmFieldGetById(field.secondsFieldId);
                REQUIRE(fieldMeta != nullptr);
                CHECK(fieldMeta->fieldType == DCGM_FT_DOUBLE);
            }
        }

        WHEN("NVML reports successful nanosecond values")
        {
            for (auto const &field : fields)
            {
                for (auto const &[nanoseconds, expectedSeconds] : conversionCases)
                {
                    injectNanoseconds(field, NVML_SUCCESS, nanoseconds);
                    auto [nanoSample, secondsSample] = samplePair(field);

                    CHECK(nanoSample.val.i64 == static_cast<long long>(nanoseconds));
                    CHECK(secondsSample.val.d == Catch::Approx(expectedSeconds));
                    REQUIRE(cacheManager.FreeSamples(&nanoSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                    REQUIRE(cacheManager.FreeSamples(&secondsSample, 1, field.secondsFieldId) == DCGM_ST_OK);
                }
            }
        }

        WHEN("NVML reports NOT_SUPPORTED")
        {
            for (auto const &field : fields)
            {
                injectNanoseconds(field, NVML_ERROR_NOT_SUPPORTED, 0);
                auto [nanoSample, secondsSample] = samplePair(field);

                CHECK(nanoSample.val.i64 == NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED));
                CHECK(secondsSample.val.d == NvmlErrorToDoubleValue(NVML_ERROR_NOT_SUPPORTED));
                REQUIRE(cacheManager.FreeSamples(&nanoSample, 1, field.legacyFieldId) == DCGM_ST_OK);
                REQUIRE(cacheManager.FreeSamples(&secondsSample, 1, field.secondsFieldId) == DCGM_ST_OK);
            }
        }
    }
}
