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
#include <catch2/catch_all.hpp>
#include <dcgm_agent.h>
#include <sstream>
#include <string_view>

#define TEST_DCGMCACHEMANAGER
#include <DcgmCacheManager.h>
#undef TEST_DCGMCACHEMANAGER
#include <DcgmCMUtils.h>
#include <Defer.hpp>
#include <UnitTestHelpers.h>
#include <dcgm_fields.h>
#include <ranges>

#if defined(NV_VMWARE)
/* No. of iterations corresponding to different sample set of vgpuIds */
#define NUM_VGPU_LISTS             5
#define TEST_MAX_NUM_VGPUS_PER_GPU 16

TEST_CASE("CacheManager: Test VgpuManageList")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

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

void callback(unsigned int gpuId, void *userData)
{
    auto gpuIdPtr = (unsigned int *)userData;
    *gpuIdPtr     = gpuId;
}

TEST_CASE("CacheManager: Test Event Register")
{
    DcgmCacheManager cm;
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
    DcgmCacheManager cm;
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
        CHECK(series[i].callReturn == DcgmCacheManager::GetFMStatusFromStruct(gpuFabricInfo, status, fmError));

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

    DcgmFieldsInit();

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    /* Initialize in manual mode */
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    // field from dedicated API
    unsigned short constexpr fieldIdUUID = DCGM_FI_DEV_UUID;
    // field from nvmlDeviceGetFieldValues
    unsigned short constexpr fieldIdECCCurrent       = DCGM_FI_DEV_ECC_CURRENT;
    std::array<unsigned short, 2> constexpr fieldIds = { fieldIdUUID, fieldIdECCCurrent };
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    for (auto fieldId : fieldIds)
    {
        bool wereFirstWatcher = false;
        auto ret              = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);
    }

    SECTION("watchInfo->isWatched == false case")
    {
        for (auto fieldId : fieldIds)
        {
            auto ret = cacheManager.RemoveFieldWatch(DCGM_FE_GPU, gpuId, fieldId, 0, watcher);
            REQUIRE(ret == DCGM_ST_OK);
        }
        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);
    }

    SECTION("Watched")
    {
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
    }
}

TEST_CASE("DcgmCacheManager::IsGpuMigEnabled & IsMigEnabledAnywhere")
{
    DcgmCacheManager cacheManager;
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

TEST_CASE("GetMultipleLatestLiveSamples: MIG entity remapping for mapped fields")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("SKU file not found: H200.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer fieldsTerm([] { DcgmFieldsTerm(); });

    auto nvmlRet = nvmlInit_v2();
    REQUIRE(nvmlRet == NVML_SUCCESS);
    DcgmNs::Defer nvmlCleanup([&] { nvmlShutdown(); });

    DcgmCacheManager cm;
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

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    SECTION("DCGM_FI_DEV_VGPU_TYPE_NAME")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        // vGPU.yaml GPU 0 has vGPU type 555 with Name="NVIDIA A40-1B"
        auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
        REQUIRE(std::string_view(vgpuTypeNames[0]) == "NVIDIA A40-1B");

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }

    SECTION("DCGM_FI_DEV_VGPU_TYPE_CLASS")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        // vGPU.yaml GPU 0 has vGPU type 555 with Class="NVS"
        auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
        REQUIRE(std::string_view(vgpuTypeClasses[0]) == "NVS");

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }

    SECTION("DCGM_FI_DEV_VGPU_TYPE_LICENSE")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        // vGPU.yaml GPU 0 has vGPU type 555 with License containing "GRID-Virtual-PC"
        auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);
        std::string_view license(vgpuTypeLicenses[0]);
        REQUIRE(license.find("GRID-Virtual-PC") != std::string_view::npos);

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }
}

TEST_CASE("DcgmCacheManager: vGPU Type Fields when vGPU Not Supported")
{
    // H200.yaml returns FunctionReturn: 3 (NVML_ERROR_NOT_SUPPORTED) for SupportedVgpus
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    unsigned int constexpr gpuId = 0;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    SECTION("DCGM_FI_DEV_VGPU_TYPE_NAME - No crash with unsupported vGPU")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        // Should not crash even when vGPU is not supported
        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
        REQUIRE(std::string_view(vgpuTypeNames[0]) == std::string_view("Unknown"));

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }

    SECTION("DCGM_FI_DEV_VGPU_TYPE_CLASS - No crash with unsupported vGPU")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
        REQUIRE(std::string_view(vgpuTypeClasses[0]) == std::string_view("Unknown"));

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }

    SECTION("DCGM_FI_DEV_VGPU_TYPE_LICENSE - No crash with unsupported vGPU")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
        bool wereFirstWatcher            = false;
        auto ret                         = cacheManager.AddFieldWatch(
            DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

        dcgmcm_sample_t sample {};
        REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
        REQUIRE(sample.val.blob != nullptr);

        auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);
        REQUIRE(std::string_view(vgpuTypeLicenses[0]) == std::string_view("Unknown"));

        REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
    }
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

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    REQUIRE(cacheManager.Init(1, 14400.0, true) == DCGM_ST_OK);
    cacheManager.Start();

    DcgmWatcher watcher(DcgmWatcherTypeClient, 5566);

    SECTION("vgpuCount = 0 - No stack data leak")
    {
        // GPU index 3 has 0 vGPU types
        unsigned int constexpr gpuId = 3;

        // Test NAME field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(std::string_view(vgpuTypeNames[i]) == std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test CLASS field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(std::string_view(vgpuTypeClasses[i]) == std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test LICENSE field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(std::string_view(vgpuTypeLicenses[i]) == std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }
    }

    SECTION("vgpuCount = 29 - Normal operation below limit")
    {
        // GPU index 0 has 29 vGPU types
        unsigned int constexpr gpuId = 0;

        // Test NAME field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            // First entry should have a valid name (not empty, not "Unknown")
            REQUIRE(vgpuTypeNames[0][0] != '\0');
            REQUIRE(std::string_view(vgpuTypeNames[0]) != std::string_view("Unknown"));
            // 29th entry (index 28) should be populated
            REQUIRE(vgpuTypeNames[28][0] != '\0');
            REQUIRE(std::string_view(vgpuTypeNames[28]) != std::string_view("Unknown"));
            // 30th entry (index 29) and beyond should be "Unknown"
            REQUIRE(std::string_view(vgpuTypeNames[29]) == std::string_view("Unknown"));
            REQUIRE(std::string_view(vgpuTypeNames[31]) == std::string_view("Unknown"));

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test CLASS field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            REQUIRE(std::string_view(vgpuTypeClasses[0]) != std::string_view("Unknown"));
            REQUIRE(std::string_view(vgpuTypeClasses[28]) != std::string_view("Unknown"));
            REQUIRE(std::string_view(vgpuTypeClasses[29]) == std::string_view("Unknown"));

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test LICENSE field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);

            REQUIRE(vgpuTypeLicenses[0][0] != '\0');
            REQUIRE(std::string_view(vgpuTypeLicenses[0]) != std::string_view("Unknown"));

            REQUIRE(vgpuTypeLicenses[28][0] != '\0');
            REQUIRE(std::string_view(vgpuTypeLicenses[28]) != std::string_view("Unknown"));

            REQUIRE(std::string_view(vgpuTypeLicenses[29]) == std::string_view("Unknown"));

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }
    }

    SECTION("vgpuCount = 32 - Boundary condition (exact limit)")
    {
        // GPU index 4 has exactly 32 vGPU types
        unsigned int constexpr gpuId = 4;

        // Test NAME field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            // Verify all 32 are populated
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeNames[i][0] != '\0');
                REQUIRE(std::string_view(vgpuTypeNames[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test CLASS field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeClasses[i][0] != '\0');
                REQUIRE(std::string_view(vgpuTypeClasses[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test LICENSE field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeLicenses[i][0] != '\0');
                REQUIRE(std::string_view(vgpuTypeLicenses[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }
    }

    SECTION("vgpuCount = 40 - Buffer overflow protection (capped to 32)")
    {
        // GPU index 5 has 40 vGPU types, but only 32 should be stored
        unsigned int constexpr gpuId = 5;

        // Test NAME field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_NAME;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeNames = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            // First 32 entries should be populated (capped from 40)
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeNames[i][0] != '\0'); // All 32 should be populated (capped from 40)
                REQUIRE(std::string_view(vgpuTypeNames[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test CLASS field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_CLASS;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeClasses = static_cast<char (*)[DCGM_VGPU_NAME_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeClasses[i][0] != '\0');
                REQUIRE(std::string_view(vgpuTypeClasses[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }

        // Test LICENSE field
        {
            unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE_LICENSE;
            bool wereFirstWatcher            = false;
            auto ret                         = cacheManager.AddFieldWatch(
                DCGM_FE_GPU, gpuId, fieldId, 0, 14400.0, 2, watcher, false, false, wereFirstWatcher);
            REQUIRE(ret == DCGM_ST_OK);

            REQUIRE(cacheManager.UpdateAllFields(1) == DCGM_ST_OK);

            dcgmcm_sample_t sample {};
            REQUIRE(cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldId, &sample, nullptr) == DCGM_ST_OK);
            REQUIRE(sample.val.blob != nullptr);

            auto *vgpuTypeLicenses = static_cast<char (*)[DCGM_GRID_LICENSE_BUFFER_SIZE]>(sample.val.blob);
            for (int i = 0; i < DCGM_MAX_VGPU_TYPES_PER_PGPU; ++i)
            {
                REQUIRE(vgpuTypeLicenses[i][0] != '\0');
                REQUIRE(std::string_view(vgpuTypeLicenses[i]) != std::string_view("Unknown"));
            }

            REQUIRE(cacheManager.FreeSamples(&sample, 1, fieldId) == DCGM_ST_OK);
        }
    }
}
