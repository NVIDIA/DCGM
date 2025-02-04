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
#include <catch2/catch_all.hpp>
#include <dcgm_agent.h>
#include <sstream>

#include <DcgmCacheManager.h>
#include <Defer.hpp>

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
