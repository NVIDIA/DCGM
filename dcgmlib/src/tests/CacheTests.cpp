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
#include <sstream>
#include <string_view>
#include <utility>

#define TEST_DCGMCACHEMANAGER
#include <DcgmCacheManager.h>
#undef TEST_DCGMCACHEMANAGER
#include <DcgmCMUtils.h>
#include <Defer.hpp>
#include <UnitTestHelpers.h>
#include <dcgm_fields.h>
#include <nvml_injection.h>
#include <vector>

namespace
{
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
} // namespace

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

TEST_CASE("CacheManager: fake entity query helpers")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cm;
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
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &fastWatcher) == DCGM_ST_OK);
            CHECK(wasAdded);

            slowWatcher.monitorIntervalUsec = 3000;
            slowWatcher.maxAgeUsec          = 40000;
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            CHECK_FALSE(wasAdded);

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
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &slowWatcher) == DCGM_ST_OK);
            REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &fastWatcher) == DCGM_ST_OK);
            watchInfo->isWatched = 1;

            REQUIRE(cm.RemoveWatcher(watchInfo, &fastWatcher) == DCGM_ST_OK);

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
            }
        }
    }
}

TEST_CASE("CacheManager: watch precheck and clear helpers")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cm;
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
        watcher.watcher             = DcgmWatcher(DcgmWatcherTypeClient, 201);
        watcher.monitorIntervalUsec = 10;
        watcher.maxAgeUsec          = 20;
        watcher.isSubscribed        = 1;
        bool wasAdded               = false;
        REQUIRE(cm.AddOrUpdateWatcher(watchInfo, &wasAdded, &watcher) == DCGM_ST_OK);
        watchInfo->isWatched       = 1;
        watchInfo->pushedByModule  = true;
        watchInfo->lastStatus      = NVML_ERROR_TIMEOUT;
        watchInfo->lastQueriedUsec = 99;

        WHEN("the watch info is cleared without cache destruction")
        {
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
            }
        }
    }
}

TEST_CASE("CacheManager: field watch lifecycle helpers")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cm;
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
    }
}

TEST_CASE("CacheManager: watch key and practical entity helpers")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

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

TEST_CASE("CacheManager: sample API edge cases")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;
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
    unsigned short constexpr fieldIdUUID = DCGM_FI_DEV_GPU_UUID;
    // field from nvmlDeviceGetFieldValues
    unsigned short constexpr fieldIdECCCurrent       = DCGM_FI_DEV_ECC_MODE;
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

TEST_CASE("DcgmCacheManager: NVLink TX/RX throughput watches with NVML field injection")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    auto nvmlRet = nvmlInit_v2();
    REQUIRE(nvmlRet == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([&] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
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

TEST_CASE("DcgmCacheManager: pure state helpers")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    SECTION("GIVEN constructor environment WHEN force-GPM is set THEN the flag is enabled")
    {
        setenv("__DCGM_FORCE_PROF_METRICS_THROUGH_GPM", "1", 1);
        DcgmNs::Defer restoreEnv([] { unsetenv("__DCGM_FORCE_PROF_METRICS_THROUGH_GPM"); });

        DcgmCacheManager cacheManager;

        CHECK(cacheManager.m_forceProfMetricsThroughGpm);
    }

    SECTION("GIVEN global watch fields WHEN global watch info is requested THEN create and lookup paths agree")
    {
        DcgmCacheManager cacheManager;

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
        DcgmCacheManager cacheManager;
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
        DcgmCacheManager cacheManager;
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
        DcgmCacheManager cacheManager;
        dcgmcm_gpu_info_t gpuInfo {};

        gpuInfo.status = DcgmEntityStatusLost;
        CHECK(cacheManager.InitializeGpuInstances(gpuInfo) == DCGM_ST_GPU_IS_LOST);

        gpuInfo.status = DcgmEntityStatusOk;
        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK(cacheManager.InitializeGpuInstances(gpuInfo) == DCGM_ST_NVML_NOT_LOADED);
    }

    SECTION("GIVEN NVML is not loaded WHEN event set is initialized THEN no event set is created")
    {
        DcgmCacheManager cacheManager;

        cacheManager.m_nvmlLoaded.store(false, std::memory_order_release);
        CHECK(cacheManager.InitializeNvmlEventSet() == DCGM_ST_OK);
        CHECK_FALSE(cacheManager.m_nvmlEventSetInitialized);
    }

    SECTION("GIVEN fake GPUs and MIG entities WHEN entity helpers are queried THEN cached state is reported")
    {
        DcgmCacheManager cacheManager;
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
        DcgmCacheManager cacheManager;
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
        DcgmCacheManager cacheManager;
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

TEST_CASE("DcgmCacheManager::IsModulePushedFieldId")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

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
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK));
    static_assert(DCGM_FIELD_ID_IS_PROF_FIELD(DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK));

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
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_L0_TX_BYTES));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_L17_RX_BYTES));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_TX_BYTES_PER_LINK));
    static_assert(DCGM_FIELD_ID_IS_GPM_MIB_BANDWIDTH(DCGM_FI_PROF_NVLINK_RX_BYTES_PER_LINK));
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
            DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO,  DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0,
            DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15,
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cm;
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
                             DCGM_FI_DEV_NVLINK_GET_STATE,
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cacheManager;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cacheManager;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    dcgm_link_t link              = {};
    link.parsed.type              = DCGM_FE_GPU;
    link.parsed.gpuId             = 0;
    link.parsed.index             = 1;
    dcgm_field_eid_t linkEntityId = link.raw;

    DcgmCacheManager cm;
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
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cm;
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
    requireUnsupportedPort(DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT, DCGM_NVLINK_MAX_LINKS_PER_GPU + 1);
}
