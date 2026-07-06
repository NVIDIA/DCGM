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
#include <DcgmVgpu.hpp>

#include <DcgmCacheManager.h>
#include <Defer.hpp>
#include <UnitTestHelpers.h>
#include <dcgm_fields.h>
#include <nvml_injection.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace
{
nvmlVgpuInstance_t constexpr VGPU_ID = 3251816820;

struct VgpuUpdateContext
{
    DcgmFvBuffer fvBuffer;
    dcgmcm_update_thread_t threadCtx {};
};

VgpuUpdateContext MakeVgpuUpdateContext(unsigned short fieldId)
{
    VgpuUpdateContext context;
    context.threadCtx.watchInfo               = nullptr;
    context.threadCtx.fvBuffer                = &context.fvBuffer;
    context.threadCtx.entityKey.entityGroupId = DCGM_FE_VGPU;
    context.threadCtx.entityKey.entityId      = VGPU_ID;
    context.threadCtx.entityKey.fieldId       = fieldId;
    return context;
}

SafeVgpuInstance MakeSafeVgpuInstance(NvmlTaskRunner const &nvmlDriver)
{
    SafeVgpuInstance vgpuId {};
    vgpuId.vgpuInstance = VGPU_ID;
    vgpuId.generation   = nvmlDriver.GetGeneration();
    return vgpuId;
}

void RequireBufferedInt64(DcgmFvBuffer &fvBuffer, unsigned short fieldId, long long expectedValue)
{
    dcgmBufferedFvCursor_t cursor = 0;
    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
    REQUIRE(fv != nullptr);
    CHECK(fv->fieldType == DCGM_FT_INT64);
    CHECK(fv->fieldId == fieldId);
    CHECK(fv->entityGroupId == DCGM_FE_VGPU);
    CHECK(fv->entityId == VGPU_ID);
    CHECK(fv->value.i64 == expectedValue);
    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
}

void RequireBufferedString(DcgmFvBuffer &fvBuffer, unsigned short fieldId, std::string_view expectedValue)
{
    dcgmBufferedFvCursor_t cursor = 0;
    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
    REQUIRE(fv != nullptr);
    CHECK(fv->fieldType == DCGM_FT_STRING);
    CHECK(fv->fieldId == fieldId);
    CHECK(fv->entityGroupId == DCGM_FE_VGPU);
    CHECK(fv->entityId == VGPU_ID);
    CHECK(std::string_view(fv->value.str) == expectedValue);
    CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
}
} // namespace

SCENARIO("BufferOrCacheLatestVgpuValue validates its field metadata")
{
    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    DcgmCacheManager cacheManager;
    dcgmcm_update_thread_t threadCtx {};
    NvmlTaskRunner nvmlDriver;
    SafeVgpuInstance vgpuId {};

    WHEN("field metadata is null")
    {
        dcgmReturn_t const result = BufferOrCacheLatestVgpuValue(cacheManager, threadCtx, nvmlDriver, vgpuId, nullptr);

        THEN("the call reports a bad parameter")
        {
            REQUIRE(result == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("field metadata for a non-device field")
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(DCGM_FI_SYSTEM_DRIVER_VERSION);
        REQUIRE(fieldMeta != nullptr);
        REQUIRE(fieldMeta->scope != DCGM_FS_DEVICE);

        WHEN("the vGPU value is requested")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the call reports a bad parameter")
            {
                REQUIRE(result == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO("BufferOrCacheLatestVgpuValue caches scalar vGPU values from NVML")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    nvmlDriver.Start();
    SafeVgpuInstance const vgpuId = MakeSafeVgpuInstance(nvmlDriver);

    GIVEN("a watched vGPU type field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_TYPE;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the vGPU type id is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedInt64(context.fvBuffer, fieldId, 564);
            }
        }
    }

    GIVEN("a watched vGPU memory usage field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_MEMORY_USAGE;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the framebuffer usage is converted to MiB in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedInt64(context.fvBuffer, fieldId, 1280);
            }
        }
    }

    GIVEN("a watched vGPU frame-rate-limit field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the frame-rate limit is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedInt64(context.fvBuffer, fieldId, 60);
            }
        }
    }

    GIVEN("a watched vGPU GPU-instance-id field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the GPU instance id is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedInt64(context.fvBuffer, fieldId, 4294967295);
            }
        }
    }
}

SCENARIO("BufferOrCacheLatestVgpuValue caches string vGPU values from NVML")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    nvmlDriver.Start();
    SafeVgpuInstance const vgpuId = MakeSafeVgpuInstance(nvmlDriver);

    GIVEN("a watched VM id field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_VM_ID;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the VM id is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedString(context.fvBuffer, fieldId, "227911");
            }
        }
    }

    GIVEN("a watched vGPU UUID field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_UUID;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the mapped UUID error is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_BADPARAM);
                RequireBufferedString(context.fvBuffer, fieldId, DCGM_STR_BLANK);
            }
        }
    }

    GIVEN("a watched VM driver version field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_DRIVER_VERSION;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the mapped driver-version error is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_BADPARAM);
                RequireBufferedString(context.fvBuffer, fieldId, DCGM_STR_BLANK);
            }
        }
    }

    GIVEN("a watched vGPU license-status field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the formatted license status is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_OK);
                RequireBufferedString(context.fvBuffer, fieldId, "Licensed (Expiry: 2024-2-1 2:36:3 GMT)");
            }
        }
    }

    GIVEN("a watched vGPU PCI id field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_PCI_ID;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the mapped PCI-id error is observable in the buffer")
            {
                REQUIRE(result == DCGM_ST_BADPARAM);
                RequireBufferedString(context.fvBuffer, fieldId, DCGM_STR_BLANK);
            }
        }
    }
}

namespace
{
template <typename ValueType>
ValueType const &RequireBufferedBlob(DcgmFvBuffer &fvBuffer, unsigned short fieldId, std::size_t expectedSize)
{
    dcgmBufferedFvCursor_t cursor = 0;
    dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
    REQUIRE(fv != nullptr);
    REQUIRE(fv->fieldType == DCGM_FT_BINARY);
    REQUIRE(fv->fieldId == fieldId);
    REQUIRE(fv->entityGroupId == DCGM_FE_VGPU);
    REQUIRE(fv->entityId == VGPU_ID);
    REQUIRE(fv->length == (sizeof(*fv) - sizeof(fv->value)) + expectedSize);
    REQUIRE(fvBuffer.GetNextFv(&cursor) == nullptr);
    return *reinterpret_cast<ValueType const *>(fv->value.blob);
}
} // namespace

SCENARIO("BufferOrCacheLatestVgpuValue caches blob vGPU values from NVML")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    nvmlDriver.Start();
    SafeVgpuInstance const vgpuId = MakeSafeVgpuInstance(nvmlDriver);

    GIVEN("a watched vGPU encoder stats field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_ENC_STATS;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the encoder stats blob reports the NVML values")
            {
                REQUIRE(result == DCGM_ST_OK);
                auto const &stats = RequireBufferedBlob<dcgmDeviceEncStats_t>(
                    context.fvBuffer, fieldId, sizeof(dcgmDeviceEncStats_t));
                CHECK(stats.version == dcgmDeviceEncStats_version);
                CHECK(stats.sessionCount == 0);
                CHECK(stats.averageFps == 0);
                CHECK(stats.averageLatency == 0);
            }
        }
    }

    GIVEN("a watched vGPU encoder sessions field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the encoder sessions blob reports an empty session list")
            {
                REQUIRE(result == DCGM_ST_OK);
                auto const &sessions = RequireBufferedBlob<dcgmDeviceVgpuEncSessions_t>(
                    context.fvBuffer, fieldId, sizeof(dcgmDeviceVgpuEncSessions_t));
                CHECK(sessions.version == dcgmDeviceVgpuEncSessions_version);
                CHECK(sessions.encoderSessionInfo.sessionCount == 0);
            }
        }
    }

    GIVEN("a watched vGPU FBC stats field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_FBC_STATS;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the FBC stats blob reports the NVML values")
            {
                REQUIRE(result == DCGM_ST_OK);
                auto const &stats = RequireBufferedBlob<dcgmDeviceFbcStats_t>(
                    context.fvBuffer, fieldId, sizeof(dcgmDeviceFbcStats_t));
                CHECK(stats.version == dcgmDeviceFbcStats_version);
                CHECK(stats.sessionCount == 0);
                CHECK(stats.averageFps == 0);
                CHECK(stats.averageLatency == 0);
            }
        }
    }

    GIVEN("a watched vGPU FBC sessions field")
    {
        unsigned short constexpr fieldId = DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO;
        dcgm_field_meta_p fieldMeta      = DcgmFieldGetById(fieldId);
        REQUIRE(fieldMeta != nullptr);
        auto context = MakeVgpuUpdateContext(fieldId);

        WHEN("the latest value is buffered or cached")
        {
            dcgmReturn_t const result
                = BufferOrCacheLatestVgpuValue(cacheManager, context.threadCtx, nvmlDriver, vgpuId, fieldMeta);

            THEN("the FBC sessions blob reports an empty session list")
            {
                REQUIRE(result == DCGM_ST_OK);
                auto const &sessions = RequireBufferedBlob<dcgmDeviceFbcSessions_t>(
                    context.fvBuffer,
                    fieldId,
                    sizeof(dcgmDeviceFbcSessions_t) - (DCGM_MAX_FBC_SESSIONS * sizeof(dcgmDeviceFbcSessionInfo_t)));
                CHECK(sessions.version == dcgmDeviceFbcSessions_version);
                CHECK(sessions.sessionCount == 0);
            }
        }
    }
}

SCENARIO("GetDeviceFBCSessionsInfo buffers empty device FBC session lists")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("vGPU.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: vGPU.yaml");
    }

    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer deferNvml([] { nvmlShutdown(); });

    NvmlTaskRunner nvmlDriver;
    nvmlDriver.Start();
    auto safeHandles = nvmlDriver.GetSafeNvmlHandles();
    REQUIRE(safeHandles.has_value());
    REQUIRE_FALSE(safeHandles->empty());

    DcgmCacheManager cacheManager;
    DcgmFvBuffer fvBuffer;
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.watchInfo               = nullptr;
    threadCtx.fvBuffer                = &fvBuffer;
    threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx.entityKey.entityId      = 0;
    threadCtx.entityKey.fieldId       = DCGM_FI_DEV_FBC_SESSIONS_INFO;

    WHEN("device FBC session information is requested for a GPU with no sessions")
    {
        dcgmReturn_t const result = GetDeviceFBCSessionsInfo(
            cacheManager, nvmlDriver, safeHandles->front().first, threadCtx, nullptr, timelib_usecSince1970(), 0);

        THEN("an empty session list is buffered")
        {
            REQUIRE(result == DCGM_ST_OK);
            dcgmBufferedFvCursor_t cursor = 0;
            dcgmBufferedFv_t *fv          = fvBuffer.GetNextFv(&cursor);
            REQUIRE(fv != nullptr);
            REQUIRE(fv->fieldType == DCGM_FT_BINARY);
            REQUIRE(fv->fieldId == DCGM_FI_DEV_FBC_SESSIONS_INFO);
            REQUIRE(fv->entityGroupId == DCGM_FE_GPU);
            REQUIRE(fv->entityId == 0);
            REQUIRE(fv->length
                    == (sizeof(*fv) - sizeof(fv->value)) + sizeof(dcgmDeviceFbcSessions_t)
                           - (DCGM_MAX_FBC_SESSIONS * sizeof(dcgmDeviceFbcSessionInfo_t)));

            auto const &sessions = *reinterpret_cast<dcgmDeviceFbcSessions_t const *>(fv->value.blob);
            CHECK(sessions.version == dcgmDeviceFbcSessions_version);
            CHECK(sessions.sessionCount == 0);
            CHECK(fvBuffer.GetNextFv(&cursor) == nullptr);
        }
    }
}

namespace
{
struct CapturedBlob
{
    unsigned short fieldId {};
    dcgm_field_entity_group_t entityGroupId {};
    dcgm_field_eid_t entityId {};
    int valueSize {};
    timelib64_t timestamp {};
    timelib64_t expireTime {};
    std::vector<std::byte> bytes;
};

struct UnsupportedVgpuDependencies
{
    void *Malloc(std::size_t size)
    {
        return std::malloc(size);
    }

    void Free(void *ptr)
    {
        std::free(ptr);
    }

    nvmlReturn_t NvmlDeviceGetFBCSessions(NvmlTaskRunner &, SafeNvmlHandle, unsigned int *, nvmlFBCSessionInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFBCSessions(NvmlTaskRunner &,
                                                SafeVgpuInstance,
                                                unsigned int *,
                                                nvmlFBCSessionInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    dcgmReturn_t AppendEntityBlob(DcgmCacheManager &, dcgmcm_update_thread_t &, void *, int, timelib64_t, timelib64_t)
    {
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmReturn_t AppendEntityString(DcgmCacheManager &,
                                    dcgmcm_update_thread_t &,
                                    char const *,
                                    timelib64_t,
                                    timelib64_t)
    {
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmReturn_t AppendEntityInt64(DcgmCacheManager &,
                                   dcgmcm_update_thread_t &,
                                   long long,
                                   long long,
                                   timelib64_t,
                                   timelib64_t)
    {
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmReturn_t UpdateFieldWatch(DcgmCacheManager &, dcgmcm_watch_info_p, timelib64_t, double, int, DcgmWatcher)
    {
        return DCGM_ST_NOT_SUPPORTED;
    }

    timelib64_t Now()
    {
        return 0;
    }

    char const *NvmlErrorToStringValue(nvmlReturn_t)
    {
        return "NVML_ERROR";
    }

    long long NvmlErrorToInt64Value(nvmlReturn_t)
    {
        return DCGM_INT64_BLANK;
    }

    char const *NvmlErrorString(NvmlTaskRunner &, nvmlReturn_t)
    {
        return "NVML_ERROR";
    }

    dcgmReturn_t NvmlReturnToDcgmReturn(nvmlReturn_t)
    {
        return DCGM_ST_NVML_ERROR;
    }

    nvmlReturn_t NvmlVgpuInstanceGetVmID(NvmlTaskRunner &, SafeVgpuInstance, char *, unsigned int, nvmlVgpuVmIdType_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetType(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetUUID(NvmlTaskRunner &, SafeVgpuInstance, char *, unsigned int)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetVmDriverVersion(NvmlTaskRunner &, SafeVgpuInstance, char *, unsigned int)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFbUsage(NvmlTaskRunner &, SafeVgpuInstance, unsigned long long *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetLicenseInfo_v2(NvmlTaskRunner &, SafeVgpuInstance, nvmlVgpuLicenseInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFrameRateLimit(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetGpuPciId(NvmlTaskRunner &, SafeVgpuInstance, char *, unsigned int *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetEncoderStats(NvmlTaskRunner &,
                                                 SafeVgpuInstance,
                                                 unsigned int *,
                                                 unsigned int *,
                                                 unsigned int *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetEncoderSessions(NvmlTaskRunner &,
                                                    SafeVgpuInstance,
                                                    unsigned int *,
                                                    nvmlEncoderSessionInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFBCStats(NvmlTaskRunner &, SafeVgpuInstance, nvmlFBCStats_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetGpuInstanceId(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }
};

struct MockFbcDependencies
{
    int failAllocationCall = 0;
    int mallocCalls        = 0;
    int freeCalls          = 0;

    unsigned int reportedSessionCount = 0;
    std::vector<nvmlFBCSessionInfo_t> sessions;
    std::vector<nvmlReturn_t> deviceReturns { NVML_SUCCESS, NVML_SUCCESS };
    std::vector<nvmlReturn_t> vgpuReturns { NVML_SUCCESS, NVML_SUCCESS };
    std::size_t deviceCallCount = 0;
    std::size_t vgpuCallCount   = 0;
    dcgmReturn_t mappedReturn   = DCGM_ST_NVML_ERROR;
    std::vector<CapturedBlob> appends;

    void *Malloc(std::size_t size)
    {
        mallocCalls++;
        if (mallocCalls == failAllocationCall)
        {
            return nullptr;
        }
        return std::malloc(size);
    }

    void Free(void *ptr)
    {
        freeCalls++;
        std::free(ptr);
    }

    nvmlReturn_t NvmlDeviceGetFBCSessions(NvmlTaskRunner &,
                                          SafeNvmlHandle,
                                          unsigned int *sessionCount,
                                          nvmlFBCSessionInfo_t *sessionInfo)
    {
        return GetFBCSessions(deviceReturns, deviceCallCount, sessionCount, sessionInfo);
    }

    nvmlReturn_t NvmlVgpuInstanceGetFBCSessions(NvmlTaskRunner &,
                                                SafeVgpuInstance,
                                                unsigned int *sessionCount,
                                                nvmlFBCSessionInfo_t *sessionInfo)
    {
        return GetFBCSessions(vgpuReturns, vgpuCallCount, sessionCount, sessionInfo);
    }

    dcgmReturn_t AppendEntityBlob(DcgmCacheManager &,
                                  dcgmcm_update_thread_t &threadCtx,
                                  void *value,
                                  int valueSize,
                                  timelib64_t timestamp,
                                  timelib64_t oldestKeepTimestamp)
    {
        CapturedBlob blob;
        blob.fieldId       = threadCtx.entityKey.fieldId;
        blob.entityGroupId = static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId);
        blob.entityId      = threadCtx.entityKey.entityId;
        blob.valueSize     = valueSize;
        blob.timestamp     = timestamp;
        blob.expireTime    = oldestKeepTimestamp;
        auto *begin        = static_cast<std::byte *>(value);
        blob.bytes.assign(begin, begin + valueSize);
        appends.push_back(std::move(blob));
        return DCGM_ST_OK;
    }

    dcgmReturn_t NvmlReturnToDcgmReturn(nvmlReturn_t)
    {
        return mappedReturn;
    }

private:
    nvmlReturn_t GetFBCSessions(std::vector<nvmlReturn_t> const &returns,
                                std::size_t &callCount,
                                unsigned int *sessionCount,
                                nvmlFBCSessionInfo_t *sessionInfo)
    {
        nvmlReturn_t const result = returns.at(callCount++);
        if (sessionInfo == nullptr)
        {
            *sessionCount = reportedSessionCount;
            return result;
        }

        *sessionCount = static_cast<unsigned int>(sessions.size());
        for (std::size_t i = 0; i < sessions.size(); i++)
        {
            sessionInfo[i] = sessions[i];
        }
        return result;
    }
};

dcgmDeviceFbcSessions_t const &CapturedSessions(CapturedBlob const &blob)
{
    return *reinterpret_cast<dcgmDeviceFbcSessions_t const *>(blob.bytes.data());
}

nvmlFBCSessionInfo_t MakeNvmlFbcSession(unsigned int ordinal)
{
    nvmlFBCSessionInfo_t session {};
    session.vgpuInstance   = 1000 + ordinal;
    session.sessionId      = 2000 + ordinal;
    session.pid            = 3000 + ordinal;
    session.displayOrdinal = 4000 + ordinal;
    session.sessionType    = NVML_FBC_SESSION_TYPE_TOSYS;
    session.sessionFlags   = 5000 + ordinal;
    session.hMaxResolution = 6000 + ordinal;
    session.vMaxResolution = 7000 + ordinal;
    session.hResolution    = 8000 + ordinal;
    session.vResolution    = 9000 + ordinal;
    session.averageFPS     = 100 + ordinal;
    session.averageLatency = 200 + ordinal;
    return session;
}
} // namespace

SCENARIO("DcgmVgpuDetail FBC session helpers use injectable dependencies")
{
    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.entityKey.entityGroupId = DCGM_FE_VGPU;
    threadCtx.entityKey.entityId      = VGPU_ID;
    threadCtx.entityKey.fieldId       = DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO;
    dcgmcm_watch_info_t watchInfo {};
    SafeVgpuInstance vgpuId { .vgpuInstance = VGPU_ID, .generation = 7 };
    timelib64_t constexpr now        = 12345;
    timelib64_t constexpr expireTime = 67890;

    GIVEN("allocation of the output structure fails")
    {
        MockFbcDependencies deps;
        deps.failAllocationCall = 1;

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("the call reports memory failure without invoking NVML or appending a payload")
            {
                REQUIRE(result == DCGM_ST_MEMORY);
                CHECK(deps.vgpuCallCount == 0);
                CHECK(deps.appends.empty());
                CHECK(deps.freeCalls == 0);
            }
        }
    }

    GIVEN("the initial NVML count query fails")
    {
        MockFbcDependencies deps;
        deps.vgpuReturns          = { NVML_ERROR_NOT_SUPPORTED };
        deps.reportedSessionCount = 4;
        deps.mappedReturn         = DCGM_ST_NOT_SUPPORTED;

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("an empty session list is appended and the mapped error is returned")
            {
                REQUIRE(result == DCGM_ST_NOT_SUPPORTED);
                REQUIRE(deps.appends.size() == 1);
                CHECK(watchInfo.lastStatus == NVML_ERROR_NOT_SUPPORTED);
                CHECK(deps.freeCalls == 1);
                CHECK(deps.appends[0].valueSize
                      == static_cast<int>(sizeof(dcgmDeviceFbcSessions_t)
                                          - (DCGM_MAX_FBC_SESSIONS * sizeof(dcgmDeviceFbcSessionInfo_t))));
                auto const &sessions = CapturedSessions(deps.appends[0]);
                CHECK(sessions.version == dcgmDeviceFbcSessions_version);
                CHECK(sessions.sessionCount == 0);
            }
        }
    }

    GIVEN("session metadata allocation fails after a successful count query")
    {
        MockFbcDependencies deps;
        deps.reportedSessionCount = 2;
        deps.failAllocationCall   = 2;

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("the call reports memory failure and releases the output structure")
            {
                REQUIRE(result == DCGM_ST_MEMORY);
                CHECK(deps.vgpuCallCount == 1);
                CHECK(deps.appends.empty());
                CHECK(deps.freeCalls == 1);
            }
        }
    }

    GIVEN("the second NVML session query fails")
    {
        MockFbcDependencies deps;
        deps.reportedSessionCount = 2;
        deps.vgpuReturns          = { NVML_SUCCESS, NVML_ERROR_UNKNOWN };
        deps.mappedReturn         = DCGM_ST_NVML_ERROR;

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("the mapped error is returned and allocated memory is released")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                CHECK(watchInfo.lastStatus == NVML_ERROR_UNKNOWN);
                CHECK(deps.vgpuCallCount == 2);
                CHECK(deps.appends.empty());
                CHECK(deps.freeCalls == 2);
            }
        }
    }

    GIVEN("NVML reports exactly as many sessions as the DCGM payload can size")
    {
        MockFbcDependencies deps;
        deps.reportedSessionCount = DCGM_MAX_FBC_SESSIONS;
        for (unsigned int i = 0; i < deps.reportedSessionCount; i++)
        {
            deps.sessions.push_back(MakeNvmlFbcSession(i));
        }

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("the appended payload uses the full destination size")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appends.size() == 1);
                CHECK(deps.appends[0].valueSize == static_cast<int>(sizeof(dcgmDeviceFbcSessions_t)));
                auto const &sessions = CapturedSessions(deps.appends[0]);
                CHECK(sessions.sessionCount == DCGM_MAX_FBC_SESSIONS);
            }
        }
    }

    GIVEN("NVML reports two active vGPU FBC sessions")
    {
        MockFbcDependencies deps;
        deps.reportedSessionCount = 2;
        deps.sessions             = { MakeNvmlFbcSession(1), MakeNvmlFbcSession(2) };

        WHEN("vGPU FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
                cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, now, expireTime, depsWrapper);

            THEN("the appended DCGM payload mirrors the NVML session data")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appends.size() == 1);
                CHECK(deps.appends[0].fieldId == DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO);
                CHECK(deps.appends[0].entityGroupId == DCGM_FE_VGPU);
                CHECK(deps.appends[0].entityId == VGPU_ID);
                CHECK(deps.appends[0].timestamp == now);
                CHECK(deps.appends[0].expireTime == expireTime);
                CHECK(deps.freeCalls == 2);

                auto const &sessions = CapturedSessions(deps.appends[0]);
                REQUIRE(sessions.sessionCount == 2);
                CHECK(sessions.sessionInfo[0].version == dcgmDeviceFbcSessionInfo_version);
                CHECK(sessions.sessionInfo[0].vgpuId == deps.sessions[0].vgpuInstance);
                CHECK(sessions.sessionInfo[0].sessionId == deps.sessions[0].sessionId);
                CHECK(sessions.sessionInfo[0].pid == deps.sessions[0].pid);
                CHECK(sessions.sessionInfo[0].displayOrdinal == deps.sessions[0].displayOrdinal);
                CHECK(sessions.sessionInfo[0].sessionType
                      == static_cast<dcgmFBCSessionType_t>(deps.sessions[0].sessionType));
                CHECK(sessions.sessionInfo[0].sessionFlags == deps.sessions[0].sessionFlags);
                CHECK(sessions.sessionInfo[0].hMaxResolution == deps.sessions[0].hMaxResolution);
                CHECK(sessions.sessionInfo[0].vMaxResolution == deps.sessions[0].vMaxResolution);
                CHECK(sessions.sessionInfo[0].hResolution == deps.sessions[0].hResolution);
                CHECK(sessions.sessionInfo[0].vResolution == deps.sessions[0].vResolution);
                CHECK(sessions.sessionInfo[0].averageFps == deps.sessions[0].averageFPS);
                CHECK(sessions.sessionInfo[0].averageLatency == deps.sessions[0].averageLatency);
            }
        }
    }
}

SCENARIO("DcgmVgpuDetail device FBC sessions use the same operation bundle")
{
    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx.entityKey.entityId      = 3;
    threadCtx.entityKey.fieldId       = DCGM_FI_DEV_FBC_SESSIONS_INFO;
    dcgmcm_watch_info_t watchInfo {};
    SafeNvmlHandle device;
    device.nvmlDevice = reinterpret_cast<nvmlDevice_t>(0x1234);
    device.generation = 7;

    GIVEN("NVML reports one active device FBC session")
    {
        MockFbcDependencies deps;
        deps.reportedSessionCount = 1;
        deps.sessions             = { MakeNvmlFbcSession(3) };

        WHEN("device FBC session information is requested")
        {
            DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::GetDeviceFBCSessionsInfo(
                cacheManager, nvmlDriver, device, threadCtx, &watchInfo, 111, 222, depsWrapper);

            THEN("the dependency-provided session is appended as a DCGM payload")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appends.size() == 1);
                CHECK(deps.deviceCallCount == 2);
                CHECK(deps.vgpuCallCount == 0);
                auto const &sessions = CapturedSessions(deps.appends[0]);
                REQUIRE(sessions.sessionCount == 1);
                CHECK(sessions.sessionInfo[0].vgpuId == deps.sessions[0].vgpuInstance);
                CHECK(sessions.sessionInfo[0].averageLatency == deps.sessions[0].averageLatency);
            }
        }
    }
}

SCENARIO("DcgmVgpuDetail FBC session helpers handle empty successful session lists")
{
    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.entityKey.entityGroupId = DCGM_FE_VGPU;
    threadCtx.entityKey.entityId      = VGPU_ID;
    threadCtx.entityKey.fieldId       = DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO;
    dcgmcm_watch_info_t watchInfo {};
    SafeVgpuInstance vgpuId { .vgpuInstance = VGPU_ID, .generation = 7 };
    MockFbcDependencies deps;
    deps.reportedSessionCount = 0;
    deps.mappedReturn         = DCGM_ST_OK;

    WHEN("NVML succeeds and reports no active sessions")
    {
        DcgmVgpuDetail::FbcSessionOps depsWrapper(&deps);
        dcgmReturn_t const result = DcgmVgpuDetail::GetVgpuInstanceFBCSessionsInfo(
            cacheManager, nvmlDriver, vgpuId, threadCtx, &watchInfo, 333, 444, depsWrapper);

        THEN("an empty payload is appended and success is returned")
        {
            REQUIRE(result == DCGM_ST_OK);
            CHECK(deps.vgpuCallCount == 1);
            REQUIRE(deps.appends.size() == 1);
            CHECK(deps.appends[0].timestamp == 333);
            CHECK(deps.appends[0].expireTime == 444);
            auto const &sessions = CapturedSessions(deps.appends[0]);
            CHECK(sessions.version == dcgmDeviceFbcSessions_version);
            CHECK(sessions.sessionCount == 0);
            CHECK(watchInfo.lastStatus == NVML_SUCCESS);
            CHECK(deps.freeCalls == 1);
        }
    }
}

namespace
{
struct WatchUpdate
{
    timelib64_t monitorIntervalUsec {};
    double maxAgeUsec {};
    int maxKeepSamples {};
};

struct AppendedInt64
{
    long long value1 {};
    long long value2 {};
    timelib64_t timestamp {};
    timelib64_t expireTime {};
};

struct MockBufferDependencies : UnsupportedVgpuDependencies
{
    timelib64_t now = 123456;

    nvmlReturn_t vmIdReturn = NVML_SUCCESS;
    std::string vmId        = "vm-123";

    nvmlReturn_t typeReturn = NVML_SUCCESS;
    unsigned int vgpuTypeId = 17;

    nvmlReturn_t uuidReturn = NVML_SUCCESS;
    std::string uuid        = "GPU-vgpu-uuid";

    nvmlReturn_t driverVersionReturn = NVML_SUCCESS;
    std::string driverVersion        = "535.104";

    nvmlReturn_t fbUsageReturn      = NVML_SUCCESS;
    unsigned long long fbUsageBytes = 512ULL * 1024ULL * 1024ULL;

    nvmlReturn_t licenseInfoReturn = NVML_SUCCESS;
    nvmlVgpuLicenseInfo_t licenseInfo {};

    nvmlReturn_t frameRateLimitReturn = NVML_SUCCESS;
    unsigned int frameRateLimit       = 60;

    nvmlReturn_t pciIdReturn = NVML_SUCCESS;
    std::string pciId        = "00000000:65:00.4";

    nvmlReturn_t encoderStatsReturn  = NVML_SUCCESS;
    unsigned int encoderSessionCount = 2;
    unsigned int encoderAverageFps   = 30;
    unsigned int encoderLatency      = 12;

    std::vector<nvmlReturn_t> encoderSessionsReturns { NVML_SUCCESS, NVML_SUCCESS };
    std::vector<nvmlEncoderSessionInfo_t> encoderSessions;
    std::size_t encoderSessionsCallCount = 0;

    nvmlReturn_t fbcStatsReturn = NVML_SUCCESS;
    nvmlFBCStats_t fbcStats { .sessionsCount = 3, .averageFPS = 45, .averageLatency = 9 };

    nvmlReturn_t gpuInstanceIdReturn = NVML_SUCCESS;
    unsigned int gpuInstanceId       = 93;

    dcgmReturn_t mappedReturn           = DCGM_ST_NVML_ERROR;
    dcgmReturn_t updateFieldWatchReturn = DCGM_ST_OK;
    char const *errorStringValue        = "NVML_ERROR";
    long long errorInt64Value           = -999;

    std::vector<std::string> appendedStrings;
    std::vector<AppendedInt64> appendedInt64s;
    std::vector<CapturedBlob> appendedBlobs;
    std::vector<WatchUpdate> watchUpdates;

    timelib64_t Now()
    {
        return now;
    }

    nvmlReturn_t NvmlVgpuInstanceGetVmID(NvmlTaskRunner &,
                                         SafeVgpuInstance,
                                         char *buffer,
                                         unsigned int bufferSize,
                                         nvmlVgpuVmIdType_t *vmIdType)
    {
        std::snprintf(buffer, bufferSize, "%s", vmId.c_str());
        *vmIdType = NVML_VGPU_VM_ID_DOMAIN_ID;
        return vmIdReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetType(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *typeId)
    {
        *typeId = vgpuTypeId;
        return typeReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetUUID(NvmlTaskRunner &, SafeVgpuInstance, char *buffer, unsigned int bufferSize)
    {
        std::snprintf(buffer, bufferSize, "%s", uuid.c_str());
        return uuidReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetVmDriverVersion(NvmlTaskRunner &,
                                                    SafeVgpuInstance,
                                                    char *buffer,
                                                    unsigned int bufferSize)
    {
        std::snprintf(buffer, bufferSize, "%s", driverVersion.c_str());
        return driverVersionReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFbUsage(NvmlTaskRunner &, SafeVgpuInstance, unsigned long long *fbUsage)
    {
        *fbUsage = fbUsageBytes;
        return fbUsageReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetLicenseInfo_v2(NvmlTaskRunner &, SafeVgpuInstance, nvmlVgpuLicenseInfo_t *info)
    {
        *info = licenseInfo;
        return licenseInfoReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFrameRateLimit(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *limit)
    {
        *limit = frameRateLimit;
        return frameRateLimitReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetGpuPciId(NvmlTaskRunner &, SafeVgpuInstance, char *buffer, unsigned int *length)
    {
        std::snprintf(buffer, *length, "%s", pciId.c_str());
        return pciIdReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetEncoderStats(NvmlTaskRunner &,
                                                 SafeVgpuInstance,
                                                 unsigned int *sessionCount,
                                                 unsigned int *averageFps,
                                                 unsigned int *averageLatency)
    {
        *sessionCount   = encoderSessionCount;
        *averageFps     = encoderAverageFps;
        *averageLatency = encoderLatency;
        return encoderStatsReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetEncoderSessions(NvmlTaskRunner &,
                                                    SafeVgpuInstance,
                                                    unsigned int *sessionCount,
                                                    nvmlEncoderSessionInfo_t *sessionInfo)
    {
        nvmlReturn_t const result = encoderSessionsReturns.at(encoderSessionsCallCount++);
        if (sessionInfo == nullptr)
        {
            *sessionCount = static_cast<unsigned int>(encoderSessions.size());
            return result;
        }

        *sessionCount = static_cast<unsigned int>(encoderSessions.size());
        for (std::size_t i = 0; i < encoderSessions.size(); i++)
        {
            sessionInfo[i] = encoderSessions[i];
        }
        return result;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFBCStats(NvmlTaskRunner &, SafeVgpuInstance, nvmlFBCStats_t *stats)
    {
        *stats = fbcStats;
        return fbcStatsReturn;
    }

    nvmlReturn_t NvmlVgpuInstanceGetGpuInstanceId(NvmlTaskRunner &, SafeVgpuInstance, unsigned int *id)
    {
        *id = gpuInstanceId;
        return gpuInstanceIdReturn;
    }

    dcgmReturn_t AppendEntityString(DcgmCacheManager &,
                                    dcgmcm_update_thread_t &,
                                    char const *value,
                                    timelib64_t,
                                    timelib64_t)
    {
        appendedStrings.emplace_back(value);
        return DCGM_ST_OK;
    }

    dcgmReturn_t AppendEntityInt64(DcgmCacheManager &,
                                   dcgmcm_update_thread_t &,
                                   long long value1,
                                   long long value2,
                                   timelib64_t timestamp,
                                   timelib64_t oldestKeepTimestamp)
    {
        appendedInt64s.push_back({ value1, value2, timestamp, oldestKeepTimestamp });
        return DCGM_ST_OK;
    }

    dcgmReturn_t UpdateFieldWatch(DcgmCacheManager &,
                                  dcgmcm_watch_info_p,
                                  timelib64_t monitorIntervalUsec,
                                  double maxAgeUsec,
                                  int maxKeepSamples,
                                  DcgmWatcher)
    {
        watchUpdates.push_back({ monitorIntervalUsec, maxAgeUsec, maxKeepSamples });
        return updateFieldWatchReturn;
    }

    char const *NvmlErrorToStringValue(nvmlReturn_t)
    {
        return errorStringValue;
    }

    long long NvmlErrorToInt64Value(nvmlReturn_t)
    {
        return errorInt64Value;
    }

    char const *NvmlErrorString(NvmlTaskRunner &, nvmlReturn_t)
    {
        return errorStringValue;
    }

    dcgmReturn_t NvmlReturnToDcgmReturn(nvmlReturn_t)
    {
        return mappedReturn;
    }

    void *Malloc(std::size_t size)
    {
        return std::malloc(size);
    }

    void Free(void *ptr)
    {
        std::free(ptr);
    }

    nvmlReturn_t NvmlDeviceGetFBCSessions(NvmlTaskRunner &, SafeNvmlHandle, unsigned int *, nvmlFBCSessionInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    nvmlReturn_t NvmlVgpuInstanceGetFBCSessions(NvmlTaskRunner &,
                                                SafeVgpuInstance,
                                                unsigned int *,
                                                nvmlFBCSessionInfo_t *)
    {
        return NVML_ERROR_NOT_SUPPORTED;
    }

    dcgmReturn_t AppendEntityBlob(DcgmCacheManager &,
                                  dcgmcm_update_thread_t &threadCtx,
                                  void *value,
                                  int valueSize,
                                  timelib64_t timestamp,
                                  timelib64_t oldestKeepTimestamp)
    {
        CapturedBlob blob;
        blob.fieldId       = threadCtx.entityKey.fieldId;
        blob.entityGroupId = static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId);
        blob.entityId      = threadCtx.entityKey.entityId;
        blob.valueSize     = valueSize;
        blob.timestamp     = timestamp;
        blob.expireTime    = oldestKeepTimestamp;
        auto *begin        = static_cast<std::byte *>(value);
        blob.bytes.assign(begin, begin + valueSize);
        appendedBlobs.push_back(std::move(blob));
        return DCGM_ST_OK;
    }
};

dcgmReturn_t BufferWithMockDeps(unsigned short fieldId, MockBufferDependencies &deps, dcgmcm_watch_info_t &watchInfo)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    REQUIRE(fieldMeta != nullptr);

    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    SafeVgpuInstance vgpuId { .vgpuInstance = VGPU_ID, .generation = 7 };
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.watchInfo               = &watchInfo;
    threadCtx.entityKey.entityGroupId = DCGM_FE_VGPU;
    threadCtx.entityKey.entityId      = VGPU_ID;
    threadCtx.entityKey.fieldId       = fieldId;

    DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
    DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
    return DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
        cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);
}
} // namespace

SCENARIO("DcgmVgpuDetail BufferOrCacheLatestVgpuValue buffers scalar and string fields through injectable dependencies")
{
    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    dcgmcm_watch_info_t watchInfo {};
    watchInfo.monitorIntervalUsec = 1;
    watchInfo.maxAgeUsec          = 50;

    GIVEN("NVML returns a VM id")
    {
        MockBufferDependencies deps;
        deps.vmId = "vm-domain-7";

        WHEN("the VM id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_VM_ID, deps, watchInfo);

            THEN("the dependency-provided VM id is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "vm-domain-7");
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
                CHECK(watchInfo.lastQueriedUsec == deps.now);
            }
        }
    }

    GIVEN("NVML cannot return the VM id")
    {
        MockBufferDependencies deps;
        deps.vmIdReturn   = NVML_ERROR_NOT_FOUND;
        deps.mappedReturn = DCGM_ST_NO_DATA;

        WHEN("the VM id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_VM_ID, deps, watchInfo);

            THEN("the mapped error string is appended and the mapped status is returned")
            {
                REQUIRE(result == DCGM_ST_NO_DATA);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == deps.errorStringValue);
                CHECK(watchInfo.lastStatus == NVML_ERROR_NOT_FOUND);
            }
        }
    }

    GIVEN("the VM name field is requested outside VMware")
    {
        MockBufferDependencies deps;
        deps.errorStringValue = "not-supported";

        WHEN("the VM name field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_VM_NAME, deps, watchInfo);

            THEN("the unsupported marker is appended without treating the query as a failed VM id lookup")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "not-supported");
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
                CHECK(deps.appendedInt64s.empty());
            }
        }
    }

    GIVEN("a device-scoped field is not implemented for vGPU updates")
    {
        MockBufferDependencies deps;

        WHEN("the unknown vGPU field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_GPU_NAME, deps, watchInfo);

            THEN("the helper reports a generic error without appending data")
            {
                REQUIRE(result == DCGM_ST_GENERIC_ERROR);
                CHECK(deps.appendedStrings.empty());
                CHECK(deps.appendedInt64s.empty());
                CHECK(deps.watchUpdates.empty());
                CHECK(watchInfo.lastQueriedUsec == deps.now);
            }
        }
    }

    GIVEN("NVML returns a vGPU type id")
    {
        MockBufferDependencies deps;
        deps.vgpuTypeId = 564;

        WHEN("the type field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_TYPE, deps, watchInfo);

            THEN("the type id is appended as an integer value")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == 564);
                CHECK(deps.appendedInt64s[0].value2 == 0);
                CHECK(deps.appendedInt64s[0].timestamp == deps.now);
                CHECK(deps.appendedInt64s[0].expireTime == deps.now - watchInfo.maxAgeUsec);
            }
        }
    }

    GIVEN("NVML cannot return a vGPU type id")
    {
        MockBufferDependencies deps;
        deps.typeReturn = NVML_ERROR_UNKNOWN;

        WHEN("the type field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_TYPE, deps, watchInfo);

            THEN("the mapped integer error is appended and the mapped status is returned")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == deps.errorInt64Value);
                CHECK(deps.appendedStrings.empty());
            }
        }
    }

    GIVEN("NVML returns a vGPU UUID")
    {
        MockBufferDependencies deps;
        deps.uuid = "GPU-1234";

        WHEN("the UUID field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_UUID, deps, watchInfo);

            THEN("the UUID string is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "GPU-1234");
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
            }
        }
    }

    GIVEN("NVML cannot return a vGPU UUID")
    {
        MockBufferDependencies deps;
        deps.uuidReturn   = NVML_ERROR_NOT_FOUND;
        deps.mappedReturn = DCGM_ST_NO_DATA;

        WHEN("the UUID field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_UUID, deps, watchInfo);

            THEN("the mapped error string is appended")
            {
                REQUIRE(result == DCGM_ST_NO_DATA);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == deps.errorStringValue);
                CHECK(watchInfo.lastStatus == NVML_ERROR_NOT_FOUND);
            }
        }
    }

    GIVEN("NVML returns a known VM driver version")
    {
        MockBufferDependencies deps;
        deps.driverVersion = "535.104.05";

        WHEN("the driver version field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_DRIVER_VERSION, deps, watchInfo);

            THEN("the version is appended and the watch interval is relaxed")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "535.104.05");
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.watchUpdates[0].monitorIntervalUsec == 900000000);
                CHECK(deps.watchUpdates[0].maxAgeUsec == 900.0);
                CHECK(deps.watchUpdates[0].maxKeepSamples == 1);
            }
        }
    }

    GIVEN("NVML reports an unavailable VM driver version")
    {
        MockBufferDependencies deps;
        deps.driverVersion = "Not Available";

        WHEN("the driver version field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_DRIVER_VERSION, deps, watchInfo);

            THEN("the unavailable value is appended without changing the watch interval")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Not Available");
                CHECK(deps.watchUpdates.empty());
            }
        }
    }

    GIVEN("NVML returns framebuffer usage")
    {
        MockBufferDependencies deps;
        deps.fbUsageBytes = 768ULL * 1024ULL * 1024ULL;

        WHEN("the memory usage field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_MEMORY_USAGE, deps, watchInfo);

            THEN("the usage is appended in MiB")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == 768);
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
            }
        }
    }

    GIVEN("NVML cannot return framebuffer usage")
    {
        MockBufferDependencies deps;
        deps.fbUsageReturn = NVML_ERROR_UNKNOWN;

        WHEN("the memory usage field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_MEMORY_USAGE, deps, watchInfo);

            THEN("the mapped integer error is appended")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == deps.errorInt64Value);
                CHECK(watchInfo.lastStatus == NVML_ERROR_UNKNOWN);
            }
        }
    }

    GIVEN("NVML returns a frame rate limit")
    {
        MockBufferDependencies deps;
        deps.frameRateLimit = 120;

        WHEN("the frame rate limit field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT, deps, watchInfo);

            THEN("the limit is appended as an integer value")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == 120);
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
            }
        }
    }

    GIVEN("NVML cannot return a frame rate limit")
    {
        MockBufferDependencies deps;
        deps.frameRateLimitReturn = NVML_ERROR_NOT_SUPPORTED;
        deps.mappedReturn         = DCGM_ST_NOT_SUPPORTED;

        WHEN("the frame rate limit field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT, deps, watchInfo);

            THEN("the mapped integer error is appended")
            {
                REQUIRE(result == DCGM_ST_NOT_SUPPORTED);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == deps.errorInt64Value);
                CHECK(watchInfo.lastStatus == NVML_ERROR_NOT_SUPPORTED);
            }
        }
    }

    GIVEN("NVML returns a valid vGPU PCI id")
    {
        MockBufferDependencies deps;
        deps.pciId = "00000000:65:00.4";

        WHEN("the PCI id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_PCI_ID, deps, watchInfo);

            THEN("the PCI id is appended and the watch interval is relaxed")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "00000000:65:00.4");
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.watchUpdates[0].monitorIntervalUsec == 3600000000);
                CHECK(deps.watchUpdates[0].maxAgeUsec == 3600.0);
                CHECK(deps.watchUpdates[0].maxKeepSamples == 1);
            }
        }
    }

    GIVEN("NVML returns the placeholder vGPU PCI id")
    {
        MockBufferDependencies deps;
        deps.pciId = "00000000:00:00.0";

        WHEN("the PCI id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_PCI_ID, deps, watchInfo);

            THEN("the placeholder is appended without changing the watch interval")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "00000000:00:00.0");
                CHECK(deps.watchUpdates.empty());
            }
        }
    }

    GIVEN("NVML cannot return a vGPU PCI id")
    {
        MockBufferDependencies deps;
        deps.pciIdReturn  = NVML_ERROR_UNKNOWN;
        deps.mappedReturn = DCGM_ST_NVML_ERROR;

        WHEN("the PCI id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_PCI_ID, deps, watchInfo);

            THEN("the mapped error string is appended")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == deps.errorStringValue);
                CHECK(deps.watchUpdates.empty());
            }
        }
    }

    GIVEN("NVML returns encoder stats")
    {
        MockBufferDependencies deps;
        deps.encoderSessionCount = 4;
        deps.encoderAverageFps   = 57;
        deps.encoderLatency      = 6;

        WHEN("the encoder stats field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_ENC_STATS, deps, watchInfo);

            THEN("the encoder stats payload is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedBlobs.size() == 1);
                auto const &stats = *reinterpret_cast<dcgmDeviceEncStats_t const *>(deps.appendedBlobs[0].bytes.data());
                CHECK(stats.version == dcgmDeviceEncStats_version);
                CHECK(stats.sessionCount == 4);
                CHECK(stats.averageFps == 57);
                CHECK(stats.averageLatency == 6);
            }
        }
    }

    GIVEN("NVML cannot return encoder stats")
    {
        MockBufferDependencies deps;
        deps.encoderStatsReturn = NVML_ERROR_UNKNOWN;

        WHEN("the encoder stats field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_ENC_STATS, deps, watchInfo);

            THEN("a zeroed stats payload is appended with the mapped status")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedBlobs.size() == 1);
                auto const &stats = *reinterpret_cast<dcgmDeviceEncStats_t const *>(deps.appendedBlobs[0].bytes.data());
                CHECK(stats.sessionCount == 0);
                CHECK(stats.averageFps == 0);
                CHECK(stats.averageLatency == 0);
            }
        }
    }

    GIVEN("NVML returns encoder session details")
    {
        MockBufferDependencies deps;
        nvmlEncoderSessionInfo_t session {};
        session.vgpuInstance   = 31;
        session.sessionId      = 42;
        session.pid            = 53;
        session.codecType      = NVML_ENCODER_QUERY_H264;
        session.hResolution    = 1920;
        session.vResolution    = 1080;
        session.averageFps     = 60;
        session.averageLatency = 8;
        deps.encoderSessions.push_back(session);

        WHEN("the encoder sessions field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO, deps, watchInfo);

            THEN("the session count and session detail records are appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedBlobs.size() == 1);
                auto const *sessions
                    = reinterpret_cast<dcgmDeviceVgpuEncSessions_t const *>(deps.appendedBlobs[0].bytes.data());
                CHECK(sessions[0].encoderSessionInfo.sessionCount == 1);
                CHECK(sessions[1].encoderSessionInfo.vgpuId == 31);
                CHECK(sessions[1].sessionId == 42);
                CHECK(sessions[1].pid == 53);
                CHECK(sessions[1].hResolution == 1920);
                CHECK(sessions[1].vResolution == 1080);
            }
        }
    }

    GIVEN("the second encoder session query fails")
    {
        MockBufferDependencies deps;
        deps.encoderSessionsReturns = { NVML_SUCCESS, NVML_ERROR_UNKNOWN };
        deps.encoderSessions.resize(1);

        WHEN("the encoder sessions field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO, deps, watchInfo);

            THEN("the mapped error is returned without appending a partial payload")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                CHECK(deps.appendedBlobs.empty());
                CHECK(watchInfo.lastStatus == NVML_ERROR_UNKNOWN);
            }
        }
    }

    GIVEN("NVML returns FBC stats")
    {
        MockBufferDependencies deps;
        deps.fbcStats.sessionsCount  = 5;
        deps.fbcStats.averageFPS     = 44;
        deps.fbcStats.averageLatency = 7;

        WHEN("the FBC stats field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_FBC_STATS, deps, watchInfo);

            THEN("the FBC stats payload is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedBlobs.size() == 1);
                auto const &stats = *reinterpret_cast<dcgmDeviceFbcStats_t const *>(deps.appendedBlobs[0].bytes.data());
                CHECK(stats.version == dcgmDeviceFbcStats_version);
                CHECK(stats.sessionCount == 5);
                CHECK(stats.averageFps == 44);
                CHECK(stats.averageLatency == 7);
            }
        }
    }

    GIVEN("NVML cannot return FBC stats")
    {
        MockBufferDependencies deps;
        deps.fbcStatsReturn = NVML_ERROR_UNKNOWN;

        WHEN("the FBC stats field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_FBC_STATS, deps, watchInfo);

            THEN("a zeroed FBC stats payload is appended with the mapped status")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedBlobs.size() == 1);
                auto const &stats = *reinterpret_cast<dcgmDeviceFbcStats_t const *>(deps.appendedBlobs[0].bytes.data());
                CHECK(stats.sessionCount == 0);
                CHECK(stats.averageFps == 0);
                CHECK(stats.averageLatency == 0);
            }
        }
    }

    GIVEN("NVML returns a GPU instance id")
    {
        MockBufferDependencies deps;
        deps.gpuInstanceId = 13;

        WHEN("the GPU instance id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID, deps, watchInfo);

            THEN("the GPU instance id is appended as an integer value")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == 13);
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
            }
        }
    }

    GIVEN("NVML cannot return a GPU instance id")
    {
        MockBufferDependencies deps;
        deps.gpuInstanceIdReturn = NVML_ERROR_UNKNOWN;

        WHEN("the GPU instance id field is buffered")
        {
            dcgmReturn_t const result = BufferWithMockDeps(DCGM_FI_DEV_VGPU_GPU_INSTANCE_ID, deps, watchInfo);

            THEN("the mapped integer error is appended")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedInt64s.size() == 1);
                CHECK(deps.appendedInt64s[0].value1 == deps.errorInt64Value);
                CHECK(watchInfo.lastStatus == NVML_ERROR_UNKNOWN);
            }
        }
    }
}

SCENARIO("DcgmVgpuDetail BufferOrCacheLatestVgpuValue formats license status through injectable dependencies")
{
    DcgmFieldsInit();
    DcgmNs::Defer deferFields([] { DcgmFieldsTerm(); });

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS);
    REQUIRE(fieldMeta != nullptr);

    DcgmCacheManager cacheManager;
    NvmlTaskRunner nvmlDriver;
    SafeVgpuInstance vgpuId { .vgpuInstance = VGPU_ID, .generation = 7 };
    dcgmcm_watch_info_t watchInfo {};
    watchInfo.monitorIntervalUsec = 1;
    dcgmcm_update_thread_t threadCtx {};
    threadCtx.watchInfo               = &watchInfo;
    threadCtx.entityKey.entityGroupId = DCGM_FE_VGPU;
    threadCtx.entityKey.entityId      = VGPU_ID;
    threadCtx.entityKey.fieldId       = DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATUS;

    GIVEN("NVML reports a licensed vGPU with a valid expiry timestamp")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState         = NVML_GRID_LICENSE_STATE_LICENSED;
        deps.licenseInfo.licenseExpiry.status = NVML_GRID_LICENSE_EXPIRY_VALID;
        deps.licenseInfo.licenseExpiry.year   = 2024;
        deps.licenseInfo.licenseExpiry.month  = 2;
        deps.licenseInfo.licenseExpiry.day    = 1;
        deps.licenseInfo.licenseExpiry.hour   = 2;
        deps.licenseInfo.licenseExpiry.min    = 36;
        deps.licenseInfo.licenseExpiry.sec    = 3;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the formatted license string is appended and the watch interval is relaxed")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Licensed (Expiry: 2024-2-1 2:36:3 GMT)");
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.watchUpdates[0].monitorIntervalUsec == 20000000);
                CHECK(deps.watchUpdates[0].maxAgeUsec == 20.0);
                CHECK(deps.watchUpdates[0].maxKeepSamples == 1);
                CHECK(watchInfo.lastStatus == NVML_SUCCESS);
                CHECK(watchInfo.lastQueriedUsec == deps.now);
            }
        }
    }

    GIVEN("NVML reports an unlicensed restricted vGPU")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState = NVML_GRID_LICENSE_STATE_UNLICENSED_RESTRICTED;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the unlicensed state is appended and the watch interval is tightened")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Unlicensed (Restricted)");
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.watchUpdates[0].monitorIntervalUsec == 1000000);
                CHECK(deps.watchUpdates[0].maxAgeUsec == 600.0);
                CHECK(deps.watchUpdates[0].maxKeepSamples == 600);
            }
        }
    }

    GIVEN("NVML reports a licensed vGPU with permanent expiry")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState         = NVML_GRID_LICENSE_STATE_LICENSED;
        deps.licenseInfo.licenseExpiry.status = NVML_GRID_LICENSE_EXPIRY_PERMANENT;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the permanent expiry text is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Licensed (Expiry: Permanent)");
            }
        }
    }

    GIVEN("NVML reports a licensed vGPU with invalid expiry")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState         = NVML_GRID_LICENSE_STATE_LICENSED;
        deps.licenseInfo.licenseExpiry.status = NVML_GRID_LICENSE_EXPIRY_INVALID;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the invalid expiry text is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Licensed (Expiry: ERR!)");
            }
        }
    }

    GIVEN("NVML reports a licensed vGPU with a not-applicable expiry")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState         = NVML_GRID_LICENSE_STATE_LICENSED;
        deps.licenseInfo.licenseExpiry.status = NVML_GRID_LICENSE_EXPIRY_NOT_APPLICABLE;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the not-applicable expiry text is appended")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Licensed (Expiry: N/A)");
            }
        }
    }

    GIVEN("NVML reports an unknown license state")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState = 999;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the unknown state is appended and the unlicensed watch interval is used")
            {
                REQUIRE(result == DCGM_ST_OK);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Unknown");
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.watchUpdates[0].monitorIntervalUsec == 1000000);
            }
        }
    }

    GIVEN("NVML cannot return license information")
    {
        MockBufferDependencies deps;
        deps.licenseInfoReturn = NVML_ERROR_UNKNOWN;
        deps.mappedReturn      = DCGM_ST_NVML_ERROR;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("Not Available is appended and the mapped status is returned")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.appendedStrings.size() == 1);
                CHECK(deps.appendedStrings[0] == "Not Available");
                CHECK(deps.watchUpdates.empty());
                CHECK(watchInfo.lastStatus == NVML_ERROR_UNKNOWN);
            }
        }
    }

    GIVEN("the cache watch update fails for a licensed vGPU")
    {
        MockBufferDependencies deps;
        deps.licenseInfo.currentState         = NVML_GRID_LICENSE_STATE_LICENSED;
        deps.licenseInfo.licenseExpiry.status = NVML_GRID_LICENSE_EXPIRY_PERMANENT;
        deps.updateFieldWatchReturn           = DCGM_ST_NVML_ERROR;

        WHEN("the license status field is buffered or cached")
        {
            DcgmVgpuDetail::VgpuFieldValueOps depsWrapper(&deps);
            DcgmVgpuDetail::FbcSessionOps fbcOps(&deps);
            dcgmReturn_t const result = DcgmVgpuDetail::BufferOrCacheLatestVgpuValue(
                cacheManager, threadCtx, nvmlDriver, vgpuId, fieldMeta, depsWrapper, fbcOps);

            THEN("the update error is returned before appending a license string")
            {
                REQUIRE(result == DCGM_ST_NVML_ERROR);
                REQUIRE(deps.watchUpdates.size() == 1);
                CHECK(deps.appendedStrings.empty());
            }
        }
    }
}
