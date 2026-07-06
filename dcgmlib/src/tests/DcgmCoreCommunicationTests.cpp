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

#include <catch2/catch_all.hpp>

#include <DcgmCoreCommunication.h>
#include <DcgmGroupManager.h>
#include <DcgmProtocol.h>
#include <Defer.hpp>
#include <dcgm_fields.h>

#include <array>
#include <cstddef>

namespace
{
struct CoreCommunicationFixture
{
    DcgmCacheManager cacheManager;
    HostEngineHandler hostEngine;
    DcgmGroupManager groupManager;
    DcgmCoreCommunication communication;

    CoreCommunicationFixture()
        : groupManager(&cacheManager, hostEngine, false)
    {
        communication.Init(&cacheManager, &groupManager);
    }
};

template <typename T>
void InitCoreHeader(T &msg, unsigned int subCommand, unsigned int version)
{
    msg.header.length     = sizeof(T);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = subCommand;
    msg.header.version    = version;
}
} // namespace

TEST_CASE("DcgmCoreCommunication validates request preconditions")
{
    DcgmCoreCommunication communication;

    GIVEN("a null request")
    {
        WHEN("it is posted to core")
        {
            auto postResult    = PostRequestToCore(nullptr, &communication);
            auto processResult = communication.ProcessRequestInCore(nullptr);

            THEN("bad parameter is returned")
            {
                CHECK(postResult == DCGM_ST_BADPARAM);
                CHECK(processResult == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a null poster")
    {
        dcgm_module_command_header_t header {};

        WHEN("a request is posted through the callback")
        {
            auto result = PostRequestToCore(&header, nullptr);

            THEN("bad parameter is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("an uninitialized communicator")
    {
        dcgm_module_command_header_t header {};
        header.subCommand = DcgmCoreReqIdCMGetGpuIds;

        WHEN("a request is processed")
        {
            auto result = communication.ProcessRequestInCore(&header);

            THEN("uninitialized is returned")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
            }
        }
    }

    GIVEN("an initialized communicator")
    {
        CoreCommunicationFixture fixture;

        dcgm_module_command_header_t header {};
        header.subCommand = 0xFFFFFFFF;

        WHEN("an unknown request is processed")
        {
            auto result = fixture.communication.ProcessRequestInCore(&header);

            THEN("the request is ignored")
            {
                CHECK(result == DCGM_ST_OK);
            }
        }
    }
}

TEST_CASE("DcgmCoreCommunication rejects known requests with version mismatches")
{
    CoreCommunicationFixture fixture;

    constexpr std::array subcommands {
        DcgmCoreReqIdCMGetGpuIds,
        DcgmCoreReqIdGetGpuStatus,
        DcgmCoreReqIdCMAreAllGpuIdsSameSku,
        DcgmCoreReqIdCMGetGpuCount,
        DcgmCoreReqIdCMAddFieldWatch,
        DcgmCoreReqIdCMGetInt64SummaryData,
        DcgmCoreReqIdCMGetLatestSample,
        DcgmCoreReqIdCMGetEntityNvLinkLinkStatus,
        DcgmCoreReqIdCMGetSamples,
        DcgmCoreReqIdCMGpuIdToNvmlIndex,
        DcgmCoreReqIdCMGetMultipleLatestLiveSamples,
        DcgmCoreReqIdCMRemoveFieldWatch,
        DcgmCoreReqIdCMGetAllGpuInfo,
        DcgmCoreReqIdCMAppendSamples,
        DcgmCoreReqIdCMSetValue,
        DcgmCoreReqIdCMNvmlIndexToGpuId,
        DcgmCoreReqIdCMUpdateAllFields,
        DcgmCoreReqIdGMVerifyAndUpdateGroupId,
        DcgmCoreReqIdGMGetGroupEntities,
        DcgmCoreReqIdGMAreAllTheSameSku,
        DcgmCoreReqIdGMGetGroupGpuIds,
        DcgmCoreReqIdLoggingGetSeverity,
        DcgmCoreReqIdSendModuleCommand,
        DcgmCoreReqIdSendRawMessage,
        DcgmCoreReqIdFGMPopulateFieldGroups,
        DcgmCoreReqIdNotifyRequestOfCompletion,
        DcgmCoreReqIdFGMGetFieldGroupFields,
        DcgmCoreReqIdGetMigInstanceEntityId,
        DcgmCoreReqIdGetMigUtilization,
        DcgmCoreReqMigIndicesForEntity,
        DcgmCoreReqGetServiceAccount,
        DcgmCoreReqPopulateMigHierarchy,
        DcgmCoreReqIdResourceReserve,
        DcgmCoreReqIdResourceFree,
        DcgmCoreReqIdChildProcessSpawn,
        DcgmCoreReqIdChildProcessStop,
        DcgmCoreReqIdChildProcessGetStatus,
        DcgmCoreReqIdChildProcessWait,
        DcgmCoreReqIdChildProcessDestroy,
        DcgmCoreReqIdChildProcessGetStdErrHandle,
        DcgmCoreReqIdChildProcessGetStdOutHandle,
        DcgmCoreReqIdChildProcessGetDataChannelHandle,
        DcgmCoreReqIdGetDriverVersion,
        DcgmCoreReqIdChildProcessManagerReset,
        DcgmCoreReqIdGetCudaVersion,
    };

    for (auto subcommand : subcommands)
    {
        CAPTURE(subcommand);

        alignas(std::max_align_t) std::array<std::byte, DCGM_PROTO_MAX_MESSAGE_SIZE> buffer {};
        auto *header       = reinterpret_cast<dcgm_module_command_header_t *>(buffer.data());
        header->length     = DCGM_PROTO_MAX_MESSAGE_SIZE;
        header->moduleId   = DcgmModuleIdCore;
        header->subCommand = subcommand;
        header->version    = 0xDEADBEEF;

        CHECK(fixture.communication.ProcessRequestInCore(header) != DCGM_ST_OK);
    }
}

TEST_CASE("DcgmCoreCommunication processes fake GPU cache-manager requests")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

    DcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();
    HostEngineHandler hostEngine;
    DcgmGroupManager groupManager(&cacheManager, hostEngine, false);

    DcgmCoreCommunication communication;
    communication.Init(&cacheManager, &groupManager);

    GIVEN("a fake GPU")
    {
        dcgmCoreGetGpuCount_t count {};
        InitCoreHeader(count, DcgmCoreReqIdCMGetGpuCount, dcgmCoreGetGpuCount_version);
        count.request.flag = 1;

        dcgmCoreGetGpuList_t list {};
        InitCoreHeader(list, DcgmCoreReqIdCMGetGpuIds, dcgmCoreGetGpuList_version);
        list.request.flag = 1;

        WHEN("count and list requests are processed")
        {
            auto countResult = communication.ProcessRequestInCore(&count.header);
            auto listResult  = communication.ProcessRequestInCore(&list.header);

            THEN("the fake GPU is reported")
            {
                REQUIRE(countResult == DCGM_ST_OK);
                CHECK(count.response.uintAnswer == 1);
                REQUIRE(listResult == DCGM_ST_OK);
                CHECK(list.response.ret == DCGM_ST_OK);
                REQUIRE(list.response.gpuCount == 1);
                CHECK(list.response.gpuIds[0] == gpuId);
            }
        }
    }

    GIVEN("a fake GPU")
    {
        dcgmCoreGetGpuStatus_t status {};
        InitCoreHeader(status, DcgmCoreReqIdGetGpuStatus, dcgmCoreGetGpuStatus_version);
        status.gpuId = gpuId;

        dcgmCoreQueryGpuInfo_t info {};
        InitCoreHeader(info, DcgmCoreReqIdCMGetAllGpuInfo, dcgmCoreQueryGpuInfo_version);

        WHEN("status and info requests are processed")
        {
            auto statusResult = communication.ProcessRequestInCore(&status.header);
            auto infoResult   = communication.ProcessRequestInCore(&info.header);

            THEN("cache state is returned")
            {
                REQUIRE(statusResult == DCGM_ST_OK);
                CHECK(status.status == DcgmEntityStatusFake);
                REQUIRE(infoResult == DCGM_ST_OK);
                CHECK(info.response.ret == DCGM_ST_OK);
                REQUIRE(info.response.infoCount == 1);
                CHECK(info.response.info[0].gpuId == gpuId);
                CHECK(info.response.info[0].status == DcgmEntityStatusFake);
            }
        }
    }

    GIVEN("a fake GPU")
    {
        dcgmCoreBasicQuery_t gpuToNvml {};
        InitCoreHeader(gpuToNvml, DcgmCoreReqIdCMGpuIdToNvmlIndex, dcgmCoreBasicQuery_version);
        gpuToNvml.request.entityId = gpuId;

        dcgmCoreBasicQuery_t nvmlToGpu {};
        InitCoreHeader(nvmlToGpu, DcgmCoreReqIdCMNvmlIndexToGpuId, dcgmCoreBasicQuery_version);
        nvmlToGpu.request.entityId = gpuId;

        dcgmCoreBasicQuery_t invalidGpuToNvml {};
        InitCoreHeader(invalidGpuToNvml, DcgmCoreReqIdCMGpuIdToNvmlIndex, dcgmCoreBasicQuery_version);
        invalidGpuToNvml.request.entityId = DCGM_MAX_NUM_DEVICES;

        dcgmCoreBasicQuery_t invalidNvmlToGpu {};
        InitCoreHeader(invalidNvmlToGpu, DcgmCoreReqIdCMNvmlIndexToGpuId, dcgmCoreBasicQuery_version);
        invalidNvmlToGpu.request.entityId = DCGM_MAX_NUM_DEVICES;

        WHEN("id and NVML index mappings are processed")
        {
            auto gpuToNvmlResult        = communication.ProcessRequestInCore(&gpuToNvml.header);
            auto nvmlToGpuResult        = communication.ProcessRequestInCore(&nvmlToGpu.header);
            auto invalidGpuToNvmlResult = communication.ProcessRequestInCore(&invalidGpuToNvml.header);
            auto invalidNvmlToGpuResult = communication.ProcessRequestInCore(&invalidNvmlToGpu.header);

            THEN("both valid directions resolve and invalid ids are rejected")
            {
                REQUIRE(gpuToNvmlResult == DCGM_ST_OK);
                CHECK(gpuToNvml.response.uintAnswer == gpuId);
                REQUIRE(nvmlToGpuResult == DCGM_ST_OK);
                CHECK(nvmlToGpu.response.uintAnswer == gpuId);
                CHECK(invalidGpuToNvmlResult == DCGM_ST_BADPARAM);
                CHECK(invalidNvmlToGpuResult == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("fake GPU ids")
    {
        dcgmCoreQueryGpuList_t sameSku {};
        InitCoreHeader(sameSku, DcgmCoreReqIdCMAreAllGpuIdsSameSku, dcgmCoreQueryGpuList_version);
        sameSku.request.gpuCount  = 1;
        sameSku.request.gpuIds[0] = gpuId;

        WHEN("SKU comparison is requested")
        {
            auto result = communication.ProcessRequestInCore(&sameSku.header);

            THEN("the result is true")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(sameSku.response.uintAnswer == 1);
            }
        }
    }
}
