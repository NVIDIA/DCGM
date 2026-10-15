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

#include "DcgmCoreProxy.h"
#include "DcgmFvBuffer.h"

#include <catch2/catch_test_macros.hpp>

#include <DcgmStringHelpers.h>

#include <string>
#include <vector>

namespace
{
constexpr auto c_expectedDriverVersion = "555.42.01";
constexpr int c_expectedCudaVersion    = 12080;

struct CorePostState
{
    dcgmReturn_t postReturn = DCGM_ST_OK;
    unsigned int calls      = 0;

    dcgmCoreReqCmd_t lastSubCommand = DcgmCoreReqIdCount;
    unsigned int lastVersion        = 0;
    size_t lastLength               = 0;
};

struct GpuListPostState : CorePostState
{
    int requestedActiveOnly  = 0;
    dcgmReturn_t responseRet = DCGM_ST_OK;
    std::vector<unsigned int> gpuIds;
};

struct GpuInfoPostState : CorePostState
{
    dcgmReturn_t responseRet = DCGM_ST_OK;
    std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
};

struct DriverVersionPostState : CorePostState
{
    std::string driverVersion;
};

struct ModuleCommandPostState : CorePostState
{
    void *command         = nullptr;
    dcgmReturn_t response = DCGM_ST_OK;
};

struct RawMessagePostState : CorePostState
{
    dcgmCoreRawMessage_t request {};
    dcgmReturn_t response = DCGM_ST_OK;
};

struct CudaVersionPostState : CorePostState
{
    int cudaVersion          = 0;
    dcgmReturn_t responseRet = DCGM_ST_OK;
};

template <typename StateT>
dcgmCoreCallbacks_t MakeCallbacks(StateT &state, dcgmCoreReqPost_f postfunc)
{
    dcgmCoreCallbacks_t callbacks {};
    callbacks.version  = dcgmCoreCallbacks_version;
    callbacks.postfunc = postfunc;
    callbacks.poster   = &state;
    return callbacks;
}

void RecordHeader(CorePostState &state, dcgm_module_command_header_t const &header)
{
    ++state.calls;
    state.lastSubCommand = static_cast<dcgmCoreReqCmd_t>(header.subCommand);
    state.lastVersion    = header.version;
    state.lastLength     = header.length;
}

dcgmReturn_t PostGpuList(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<GpuListPostState *>(poster);
    RecordHeader(state, *header);

    auto *msg                 = reinterpret_cast<dcgmCoreGetGpuList_t *>(header);
    state.requestedActiveOnly = msg->request.flag;
    msg->response.ret         = state.responseRet;
    msg->response.gpuCount    = static_cast<unsigned int>(state.gpuIds.size());
    for (size_t i = 0; i < state.gpuIds.size(); ++i)
    {
        msg->response.gpuIds[i] = state.gpuIds[i];
    }

    return state.postReturn;
}

dcgmReturn_t PostGpuInfo(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<GpuInfoPostState *>(poster);
    RecordHeader(state, *header);

    auto *msg               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
    msg->response.ret       = state.responseRet;
    msg->response.infoCount = static_cast<unsigned int>(state.gpuInfo.size());
    for (size_t i = 0; i < state.gpuInfo.size(); ++i)
    {
        msg->response.info[i] = state.gpuInfo[i];
    }

    return state.postReturn;
}

dcgmReturn_t PostDriverVersion(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<DriverVersionPostState *>(poster);
    RecordHeader(state, *header);

    auto *msg = reinterpret_cast<dcgmCoreReqIdGetDriverVersion_t *>(header);
    SafeCopyTo(msg->driverVersion, state.driverVersion.c_str());

    return state.postReturn;
}

dcgmReturn_t PostModuleCommand(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<ModuleCommandPostState *>(poster);
    RecordHeader(state, *header);

    auto *msg     = reinterpret_cast<dcgmCoreSendModuleCommand_t *>(header);
    state.command = msg->command;
    msg->response = state.response;

    return state.postReturn;
}

dcgmReturn_t PostRawMessage(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<RawMessagePostState *>(poster);
    RecordHeader(state, *header);

    auto *msg     = reinterpret_cast<dcgmCoreSendRawMessage_t *>(header);
    state.request = msg->request;
    msg->response = state.response;

    return state.postReturn;
}

dcgmReturn_t PostCudaVersion(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<CudaVersionPostState *>(poster);
    RecordHeader(state, *header);

    auto *msg         = reinterpret_cast<dcgmCoreGetCudaVersion_v1 *>(header);
    msg->cudaVersion  = state.cudaVersion;
    msg->response.ret = state.responseRet;

    return state.postReturn;
}
} // namespace

SCENARIO("DcgmCoreProxy GPU list routing uses core callbacks", "[core][common]")
{
    GIVEN("a proxy with a GPU-list callback response")
    {
        GpuListPostState state;
        state.gpuIds   = { 7, 11 };
        auto callbacks = MakeCallbacks(state, PostGpuList);
        DcgmCoreProxy proxy(callbacks);
        std::vector<unsigned int> gpuIds { 99 };

        WHEN("GPU ids are requested")
        {
            auto const ret = proxy.GetGpuIds(1, gpuIds);

            THEN("the request is routed with the expected core header and response")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(state.calls == 1U);
                CHECK(state.lastSubCommand == DcgmCoreReqIdCMGetGpuIds);
                CHECK(state.lastVersion == dcgmCoreGetGpuList_version);
                CHECK(state.lastLength == sizeof(dcgmCoreGetGpuList_t));
                CHECK(state.requestedActiveOnly == 1);
                std::vector<unsigned int> const expectedGpuIds { 7, 11 };
                CHECK(gpuIds == expectedGpuIds);
            }
        }
    }

    GIVEN("a proxy whose GPU-list post fails")
    {
        GpuListPostState state;
        state.postReturn = DCGM_ST_GENERIC_ERROR;
        state.gpuIds     = { 3 };
        auto callbacks   = MakeCallbacks(state, PostGpuList);
        DcgmCoreProxy proxy(callbacks);
        std::vector<unsigned int> gpuIds { 99 };

        WHEN("GPU ids are requested")
        {
            auto const ret = proxy.GetGpuIds(0, gpuIds);

            THEN("the post error is returned and the caller's vector is cleared")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
                CHECK(state.calls == 1U);
                CHECK(gpuIds.empty());
            }
        }
    }
}

SCENARIO("DcgmCoreProxy GPU info routing feeds chip architecture lookup", "[core][common]")
{
    GIVEN("a proxy with cached GPU info returned by the callback")
    {
        GpuInfoPostState state;
        dcgmcm_gpu_info_cached_t first {};
        first.gpuId = 4;
        first.arch  = DCGM_CHIP_ARCH_AMPERE;
        dcgmcm_gpu_info_cached_t second {};
        second.gpuId   = 9;
        second.arch    = DCGM_CHIP_ARCH_HOPPER;
        state.gpuInfo  = { first, second };
        auto callbacks = MakeCallbacks(state, PostGpuInfo);
        DcgmCoreProxy proxy(callbacks);

        WHEN("all GPU info is requested")
        {
            std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
            auto const ret = proxy.GetAllGpuInfo(gpuInfo);

            THEN("the routed response is copied to the caller")
            {
                CHECK(ret == DCGM_ST_OK);
                REQUIRE(gpuInfo.size() == 2U);
                CHECK(gpuInfo[0].gpuId == 4U);
                CHECK(gpuInfo[0].arch == DCGM_CHIP_ARCH_AMPERE);
                CHECK(gpuInfo[1].gpuId == 9U);
                CHECK(gpuInfo[1].arch == DCGM_CHIP_ARCH_HOPPER);
                CHECK(state.lastSubCommand == DcgmCoreReqIdCMGetAllGpuInfo);
                CHECK(state.lastVersion == dcgmCoreQueryGpuInfo_version);
                CHECK(state.lastLength == sizeof(dcgmCoreQueryGpuInfo_t));
            }
        }

        WHEN("a present GPU architecture is requested")
        {
            auto const arch = proxy.GetGpuChipArch(9);

            THEN("the matching chip architecture is returned")
            {
                REQUIRE(arch.has_value());
                CHECK(*arch == DCGM_CHIP_ARCH_HOPPER);
            }
        }

        WHEN("a missing GPU architecture is requested")
        {
            auto const arch = proxy.GetGpuChipArch(123);

            THEN("a bad-parameter result is returned")
            {
                REQUIRE_FALSE(arch.has_value());
                CHECK(arch.error() == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a proxy whose GPU-info response reports an error")
    {
        GpuInfoPostState state;
        state.responseRet = DCGM_ST_NO_DATA;
        auto callbacks    = MakeCallbacks(state, PostGpuInfo);
        DcgmCoreProxy proxy(callbacks);

        WHEN("a GPU architecture is requested")
        {
            auto const arch = proxy.GetGpuChipArch(0);

            THEN("the core response error is propagated")
            {
                REQUIRE_FALSE(arch.has_value());
                CHECK(arch.error() == DCGM_ST_NO_DATA);
            }
        }
    }
}

SCENARIO("DcgmCoreProxy forwards scalar and module-message requests", "[core][common]")
{
    GIVEN("a proxy with a driver-version callback response")
    {
        DriverVersionPostState state;
        state.driverVersion = c_expectedDriverVersion;
        auto callbacks      = MakeCallbacks(state, PostDriverVersion);
        DcgmCoreProxy proxy(callbacks);
        std::string driverVersion;

        WHEN("the driver version is requested")
        {
            auto const ret = proxy.GetDriverVersion(driverVersion);

            THEN("the routed string response is returned")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(driverVersion == c_expectedDriverVersion);
                CHECK(state.lastSubCommand == DcgmCoreReqIdGetDriverVersion);
                CHECK(state.lastVersion == dcgmCoreReqIdGetDriverVersion_version);
                CHECK(state.lastLength == sizeof(dcgmCoreReqIdGetDriverVersion_t));
            }
        }
    }

    GIVEN("a proxy with a CUDA-version callback response")
    {
        CudaVersionPostState state;
        state.cudaVersion = c_expectedCudaVersion;
        auto callbacks    = MakeCallbacks(state, PostCudaVersion);
        DcgmCoreProxy proxy(callbacks);
        int cudaVersion = 0;

        WHEN("the CUDA version is requested")
        {
            auto const ret = proxy.GetCudaVersion(cudaVersion);

            THEN("the routed integer response is returned")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(cudaVersion == c_expectedCudaVersion);
                CHECK(state.lastSubCommand == DcgmCoreReqIdGetCudaVersion);
                CHECK(state.lastVersion == dcgmCoreGetCudaVersion_version1);
                CHECK(state.lastLength == sizeof(dcgmCoreGetCudaVersion_v1));
            }
        }
    }

    GIVEN("a proxy with a module-command callback response")
    {
        ModuleCommandPostState state;
        state.response = DCGM_ST_NOT_SUPPORTED;
        auto callbacks = MakeCallbacks(state, PostModuleCommand);
        DcgmCoreProxy proxy(callbacks);
        int command = 42;

        WHEN("a module command is sent")
        {
            auto const ret = proxy.SendModuleCommand(&command);

            THEN("the command pointer is forwarded and the inner response is returned")
            {
                CHECK(ret == DCGM_ST_NOT_SUPPORTED);
                CHECK(state.command == &command);
                CHECK(state.lastSubCommand == DcgmCoreReqIdSendModuleCommand);
                CHECK(state.lastVersion == dcgmCoreSendModuleCommand_version);
                CHECK(state.lastLength == sizeof(dcgmCoreSendModuleCommand_t));
            }
        }
    }

    GIVEN("a proxy whose module-command post fails")
    {
        ModuleCommandPostState state;
        state.postReturn = DCGM_ST_CONNECTION_NOT_VALID;
        state.response   = DCGM_ST_OK;
        auto callbacks   = MakeCallbacks(state, PostModuleCommand);
        DcgmCoreProxy proxy(callbacks);
        int command = 7;

        WHEN("a module command is sent")
        {
            auto const ret = proxy.SendModuleCommand(&command);

            THEN("the post error is returned before the inner response")
            {
                CHECK(ret == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(state.command == &command);
            }
        }
    }

    GIVEN("a proxy with a raw-message callback response")
    {
        RawMessagePostState state;
        state.response = DCGM_ST_TIMEOUT;
        auto callbacks = MakeCallbacks(state, PostRawMessage);
        DcgmCoreProxy proxy(callbacks);
        int payload = 99;

        WHEN("a raw client message is sent")
        {
            auto const ret = proxy.SendRawMessageToClient(12, 34, 56, &payload, sizeof(payload), DCGM_ST_OK);

            THEN("the raw message fields are forwarded and the inner response is returned")
            {
                CHECK(ret == DCGM_ST_TIMEOUT);
                CHECK(state.request.connectionId == static_cast<dcgm_connection_id_t>(12));
                CHECK(state.request.msgType == 34U);
                CHECK(state.request.requestId == static_cast<dcgm_request_id_t>(56));
                CHECK(state.request.msgData == &payload);
                CHECK(state.request.msgSize == static_cast<unsigned int>(sizeof(payload)));
                CHECK(state.request.status == DCGM_ST_OK);
                CHECK(state.lastSubCommand == DcgmCoreReqIdSendRawMessage);
                CHECK(state.lastVersion == dcgmCoreSendRawMessage_version);
                CHECK(state.lastLength == sizeof(dcgmCoreSendRawMessage_t));
            }
        }
    }
}


namespace
{
struct ExtendedCorePostState : CorePostState
{
    dcgmReturn_t responseRet       = DCGM_ST_OK;
    unsigned int uintAnswer        = 0;
    DcgmEntityStatus_t gpuStatus   = DcgmEntityStatusOk;
    DcgmLoggingSeverity_t severity = DcgmLoggingSeverityInfo;
    bool wereFirstWatcher          = false;
    unsigned int lastBasicEntityId = 0;
    dcgmCoreBasicReqParams_t lastBasicRequest {};
    dcgmCoreGetInt64SummaryDataParams_t lastSummaryRequest {};
    dcgmCoreGetLatestSampleParams_t lastLatestRequest {};
    dcgmCoreGetSamplesParams_t lastSamplesRequest {};
    dcgmCoreGetMultipleLatestLiveSamplesParams_t lastMultipleLatestRequest {};
    dcgmCoreAddFieldWatchReqParams_v1 lastWatch {};
    dcgmCoreRemoveFieldWatchParams_t lastRemove {};
    dcgmCoreAppendSamplesParams_t lastAppend {};
    dcgmCoreGetGroupEntitiesParams_t lastGroupRequest {};
    dcgmCoreBasicGroupParams_t lastBasicGroup {};
    dcgmCoreGetComputeInstanceEntityIdParams_t lastMigEntityRequest {};
    dcgmCoreGetMigUtilizationParams_t lastMigUtilizationRequest {};
    dcgmCoreGetMigIndicesForEntityParams_t lastMigIndicesRequest {};
    std::vector<dcgmGroupEntityPair_t> groupEntities;
    std::vector<unsigned int> groupGpuIds;
    std::vector<unsigned short> fieldIds;
    std::vector<long long> summaryValues;
    std::vector<dcgmcm_sample_t> samples;
    dcgmcm_sample_t latestSample {};
    dcgmNvLinkLinkState_t linkState = DcgmNvLinkLinkStateUp;
    dcgmAllFieldGroup_t allFieldGroups {};
    dcgmMigHierarchy_v2 migHierarchy {};
    size_t migCapacity = 0;
    size_t migUsed     = 0;
    std::string serviceAccount;
    ChildProcessHandle_t childHandle = 0;
    int childPid                     = 0;
    int childFd                      = -1;
    dcgmChildProcessStatus_t childStatus {};
};

dcgmReturn_t PostExtendedCore(dcgm_module_command_header_t *header, void *poster)
{
    auto &state = *static_cast<ExtendedCorePostState *>(poster);
    RecordHeader(state, *header);

    switch (header->subCommand)
    {
        case DcgmCoreReqIdGetGpuStatus:
        {
            auto *msg               = reinterpret_cast<dcgmCoreGetGpuStatus_t *>(header);
            state.lastBasicEntityId = msg->gpuId;
            msg->status             = state.gpuStatus;
            break;
        }
        case DcgmCoreReqIdCMGetGpuCount:
        {
            auto *msg                = reinterpret_cast<dcgmCoreGetGpuCount_t *>(header);
            state.lastBasicEntityId  = msg->request.flag;
            msg->response.ret        = state.responseRet;
            msg->response.uintAnswer = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdCMAreAllGpuIdsSameSku:
        {
            auto *msg                = reinterpret_cast<dcgmCoreQueryGpuList_t *>(header);
            state.lastBasicEntityId  = msg->request.gpuCount;
            msg->response.ret        = state.responseRet;
            msg->response.uintAnswer = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdCMNvmlIndexToGpuId:
        case DcgmCoreReqIdCMGpuIdToNvmlIndex:
        case DcgmCoreReqIdGMVerifyAndUpdateGroupId:
        {
            auto *msg                = reinterpret_cast<dcgmCoreBasicQuery_t *>(header);
            state.lastBasicEntityId  = msg->request.entityId;
            msg->response.ret        = state.responseRet;
            msg->response.uintAnswer = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdCMUpdateAllFields:
        {
            auto *msg               = reinterpret_cast<dcgmCoreGetGpuCount_t *>(header);
            state.lastBasicEntityId = msg->request.flag;
            msg->response.ret       = state.responseRet;
            break;
        }
        case DcgmCoreReqIdCMAddFieldWatch:
        {
            auto *msg                      = reinterpret_cast<dcgmCoreAddFieldWatch_v2 *>(header);
            state.lastWatch                = msg->request;
            msg->response.wereFirstWatcher = state.wereFirstWatcher;
            msg->ret                       = state.responseRet;
            break;
        }
        case DcgmCoreReqIdCMRemoveFieldWatch:
        {
            auto *msg        = reinterpret_cast<dcgmCoreRemoveFieldWatch_t *>(header);
            state.lastRemove = msg->request;
            msg->ret         = state.responseRet;
            break;
        }
        case DcgmCoreReqIdCMGetEntityNvLinkLinkStatus:
        {
            auto *msg              = reinterpret_cast<dcgmCoreGetEntityNvLinkLinkStatus_t *>(header);
            state.lastBasicRequest = msg->request;
            msg->response.ret      = state.responseRet;
            for (auto &linkState : msg->response.linkStates)
            {
                linkState = state.linkState;
            }
            break;
        }
        case DcgmCoreReqIdCMGetInt64SummaryData:
        {
            auto *msg                     = reinterpret_cast<dcgmCoreGetInt64SummaryData_t *>(header);
            state.lastSummaryRequest      = msg->request;
            msg->response.ret             = state.responseRet;
            msg->response.numSummaryTypes = static_cast<unsigned int>(state.summaryValues.size());
            for (size_t i = 0; i < state.summaryValues.size(); ++i)
            {
                msg->response.summaryValues[i] = state.summaryValues[i];
            }
            break;
        }
        case DcgmCoreReqIdCMGetLatestSample:
        {
            auto *msg                = reinterpret_cast<dcgmCoreGetLatestSample_t *>(header);
            state.lastLatestRequest  = msg->request;
            msg->response.ret        = state.responseRet;
            msg->response.sample     = state.latestSample;
            msg->response.bufferSize = 0;
            break;
        }
        case DcgmCoreReqIdCMGetSamples:
        {
            auto *msg                = reinterpret_cast<dcgmCoreGetSamples_t *>(header);
            state.lastSamplesRequest = msg->request;
            msg->response.ret        = state.responseRet;
            msg->response.numSamples = static_cast<int>(state.samples.size());
            for (size_t i = 0; i < state.samples.size() && i < static_cast<size_t>(msg->request.maxSamples); ++i)
            {
                msg->response.samples[i] = state.samples[i];
            }
            break;
        }
        case DcgmCoreReqIdCMGetMultipleLatestLiveSamples:
        {
            auto *msg                       = reinterpret_cast<dcgmCoreGetMultipleLatestLiveSamples_t *>(header);
            state.lastMultipleLatestRequest = msg->request;
            msg->response.ret               = state.responseRet;
            msg->response.bufferSize        = 0;
            msg->response.dataDidNotFit     = 0;
            break;
        }
        case DcgmCoreReqIdCMAppendSamples:
        {
            auto *msg        = reinterpret_cast<dcgmCoreAppendSamples_t *>(header);
            state.lastAppend = msg->request;
            msg->ret         = state.responseRet;
            break;
        }
        case DcgmCoreReqIdGMGetGroupEntities:
        {
            auto *msg                      = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(header);
            state.lastGroupRequest         = msg->request;
            msg->response.ret              = state.responseRet;
            msg->response.entityPairsCount = static_cast<unsigned int>(state.groupEntities.size());
            for (size_t i = 0; i < state.groupEntities.size(); ++i)
            {
                msg->response.entityPairs[i] = state.groupEntities[i];
            }
            break;
        }
        case DcgmCoreReqIdGMAreAllTheSameSku:
        {
            auto *msg                = reinterpret_cast<dcgmCoreBasicGroup_t *>(header);
            state.lastBasicGroup     = msg->request;
            msg->response.ret        = state.responseRet;
            msg->response.uintAnswer = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdGMGetGroupGpuIds:
        {
            auto *msg              = reinterpret_cast<dcgmCoreGetGroupGpuIds_t *>(header);
            state.lastGroupRequest = msg->request;
            msg->response.ret      = state.responseRet;
            msg->response.gpuCount = static_cast<unsigned int>(state.groupGpuIds.size());
            for (size_t i = 0; i < state.groupGpuIds.size(); ++i)
            {
                msg->response.gpuIds[i] = state.groupGpuIds[i];
            }
            break;
        }
        case DcgmCoreReqIdFGMGetFieldGroupFields:
        {
            auto *msg                 = reinterpret_cast<dcgmCoreGetFieldGroupFields_t *>(header);
            msg->response.ret         = state.responseRet;
            msg->response.numFieldIds = state.fieldIds.size();
            for (size_t i = 0; i < state.fieldIds.size(); ++i)
            {
                msg->response.fieldIds[i] = state.fieldIds[i];
            }
            break;
        }
        case DcgmCoreReqIdLoggingGetSeverity:
        {
            auto *msg     = reinterpret_cast<dcgmCoreGetSeverity_t *>(header);
            msg->response = state.severity;
            break;
        }
        case DcgmCoreReqIdNotifyRequestOfCompletion:
        {
            auto *msg     = reinterpret_cast<dcgmCoreNotifyRequestOfCompletion_t *>(header);
            msg->response = state.responseRet;
            break;
        }
        case DcgmCoreReqIdFGMPopulateFieldGroups:
        {
            auto *msg            = reinterpret_cast<dcgmCorePopulateFieldGroups_t *>(header);
            msg->response.groups = state.allFieldGroups;
            msg->response.status = state.responseRet;
            break;
        }
        case DcgmCoreReqIdGetMigInstanceEntityId:
        {
            auto *msg                  = reinterpret_cast<dcgmCoreGetComputeInstanceEntityId_t *>(header);
            state.lastMigEntityRequest = msg->request;
            msg->response.ret          = state.responseRet;
            msg->response.uintAnswer   = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdGetMigUtilization:
        {
            auto *msg                       = reinterpret_cast<dcgmCoreGetMigUtilization_t *>(header);
            state.lastMigUtilizationRequest = msg->request;
            msg->response.ret               = state.responseRet;
            msg->response.capacity          = state.migCapacity;
            msg->response.usage             = state.migUsed;
            break;
        }
        case DcgmCoreReqMigIndicesForEntity:
        {
            auto *msg                   = reinterpret_cast<dcgmCoreGetMigIndicesForEntity_t *>(header);
            state.lastMigIndicesRequest = msg->request;
            msg->response.ret           = state.responseRet;
            msg->response.gpuId         = state.uintAnswer;
            msg->response.instanceId    = 77;
            break;
        }
        case DcgmCoreReqPopulateMigHierarchy:
        {
            auto *msg     = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
            msg->response = state.migHierarchy;
            break;
        }
        case DcgmCoreReqGetServiceAccount:
        {
            auto *msg = reinterpret_cast<dcgmCoreGetServiceAccount_t *>(header);
            SafeCopyTo(msg->response.serviceAccount, state.serviceAccount.c_str());
            break;
        }
        case DcgmCoreReqIdResourceReserve:
        {
            auto *msg           = reinterpret_cast<dcgmCoreResourceReserve_t *>(header);
            msg->response.ret   = state.responseRet;
            msg->response.token = state.uintAnswer;
            break;
        }
        case DcgmCoreReqIdResourceFree:
        {
            auto *msg               = reinterpret_cast<dcgmCoreResourceFree_t *>(header);
            state.lastBasicEntityId = msg->request.token;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessSpawn:
        {
            auto *msg   = reinterpret_cast<dcgmCoreChildProcessSpawn_t *>(header);
            msg->ret    = state.responseRet;
            msg->handle = state.childHandle;
            msg->pid    = state.childPid;
            break;
        }
        case DcgmCoreReqIdChildProcessStop:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessStop_t *>(header);
            state.lastBasicEntityId = msg->handle;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessGetStatus:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessGetStatus_t *>(header);
            state.lastBasicEntityId = msg->handle;
            msg->status             = state.childStatus;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessWait:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessWait_t *>(header);
            state.lastBasicEntityId = msg->timeoutSec;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessDestroy:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessDestroy_t *>(header);
            state.lastBasicEntityId = msg->sigTermTimeoutSec;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessGetStdErrHandle:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessGetStdErrHandle_t *>(header);
            state.lastBasicEntityId = msg->handle;
            msg->fd                 = state.childFd;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessGetStdOutHandle:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessGetStdOutHandle_t *>(header);
            state.lastBasicEntityId = msg->handle;
            msg->fd                 = state.childFd;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessGetDataChannelHandle:
        {
            auto *msg               = reinterpret_cast<dcgmCoreChildProcessGetDataChannelHandle_t *>(header);
            state.lastBasicEntityId = msg->handle;
            msg->fd                 = state.childFd;
            msg->ret                = state.responseRet;
            break;
        }
        case DcgmCoreReqIdChildProcessManagerReset:
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessManagerReset_t *>(header);
            msg->ret  = state.responseRet;
            break;
        }
        default:
            break;
    }

    return state.postReturn;
}
} // namespace

SCENARIO("DcgmCoreProxy forwards additional scalar, watch, and group requests", "[core][common]")
{
    GIVEN("a proxy backed by an extended core callback")
    {
        ExtendedCorePostState state;
        state.uintAnswer       = 17;
        state.responseRet      = DCGM_ST_OK;
        state.gpuStatus        = DcgmEntityStatusFake;
        state.wereFirstWatcher = true;
        state.groupEntities    = { { DCGM_FE_GPU, 3 }, { DCGM_FE_SWITCH, 9 } };
        state.groupGpuIds      = { 4, 8 };
        state.fieldIds         = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
        auto callbacks         = MakeCallbacks(state, PostExtendedCore);
        DcgmCoreProxy proxy(callbacks);

        WHEN("GPU status and SKU information are requested")
        {
            std::vector<unsigned int> gpuIds { 1, 2, 3 };
            auto const status  = proxy.GetGpuStatus(44);
            auto const sameSku = proxy.AreAllGpuIdsSameSku(gpuIds);

            THEN("the scalar core replies are returned")
            {
                CHECK(status == DcgmEntityStatusFake);
                CHECK(sameSku);
                CHECK(state.calls == 2U);
                CHECK(state.lastSubCommand == DcgmCoreReqIdCMAreAllGpuIdsSameSku);
                CHECK(state.lastBasicEntityId == gpuIds.size());
            }
        }

        WHEN("index translation and group-id validation are requested")
        {
            auto const gpuId     = proxy.NvmlIndexToGpuId(6);
            auto const nvmlIndex = proxy.GpuIdToNvmlIndex(8);
            unsigned int groupId = 3;
            auto const verifyRet = proxy.VerifyAndUpdateGroupId(&groupId);

            THEN("the common basic-query envelope is used for each request")
            {
                CHECK(gpuId == 17U);
                CHECK(nvmlIndex == 17);
                CHECK(verifyRet == DCGM_ST_OK);
                CHECK(groupId == 17U);
                CHECK(state.lastSubCommand == DcgmCoreReqIdGMVerifyAndUpdateGroupId);
            }
        }

        WHEN("field watches are added and removed")
        {
            DcgmWatcher watcher(DcgmWatcherTypeClient, 99);
            bool wereFirstWatcher = false;
            auto const addRet     = proxy.AddFieldWatch(
                DCGM_FE_GPU, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 5000, 42.0, 7, watcher, true, false, wereFirstWatcher);
            auto const removeRet = proxy.RemoveFieldWatch(DCGM_FE_GPU, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 1, watcher);

            THEN("watch parameters and watcher identity are forwarded")
            {
                CHECK(addRet == DCGM_ST_OK);
                CHECK(removeRet == DCGM_ST_OK);
                CHECK_FALSE(wereFirstWatcher);
                CHECK(state.lastWatch.entityGroupId == DCGM_FE_GPU);
                CHECK(state.lastWatch.entityId == 2U);
                CHECK(state.lastWatch.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                CHECK(state.lastWatch.monitorFreqUsec == 5000);
                CHECK(state.lastWatch.maxSampleAge == 42.0);
                CHECK(state.lastWatch.maxKeepSamples == 7);
                CHECK(state.lastWatch.watcherType == DcgmWatcherTypeClient);
                CHECK(state.lastWatch.connectionId == 99U);
                CHECK(state.lastWatch.subscribeForUpdates);
                CHECK_FALSE(state.lastWatch.updateOnFirstWatch);
                CHECK(state.lastRemove.clearCache == 1);
            }
        }

        WHEN("group entities and GPU ids are requested")
        {
            std::vector<dcgmGroupEntityPair_t> entities;
            std::vector<unsigned int> groupGpuIds { 99 };
            bool areSameSku      = false;
            auto const entityRet = proxy.GetGroupEntities(12, EntityListOption::ActiveOnly, entities);
            auto const skuRet    = proxy.AreAllTheSameSku(55, 12, &areSameSku);
            auto const gpuRet    = proxy.GetGroupGpuIds(55, 12, EntityListOption::All, groupGpuIds);

            THEN("the group response vectors and bool response are copied")
            {
                CHECK(entityRet == DCGM_ST_OK);
                CHECK(skuRet == DCGM_ST_OK);
                CHECK(gpuRet == DCGM_ST_OK);
                CHECK(areSameSku);
                REQUIRE(entities.size() == 2U);
                CHECK(entities[0].entityGroupId == DCGM_FE_GPU);
                CHECK(entities[1].entityGroupId == DCGM_FE_SWITCH);
                CHECK(groupGpuIds == std::vector<unsigned int> { 4, 8 });
                CHECK(state.lastGroupRequest.groupId == 12U);
                CHECK_FALSE(state.lastGroupRequest.activeOnly);
            }
        }

        WHEN("field group fields are requested")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_UUID };
            auto const ret = proxy.GetFieldGroupFields(static_cast<dcgmFieldGrp_t>(0x1234), fieldIds);

            THEN("the field-id response replaces the caller vector")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(fieldIds
                      == std::vector<unsigned short> { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS });
                CHECK(state.lastSubCommand == DcgmCoreReqIdFGMGetFieldGroupFields);
            }
        }
    }
}

SCENARIO("DcgmCoreProxy covers append-sample, MIG, service-account, resource, and child-process routing",
         "[core][common]")
{
    GIVEN("a proxy backed by an extended core callback")
    {
        ExtendedCorePostState state;
        state.responseRet    = DCGM_ST_OK;
        state.uintAnswer     = 21;
        state.migCapacity    = 14;
        state.migUsed        = 5;
        state.serviceAccount = "svc-dcgm";
        state.childHandle    = 0xabc;
        state.childPid       = 4242;
        state.childFd        = 12;
        state.childStatus    = dcgmChildProcessStatus_t {};
        auto callbacks       = MakeCallbacks(state, PostExtendedCore);
        DcgmCoreProxy proxy(callbacks);

        WHEN("samples are appended from a non-empty field-value buffer")
        {
            DcgmFvBuffer buffer;
            REQUIRE(buffer.AddInt64Value(DCGM_FE_GPU, 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 77, 12345, DCGM_ST_OK)
                    != nullptr);
            size_t expectedBufferSize = 0;
            REQUIRE(buffer.GetSize(&expectedBufferSize, nullptr) == DCGM_ST_OK);
            auto const ret = proxy.AppendSamples(&buffer);

            THEN("the raw buffer pointer and size are forwarded")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(state.lastSubCommand == DcgmCoreReqIdCMAppendSamples);
                CHECK(state.lastAppend.buffer == buffer.GetBuffer());
                CHECK(state.lastAppend.bufferSize == expectedBufferSize);
            }
        }

        WHEN("an empty field-value buffer is appended")
        {
            DcgmFvBuffer buffer;
            auto const ret = proxy.AppendSamples(&buffer);

            THEN("the proxy rejects it before posting to core")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("MIG entity and utilization helpers are requested")
        {
            dcgm_field_eid_t gpuInstanceEntity = 0;
            dcgm_field_eid_t computeEntity     = 0;
            size_t capacity                    = 0;
            size_t used                        = 0;
            unsigned int gpuId                 = 0;
            dcgm_field_eid_t instanceId        = 0;
            auto const gpuInstanceRet
                = proxy.GetGpuInstanceEntityId(3, DcgmNs::Mig::Nvml::GpuInstanceId { 4 }, &gpuInstanceEntity);
            auto const computeRet = proxy.GetComputeInstanceEntityId(
                3, DcgmNs::Mig::Nvml::GpuInstanceId { 4 }, DcgmNs::Mig::Nvml::ComputeInstanceId { 5 }, &computeEntity);
            auto const utilizationRet
                = proxy.GetMigInstancePopulation(3, DcgmNs::Mig::Nvml::GpuInstanceId { 4 }, &capacity, &used);
            auto const indicesRet = proxy.GetMigIndicesForEntity({ DCGM_FE_GPU_I, 33 }, &gpuId, &instanceId);

            THEN("the MIG response fields are copied to caller outputs")
            {
                CHECK(gpuInstanceRet == DCGM_ST_OK);
                CHECK(computeRet == DCGM_ST_OK);
                CHECK(utilizationRet == DCGM_ST_OK);
                CHECK(indicesRet == DCGM_ST_OK);
                CHECK(gpuInstanceEntity == 21U);
                CHECK(computeEntity == 21U);
                CHECK(capacity == 14U);
                CHECK(used == 5U);
                CHECK(gpuId == 21U);
                CHECK(instanceId == 77U);
                CHECK(state.lastMigIndicesRequest.entityGroupId == DCGM_FE_GPU_I);
                CHECK(state.lastMigIndicesRequest.entityId == 33U);
            }
        }

        WHEN("service-account and resource requests are processed")
        {
            std::string serviceAccount;
            unsigned int token    = 0;
            auto const serviceRet = proxy.GetServiceAccount(serviceAccount);
            auto const reserveRet = proxy.ReserveResources(token);
            auto const freeRet    = proxy.FreeResources(token);

            THEN("string and token responses are returned")
            {
                CHECK(serviceRet == DCGM_ST_OK);
                CHECK(serviceAccount == "svc-dcgm");
                CHECK(reserveRet == DCGM_ST_OK);
                CHECK(freeRet == DCGM_ST_OK);
                CHECK(token == 21U);
                CHECK(state.lastBasicEntityId == 21U);
            }
        }

        WHEN("child-process requests are routed through core")
        {
            dcgmChildProcessParams_t params {};
            ChildProcessHandle_t handle = 0;
            int pid                     = 0;
            int fd                      = -1;
            dcgmChildProcessStatus_t status {};

            auto const spawnRet   = proxy.ChildProcessSpawn(params, handle, pid);
            auto const statusRet  = proxy.ChildProcessGetStatus(handle, status);
            auto const stderrRet  = proxy.ChildProcessGetStdErrHandle(handle, fd);
            auto const stdoutRet  = proxy.ChildProcessGetStdOutHandle(handle, fd);
            auto const dataRet    = proxy.ChildProcessGetDataChannelHandle(handle, fd);
            auto const waitRet    = proxy.ChildProcessWait(handle, 3);
            auto const stopRet    = proxy.ChildProcessStop(handle, true);
            auto const destroyRet = proxy.ChildProcessDestroy(handle, 4);
            auto const resetRet   = proxy.ChildProcessManagerReset();

            THEN("handles, pid, fd, and status return codes are copied")
            {
                CHECK(spawnRet == DCGM_ST_OK);
                CHECK(statusRet == DCGM_ST_OK);
                CHECK(stderrRet == DCGM_ST_OK);
                CHECK(stdoutRet == DCGM_ST_OK);
                CHECK(dataRet == DCGM_ST_OK);
                CHECK(waitRet == DCGM_ST_OK);
                CHECK(stopRet == DCGM_ST_OK);
                CHECK(destroyRet == DCGM_ST_OK);
                CHECK(resetRet == DCGM_ST_OK);
                CHECK(handle == state.childHandle);
                CHECK(pid == state.childPid);
                CHECK(fd == state.childFd);
                CHECK(state.lastSubCommand == DcgmCoreReqIdChildProcessManagerReset);
            }
        }
    }
}


SCENARIO("DcgmCoreProxy forwards sample and status query requests", "[core][common]")
{
    GIVEN("a proxy backed by an extended core callback with sample data")
    {
        ExtendedCorePostState state;
        state.responseRet          = DCGM_ST_OK;
        state.uintAnswer           = 4;
        state.severity             = DcgmLoggingSeverityDebug;
        state.linkState            = DcgmNvLinkLinkStateDown;
        state.summaryValues        = { 11, 22 };
        state.latestSample.val.i64 = 0x90000;
        state.samples.resize(2);
        state.samples[0].timestamp = 101;
        state.samples[0].val.i64   = 700;
        state.samples[1].timestamp = 202;
        state.samples[1].val.i64   = 800;
        state.migHierarchy.version = dcgmMigHierarchy_version2;
        state.migHierarchy.count   = 1;
        auto callbacks             = MakeCallbacks(state, PostExtendedCore);
        DcgmCoreProxy proxy(callbacks);

        WHEN("scalar status helpers are requested")
        {
            auto const gpuCount  = proxy.GetGpuCount(1);
            auto const severity  = proxy.GetLoggerSeverity(BASE_LOGGER);
            auto const notifyRet = proxy.NotifyRequestOfCompletion(15, 16);
            dcgmMigHierarchy_v2 hierarchy {};
            auto const hierarchyRet = proxy.GetGpuInstanceHierarchy(hierarchy);

            THEN("the public return values mirror the core replies")
            {
                CHECK(gpuCount == 4U);
                CHECK(severity == DcgmLoggingSeverityDebug);
                CHECK(notifyRet == DCGM_ST_OK);
                CHECK(hierarchyRet == DCGM_ST_OK);
                CHECK(hierarchy.version == dcgmMigHierarchy_version2);
                CHECK(hierarchy.count == 1U);
                CHECK(state.lastSubCommand == DcgmCoreReqPopulateMigHierarchy);
            }
        }

        WHEN("NVLink states and summary values are requested")
        {
            constexpr timelib64_t startTime = 1000;
            constexpr timelib64_t endTime   = 2000;
            dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_GPU] {};
            DcgmcmSummaryType_t summaryTypes[] = { DcgmcmSummaryTypeMinimum, DcgmcmSummaryTypeMaximum };
            long long summaryValues[]          = { -1, -1 };
            auto const linkRet                 = proxy.GetEntityNvLinkLinkStatus(DCGM_FE_GPU, 3, linkStates);
            auto const summaryRet              = proxy.GetInt64SummaryData(DCGM_FE_GPU,
                                                                           3,
                                                                           DCGM_FI_DEV_GPU_TEMP_CELSIUS,
                                                                           2,
                                                                           summaryTypes,
                                                                           summaryValues,
                                                                           startTime,
                                                                           endTime,
                                                                           nullptr,
                                                                           nullptr);

            THEN("the request domains and response payloads are copied")
            {
                CHECK(linkRet == DCGM_ST_OK);
                CHECK(summaryRet == DCGM_ST_OK);
                CHECK(linkStates[0] == DcgmNvLinkLinkStateDown);
                CHECK(summaryValues[0] == 11);
                CHECK(summaryValues[1] == 22);
                CHECK(state.lastBasicRequest.entityGroupId == DCGM_FE_GPU);
                CHECK(state.lastBasicRequest.entityId == 3U);
                CHECK(state.lastSummaryRequest.summaryCount == 2U);
                CHECK(state.lastSummaryRequest.summaries[0] == DcgmcmSummaryTypeMinimum);
                CHECK(state.lastSummaryRequest.startTime == startTime);
                CHECK(state.lastSummaryRequest.endTime == endTime);
            }
        }

        WHEN("latest and historical samples are requested")
        {
            dcgmcm_sample_t latest {};
            dcgmcm_sample_t samples[2] {};
            int sampleCount = 2;
            auto const latestRet
                = proxy.GetLatestSample(DCGM_FE_GPU, 5, DCGM_FI_CUDA_GPU_COMPUTE_CAPABILITY, &latest, nullptr);
            auto const samplesRet = proxy.GetSamples(
                DCGM_FE_GPU, 5, DCGM_FI_DEV_GPU_TEMP_CELSIUS, samples, &sampleCount, 12, 34, DCGM_ORDER_ASCENDING);
            int arch           = 0;
            auto const archRet = proxy.GetCudaComputeCapabilityMajorVersion(5, arch);

            THEN("sample buffers and compute capability decoding are observable")
            {
                CHECK(latestRet == DCGM_ST_OK);
                CHECK(samplesRet == DCGM_ST_OK);
                CHECK(archRet == DCGM_ST_OK);
                CHECK(latest.val.i64 == 0x90000);
                CHECK(arch == 9);
                CHECK(sampleCount == 2);
                CHECK(samples[0].timestamp == 101);
                CHECK(samples[1].val.i64 == 800);
                CHECK(state.lastLatestRequest.populateSamples);
                CHECK_FALSE(state.lastLatestRequest.populateFvBuffer);
                CHECK(state.lastSamplesRequest.maxSamples == 2);
                CHECK(state.lastSamplesRequest.order == DCGM_ORDER_ASCENDING);
            }
        }

        WHEN("multiple latest live samples are requested")
        {
            std::vector<dcgmGroupEntityPair_t> entities { { DCGM_FE_GPU, 2 }, { DCGM_FE_GPU_I, 9 } };
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
            DcgmFvBuffer buffer;
            auto const ret = proxy.GetMultipleLatestLiveSamples(entities, fieldIds, &buffer);

            THEN("entity and field domains are routed through core")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(state.lastSubCommand == DcgmCoreReqIdCMGetMultipleLatestLiveSamples);
                CHECK(state.lastMultipleLatestRequest.entityPairCount == entities.size());
                CHECK(state.lastMultipleLatestRequest.entityPairs[0].entityGroupId == DCGM_FE_GPU);
                CHECK(state.lastMultipleLatestRequest.entityPairs[1].entityId == 9U);
                CHECK(state.lastMultipleLatestRequest.numFieldIds == fieldIds.size());
                CHECK(state.lastMultipleLatestRequest.fieldIds[0] == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
            }
        }

        WHEN("multiple latest live samples are requested for too many entities")
        {
            std::vector<dcgmGroupEntityPair_t> entities(DCGM_GROUP_MAX_ENTITIES_V2 + 1, { DCGM_FE_GPU, 0 });
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
            DcgmFvBuffer buffer;
            auto const ret = proxy.GetMultipleLatestLiveSamples(entities, fieldIds, &buffer);

            THEN("the proxy rejects the request before posting to core")
            {
                CHECK(ret == DCGM_ST_MAX_LIMIT);
                CHECK(state.calls == 0U);
            }
        }
    }

    GIVEN("a proxy whose core callback fails before populating outputs")
    {
        ExtendedCorePostState state;
        state.postReturn = DCGM_ST_CONNECTION_NOT_VALID;
        auto callbacks   = MakeCallbacks(state, PostExtendedCore);
        DcgmCoreProxy proxy(callbacks);

        WHEN("query helpers are invoked")
        {
            auto const gpuCount                = proxy.GetGpuCount(0);
            auto const severity                = proxy.GetLoggerSeverity(BASE_LOGGER);
            constexpr int unchangedCudaVersion = 99;
            int cudaVersion                    = unchangedCudaVersion;
            auto const cudaRet                 = proxy.GetCudaVersion(cudaVersion);

            THEN("failure defaults are returned without trusting response payloads")
            {
                CHECK(gpuCount == 0U);
                CHECK(severity == DcgmLoggingSeverityUnspecified);
                CHECK(cudaRet == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(cudaVersion == unchangedCudaVersion);
            }
        }
    }
}
