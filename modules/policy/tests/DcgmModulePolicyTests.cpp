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

#include <DcgmFvBuffer.h>
#include <DcgmModulePolicy.h>
#include <DcgmStringHelpers.h>
#include <ScopedDcgmFields.hpp>
#include <ScopedEnvVars.hpp>
#include <dcgm_core_communication.h>
#include <dcgm_fields.h>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

struct PolicyModuleCoreCallbackContext
{
    dcgmReturn_t verifyGroupRet { DCGM_ST_OK }; //!< Return code reported by group verification.
    unsigned int verifyGroupCalls {};           //!< Number of group verification requests observed.
};

/**
 * Isolates process-global policy persistence configuration for one test scope.
 */
class TestPolicyConfigGuard
{
public:
    /**
     * Set an isolated policy persistence path for one test scope.
     */
    TestPolicyConfigGuard()
    {
        static std::atomic<unsigned int> counter { 0 };
        auto const filename = "dcgm_policy_module_tests_" + std::to_string(getpid()) + "_"
                              + std::to_string(counter.fetch_add(1)) + ".yaml";
        m_path              = std::filesystem::temp_directory_path() / filename;
        m_env.Add("DCGM_POLICY_ALLOWED_DIRS", std::filesystem::temp_directory_path().string());
        m_env.Add("DCGM_POLICY_CONFIG_PATH", m_path.string());
        RemoveConfigFiles();
    }

    /**
     * Restore policy persistence environment and remove temp files.
     */
    ~TestPolicyConfigGuard()
    {
        RemoveConfigFiles();
    }

    /**
     * Disable copying because this guard owns process-global environment restoration.
     */
    TestPolicyConfigGuard(TestPolicyConfigGuard const &) = delete;

    /**
     * Disable copy assignment because this guard owns process-global environment restoration.
     *
     * @return This operator is deleted.
     */
    TestPolicyConfigGuard &operator=(TestPolicyConfigGuard const &) = delete;

private:
    /**
     * Remove this guard's policy persistence files.
     */
    void RemoveConfigFiles() noexcept
    {
        std::error_code ec;
        std::filesystem::remove(m_path, ec);
        std::filesystem::remove(m_path.string() + ".tmp", ec);
    }

    ScopedDcgmFields m_fields;    //!< Initializes field metadata required by field-policy validation.
    ScopedEnvVars m_env;          //!< Restores policy persistence environment after the test scope.
    std::filesystem::path m_path; //!< Per-test policy persistence path.
};


static constexpr dcgm_connection_id_t kConnId         = 1;
static constexpr dcgm_request_id_t kReqId             = 1;
static constexpr unsigned int kGpuId                  = 0;
static constexpr dcgmGpuGrp_t kGroupId                = static_cast<dcgmGpuGrp_t>(1);
static constexpr unsigned int kMaxThermal             = 80;
static constexpr dcgmPolicyCondition_t kAllConditions = static_cast<dcgmPolicyCondition_t>(
    DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_PCI | DCGM_POLICY_COND_MAX_PAGES_RETIRED | DCGM_POLICY_COND_THERMAL
    | DCGM_POLICY_COND_POWER | DCGM_POLICY_COND_NVLINK | DCGM_POLICY_COND_XID);

struct PolicyModuleTestContext
{
    dcgmReturn_t verifyGroupRet { DCGM_ST_OK };               //!< Return code reported by group verification.
    unsigned int verifyGroupCalls {};                         //!< Number of group verification requests observed.
    unsigned int notifyCount {};                              //!< Legacy policy violation notifications observed.
    unsigned int fieldViolationNotifyCount {};                //!< Field-policy violation notifications observed.
    unsigned short lastViolationFieldId {};                   //!< Field ID from the last field-policy notification.
    std::atomic<unsigned int> activeTasks { 0 };              //!< In-flight serialized policy tasks.
    unsigned int maxActiveTasks {};                           //!< Peak concurrent serialized policy tasks.
    std::mutex maxActiveTasksMutex;                           //!< Protects maxActiveTasks updates.
    std::atomic<unsigned int> getGroupEntitiesDelayMs { 20 }; //!< Delay used by unblocked group-entity requests.
    std::atomic<bool> blockGetGroupEntities { false };        //!< Hold GetGroupEntities until released.
    std::atomic<bool> getGroupEntitiesEntered { false };      //!< Observes when the held call starts.
    std::atomic<bool> releaseGetGroupEntities { false };      //!< Releases a held GetGroupEntities call.
};

/**
 * Populate a legacy policy-module command header for a test message.
 *
 * @param[in,out] header Header to populate.
 * @param[in] subCommand Policy subcommand identifier.
 * @param[in] version Expected message version.
 * @param[in] length Full message size.
 */
static void PopulateLegacyPolicyHeader(dcgm_module_command_header_t &header,
                                       unsigned int subCommand,
                                       unsigned int version,
                                       unsigned int length)
{
    header.length     = length;
    header.moduleId   = DcgmModuleIdPolicy;
    header.subCommand = subCommand;
    header.version    = version;
}

/**
 * Build a core attach-GPUs message for module tests.
 *
 * @return Initialized attach-GPUs message.
 */
static dcgm_core_msg_attach_gpus_t MakeAttachGpusMessage()
{
    dcgm_core_msg_attach_gpus_t msg {};
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_ATTACH_GPUS;
    msg.header.version    = dcgm_core_msg_attach_gpus_version;
    return msg;
}

/**
 * Build a core detach-GPUs message for module tests.
 *
 * @return Initialized detach-GPUs message.
 */
static dcgm_core_msg_detach_gpus_t MakeDetachGpusMessage()
{
    dcgm_core_msg_detach_gpus_t msg {};
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_DETACH_GPUS;
    msg.header.version    = dcgm_core_msg_detach_gpus_version;
    return msg;
}

/**
 * Build a core field-values-updated message for module tests.
 *
 * @param[in] buffer Field-value buffer pointer.
 * @param[in] bufferSize Field-value buffer size in bytes.
 * @return Initialized field-values-updated message.
 */
static dcgm_core_msg_field_values_updated_t MakeFieldValuesUpdatedMessage(char *buffer, size_t bufferSize)
{
    dcgm_core_msg_field_values_updated_t msg {};
    msg.header.length          = sizeof(msg);
    msg.header.moduleId        = DcgmModuleIdCore;
    msg.header.subCommand      = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    msg.header.version         = dcgm_core_msg_field_values_updated_version;
    msg.fieldValues.buffer     = buffer;
    msg.fieldValues.bufferSize = bufferSize;
    return msg;
}

/**
 * Build a core client-disconnect message for module tests.
 *
 * @param[in] connectionId Connection ID to disconnect.
 * @return Initialized client-disconnect message.
 */
static dcgm_core_msg_client_disconnect_t MakeClientDisconnectMessage(dcgm_connection_id_t connectionId)
{
    dcgm_core_msg_client_disconnect_t msg {};
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_CLIENT_DISCONNECT;
    msg.header.version    = dcgm_core_msg_client_disconnect_version;
    msg.connectionId      = connectionId;
    return msg;
}

/**
 * Build a legacy get-policies message for module tests.
 *
 * @return Initialized get-policies message.
 */
static dcgm_policy_msg_get_policies_t MakeGetPoliciesMessage()
{
    dcgm_policy_msg_get_policies_t msg {};
    PopulateLegacyPolicyHeader(
        msg.header, DCGM_POLICY_SR_GET_POLICIES, dcgm_policy_msg_get_policies_version, sizeof(msg));
    msg.groupId = kGroupId;
    return msg;
}

/**
 * Create core callbacks for module tests that do not need to inspect core messages.
 *
 * @return Core callbacks that accept all posted messages.
 */
static dcgmCoreCallbacks_t MakePolicyModuleCallbacks()
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
             .poster     = nullptr,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

/**
 * Track active serialized policy work and optionally inject attach/detach failures.
 *
 * @param[in,out] ctx Context used to configure and observe core callback behavior.
 * @return Core callbacks that accept core requests used by policy manager initialization.
 */
static dcgmCoreCallbacks_t MakePolicyModuleCallbacks(PolicyModuleTestContext &ctx)
{
    return {
        .version  = dcgmCoreCallbacks_version,
        .postfunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
            auto *ctx = static_cast<PolicyModuleTestContext *>(poster);
            switch (header->subCommand)
            {
                case DcgmCoreReqIdCMGetGpuCount:
                {
                    auto *msg                = reinterpret_cast<dcgmCoreGetGpuCount_t *>(header);
                    msg->response.ret        = DCGM_ST_OK;
                    msg->response.uintAnswer = 1;
                    return DCGM_ST_OK;
                }
                case DcgmCoreReqIdCMGetGpuIds:
                {
                    auto *msg               = reinterpret_cast<dcgmCoreGetGpuList_t *>(header);
                    msg->response.ret       = DCGM_ST_OK;
                    msg->response.gpuCount  = 1;
                    msg->response.gpuIds[0] = kGpuId;
                    return DCGM_ST_OK;
                }
                case DcgmCoreReqIdGMVerifyAndUpdateGroupId:
                {
                    ++ctx->verifyGroupCalls;
                    auto *msg                = reinterpret_cast<dcgmCoreBasicQuery_t *>(header);
                    msg->response.ret        = ctx->verifyGroupRet;
                    msg->response.uintAnswer = msg->request.entityId;
                    return DCGM_ST_OK;
                }
                case DcgmCoreReqIdGMGetGroupEntities:
                {
                    unsigned int const active = ++ctx->activeTasks;
                    {
                        std::lock_guard<std::mutex> guard(ctx->maxActiveTasksMutex);
                        ctx->maxActiveTasks = std::max(ctx->maxActiveTasks, active);
                    }
                    if (ctx->blockGetGroupEntities)
                    {
                        ctx->getGroupEntitiesEntered = true;
                        while (!ctx->releaseGetGroupEntities)
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                    else
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(ctx->getGroupEntitiesDelayMs.load()));
                    }

                    auto *msg                                  = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(header);
                    msg->response.ret                          = DCGM_ST_OK;
                    msg->response.entityPairsCount             = 1;
                    msg->response.entityPairs[0].entityGroupId = DCGM_FE_GPU;
                    msg->response.entityPairs[0].entityId      = kGpuId;

                    --ctx->activeTasks;
                    return DCGM_ST_OK;
                }
                case DcgmCoreReqIdCMAddFieldWatch:
                case DcgmCoreReqIdCMUpdateAllFields:
                    return DCGM_ST_OK;
                case DcgmCoreReqIdSendRawMessage:
                {
                    auto *msg     = reinterpret_cast<dcgmCoreSendRawMessage_t *>(header);
                    msg->response = DCGM_ST_OK;
                    if (msg->request.msgData != nullptr && msg->request.msgSize == sizeof(dcgm_msg_policy_notify_t))
                    {
                        auto *notify = static_cast<dcgm_msg_policy_notify_t *>(msg->request.msgData);
                        if (notify->response.condition == DCGM_POLICY_COND_THERMAL)
                        {
                            ++ctx->notifyCount;
                        }
                    }
                    if (msg->request.msgData != nullptr && msg->request.msgType == DCGM_MSG_POLICY_VIOLATION
                        && msg->request.msgSize == sizeof(dcgm_msg_policy_violation_t))
                    {
                        auto *notify = static_cast<dcgm_msg_policy_violation_t *>(msg->request.msgData);
                        ++ctx->fieldViolationNotifyCount;
                        ctx->lastViolationFieldId = notify->violation.fieldId;
                    }
                    return DCGM_ST_OK;
                }
                default:
                    return DCGM_ST_OK;
            }
        },
        .poster     = &ctx,
        .loggerfunc = [](void const *) { /* do nothing */ }
    };
}

namespace
{
dcgmCoreCallbacks_t MakeCallbacks()
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
             .poster     = nullptr,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

void PopulatePolicyHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdPolicy;
    header.subCommand = subCommand;
    header.version    = version;
}

void PopulateCoreHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdCore;
    header.subCommand = subCommand;
    header.version    = version;
}
} //namespace

/**
 * Register and configure a legacy thermal policy through the module IPC path.
 *
 * @param[in,out] policyModule Policy module under test.
 */
static void RegisterLegacyThermalPolicy(DcgmModulePolicy &policyModule)
{
    dcgm_policy_msg_register_t registerMsg {};
    PopulateLegacyPolicyHeader(
        registerMsg.header, DCGM_POLICY_SR_REGISTER, dcgm_policy_msg_register_version, sizeof(registerMsg));
    registerMsg.header.connectionId = kConnId;
    registerMsg.header.requestId    = kReqId;
    registerMsg.groupId             = kGroupId;
    registerMsg.condition           = kAllConditions;
    REQUIRE(policyModule.ProcessMessage(&registerMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_set_policy_t setMsg {};
    PopulateLegacyPolicyHeader(
        setMsg.header, DCGM_POLICY_SR_SET_POLICY, dcgm_policy_msg_set_policy_version, sizeof(setMsg));
    setMsg.header.connectionId                                            = kConnId;
    setMsg.groupId                                                        = kGroupId;
    setMsg.policy.version                                                 = dcgmPolicy_version;
    setMsg.policy.condition                                               = kAllConditions;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_THERMAL].tag                 = dcgmPolicyConditionParams_st::LLONG;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval           = kMaxThermal;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_POWER].tag                   = dcgmPolicyConditionParams_st::LLONG;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_POWER].val.llval             = 200;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].tag       = dcgmPolicyConditionParams_st::LLONG;
    setMsg.policy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].val.llval = 10;
    REQUIRE(policyModule.ProcessMessage(&setMsg.header) == DCGM_ST_OK);
}

struct PolicyWatchTestContext
{
    std::vector<std::pair<unsigned int, unsigned short>> addedWatches;
    std::vector<std::pair<unsigned int, unsigned short>> removedWatches;
};

/**
 * Create core callbacks that record Cache Manager watch add/remove requests.
 *
 * @param[in,out] ctx Context that receives recorded watch operations.
 * @return Core callbacks that accept all posted messages.
 */
static dcgmCoreCallbacks_t MakePolicyModuleCallbacks(PolicyWatchTestContext &ctx)
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
                 auto *ctx = static_cast<PolicyWatchTestContext *>(poster);
                 switch (header->subCommand)
                 {
                     case DcgmCoreReqIdCMAddFieldWatch:
                     {
                         auto *msg = reinterpret_cast<dcgmCoreAddFieldWatch_v2 *>(header);
                         ctx->addedWatches.emplace_back(msg->request.entityId, msg->request.fieldId);
                         msg->ret = DCGM_ST_OK;
                         return DCGM_ST_OK;
                     }
                     case DcgmCoreReqIdCMRemoveFieldWatch:
                     {
                         auto *msg = reinterpret_cast<dcgmCoreRemoveFieldWatch_t *>(header);
                         ctx->removedWatches.emplace_back(msg->request.entityId, msg->request.fieldId);
                         msg->ret = DCGM_ST_OK;
                         return DCGM_ST_OK;
                     }
                     case DcgmCoreReqIdCMGetGpuIds:
                     {
                         auto *msg               = reinterpret_cast<dcgmCoreGetGpuList_t *>(header);
                         msg->response.ret       = DCGM_ST_OK;
                         msg->response.gpuCount  = 2;
                         msg->response.gpuIds[0] = 0;
                         msg->response.gpuIds[1] = 1;
                         return DCGM_ST_OK;
                     }
                     default:
                         return DCGM_ST_OK;
                 }
             },
             .poster     = &ctx,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

/**
 * Return a sorted copy of recorded watch calls for order-insensitive assertions.
 *
 * @param[in] watches Recorded watch calls.
 * @return Sorted watch calls.
 */
static std::vector<std::pair<unsigned int, unsigned short>> SortedWatches(
    std::vector<std::pair<unsigned int, unsigned short>> watches)
{
    std::ranges::sort(watches);
    return watches;
}

/**
 * Populate a field-policy-module command header for a test message.
 *
 * @param[in,out] header Header to populate.
 * @param[in] subCommand Policy subcommand identifier.
 * @param[in] version Expected message version.
 */
static void PopulateFieldPolicyHeader(dcgm_module_command_header_t &header,
                                      unsigned int subCommand,
                                      unsigned int version)
{
    header.length     = 0;
    header.moduleId   = DcgmModuleIdPolicy;
    header.subCommand = subCommand;
    header.version    = version;
}

/**
 * Build a valid field-policy configuration for module tests.
 *
 * @param[in] fieldId DCGM field ID for the policy.
 * @param[in] threshold Threshold value for the policy.
 * @return Valid policy info request payload.
 */
static dcgmPolicySpec_t MakePolicyInfo(unsigned short fieldId, double threshold = 80.0)
{
    dcgmPolicySpec_t info {};
    info.version        = dcgmPolicySpec_version;
    info.fieldId        = fieldId;
    info.threshold      = threshold;
    info.policyOperator = DCGM_POLICY_OP_GT;
    info.entities[0]    = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    info.entityCount    = 1;
    info.channelMask    = DCGM_POLICY_CHANNEL_CONSOLE;
    info.state          = DCGM_POLICY_STATE_ENABLED;
    info.rateLimitSec   = 1;
    SafeCopyTo(info.name, fmt::format("policy_{}", static_cast<unsigned int>(fieldId)).c_str());
    return info;
}

TEST_CASE("DcgmModulePolicy::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);
    CHECK(policyModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_get_all_t msg {};
    PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_GET_ALL, dcgm_policy_msg_get_all_version);
    msg.header.length   = sizeof(msg);
    msg.capacity        = DCGM_POLICY_GET_ALL_PAGE;
    msg.header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::ProcessMessage handles policy routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    SECTION("unknown policy subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulatePolicyHeader(header, DCGM_POLICY_SR_COUNT + 10, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("policy subcommands reject version mismatches")
    {
        GIVEN("a policy message with an invalid version")
        {
            SECTION("get policies")
            {
                dcgm_policy_msg_get_policies_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_GET_POLICIES, dcgm_policy_msg_get_policies_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("set policy")
            {
                dcgm_policy_msg_set_policy_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_SET_POLICY, dcgm_policy_msg_set_policy_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("register")
            {
                dcgm_policy_msg_register_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_REGISTER, dcgm_policy_msg_register_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("unregister")
            {
                dcgm_policy_msg_unregister_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_UNREGISTER, dcgm_policy_msg_unregister_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModulePolicy::ProcessMessage handles core routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume does not require manager work")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_OK);
    }

    SECTION("core subcommands reject version mismatches")
    {
        GIVEN("a core message with an invalid version")
        {
            SECTION("client disconnect")
            {
                dcgm_core_msg_client_disconnect_t msg {};
                PopulateCoreHeader(
                    msg.header, DCGM_CORE_SR_CLIENT_DISCONNECT, dcgm_core_msg_client_disconnect_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("attach GPUs")
            {
                dcgm_core_msg_attach_gpus_t msg {};
                PopulateCoreHeader(msg.header, DCGM_CORE_SR_ATTACH_GPUS, dcgm_core_msg_attach_gpus_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("detach GPUs")
            {
                dcgm_core_msg_detach_gpus_t msg {};
                PopulateCoreHeader(msg.header, DCGM_CORE_SR_DETACH_GPUS, dcgm_core_msg_detach_gpus_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("field values updated")
            {
                dcgm_core_msg_field_values_updated_t msg {};
                PopulateCoreHeader(
                    msg.header, DCGM_CORE_SR_FIELD_VALUES_UPDATED, dcgm_core_msg_field_values_updated_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModulePolicy propagates attach GPU failures")
{
    PolicyModuleTestContext ctx;
    ctx.verifyGroupRet      = DCGM_ST_BADPARAM;
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    auto msg = MakeAttachGpusMessage();
    CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_BADPARAM);
    CHECK(ctx.verifyGroupCalls == 1);
}

TEST_CASE("DcgmModulePolicy processes detach GPU requests")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    auto attachMsg = MakeAttachGpusMessage();
    REQUIRE(policyModule.ProcessMessage(&attachMsg.header) == DCGM_ST_OK);

    auto detachMsg = MakeDetachGpusMessage();
    CHECK(policyModule.ProcessMessage(&detachMsg.header) == DCGM_ST_OK);
}

TEST_CASE("DcgmModulePolicy::routes legacy policy IPC through the sync queue")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    RegisterLegacyThermalPolicy(policyModule);

    auto getMsg = MakeGetPoliciesMessage();
    REQUIRE(policyModule.ProcessMessage(&getMsg.header) == DCGM_ST_OK);
    REQUIRE(getMsg.numPolicies == 1);
    REQUIRE(getMsg.policies[0].parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval == kMaxThermal);
}

TEST_CASE("DcgmModulePolicy::cancels timed-out queued synchronous requests")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc, std::chrono::milliseconds(20));

    ctx.getGroupEntitiesDelayMs = 0;
    RegisterLegacyThermalPolicy(policyModule);

    ctx.blockGetGroupEntities = true;
    auto blockingGetMsg       = MakeGetPoliciesMessage();
    dcgmReturn_t blockingGetRet { DCGM_ST_GENERIC_ERROR };
    std::jthread blockingGetThread([&policyModule, &blockingGetMsg, &blockingGetRet] {
        blockingGetRet = policyModule.ProcessMessage(&blockingGetMsg.header);
    });

    for (unsigned int attempt = 0; attempt < 100 && !ctx.getGroupEntitiesEntered; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!ctx.getGroupEntitiesEntered)
    {
        ctx.releaseGetGroupEntities = true;
        blockingGetThread.join();
    }
    REQUIRE(ctx.getGroupEntitiesEntered);

    auto setMsg = std::make_unique<dcgm_policy_msg_set_policy_t>();
    PopulateLegacyPolicyHeader(
        setMsg->header, DCGM_POLICY_SR_SET_POLICY, dcgm_policy_msg_set_policy_version, sizeof(*setMsg));
    setMsg->header.connectionId                                  = kConnId;
    setMsg->groupId                                              = kGroupId;
    setMsg->policy.version                                       = dcgmPolicy_version;
    setMsg->policy.condition                                     = kAllConditions;
    setMsg->policy.parms[DCGM_POLICY_COND_IDX_THERMAL].tag       = dcgmPolicyConditionParams_st::LLONG;
    setMsg->policy.parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval = kMaxThermal + 10;

    REQUIRE(policyModule.ProcessMessage(&setMsg->header) == DCGM_ST_TIMEOUT);
    setMsg.reset();

    ctx.getGroupEntitiesDelayMs = 0;
    ctx.blockGetGroupEntities   = false;
    ctx.releaseGetGroupEntities = true;
    blockingGetThread.join();
    REQUIRE(blockingGetRet == DCGM_ST_TIMEOUT);

    auto getMsg = MakeGetPoliciesMessage();
    REQUIRE(policyModule.ProcessMessage(&getMsg.header) == DCGM_ST_OK);
    REQUIRE(getMsg.policies[0].parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval == kMaxThermal);
}

TEST_CASE("DcgmModulePolicy::rejects invalid field value update buffers")
{
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    auto updateMsg = MakeFieldValuesUpdatedMessage(nullptr, 0);
    CHECK(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_BADPARAM);

    char payload[] = "unused";
    updateMsg      = MakeFieldValuesUpdatedMessage(payload, 0);
    CHECK(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::copies field value buffers before async processing")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    RegisterLegacyThermalPolicy(policyModule);

    auto attachMsg = MakeAttachGpusMessage();
    REQUIRE(policyModule.ProcessMessage(&attachMsg.header) == DCGM_ST_OK);

    DcgmFvBuffer fvBuffer;
    REQUIRE(
        fvBuffer.AddInt64Value(DCGM_FE_GPU, kGpuId, DCGM_FI_DEV_GPU_TEMP_CELSIUS, kMaxThermal + 1, 6000000, DCGM_ST_OK)
        != nullptr);

    std::vector<char> ownedBuffer;
    size_t bufferSize = 0;
    REQUIRE(fvBuffer.GetSize(&bufferSize, nullptr) == DCGM_ST_OK);
    ownedBuffer.assign(fvBuffer.GetBuffer(), fvBuffer.GetBuffer() + bufferSize);

    ctx.blockGetGroupEntities = true;
    auto blockingGetMsg       = MakeGetPoliciesMessage();
    dcgmReturn_t blockingGetRet { DCGM_ST_GENERIC_ERROR };
    std::jthread blockingGetThread([&policyModule, &blockingGetMsg, &blockingGetRet] {
        blockingGetRet = policyModule.ProcessMessage(&blockingGetMsg.header);
    });

    for (unsigned int attempt = 0; attempt < 100 && !ctx.getGroupEntitiesEntered; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!ctx.getGroupEntitiesEntered)
    {
        ctx.releaseGetGroupEntities = true;
        blockingGetThread.join();
    }
    REQUIRE(ctx.getGroupEntitiesEntered);

    auto updateMsg               = MakeFieldValuesUpdatedMessage(ownedBuffer.data(), ownedBuffer.size());
    dcgmReturn_t const updateRet = policyModule.ProcessMessage(&updateMsg.header);

    std::memset(ownedBuffer.data(), 0, ownedBuffer.size());

    ctx.blockGetGroupEntities   = false;
    ctx.releaseGetGroupEntities = true;
    blockingGetThread.join();
    REQUIRE(blockingGetRet == DCGM_ST_OK);
    REQUIRE(updateRet == DCGM_ST_OK);

    auto getMsg = MakeGetPoliciesMessage();
    REQUIRE(policyModule.ProcessMessage(&getMsg.header) == DCGM_ST_OK);
    REQUIRE(ctx.notifyCount == 1);
}

TEST_CASE("DcgmModulePolicy::enqueues client disconnect asynchronously")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    RegisterLegacyThermalPolicy(policyModule);

    auto disconnectMsg = MakeClientDisconnectMessage(kConnId);
    CHECK(policyModule.ProcessMessage(&disconnectMsg.header) == DCGM_ST_OK);
}

TEST_CASE("DcgmModulePolicy::serializes concurrent legacy policy requests")
{
    PolicyModuleTestContext ctx {};
    dcgmCoreCallbacks_t dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    RegisterLegacyThermalPolicy(policyModule);

    std::vector<dcgm_policy_msg_get_policies_t> messages(4);
    std::vector<dcgmReturn_t> statuses(messages.size(), DCGM_ST_GENERIC_ERROR);

    {
        std::vector<std::jthread> threads;
        for (unsigned int i = 0; i < messages.size(); ++i)
        {
            messages[i] = MakeGetPoliciesMessage();
            threads.emplace_back([&policyModule, &messages, &statuses, i] {
                statuses[i] = policyModule.ProcessMessage(&messages[i].header);
            });
        }
    }

    for (auto const status : statuses)
    {
        REQUIRE(status == DCGM_ST_OK);
    }
    REQUIRE(ctx.maxActiveTasks == 1);
}

TEST_CASE("DcgmModulePolicy::field-based policy commands are serialized end-to-end")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulateFieldPolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length = sizeof(createMsg);
    createMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);

    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);
    REQUIRE(createMsg.created.policyId != 0);

    dcgm_policy_msg_get_one_t getOneMsg {};
    PopulateFieldPolicyHeader(getOneMsg.header, DCGM_POLICY_SR_GET_ONE, dcgm_policy_msg_get_one_version);
    getOneMsg.header.length  = sizeof(getOneMsg);
    getOneMsg.fieldId        = createMsg.created.fieldId;
    getOneMsg.policyId       = createMsg.created.policyId;
    getOneMsg.policy.version = dcgmPolicyInfo_version;
    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_OK);
    REQUIRE(getOneMsg.policy.fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    REQUIRE(getOneMsg.policy.threshold == 70.0);

    dcgm_policy_msg_modify_t modifyMsg {};
    PopulateFieldPolicyHeader(modifyMsg.header, DCGM_POLICY_SR_MODIFY, dcgm_policy_msg_modify_version);
    modifyMsg.header.length    = sizeof(modifyMsg);
    modifyMsg.policyId         = getOneMsg.policy.policyId;
    modifyMsg.policy           = MakePolicyInfo(getOneMsg.policy.fieldId, getOneMsg.policy.threshold);
    modifyMsg.policy.threshold = 75.0;
    REQUIRE(policyModule.ProcessMessage(&modifyMsg.header) == DCGM_ST_OK);

    getOneMsg.fieldId        = modifyMsg.modified.fieldId;
    getOneMsg.policyId       = modifyMsg.modified.policyId;
    getOneMsg.policy.version = dcgmPolicyInfo_version;
    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_OK);
    REQUIRE(getOneMsg.policy.threshold == 75.0);

    dcgm_policy_msg_set_state_t disableMsg {};
    PopulateFieldPolicyHeader(disableMsg.header, DCGM_POLICY_SR_DISABLE, dcgm_policy_msg_set_state_version);
    disableMsg.header.length = sizeof(disableMsg);
    disableMsg.fieldId       = modifyMsg.modified.fieldId;
    disableMsg.policyId      = modifyMsg.modified.policyId;
    REQUIRE(policyModule.ProcessMessage(&disableMsg.header) == DCGM_ST_OK);
    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_OK);
    REQUIRE(getOneMsg.policy.state == DCGM_POLICY_STATE_DISABLED);

    dcgm_policy_msg_set_state_t enableMsg {};
    PopulateFieldPolicyHeader(enableMsg.header, DCGM_POLICY_SR_ENABLE, dcgm_policy_msg_set_state_version);
    enableMsg.header.length = sizeof(enableMsg);
    enableMsg.fieldId       = modifyMsg.modified.fieldId;
    enableMsg.policyId      = modifyMsg.modified.policyId;
    REQUIRE(policyModule.ProcessMessage(&enableMsg.header) == DCGM_ST_OK);
    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_OK);
    REQUIRE(getOneMsg.policy.state == DCGM_POLICY_STATE_ENABLED);

    dcgm_policy_msg_delete_t deleteMsg {};
    PopulateFieldPolicyHeader(deleteMsg.header, DCGM_POLICY_SR_DELETE, dcgm_policy_msg_delete_version);
    deleteMsg.header.length = sizeof(deleteMsg);
    deleteMsg.fieldId       = modifyMsg.modified.fieldId;
    deleteMsg.policyId      = modifyMsg.modified.policyId;
    REQUIRE(policyModule.ProcessMessage(&deleteMsg.header) == DCGM_ST_OK);

    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_NO_DATA);
}

TEST_CASE("DcgmModulePolicy::rejects invalid field-policy specifications")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    auto createPolicy = [&policyModule](dcgmPolicySpec_t const &policy) {
        dcgm_policy_msg_create_t createMsg {};
        PopulateFieldPolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
        createMsg.header.length = sizeof(createMsg);
        createMsg.policy        = policy;
        return policyModule.ProcessMessage(&createMsg.header);
    };

    SECTION("empty name")
    {
        auto policy    = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        policy.name[0] = '\0';
        REQUIRE(createPolicy(policy) == DCGM_ST_BADPARAM);
    }

    SECTION("unterminated name")
    {
        auto policy = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        std::memset(policy.name, 'a', sizeof(policy.name));
        REQUIRE(createPolicy(policy) == DCGM_ST_BADPARAM);
    }

    SECTION("non-numeric field")
    {
        auto policy = MakePolicyInfo(DCGM_FI_DEV_GPU_UUID);
        REQUIRE(createPolicy(policy) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("DcgmModulePolicy::serializes concurrent field-based creates")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    std::vector<dcgm_policy_msg_create_t> messages(3);
    std::vector<dcgmReturn_t> statuses(messages.size(), DCGM_ST_GENERIC_ERROR);

    // Exercise concurrent create requests to validate module-side serialization behavior.
    {
        std::vector<std::jthread> threads;
        for (unsigned int i = 0; i < messages.size(); ++i)
        {
            PopulateFieldPolicyHeader(messages[i].header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
            messages[i].header.length = sizeof(messages[i]);
            messages[i].policy
                = MakePolicyInfo(static_cast<unsigned short>(DCGM_FI_DEV_GPU_TEMP_CELSIUS + i), 70.0 + i);
            threads.emplace_back([&policyModule, &messages, &statuses, i] {
                statuses[i] = policyModule.ProcessMessage(&messages[i].header);
            });
        }
    }
    for (auto const status : statuses)
    {
        REQUIRE(status == DCGM_ST_OK);
    }

    dcgm_policy_msg_get_all_t getAllMsg {};
    PopulateFieldPolicyHeader(getAllMsg.header, DCGM_POLICY_SR_GET_ALL, dcgm_policy_msg_get_all_version);
    getAllMsg.header.length = sizeof(getAllMsg);
    getAllMsg.capacity      = DCGM_POLICY_GET_ALL_PAGE;
    REQUIRE(policyModule.ProcessMessage(&getAllMsg.header) == DCGM_ST_OK);
    REQUIRE(getAllMsg.returnedCount == 3);
    REQUIRE(getAllMsg.totalCount == 3);

    getAllMsg = {};
    PopulateFieldPolicyHeader(getAllMsg.header, DCGM_POLICY_SR_GET_ALL, dcgm_policy_msg_get_all_version);
    getAllMsg.header.length = sizeof(getAllMsg);
    getAllMsg.capacity      = 2;
    REQUIRE(policyModule.ProcessMessage(&getAllMsg.header) == DCGM_ST_INSUFFICIENT_SIZE);
    REQUIRE(getAllMsg.returnedCount == 2);
    REQUIRE(getAllMsg.totalCount == 3);
}

TEST_CASE("DcgmModulePolicy::owns new policy field watches")
{
    TestPolicyConfigGuard policyConfigGuard;
    PolicyWatchTestContext ctx;
    auto dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulateFieldPolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length      = sizeof(createMsg);
    createMsg.policy             = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    createMsg.policy.entities[1] = { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
    createMsg.policy.entityCount = 2;
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);
    REQUIRE(SortedWatches(ctx.addedWatches)
            == SortedWatches(std::vector<std::pair<unsigned int, unsigned short>> {
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
            }));
    REQUIRE(ctx.removedWatches.empty());

    dcgm_policy_msg_set_state_t disableMsg {};
    PopulateFieldPolicyHeader(disableMsg.header, DCGM_POLICY_SR_DISABLE, dcgm_policy_msg_set_state_version);
    disableMsg.header.length = sizeof(disableMsg);
    disableMsg.fieldId       = createMsg.created.fieldId;
    disableMsg.policyId      = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&disableMsg.header) == DCGM_ST_OK);
    REQUIRE(SortedWatches(ctx.removedWatches)
            == SortedWatches(std::vector<std::pair<unsigned int, unsigned short>> {
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
            }));

    dcgm_policy_msg_set_state_t enableMsg {};
    PopulateFieldPolicyHeader(enableMsg.header, DCGM_POLICY_SR_ENABLE, dcgm_policy_msg_set_state_version);
    enableMsg.header.length = sizeof(enableMsg);
    enableMsg.fieldId       = createMsg.created.fieldId;
    enableMsg.policyId      = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&enableMsg.header) == DCGM_ST_OK);
    REQUIRE(ctx.addedWatches.size() == 4);
    REQUIRE(SortedWatches(ctx.addedWatches)
            == SortedWatches(std::vector<std::pair<unsigned int, unsigned short>> {
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
            }));

    dcgm_policy_msg_delete_t deleteMsg {};
    PopulateFieldPolicyHeader(deleteMsg.header, DCGM_POLICY_SR_DELETE, dcgm_policy_msg_delete_version);
    deleteMsg.header.length = sizeof(deleteMsg);
    deleteMsg.fieldId       = createMsg.created.fieldId;
    deleteMsg.policyId      = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&deleteMsg.header) == DCGM_ST_OK);
    REQUIRE(ctx.removedWatches.size() == 4);
    REQUIRE(SortedWatches(ctx.removedWatches)
            == SortedWatches(std::vector<std::pair<unsigned int, unsigned short>> {
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
                { 1, DCGM_FI_DEV_GPU_TEMP_CELSIUS },
            }));
}

TEST_CASE("DcgmModulePolicy::evaluates and records field update violations")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulateFieldPolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length = sizeof(createMsg);
    createMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);

    DcgmFvBuffer fvBuffer;
    REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 80, 12345, DCGM_ST_OK) != nullptr);

    dcgm_core_msg_field_values_updated_t updateMsg {};
    updateMsg.header.length          = sizeof(updateMsg);
    updateMsg.header.moduleId        = DcgmModuleIdCore;
    updateMsg.header.subCommand      = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    updateMsg.header.version         = dcgm_core_msg_field_values_updated_version;
    updateMsg.fieldValues.buffer     = fvBuffer.GetBuffer();
    updateMsg.fieldValues.bufferSize = 0;
    REQUIRE(fvBuffer.GetSize(&updateMsg.fieldValues.bufferSize, nullptr) == DCGM_ST_OK);

    REQUIRE(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_get_one_t getOneMsg {};
    PopulateFieldPolicyHeader(getOneMsg.header, DCGM_POLICY_SR_GET_ONE, dcgm_policy_msg_get_one_version);
    getOneMsg.header.length  = sizeof(getOneMsg);
    getOneMsg.fieldId        = createMsg.created.fieldId;
    getOneMsg.policyId       = createMsg.created.policyId;
    getOneMsg.policy.version = dcgmPolicyInfo_version;
    REQUIRE(policyModule.ProcessMessage(&getOneMsg.header) == DCGM_ST_OK);
    REQUIRE(getOneMsg.policy.violationCount >= 1);
    REQUIRE(getOneMsg.policy.lastTriggeredTimestamp > 0);
}

TEST_CASE("DcgmModulePolicy::returns recent field update violations via get-violations IPC")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulatePolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length = sizeof(createMsg);
    createMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);

    DcgmFvBuffer fvBuffer;
    REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 80, 12345, DCGM_ST_OK) != nullptr);

    dcgm_core_msg_field_values_updated_t updateMsg {};
    updateMsg.header.length          = sizeof(updateMsg);
    updateMsg.header.moduleId        = DcgmModuleIdCore;
    updateMsg.header.subCommand      = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    updateMsg.header.version         = dcgm_core_msg_field_values_updated_version;
    updateMsg.fieldValues.buffer     = fvBuffer.GetBuffer();
    updateMsg.fieldValues.bufferSize = 0;
    REQUIRE(fvBuffer.GetSize(&updateMsg.fieldValues.bufferSize, nullptr) == DCGM_ST_OK);

    REQUIRE(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_get_violations_t violationsMsg {};
    PopulatePolicyHeader(violationsMsg.header, DCGM_POLICY_SR_GET_VIOLATIONS, dcgm_policy_msg_get_violations_version);
    violationsMsg.header.length = sizeof(violationsMsg);
    violationsMsg.count         = DCGM_POLICY_MAX_VIOLATIONS;
    REQUIRE(policyModule.ProcessMessage(&violationsMsg.header) == DCGM_ST_OK);
    REQUIRE(violationsMsg.count == 1);
    REQUIRE(violationsMsg.violations[0].fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    REQUIRE(violationsMsg.violations[0].entity.entityGroupId == DCGM_FE_GPU);
    REQUIRE(violationsMsg.violations[0].entity.entityId == 0);

    violationsMsg.sinceTimestamp = 12345;
    violationsMsg.count          = DCGM_POLICY_MAX_VIOLATIONS;
    REQUIRE(policyModule.ProcessMessage(&violationsMsg.header) == DCGM_ST_OK);
    REQUIRE(violationsMsg.count == 1);
}

TEST_CASE("DcgmModulePolicy::register and unregister field-policy callback IPC")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_register_callback_t registerMsg {};
    PopulatePolicyHeader(
        registerMsg.header, DCGM_POLICY_SR_REGISTER_CALLBACK, dcgm_policy_msg_register_callback_version);
    registerMsg.header.length       = sizeof(registerMsg);
    registerMsg.header.connectionId = kConnId;
    registerMsg.header.requestId    = kReqId;
    registerMsg.fieldId             = 0;
    registerMsg.policyId            = 0;
    REQUIRE(policyModule.ProcessMessage(&registerMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_unregister_callback_t unregisterMsg {};
    PopulatePolicyHeader(
        unregisterMsg.header, DCGM_POLICY_SR_UNREGISTER_CALLBACK, dcgm_policy_msg_unregister_callback_version);
    unregisterMsg.header.length       = sizeof(unregisterMsg);
    unregisterMsg.header.connectionId = kConnId;
    unregisterMsg.fieldId             = 0;
    unregisterMsg.policyId            = 0;
    REQUIRE(policyModule.ProcessMessage(&unregisterMsg.header) == DCGM_ST_OK);
}

TEST_CASE("DcgmModulePolicy::unregister field-policy callback returns no data when none registered")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_unregister_callback_t unregisterMsg {};
    PopulatePolicyHeader(
        unregisterMsg.header, DCGM_POLICY_SR_UNREGISTER_CALLBACK, dcgm_policy_msg_unregister_callback_version);
    unregisterMsg.header.length       = sizeof(unregisterMsg);
    unregisterMsg.header.connectionId = kConnId;
    unregisterMsg.fieldId             = 0;
    unregisterMsg.policyId            = 0;
    REQUIRE(policyModule.ProcessMessage(&unregisterMsg.header) == DCGM_ST_NO_DATA);
}

TEST_CASE("DcgmModulePolicy::unregister catch-all callback keeps specific registrations")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulatePolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length = sizeof(createMsg);
    createMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_register_callback_t catchAllRegisterMsg {};
    PopulatePolicyHeader(
        catchAllRegisterMsg.header, DCGM_POLICY_SR_REGISTER_CALLBACK, dcgm_policy_msg_register_callback_version);
    catchAllRegisterMsg.header.length       = sizeof(catchAllRegisterMsg);
    catchAllRegisterMsg.header.connectionId = kConnId;
    catchAllRegisterMsg.header.requestId    = kReqId;
    catchAllRegisterMsg.fieldId             = 0;
    catchAllRegisterMsg.policyId            = 0;
    REQUIRE(policyModule.ProcessMessage(&catchAllRegisterMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_register_callback_t specificRegisterMsg {};
    PopulatePolicyHeader(
        specificRegisterMsg.header, DCGM_POLICY_SR_REGISTER_CALLBACK, dcgm_policy_msg_register_callback_version);
    specificRegisterMsg.header.length       = sizeof(specificRegisterMsg);
    specificRegisterMsg.header.connectionId = kConnId;
    specificRegisterMsg.header.requestId    = kReqId + 1;
    specificRegisterMsg.fieldId             = createMsg.created.fieldId;
    specificRegisterMsg.policyId            = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&specificRegisterMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_unregister_callback_t catchAllUnregisterMsg {};
    PopulatePolicyHeader(
        catchAllUnregisterMsg.header, DCGM_POLICY_SR_UNREGISTER_CALLBACK, dcgm_policy_msg_unregister_callback_version);
    catchAllUnregisterMsg.header.length       = sizeof(catchAllUnregisterMsg);
    catchAllUnregisterMsg.header.connectionId = kConnId;
    catchAllUnregisterMsg.fieldId             = 0;
    catchAllUnregisterMsg.policyId            = 0;
    REQUIRE(policyModule.ProcessMessage(&catchAllUnregisterMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_unregister_callback_t specificUnregisterMsg {};
    PopulatePolicyHeader(
        specificUnregisterMsg.header, DCGM_POLICY_SR_UNREGISTER_CALLBACK, dcgm_policy_msg_unregister_callback_version);
    specificUnregisterMsg.header.length       = sizeof(specificUnregisterMsg);
    specificUnregisterMsg.header.connectionId = kConnId;
    specificUnregisterMsg.fieldId             = createMsg.created.fieldId;
    specificUnregisterMsg.policyId            = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&specificUnregisterMsg.header) == DCGM_ST_OK);
}

TEST_CASE("DcgmModulePolicy::register field-policy callback rejects missing requestId")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_register_callback_t registerMsg {};
    PopulatePolicyHeader(
        registerMsg.header, DCGM_POLICY_SR_REGISTER_CALLBACK, dcgm_policy_msg_register_callback_version);
    registerMsg.header.length       = sizeof(registerMsg);
    registerMsg.header.connectionId = kConnId;
    registerMsg.header.requestId    = DCGM_REQUEST_ID_NONE;
    REQUIRE(policyModule.ProcessMessage(&registerMsg.header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::routes field-policy violations to registered callback subscribers")
{
    TestPolicyConfigGuard policyConfigGuard;
    PolicyModuleTestContext ctx;
    auto dcc = MakePolicyModuleCallbacks(ctx);
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulatePolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length      = sizeof(createMsg);
    createMsg.policy             = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    createMsg.policy.channelMask = DCGM_POLICY_CHANNEL_CALLBACK;
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_register_callback_t registerMsg {};
    PopulatePolicyHeader(
        registerMsg.header, DCGM_POLICY_SR_REGISTER_CALLBACK, dcgm_policy_msg_register_callback_version);
    registerMsg.header.length       = sizeof(registerMsg);
    registerMsg.header.connectionId = kConnId;
    registerMsg.header.requestId    = kReqId;
    registerMsg.fieldId             = createMsg.created.fieldId;
    registerMsg.policyId            = createMsg.created.policyId;
    REQUIRE(policyModule.ProcessMessage(&registerMsg.header) == DCGM_ST_OK);

    DcgmFvBuffer fvBuffer;
    REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 80, 12345, DCGM_ST_OK) != nullptr);

    dcgm_core_msg_field_values_updated_t updateMsg {};
    updateMsg.header.length          = sizeof(updateMsg);
    updateMsg.header.moduleId        = DcgmModuleIdCore;
    updateMsg.header.subCommand      = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    updateMsg.header.version         = dcgm_core_msg_field_values_updated_version;
    updateMsg.fieldValues.buffer     = fvBuffer.GetBuffer();
    updateMsg.fieldValues.bufferSize = 0;
    REQUIRE(fvBuffer.GetSize(&updateMsg.fieldValues.bufferSize, nullptr) == DCGM_ST_OK);
    REQUIRE(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_get_violations_t violationsMsg {};
    PopulatePolicyHeader(violationsMsg.header, DCGM_POLICY_SR_GET_VIOLATIONS, dcgm_policy_msg_get_violations_version);
    violationsMsg.header.length = sizeof(violationsMsg);
    violationsMsg.count         = DCGM_POLICY_MAX_VIOLATIONS;
    REQUIRE(policyModule.ProcessMessage(&violationsMsg.header) == DCGM_ST_OK);

    REQUIRE(ctx.fieldViolationNotifyCount == 1);
    REQUIRE(ctx.lastViolationFieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
}

TEST_CASE("DcgmModulePolicy::get-violations returns insufficient size when caller buffer is too small")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t tempPolicyMsg {};
    PopulatePolicyHeader(tempPolicyMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    tempPolicyMsg.header.length = sizeof(tempPolicyMsg);
    tempPolicyMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    REQUIRE(policyModule.ProcessMessage(&tempPolicyMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_create_t clockPolicyMsg {};
    PopulatePolicyHeader(clockPolicyMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    clockPolicyMsg.header.length = sizeof(clockPolicyMsg);
    clockPolicyMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_SM_CLOCK, 200.0);
    REQUIRE(policyModule.ProcessMessage(&clockPolicyMsg.header) == DCGM_ST_OK);

    DcgmFvBuffer fvBuffer;
    REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, 80, 12345, DCGM_ST_OK) != nullptr);
    REQUIRE(fvBuffer.AddInt64Value(DCGM_FE_GPU, 0, DCGM_FI_DEV_SM_CLOCK, 250, 12345, DCGM_ST_OK) != nullptr);

    dcgm_core_msg_field_values_updated_t updateMsg {};
    updateMsg.header.length          = sizeof(updateMsg);
    updateMsg.header.moduleId        = DcgmModuleIdCore;
    updateMsg.header.subCommand      = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    updateMsg.header.version         = dcgm_core_msg_field_values_updated_version;
    updateMsg.fieldValues.buffer     = fvBuffer.GetBuffer();
    updateMsg.fieldValues.bufferSize = 0;
    REQUIRE(fvBuffer.GetSize(&updateMsg.fieldValues.bufferSize, nullptr) == DCGM_ST_OK);
    REQUIRE(policyModule.ProcessMessage(&updateMsg.header) == DCGM_ST_OK);

    dcgm_policy_msg_get_violations_t violationsMsg {};
    PopulatePolicyHeader(violationsMsg.header, DCGM_POLICY_SR_GET_VIOLATIONS, dcgm_policy_msg_get_violations_version);
    violationsMsg.header.length = sizeof(violationsMsg);
    violationsMsg.count         = 1;
    REQUIRE(policyModule.ProcessMessage(&violationsMsg.header) == DCGM_ST_INSUFFICIENT_SIZE);
    REQUIRE(violationsMsg.count == 2);
    REQUIRE((violationsMsg.violations[0].fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS
             || violationsMsg.violations[0].fieldId == DCGM_FI_DEV_SM_CLOCK));
}

TEST_CASE("DcgmModulePolicy::serializes concurrent commands for the same field policy")
{
    TestPolicyConfigGuard policyConfigGuard;
    auto dcc = MakePolicyModuleCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_policy_msg_create_t createMsg {};
    PopulateFieldPolicyHeader(createMsg.header, DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version);
    createMsg.header.length = sizeof(createMsg);
    createMsg.policy        = MakePolicyInfo(DCGM_FI_DEV_GPU_TEMP_CELSIUS, 70.0);
    REQUIRE(policyModule.ProcessMessage(&createMsg.header) == DCGM_ST_OK);
    REQUIRE(createMsg.created.policyId != 0);

    dcgm_policy_msg_modify_t modifyMsg {};
    PopulateFieldPolicyHeader(modifyMsg.header, DCGM_POLICY_SR_MODIFY, dcgm_policy_msg_modify_version);
    modifyMsg.header.length    = sizeof(modifyMsg);
    modifyMsg.policyId         = createMsg.created.policyId;
    modifyMsg.policy           = createMsg.policy;
    modifyMsg.policy.threshold = 75.0;

    dcgm_policy_msg_get_one_t getOneMsg {};
    PopulateFieldPolicyHeader(getOneMsg.header, DCGM_POLICY_SR_GET_ONE, dcgm_policy_msg_get_one_version);
    getOneMsg.header.length  = sizeof(getOneMsg);
    getOneMsg.fieldId        = createMsg.created.fieldId;
    getOneMsg.policyId       = createMsg.created.policyId;
    getOneMsg.policy.version = dcgmPolicyInfo_version;

    dcgm_policy_msg_set_state_t disableMsg {};
    PopulateFieldPolicyHeader(disableMsg.header, DCGM_POLICY_SR_DISABLE, dcgm_policy_msg_set_state_version);
    disableMsg.header.length = sizeof(disableMsg);
    disableMsg.fieldId       = createMsg.created.fieldId;
    disableMsg.policyId      = createMsg.created.policyId;

    dcgm_policy_msg_set_state_t enableMsg {};
    PopulateFieldPolicyHeader(enableMsg.header, DCGM_POLICY_SR_ENABLE, dcgm_policy_msg_set_state_version);
    enableMsg.header.length = sizeof(enableMsg);
    enableMsg.fieldId       = createMsg.created.fieldId;
    enableMsg.policyId      = createMsg.created.policyId;

    std::vector<dcgmReturn_t> returns(4, DCGM_ST_GENERIC_ERROR);

    {
        std::vector<std::jthread> threads;
        threads.emplace_back(
            [&policyModule, &modifyMsg, &returns] { returns[0] = policyModule.ProcessMessage(&modifyMsg.header); });
        threads.emplace_back(
            [&policyModule, &getOneMsg, &returns] { returns[1] = policyModule.ProcessMessage(&getOneMsg.header); });
        threads.emplace_back(
            [&policyModule, &disableMsg, &returns] { returns[2] = policyModule.ProcessMessage(&disableMsg.header); });
        threads.emplace_back(
            [&policyModule, &enableMsg, &returns] { returns[3] = policyModule.ProcessMessage(&enableMsg.header); });
    }

    for (auto ret : returns)
    {
        REQUIRE(ret == DCGM_ST_OK);
    }

    dcgm_policy_msg_get_all_t getAllMsg {};
    PopulateFieldPolicyHeader(getAllMsg.header, DCGM_POLICY_SR_GET_ALL, dcgm_policy_msg_get_all_version);
    getAllMsg.header.length = sizeof(getAllMsg);
    getAllMsg.capacity      = DCGM_POLICY_GET_ALL_PAGE;
    REQUIRE(policyModule.ProcessMessage(&getAllMsg.header) == DCGM_ST_OK);
    REQUIRE(getAllMsg.returnedCount == 1);
    REQUIRE(getAllMsg.totalCount == 1);
    REQUIRE(getAllMsg.policies[0].fieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    REQUIRE(getAllMsg.policies[0].policyId == createMsg.created.policyId);
    REQUIRE(getAllMsg.policies[0].threshold == 75.0);
    REQUIRE((getAllMsg.policies[0].state == DCGM_POLICY_STATE_ENABLED
             || getAllMsg.policies[0].state == DCGM_POLICY_STATE_DISABLED));
}
