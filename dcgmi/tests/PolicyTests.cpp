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

#include <CoutCapture.hpp>

#include <MockDcgmOps.h>
#include <Policy.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <string>

namespace
{
class TestSetPolicy : public SetPolicy
{
public:
    using SetPolicy::SetPolicy;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestRegPolicy : public RegPolicy
{
public:
    using RegPolicy::RegPolicy;

    void SetHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
    }
};

dcgmPolicy_t MakePolicy()
{
    dcgmPolicy_t policy {};
    policy.version    = dcgmPolicy_version;
    policy.condition  = static_cast<dcgmPolicyCondition_t>(DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_POWER);
    policy.mode       = DCGM_POLICY_MODE_AUTOMATED;
    policy.action     = DCGM_POLICY_ACTION_GPURESET;
    policy.validation = DCGM_POLICY_VALID_SV_SHORT;
    policy.response   = DCGM_POLICY_FAILURE_NONE;
    policy.parms[DCGM_POLICY_COND_IDX_POWER].tag       = dcgmPolicyConditionParams_t::LLONG;
    policy.parms[DCGM_POLICY_COND_IDX_POWER].val.llval = 250;
    return policy;
}

} //namespace

TEST_CASE("Policy::SetCurrentViolationPolicy")
{
    GIVEN("a policy manager with mocked status and policy APIs")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Policy policyManager(&mockDcgmOps);
        dcgmPolicy_t policy  = MakePolicy();
        auto handle          = static_cast<dcgmHandle_t>(0x11);
        unsigned int groupId = 7;

        WHEN("the policy is applied successfully")
        {
            CoutCapture capture;
            dcgmReturn_t result = policyManager.SetCurrentViolationPolicy(handle, groupId, policy);

            THEN("the policy API receives the handle, group, policy, and status handle")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.setCallCount == 1);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
                CHECK(mockDcgmOps.m_policyState.lastGroupId == static_cast<dcgmGpuGrp_t>(groupId));
                CHECK(mockDcgmOps.m_policyState.lastStatusHandle == mockDcgmOps.m_statusState.createdHandle);
                CHECK(mockDcgmOps.m_policyState.lastPolicy.condition == policy.condition);
                CHECK(mockDcgmOps.m_policyState.lastPolicy.parms[DCGM_POLICY_COND_IDX_POWER].val.llval == 250);
                CHECK(mockDcgmOps.m_policyState.lastStatusHandle == mockDcgmOps.m_statusState.createdHandle);
                CHECK(mockDcgmOps.m_statusState.lastDestroyedStatus == mockDcgmOps.m_statusState.createdHandle);
            }
        }

        WHEN("creating the status handle fails")
        {
            mockDcgmOps.m_statusState.createReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.SetCurrentViolationPolicy(handle, groupId, policy);

            THEN("the policy call is skipped and the error is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.setCallCount == 0);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 0);
            }
        }

        WHEN("destroying the status handle fails after a successful set")
        {
            mockDcgmOps.m_statusState.destroyReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.SetCurrentViolationPolicy(handle, groupId, policy);

            THEN("the set still succeeds and the cleanup failure is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_policyState.setCallCount == 1);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
                CHECK(capture.str().find("Error: Cannot delete status handler") != std::string::npos);
            }
        }
    }
}

TEST_CASE("Policy::RunFieldPolicyCommand uses DcgmOps for field policies")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    Policy policyManager(&mockDcgmOps);
    auto const handle = static_cast<dcgmHandle_t>(0x21);

    FieldPolicyOptions options;
    options.action         = FieldPolicyAction::Create;
    options.fieldId        = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
    options.name           = "temperature";
    options.threshold      = 80.0;
    options.policyOperator = DCGM_POLICY_OP_GT;
    options.channelMask    = DCGM_POLICY_CHANNEL_CONSOLE;

    CoutCapture capture;
    REQUIRE(policyManager.RunFieldPolicyCommand(handle, options, false) == DCGM_ST_OK);
    REQUIRE(mockDcgmOps.m_fieldPolicyState.createCallCount == 1);
    REQUIRE(mockDcgmOps.m_fieldPolicyState.lastHandle == handle);
    REQUIRE(mockDcgmOps.m_fieldPolicyState.lastFieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    REQUIRE(mockDcgmOps.m_fieldPolicyState.lastName == "temperature");
}

TEST_CASE("Policy::DisplayCurrentViolationPolicy")
{
    GIVEN("a group with current policies")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Policy policyManager(&mockDcgmOps);
        auto handle          = static_cast<dcgmHandle_t>(0x18);
        unsigned int groupId = 5;
        auto policy          = MakePolicy();

        mockDcgmOps.m_groupInfoState.overrideInfo
            = DcgmNs::MockDcgmOps::MakeGroupInfo("display-group", { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 } });
        mockDcgmOps.m_policyState.policies = { policy, policy };

        WHEN("compact text output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("group policy details are fetched and rendered once")
            {
                CHECK(result == DCGM_ST_OK);
                auto output = capture.str();
                CHECK(output.find("Policy information") != std::string::npos);
                CHECK(output.find("display-group") != std::string::npos);
                CHECK(output.find("Double-bit ECC errors") != std::string::npos);
                CHECK(output.find("Max power threshold") != std::string::npos);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.getCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastPolicyGetCount == 2);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
            }
        }

        WHEN("verbose JSON output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, true, true);

            THEN("each entity is rendered")
            {
                auto output = capture.str();
                CHECK(output.find("Policy information") != std::string::npos);
                CHECK(output.find("GPU ID: 1") != std::string::npos);
                CHECK(output.find("GPU ID: 2") != std::string::npos);
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_policyState.getCallCount == 1);
            }
        }

        WHEN("verbose text output contains all policy condition types")
        {
            auto fullPolicy      = MakePolicy();
            fullPolicy.condition = static_cast<dcgmPolicyCondition_t>(
                DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_PCI | DCGM_POLICY_COND_MAX_PAGES_RETIRED
                | DCGM_POLICY_COND_THERMAL | DCGM_POLICY_COND_POWER | DCGM_POLICY_COND_NVLINK | DCGM_POLICY_COND_XID);
            fullPolicy.mode                                                    = DCGM_POLICY_MODE_MANUAL;
            fullPolicy.action                                                  = DCGM_POLICY_ACTION_NONE;
            fullPolicy.validation                                              = DCGM_POLICY_VALID_SV_XLONG;
            fullPolicy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].tag       = dcgmPolicyConditionParams_t::LLONG;
            fullPolicy.parms[DCGM_POLICY_COND_IDX_MAX_PAGES_RETIRED].val.llval = 12;
            fullPolicy.parms[DCGM_POLICY_COND_IDX_THERMAL].tag                 = dcgmPolicyConditionParams_t::LLONG;
            fullPolicy.parms[DCGM_POLICY_COND_IDX_THERMAL].val.llval           = 88;
            mockDcgmOps.m_policyState.policies                                 = { fullPolicy, fullPolicy };

            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, true, false);

            THEN("every condition branch is rendered")
            {
                auto output = capture.str();
                CHECK(output.find("Max retired pages threshold") != std::string::npos);
                CHECK(output.find("Max temperature threshold") != std::string::npos);
                CHECK(output.find("NVLink Errors") != std::string::npos);
                CHECK(output.find("XID error detected") != std::string::npos);
                CHECK(result == DCGM_ST_OK);
            }
        }

        WHEN("compact output sees non-homogeneous policy fields")
        {
            auto otherPolicy                   = MakePolicy();
            otherPolicy.condition              = DCGM_POLICY_COND_PCI;
            otherPolicy.mode                   = DCGM_POLICY_MODE_MANUAL;
            otherPolicy.action                 = DCGM_POLICY_ACTION_NONE;
            otherPolicy.validation             = DCGM_POLICY_VALID_SV_LONG;
            otherPolicy.response               = DCGM_POLICY_FAILURE_NONE;
            mockDcgmOps.m_policyState.policies = { policy, otherPolicy };

            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("mixed fields are marked not applicable")
            {
                auto output = capture.str();
                CHECK(output.find("****") != std::string::npos);
                CHECK(output.find("Non-homogenous settings across group. Use with -v flag") != std::string::npos);
                CHECK(result == DCGM_ST_OK);
            }
        }

        WHEN("the group lookup fails")
        {
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("policy lookup is skipped")
            {
                CHECK(result == DCGM_ST_NOT_CONFIGURED);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 0);
                CHECK(mockDcgmOps.m_policyState.getCallCount == 0);
            }
        }

        WHEN("creating the status handle fails")
        {
            mockDcgmOps.m_statusState.createReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("the allocated policy buffer is cleaned up without a policy query")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.getCallCount == 0);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 0);
            }
        }
    }
}

TEST_CASE("SetPolicy::DoExecuteConnected")
{
    GIVEN("a set-policy command with a configured DCGM handle")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        dcgmPolicy_t policy = MakePolicy();
        TestSetPolicy command("localhost", policy, 12, &mockDcgmOps);
        auto handle = static_cast<dcgmHandle_t>(0x22);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("it forwards the request to Policy::SetCurrentViolationPolicy")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_policyState.setCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
                CHECK(mockDcgmOps.m_policyState.lastGroupId == 12);
                CHECK(mockDcgmOps.m_policyState.lastPolicy.condition == policy.condition);
            }
        }
    }
}

TEST_CASE("Policy::UnregisterPolicyUpdates")
{
    GIVEN("a policy manager with a mocked unregister API")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Policy policyManager(&mockDcgmOps);
        auto handle            = static_cast<dcgmHandle_t>(0x33);
        unsigned int groupId   = 13;
        unsigned int condition = DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_XID;

        WHEN("unregister succeeds")
        {
            dcgmReturn_t result = policyManager.UnregisterPolicyUpdates(handle, groupId, condition);

            THEN("the condition is forwarded as a policy condition")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_policyState.unregisterCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
                CHECK(mockDcgmOps.m_policyState.lastGroupId == static_cast<dcgmGpuGrp_t>(groupId));
                CHECK(mockDcgmOps.m_policyState.lastCondition == static_cast<dcgmPolicyCondition_t>(condition));
            }
        }

        WHEN("unregister fails")
        {
            mockDcgmOps.m_policyState.unregisterReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.UnregisterPolicyUpdates(handle, groupId, condition);

            THEN("the error is returned and printed")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_policyState.unregisterCallCount == 1);
                CHECK(capture.str().find("Error: Cannot unregister to receive policy violations") != std::string::npos);
            }
        }
    }
}

TEST_CASE("RegPolicy::~RegPolicy")
{
    GIVEN("a registration command with a live handle")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        auto handle = static_cast<dcgmHandle_t>(0x44);

        WHEN("the command is destroyed")
        {
            {
                TestRegPolicy command("localhost", 4, DCGM_POLICY_COND_THERMAL, &mockDcgmOps);
                command.SetHandle(handle);
            }

            THEN("the registered condition is unregistered")
            {
                CHECK(mockDcgmOps.m_policyState.unregisterCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
                CHECK(mockDcgmOps.m_policyState.lastGroupId == 4);
                CHECK(mockDcgmOps.m_policyState.lastCondition == DCGM_POLICY_COND_THERMAL);
            }
        }
    }
}

TEST_CASE("Policy::RegisterForPolicyUpdates reports registration failures")
{
    GIVEN("a policy manager with a mocked register API")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Policy policyManager(&mockDcgmOps);
        auto handle          = static_cast<dcgmHandle_t>(0x55);
        unsigned int groupId = 14;

        WHEN("registration fails before entering the listen loop")
        {
            mockDcgmOps.m_policyState.registerReturn = DCGM_ST_NOT_SUPPORTED;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.RegisterForPolicyUpdates(handle, groupId, DCGM_POLICY_COND_DBE);

            THEN("all policy conditions are requested and the error is returned")
            {
                auto allConditions = static_cast<dcgmPolicyCondition_t>(
                    DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_PCI | DCGM_POLICY_COND_MAX_PAGES_RETIRED
                    | DCGM_POLICY_COND_THERMAL | DCGM_POLICY_COND_POWER | DCGM_POLICY_COND_NVLINK
                    | DCGM_POLICY_COND_XID);

                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_policyState.registerCallCount == 1);
                CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
                CHECK(mockDcgmOps.m_policyState.lastGroupId == static_cast<dcgmGpuGrp_t>(groupId));
                CHECK(mockDcgmOps.m_policyState.lastRegisteredCondition == allConditions);
                CHECK(mockDcgmOps.m_policyState.lastCallback != nullptr);
                CHECK(mockDcgmOps.m_policyState.lastUserData == 0);
                CHECK(capture.str().find("Policy management is only supported on Tesla GPUs") != std::string::npos);
            }
        }
    }
}

TEST_CASE("Captured policy registration callback formats each policy condition")
{
    GIVEN("a captured production callback without entering the listen loop")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        Policy policyManager(&mockDcgmOps);
        mockDcgmOps.m_policyState.registerReturn = DCGM_ST_NOT_SUPPORTED;
        auto handle                              = static_cast<dcgmHandle_t>(0x66);

        {
            CoutCapture capture;
            CHECK(policyManager.RegisterForPolicyUpdates(handle, 15, DCGM_POLICY_COND_DBE) == DCGM_ST_NOT_SUPPORTED);
        }
        REQUIRE(mockDcgmOps.m_policyState.lastCallback != nullptr);
        auto callback = mockDcgmOps.m_policyState.lastCallback;
        auto userData = mockDcgmOps.m_policyState.lastUserData;

        struct TestCase
        {
            dcgmPolicyCondition_t condition;
            char const *expected;
        };

        TestCase cases[] = {
            { DCGM_POLICY_COND_DBE, "double-bit ECC" },
            { DCGM_POLICY_COND_PCI, "PCIe replay" },
            { DCGM_POLICY_COND_MAX_PAGES_RETIRED, "maximum number of retired pages" },
            { DCGM_POLICY_COND_THERMAL, "maximum thermal limit" },
            { DCGM_POLICY_COND_POWER, "maximum power limit" },
            { DCGM_POLICY_COND_NVLINK, "NVLink Counter" },
            { DCGM_POLICY_COND_XID, "XID error" },
            { static_cast<dcgmPolicyCondition_t>(0), "unknown error" },
        };

        THEN("the captured callback emits the user-facing diagnostic text for each condition")
        {
            for (auto const &testCase : cases)
            {
                CAPTURE(testCase.condition);
                dcgmPolicyCallbackResponse_t response {};
                response.condition                    = testCase.condition;
                response.val.pci.timestamp            = 1700000000;
                response.val.pci.counter              = 11;
                response.val.mpr.timestamp            = 1700000001;
                response.val.mpr.sbepages             = 3;
                response.val.mpr.dbepages             = 4;
                response.val.thermal.timestamp        = 1700000002;
                response.val.thermal.thermalViolation = 91;
                response.val.power.timestamp          = 1700000003;
                response.val.power.powerViolation     = 255;
                response.val.nvlink.timestamp         = 1700000004;
                response.val.nvlink.fieldId           = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_TOTAL;
                response.val.nvlink.counter           = 7;
                response.val.xid.timestamp            = 1700000005;
                response.val.xid.errnum               = 79;

                CoutCapture capture;
                CHECK(callback(&response, userData) == 0);
                CHECK(capture.str().find(testCase.expected) != std::string::npos);
            }
        }
    }
}
