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

#include "TestHelpers.hpp"
#include "mock/MockDcgmiGroupInfo.hpp"
#include "mock/MockDcgmiStatus.hpp"

#include <Policy.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <cstring>
#include <string>
#include <vector>

namespace
{
struct PolicyApiState
{
    dcgmReturn_t policyGetReturn  = DCGM_ST_OK;
    dcgmReturn_t policySetReturn  = DCGM_ST_OK;
    dcgmReturn_t unregisterReturn = DCGM_ST_OK;

    int policyGetCallCount  = 0;
    int policySetCallCount  = 0;
    int unregisterCallCount = 0;

    dcgmStatus_t lastStatusHandle       = 0;
    dcgmHandle_t lastHandle             = 0;
    dcgmGpuGrp_t lastGroupId            = 0;
    int lastPolicyGetCount              = 0;
    dcgmPolicyCondition_t lastCondition = {};
    dcgmPolicy_t lastPolicy {};
    std::vector<dcgmPolicy_t> policies;
};

PolicyApiState g_policyApi;

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

void ResetPolicyApi()
{
    g_policyApi                  = {};
    g_policyApi.policyGetReturn  = DCGM_ST_OK;
    g_policyApi.policySetReturn  = DCGM_ST_OK;
    g_policyApi.unregisterReturn = DCGM_ST_OK;
    ResetMockDcgmiStatus();
    ResetMockDcgmiGroupInfo();
}

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

void SetGroupInfo(std::string const &name, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    REQUIRE(entities.size() <= std::size(g_mockDcgmiGroupInfoData.m_groupInfo.entityList));

    g_mockDcgmiGroupInfoData.m_groupInfo.count = static_cast<unsigned int>(entities.size());
    std::strncpy(g_mockDcgmiGroupInfoData.m_groupInfo.groupName,
                 name.c_str(),
                 sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1);
    g_mockDcgmiGroupInfoData.m_groupInfo.groupName[sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1] = '\0';
    for (size_t i = 0; i < entities.size(); ++i)
    {
        g_mockDcgmiGroupInfoData.m_groupInfo.entityList[i] = entities[i];
    }
}
} //namespace

extern "C" dcgmReturn_t dcgmPolicySet(dcgmHandle_t handle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmPolicy_t *policy,
                                      dcgmStatus_t statusHandle)
{
    g_policyApi.policySetCallCount++;
    g_policyApi.lastHandle       = handle;
    g_policyApi.lastGroupId      = groupId;
    g_policyApi.lastStatusHandle = statusHandle;

    if (policy != nullptr)
    {
        g_policyApi.lastPolicy = *policy;
    }

    return g_policyApi.policySetReturn;
}

extern "C" dcgmReturn_t dcgmPolicyGet(dcgmHandle_t handle,
                                      dcgmGpuGrp_t groupId,
                                      int count,
                                      dcgmPolicy_t *policy,
                                      dcgmStatus_t statusHandle)
{
    g_policyApi.policyGetCallCount++;
    g_policyApi.lastHandle         = handle;
    g_policyApi.lastGroupId        = groupId;
    g_policyApi.lastPolicyGetCount = count;
    g_policyApi.lastStatusHandle   = statusHandle;

    if (policy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_policyApi.policyGetReturn != DCGM_ST_OK)
    {
        return g_policyApi.policyGetReturn;
    }

    for (int i = 0; i < count && i < static_cast<int>(g_policyApi.policies.size()); ++i)
    {
        policy[i] = g_policyApi.policies[i];
    }

    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmPolicyUnregister(dcgmHandle_t handle, dcgmGpuGrp_t groupId, dcgmPolicyCondition_t condition)
{
    g_policyApi.unregisterCallCount++;
    g_policyApi.lastHandle    = handle;
    g_policyApi.lastGroupId   = groupId;
    g_policyApi.lastCondition = condition;
    return g_policyApi.unregisterReturn;
}

TEST_CASE("Policy::SetCurrentViolationPolicy")
{
    GIVEN("a policy manager with mocked status and policy APIs")
    {
        ResetPolicyApi();
        Policy policyManager;
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
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
                CHECK(g_policyApi.policySetCallCount == 1);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
                CHECK(g_policyApi.lastHandle == handle);
                CHECK(g_policyApi.lastGroupId == static_cast<dcgmGpuGrp_t>(groupId));
                CHECK(g_policyApi.lastStatusHandle == g_mockDcgmiStatusData.statusHandle);
                CHECK(g_policyApi.lastPolicy.condition == policy.condition);
                CHECK(g_policyApi.lastPolicy.parms[DCGM_POLICY_COND_IDX_POWER].val.llval == 250);
                CHECK(g_policyApi.lastStatusHandle == g_mockDcgmiStatusData.statusHandle);
                CHECK(g_mockDcgmiStatusData.lastDestroyedStatus == g_mockDcgmiStatusData.statusHandle);
            }
        }

        WHEN("creating the status handle fails")
        {
            g_mockDcgmiStatusData.statusCreateReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.SetCurrentViolationPolicy(handle, groupId, policy);

            THEN("the policy call is skipped and the error is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
                CHECK(g_policyApi.policySetCallCount == 0);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 0);
            }
        }

        WHEN("destroying the status handle fails after a successful set")
        {
            g_mockDcgmiStatusData.statusDestroyReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.SetCurrentViolationPolicy(handle, groupId, policy);

            THEN("the set still succeeds and the cleanup failure is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_policyApi.policySetCallCount == 1);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
                CHECK(capture.str().find("Error: Cannot delete status handler") != std::string::npos);
            }
        }
    }
}

TEST_CASE("Policy::DisplayCurrentViolationPolicy")
{
    GIVEN("a group with current policies")
    {
        ResetPolicyApi();
        Policy policyManager;
        auto handle          = static_cast<dcgmHandle_t>(0x18);
        unsigned int groupId = 5;
        auto policy          = MakePolicy();

        SetGroupInfo("display-group", { { DCGM_FE_GPU, 1 }, { DCGM_FE_GPU, 2 } });
        g_policyApi.policies = { policy, policy };

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
                CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
                CHECK(g_policyApi.policyGetCallCount == 1);
                CHECK(g_policyApi.lastPolicyGetCount == 2);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
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
                CHECK(g_policyApi.policyGetCallCount == 1);
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
            g_policyApi.policies                                               = { fullPolicy, fullPolicy };

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
            auto otherPolicy       = MakePolicy();
            otherPolicy.condition  = DCGM_POLICY_COND_PCI;
            otherPolicy.mode       = DCGM_POLICY_MODE_MANUAL;
            otherPolicy.action     = DCGM_POLICY_ACTION_NONE;
            otherPolicy.validation = DCGM_POLICY_VALID_SV_LONG;
            otherPolicy.response   = DCGM_POLICY_FAILURE_NONE;
            g_policyApi.policies   = { policy, otherPolicy };

            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("mixed fields are marked not applicable")
            {
                auto output = capture.str();
                CHECK(output.find("****") != std::string::npos);
                CHECK(output.find("Non-homogenous settings") != std::string::npos);
                CHECK(result == DCGM_ST_OK);
            }
        }

        WHEN("the group lookup fails")
        {
            g_mockDcgmiGroupInfoData.m_groupInfoReturn = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("policy lookup is skipped")
            {
                CHECK(result == DCGM_ST_NOT_CONFIGURED);
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 0);
                CHECK(g_policyApi.policyGetCallCount == 0);
            }
        }

        WHEN("creating the status handle fails")
        {
            g_mockDcgmiStatusData.statusCreateReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.DisplayCurrentViolationPolicy(handle, groupId, false, false);

            THEN("the allocated policy buffer is cleaned up without a policy query")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
                CHECK(g_policyApi.policyGetCallCount == 0);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 0);
            }
        }
    }
}

TEST_CASE("SetPolicy::DoExecuteConnected")
{
    GIVEN("a set-policy command with a configured DCGM handle")
    {
        ResetPolicyApi();
        dcgmPolicy_t policy = MakePolicy();
        TestSetPolicy command("localhost", policy, 12);
        auto handle = static_cast<dcgmHandle_t>(0x22);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("it forwards the request to Policy::SetCurrentViolationPolicy")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_policyApi.policySetCallCount == 1);
                CHECK(g_policyApi.lastHandle == handle);
                CHECK(g_policyApi.lastGroupId == 12);
                CHECK(g_policyApi.lastPolicy.condition == policy.condition);
            }
        }
    }
}

TEST_CASE("Policy::UnregisterPolicyUpdates")
{
    GIVEN("a policy manager with a mocked unregister API")
    {
        ResetPolicyApi();
        Policy policyManager;
        auto handle            = static_cast<dcgmHandle_t>(0x33);
        unsigned int groupId   = 13;
        unsigned int condition = DCGM_POLICY_COND_DBE | DCGM_POLICY_COND_XID;

        WHEN("unregister succeeds")
        {
            dcgmReturn_t result = policyManager.UnregisterPolicyUpdates(handle, groupId, condition);

            THEN("the condition is forwarded as a policy condition")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_policyApi.unregisterCallCount == 1);
                CHECK(g_policyApi.lastHandle == handle);
                CHECK(g_policyApi.lastGroupId == static_cast<dcgmGpuGrp_t>(groupId));
                CHECK(g_policyApi.lastCondition == static_cast<dcgmPolicyCondition_t>(condition));
            }
        }

        WHEN("unregister fails")
        {
            g_policyApi.unregisterReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = policyManager.UnregisterPolicyUpdates(handle, groupId, condition);

            THEN("the error is returned and printed")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(g_policyApi.unregisterCallCount == 1);
                CHECK(capture.str().find("Error: Cannot unregister to receive policy violations") != std::string::npos);
            }
        }
    }
}

TEST_CASE("RegPolicy::~RegPolicy")
{
    GIVEN("a registration command with a live handle")
    {
        ResetPolicyApi();
        auto handle = static_cast<dcgmHandle_t>(0x44);

        WHEN("the command is destroyed")
        {
            {
                TestRegPolicy command("localhost", 4, DCGM_POLICY_COND_THERMAL);
                command.SetHandle(handle);
            }

            THEN("the registered condition is unregistered")
            {
                CHECK(g_policyApi.unregisterCallCount == 1);
                CHECK(g_policyApi.lastHandle == handle);
                CHECK(g_policyApi.lastGroupId == 4);
                CHECK(g_policyApi.lastCondition == DCGM_POLICY_COND_THERMAL);
            }
        }
    }
}
