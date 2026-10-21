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
/*
 * Policy.h
 *
 *  Created on: Oct 5, 2015
 *      Author: chris
 */

#ifndef POLICY_H_
#define POLICY_H_

#include "Command.h"

#include <DcgmOpsAnyPtr.hpp>
#include <dcgm_structs.h>
#include <timelib.h>

#include <string>
#include <vector>

enum class FieldPolicyAction
{
    Create,
    List,
    Get,
    Modify,
    Delete,
    Enable,
    Disable,
    Import,
    Watch,
};

struct FieldPolicyOptions
{
    FieldPolicyAction action { FieldPolicyAction::List };
    unsigned short fieldId {};
    uint64_t policyId {};
    std::string name;
    double threshold {};
    dcgmPolicyOperator_t policyOperator { DCGM_POLICY_OP_GT };
    std::vector<dcgmGroupEntityPair_t> entities;
    unsigned int channelMask { DCGM_POLICY_CHANNEL_CONSOLE };
    bool enabled { true };
    unsigned int rateLimitSec { 60 };
    bool deleteAll {};
    bool hasFieldId {};
    bool hasPolicyId {};
    bool hasName {};
    bool hasThreshold {};
    bool hasOperator {};
    bool hasEntities {};
    bool hasChannels {};
    bool hasRateLimit {};
    std::string importPath;
    timelib64_t sinceTimestamp {};
    unsigned int maxViolations { DCGM_POLICY_MAX_VIOLATIONS };
    unsigned int watchIntervalSec { 1 };
    bool watchOnce {};
};

class Policy
{
public:
    explicit Policy(DcgmNs::DcgmOpsAnyPtr dcgmOps);
    virtual ~Policy();

    dcgmReturn_t DisplayCurrentViolationPolicy(dcgmHandle_t mNvcmHandle, unsigned int groupId, bool verbose, bool json);
    dcgmReturn_t SetCurrentViolationPolicy(dcgmHandle_t mNvcmHandle, unsigned int groupId, dcgmPolicy_t &policy);
    dcgmReturn_t RegisterForPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition);
    dcgmReturn_t UnregisterPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition);

    /**
     * Execute a field-policy command against the hostengine.
     *
     * @param[in] dcgmHandle Connected DCGM hostengine handle.
     * @param[in] options Parsed field-policy action and its arguments.
     * @param[in] json Whether to render successful command output as JSON.
     * @return DCGM_ST_OK on success; DCGM_ST_BADPARAM for invalid command options; otherwise an error returned by the
     *         corresponding field-policy API.
     */
    dcgmReturn_t RunFieldPolicyCommand(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json);

private:
    dcgmReturn_t RunFieldPolicyCreate(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json);
    dcgmReturn_t RunFieldPolicyList(dcgmHandle_t dcgmHandle, bool json);
    dcgmReturn_t RunFieldPolicyGet(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json);
    dcgmReturn_t RunFieldPolicyModify(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options);
    dcgmReturn_t RunFieldPolicyDelete(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options);
    dcgmReturn_t RunFieldPolicyEnableDisable(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options);
    dcgmReturn_t RunFieldPolicyImport(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options);
    dcgmReturn_t RunFieldPolicyWatch(dcgmHandle_t dcgmHandle, FieldPolicyOptions const &options, bool json);
    dcgmReturn_t GetAllFieldPolicies(dcgmHandle_t dcgmHandle, std::vector<dcgmPolicyInfo_t> &policies);
    void PrintFieldPolicyDetails(dcgmHandle_t dcgmHandle, dcgmPolicyInfo_t const &policy);

    static int ListenForViolations(dcgmPolicyCallbackResponse_t *callbackResponse, uint64_t userData);
    static std::string HelperFormatTimestamp(long long timestamp);

    DcgmNs::DcgmOpsAnyPtr m_dcgmOps;
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Get Policy invoker class
 */
class GetPolicy : public Command
{
public:
    GetPolicy(std::string hostname, unsigned int groupId, bool verbose, bool json, DcgmNs::DcgmOpsAnyPtr dcgmOps);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Policy policyObj;
    unsigned int groupId;
    bool verbose;
};

/**
 * Set Policy invoker class
 */
class SetPolicy : public Command
{
public:
    SetPolicy(std::string hostname, dcgmPolicy_t &setPolicy, unsigned int groupId, DcgmNs::DcgmOpsAnyPtr dcgmOps);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Policy policyObj;
    dcgmPolicy_t setPolicy;
    unsigned int groupId;
};

/**
 * Register Policy invoker class
 */
class RegPolicy : public Command
{
public:
    RegPolicy(std::string hostname, unsigned int groupId, unsigned int condition, DcgmNs::DcgmOpsAnyPtr dcgmOps);
    ~RegPolicy() override;

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Policy policyObj;
    unsigned int groupId;
    unsigned int condition;
};

class FieldPolicy : public Command
{
public:
    FieldPolicy(std::string hostname, FieldPolicyOptions options, bool json, DcgmNs::DcgmOpsAnyPtr dcgmOps);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Policy policyObj;
    FieldPolicyOptions m_options;
};
#endif /* POLICY_H_ */
