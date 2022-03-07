/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

class Policy
{
public:
    Policy();
    virtual ~Policy();

    dcgmReturn_t DisplayCurrentViolationPolicy(dcgmHandle_t mNvcmHandle, unsigned int groupId, bool verbose, bool json);
    dcgmReturn_t SetCurrentViolationPolicy(dcgmHandle_t mNvcmHandle, unsigned int groupId, dcgmPolicy_t &policy);
    dcgmReturn_t RegisterForPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition);
    dcgmReturn_t UnregisterPolicyUpdates(dcgmHandle_t mNvcmHandle, unsigned int groupId, unsigned int condition);

private:
    static int ListenForViolations(void *data);
    static std::string HelperFormatTimestamp(long long timestamp);
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
    GetPolicy(std::string hostname, unsigned int groupId, bool verbose, bool json);

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
    SetPolicy(std::string hostname, dcgmPolicy_t &setPolicy, unsigned int groupId);

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
    RegPolicy(std::string hostname, unsigned int groupId, unsigned int condition);
    ~RegPolicy() override;

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Policy policyObj;
    unsigned int groupId;
    unsigned int condition;
};
#endif /* POLICY_H_ */
