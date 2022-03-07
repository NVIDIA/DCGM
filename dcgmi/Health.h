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
 * Health.h
 *
 *  Created on: Oct 6, 2015
 *      Author: chris
 */

#ifndef HEALTH_H_
#define HEALTH_H_

#include "Command.h"

class Health : public Command
{
public:
    Health() = default;
    dcgmReturn_t GetWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json);
    dcgmReturn_t SetWatches(dcgmHandle_t mDcgmHandle,
                            dcgmGpuGrp_t groupId,
                            dcgmHealthSystems_t systems,
                            double updateInterval,
                            double maxKeepAge);
    dcgmReturn_t CheckWatches(dcgmHandle_t mDcgmHandle, dcgmGpuGrp_t groupId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    std::string HelperHealthToString(dcgmHealthWatchResults_t health);
    std::string HelperSystemToString(dcgmHealthSystems_t system);
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Get Watches Invoker
 */
class GetHealth : public Command
{
public:
    GetHealth(std::string hostname, unsigned int groupId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Health healthObj;
    dcgmGpuGrp_t groupId;
};

/**
 * Set Watches Invoker
 */
class SetHealth : public Command
{
public:
    SetHealth(std::string hostname,
              unsigned int groupId,
              unsigned int system,
              double updateInterval,
              double maxKeepAge);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Health healthObj;
    dcgmGpuGrp_t mGroupId;
    dcgmHealthSystems_t mSystems;
    double mUpdateInterval;
    double mMaxKeepAge;
};

/**
 * Check Watches Invoker
 */
class CheckHealth : public Command
{
public:
    CheckHealth(std::string hostname, unsigned int groupId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Health healthObj;
    dcgmGpuGrp_t groupId;
};

#endif /* HEALTH_H_ */
