/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmiOutput.h"

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


    /****************************************************************************/
    /*
     * AppendSystemIncidents
     * Looks at the response struct and appends the messages from incidents that
     * occurred on the same system to facilitate display later
     *
     * @param response      (in)  - the health response struct
     * @param startingIndex (in)  - the first index to check for a matching incident
     * @param entityId      (in)  - the ID of the entity we are matching
     * @param entityGroupId (in)  - the entity type we are matching
     * @param system        (in)  - the system for the incident we are matching
     * @param buf           (out) - the buffer we're appending the error messages to
     * @param systemHealth  (out) - the overall system health from the incidents
     *
     * @return the number of matching incidents
     */
    unsigned int AppendSystemIncidents(const dcgmHealthResponse_t &response,
                                       unsigned int startingIndex,
                                       dcgm_field_eid_t entityId,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgmHealthSystems_t system,
                                       std::stringstream &buf,
                                       dcgmHealthWatchResults_t &systemHealth);

    /****************************************************************************/
    /*
     * AddErrorMessage
     *
     * Adds the error message piece by piece if it's longer than what fits on a line.
     *
     * @param outErrors (out) - the output boxer object that displays the message
     * @param errorMsg  (in)  - the error message
     * @param systemStr (in)  - the string representation of the string having an error
     */
    void AddErrorMessage(DcgmiOutputBoxer &outErrors, const std::string &errorMsg, const std::string &systemStr);

    /****************************************************************************/
    /*
     * HandleOneEntity
     *
     * Reads and outputs all of the incidents related to a single entity and returns the number of incidents
     * processed
     *
     * @param response      (in)  - the health response struct received from the hostengine
     * @param startingIndex (in)  - the first incident index to inspect
     * @param entityId      (in)  - the ID of the entity we're processing
     * @param entityGroupId (in)  - the group ID of the entity we're processing
     * @param out           (out) - the dcgmi output manager we're adding to
     *
     * @return the string to be written to stdout from dcgmi
     */
    unsigned int HandleOneEntity(const dcgmHealthResponse_t &response,
                                 unsigned int startingIndex,
                                 dcgm_field_eid_t entityId,
                                 dcgm_field_entity_group_t entityGroupId,
                                 DcgmiOutput &out);

    /****************************************************************************/
    /*
     * GenerateOutputFromResponse
     *
     * Returns the string output based on the health response received
     *
     * @param response (in)  - the health response struct received from the hostengine
     * @param out      (out) - the dcgmi output manager we're adding to
     *
     * @return the string to be written to stdout from dcgmi
     */
    std::string GenerateOutputFromResponse(const dcgmHealthResponse_t &response, DcgmiOutput &out);

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
