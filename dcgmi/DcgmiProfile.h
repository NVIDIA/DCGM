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
#pragma once

#include "Command.h"
#include "DcgmiOutput.h"
#include "dcgm_structs.h"

class DcgmiProfile
{
public:
    DcgmiProfile()          = default;
    virtual ~DcgmiProfile() = default;
    dcgmReturn_t RunProfileList(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, bool outputAsJson);
    dcgmReturn_t RunProfileSetPause(dcgmHandle_t dcgmHandle, bool pause);

private:
};

class DcgmiProfileList : public Command
{
public:
    DcgmiProfileList(std::string hostname, std::string gpuIdsStr, std::string groupIdStr, bool outputAsJson);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    DcgmiProfile mProfileObj;
    std::string mGpuIdsStr;
    std::string mGroupIdStr;

    dcgmGpuGrp_t mGroupId;

    dcgmReturn_t ValidateOrCreateEntityGroup(void);     /* Set mGroupId based on mGpuIdsStr and mGroupIdStr */
    dcgmReturn_t CreateEntityGroupFromEntityList(void); /* Helper called by ValidateOrCreateEntityGroup() */
};

class DcgmiProfileSetPause : public Command
{
public:
    DcgmiProfileSetPause(std::string hostname, bool pause);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    DcgmiProfile mProfileObj;
    bool m_pause; /* Should we pause (true) or resume (false) */
};
