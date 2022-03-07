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
 * Group.h
 *
 */

#ifndef GROUP_H_
#define GROUP_H_

#include "Command.h"
#include "DcgmiOutput.h"
#include <vector>

class Group
{
public:
    Group()          = default;
    virtual ~Group() = default;

    /*****************************************************************************
     * This method is used to list the groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupList(dcgmHandle_t pDcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to create groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupCreate(dcgmHandle_t pDcgmHandle, dcgmGroupType_t groupType);

    /*****************************************************************************
     * This method is used to remove groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupDestroy(dcgmHandle_t mDcgmHandle);

    /*****************************************************************************
     * This method is used to get the info for a group on the host-engine represented
     * by the DCGM handle
     * It is overloaded in order to allow reuse in RunGroupList
     *****************************************************************************/
    dcgmReturn_t RunGroupInfo(dcgmHandle_t dcgmHandle, bool outputJson);
    dcgmReturn_t RunGroupInfo(dcgmHandle_t dcgmHandle, DcgmiOutputBoxer &outGroup);

    /*****************************************************************************
     * This method is used to add to or remove from a group on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupManageDevice(dcgmHandle_t dcgmHandle, bool add);

    /******************************************************************************
     * Getters and setters
     ******************************************************************************/
    void SetGroupId(unsigned int id);
    unsigned int GetGroupId() const;
    void SetGroupName(std::string name);
    std::string getGroupName();
    void SetGroupInfo(std::string name);
    std::string getGroupInfo();

private:
    dcgmGpuGrp_t groupId = 0;
    std::string groupName;
    std::string groupInfo;
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * List Group Invoker class
 */
class GroupList : public Command
{
public:
    GroupList(std::string hostname, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
};

/**
 * Create Group Invoker class
 */
class GroupCreate : public Command
{
public:
    GroupCreate(std::string hostname, Group &obj, dcgmGroupType_t groupType);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
    dcgmGroupType_t groupType;
};

/**
 * Destroy Group Invoker class
 */
class GroupDestroy : public Command
{
public:
    GroupDestroy(std::string hostname, Group &obj);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
};

/**
 * Group Info Invoker class
 */
class GroupInfo : public Command
{
public:
    GroupInfo(std::string hostname, Group &obj, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
};

/**
 * Add to Group Invoker class
 */
class GroupAddTo : public Command
{
public:
    GroupAddTo(std::string hostname, Group &obj);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
};

/**
 * Add to Group Invoker class
 */
class GroupDeleteFrom : public Command
{
public:
    GroupDeleteFrom(std::string hostname, Group &obj);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Group groupObj;
};

#endif /* GROUP_H_ */
