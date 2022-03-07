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
#ifndef FIELDGROUP_H
#define FIELDGROUP_H

#include "Command.h"
#include <vector>

class FieldGroup
{
public:
    FieldGroup()          = default;
    virtual ~FieldGroup() = default;

    /*****************************************************************************
     * This method is used to list the field groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupListAll(dcgmHandle_t dcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to query field group info on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupInfo(dcgmHandle_t dcgmHandle, bool json);

    /*****************************************************************************
     * This method is used to create groups on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupCreate(dcgmHandle_t dcgmHandle);

    /*****************************************************************************
     * This method is used to delete a group on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RunGroupDestroy(dcgmHandle_t pNvcmHandle);

    /******************************************************************************
     * Getters and setters
     ******************************************************************************/
    void SetFieldGroupId(unsigned int id);
    unsigned int GetFieldGroupId();
    void SetFieldGroupName(std::string name);
    std::string GetFieldGroupName();
    void SetFieldIdsString(std::string fieldIdsStr);

private:
    /*****************************************************************************
     * helper method to take a string of form x,y,z... and
     * transform that into a list of field IDs
     *****************************************************************************/
    dcgmReturn_t HelperFieldIdStringParse(std::string input, std::vector<unsigned short> &fieldIds);

    unsigned int m_fieldGroupId = 0;
    std::string m_fieldGroupName;
    std::string m_fieldIdsStr;
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * List Group Invoker class
 */
class FieldGroupListAll : public Command
{
public:
    FieldGroupListAll(std::string hostname, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    FieldGroup fieldGroupObj;
};

/**
 * Create Field Group Invoker class
 */
class FieldGroupCreate : public Command
{
public:
    FieldGroupCreate(std::string hostname, FieldGroup &obj);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    FieldGroup fieldGroupObj;
};

/**
 * Destroy Field Group Invoker class
 */
class FieldGroupDestroy : public Command
{
public:
    FieldGroupDestroy(std::string hostname, FieldGroup &obj);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    FieldGroup fieldGroupObj;
};

/**
 * Field Group Info Invoker class
 */
class FieldGroupInfo : public Command
{
public:
    FieldGroupInfo(std::string hostname, FieldGroup &obj, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    FieldGroup fieldGroupObj;
};


#endif /* GROUP_H_ */
