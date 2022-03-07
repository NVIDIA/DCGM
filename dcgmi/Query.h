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
 * Query.h
 *
 */

#ifndef QUERY_H_
#define QUERY_H_

#include "Command.h"
#include <vector>

class Query
{
public:
    Query();
    virtual ~Query();

    /*****************************************************************************
     * This method is used to display the GPUs on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayDiscoveredDevices(dcgmHandle_t mNvcmHandle);

    /*****************************************************************************
     * This method is used to display GPU info for the specified device on the
     * host engine represented by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayDeviceInfo(dcgmHandle_t dcgmHandle, unsigned int requestedGpuId, std::string const &attributes);

    /*****************************************************************************
     * This method is used to display the gpus in the specified group on the
     * host-engine represented by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGroupInfo(dcgmHandle_t mNvcmHandle,
                                  unsigned int requestedGroupId,
                                  std::string const &attributes,
                                  bool verbose);

    /*****************************************************************************
     * This method is used to display the hierarchy of GPUs, instances, and
     * compute instances
     *****************************************************************************/
    dcgmReturn_t DisplayHierarchyInfo(dcgmHandle_t handle);

private:
    /*****************************************************************************
     * Helper method to acquire a list of All entities of a given entityGroup on the system
     *****************************************************************************/
    dcgmReturn_t HelperGetEntityList(dcgmHandle_t dcgmHandle,
                                     dcgm_field_entity_group_t entityGroup,
                                     std::vector<dcgm_field_eid_t> &entityIds);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    std::string HelperFormatClock(dcgmClockSet_t clock);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    dcgmReturn_t HelperValidInput(std::string const &attributes);

    /*****************************************************************************
     * Helper method to format the display of clock information
     *****************************************************************************/
    dcgmReturn_t HelperDisplayNonVerboseGroup(dcgmHandle_t mNvcmHandle,
                                              dcgmGroupInfo_t &stNvcmGroupInfo,
                                              std::string attributes);

    /*****************************************************************************
     * These functions pass the information to the output controller to be displayed
     * bitvectors are used if the function replaces values with "****" to indicate
     * that the values are not homogenous across the group. Each bit references one
     * of the output values in order of how its displayed from top (bit 0) to bottom
     *****************************************************************************/
    void HelperDisplayClocks(dcgmDeviceSupportedClockSets_t &clocks);
    void HelperDisplayThermals(dcgmDeviceThermals_t thermals, unsigned int bitvector);
    void HelperDisplayPowerLimits(dcgmDevicePowerLimits_t powerLimits, unsigned int bitvector);
    void HelperDisplayIdentifiers(dcgmDeviceIdentifiers_t &identifiers, unsigned int bitvector);
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Query info invoker class
 */
class QueryDeviceInfo : public Command
{
public:
    QueryDeviceInfo(std::string hostname, unsigned int device, std::string attributes);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Query queryObj;
    unsigned int deviceNum;
    std::string attributes;
};

/**
 * Query info invoker class
 */
class QueryGroupInfo : public Command
{
public:
    QueryGroupInfo(std::string hostname, unsigned int device, std::string attributes, bool verbose);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Query queryObj;
    unsigned int groupNum;
    std::string attributes;
    bool verbose;
};

/**
 * Query device list invoker class
 */
class QueryDeviceList : public Command
{
public:
    QueryDeviceList(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Query queryObj;
};

class QueryHierarchyInfo : public Command
{
public:
    explicit QueryHierarchyInfo(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Query m_queryObj;
};


void TopologicalSort(dcgmMigHierarchy_v2 &hierarchy);
std::string FormatMigHierarchy(dcgmMigHierarchy_v2 &hierarchy);

#endif /* QUERY_H_ */
