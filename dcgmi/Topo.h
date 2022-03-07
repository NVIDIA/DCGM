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
 * Topo.h
 *
 *  Created on: Dec 10, 2015
 *      Author: chris
 */

#ifndef TOPO_H_
#define TOPO_H_

#include "Command.h"

class Topo
{
public:
    Topo()          = default;
    virtual ~Topo() = default;

    /*****************************************************************************
     * This method is used to display the GPU topology on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGPUTopology(dcgmHandle_t mNvcmHandle, unsigned int requestedGPUId, bool json);

    /*****************************************************************************
     * This method is used to display the Group topology on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisplayGroupTopology(dcgmHandle_t mNvcmHandle, dcgmGpuGrp_t requestedGroupId, bool json);


    static std::string HelperGetAffinity(unsigned long const *cpuAffinity);
    std::string HelperGetPciPath(dcgmGpuTopologyLevel_t &path);
    std::string HelperGetNvLinkPath(dcgmGpuTopologyLevel_t &path, unsigned int linkMask);

private:
    /*****************************************************************************
     * Helper method to acquire a list of All GPUs on the system
     *****************************************************************************/
};

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Query info invoker class
 */
class GetGPUTopo : public Command
{
public:
    GetGPUTopo(std::string hostname, unsigned int gpuId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Topo topoObj;
    unsigned int mGpuId;
};


/**
 * Query info invoker class
 */
class GetGroupTopo : public Command
{
public:
    GetGroupTopo(std::string hostname, unsigned int groupId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Topo topoObj;
    dcgmGpuGrp_t mGroupId;
};


#endif /* TOPO_H_ */
