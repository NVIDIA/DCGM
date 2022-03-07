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
#ifndef NVLINK_H_
#define NVLINK_H_

#include "Command.h"


class Nvlink
{
public:
    Nvlink();
    virtual ~Nvlink();

    /***********************************************************************************
     * This method is used to display the NvLink error counter values for all the lanes
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkErrorCountsForGpu(dcgmHandle_t dcgmHandle, unsigned int gpuId, bool json);

    /***********************************************************************************
     * This method is used to display the link statuses for the GPUs and NvSwitches in the system
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkLinkStatus(dcgmHandle_t dcgmHandle);

private:
    /****************************************************************************************
     * This method is used to convert the nvlink error count fieldIds to the error type string
     *****************************************************************************************/
    std::string HelperGetNvlinkErrorCountType(unsigned short fieldId);
};

/*****************************************************************************
 * Get NvLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info invoker class
 */
class GetGpuNvlinkErrorCounts : public Command
{
public:
    GetGpuNvlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Nvlink mNvlinkObj;
    unsigned int mGpuId;
};

/*****************************************************************************
 * Get NvLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info invoker class
 */
class GetNvLinkLinkStatuses : public Command
{
public:
    GetNvLinkLinkStatuses(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Nvlink mNvlinkObj;
};


#endif /* NVLINK_H_ */
