/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef PCIEMAIN_H
#define PCIEMAIN_H

#include "DcgmRecorder.h"
#include "NvvsDeviceList.h"
#include "Pcie.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include "cuda_runtime.h"
#include <PluginDevice.h>

#define PCIE_MAX_GPUS 16

/*****************************************************************************/
int main_init(BusGrind &bg, const dcgmDiagPluginGpuList_t &gpuInfo);

/*****************************************************************************/
int main_entry(BusGrind *bg, const dcgmDiagPluginGpuList_t &gpuInfo);

/*****************************************************************************/
bool pcie_gpu_id_in_list(unsigned int gpuId, const dcgmDiagPluginGpuList_t &gpuInfo);

/*****************************************************************************/
void pcie_check_nvlink_status(BusGrind *bg, const dcgmDiagPluginGpuList_t &gpuInfo);

typedef struct
{
    pid_t pid;
    std::string stdoutStr;
    int readOutputRet;
    unsigned int outputFdIndex;
} dcgmChildInfo_t;

/*****************************************************************************/
// Declared here for unit tests
unsigned int ProcessChildrenOutputs(std::vector<dcgmChildInfo_t> &childrenInfo,
                                    BusGrind &bg,
                                    const std::string &groupName);

#endif // PCIEMAIN_H
