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
#ifndef PCIEMAIN_H
#define PCIEMAIN_H

#include "Pcie.h"

#include <PluginDevice.h>
#include <string_view>

#define PCIE_MAX_GPUS 16

/*****************************************************************************/
int main_init(BusGrind &bg, const dcgmDiagPluginEntityList_v1 &entityInfo);

/*****************************************************************************/
int main_entry(BusGrind *bg, const dcgmDiagPluginEntityList_v1 &entityInfo);

/*****************************************************************************/
bool pcie_gpu_id_in_list(unsigned int gpuId, const dcgmDiagPluginEntityList_v1 &entityInfo);

/*****************************************************************************/
void pcie_check_nvlink_status(BusGrind *bg, const dcgmDiagPluginEntityList_v1 &entityInfo);

typedef struct
{
    pid_t pid;
    std::string stdoutStr;
    std::string stderrStr;
    int readOutputRet;
    int readErrorRet;
    unsigned int outputFdIndex;
} dcgmChildInfo_t;

/*****************************************************************************/
// Declared here for unit tests
unsigned int ProcessChildrenOutputs(std::vector<dcgmChildInfo_t> &childrenInfo,
                                    BusGrind &bg,
                                    const std::string &groupName);

std::unique_ptr<DcgmGroup> StartDcgmGroupWatch(BusGrind *bg,
                                               std::vector<unsigned short> const &fieldIds,
                                               std::vector<unsigned int> const &gpuIds);

std::string_view ExtractJson(std::string_view output);

/**
 * Output a pairs P2P bandwidth matrix for concurrent GPU transfers
 *
 * Computes and records bandwidth measurements for P2P transfers between GPU pairs,
 * with optional P2P enablement. Skips the test if fewer than 2 GPUs are available.
 *
 * @param[in] bg  BusGrind context containing GPUs, test parameters, and logging helpers
 * @param[in] p2p Whether P2P is enabled for the test
 *
 * @return 0 when the subtest is skipped or completes successfully
 */
int outputConcurrentPairsP2PBandwidthMatrix(BusGrind *bg, bool p2p);

/**
 * Output a 1D exchange bandwidth matrix for concurrent neighbor GPU transfers
 *
 * Computes and records bandwidth measurements for 1D exchange transfers across
 * adjacent GPUs, with optional P2P enablement. Skips the test if fewer than 2 GPUs
 * are available.
 *
 * @param[in] bg  BusGrind context containing GPUs, test parameters, and logging helpers
 * @param[in] p2p Whether P2P is enabled for the test
 *
 * @return 0 when the subtest is skipped or completes successfully
 */
int outputConcurrent1DExchangeBandwidthMatrix(BusGrind *bg, bool p2p);

/**
 * Check PCIe link width and generation for all GPUs
 *
 * Verifies that each GPU's PCIe link width and generation meet the configured
 * minimum thresholds. Skips GPUs not in P-state 0.
 *
 * @param[in,out] bg      BusGrind context containing GPUs, recorder, and error reporting
 * @param[in]     subTest Name of the subtest to read threshold parameters from
 *
 * @return 0 if all GPUs pass, 1 if any GPU fails
 */
int bg_check_pci_link(BusGrind &bg, std::string subTest);
#endif // PCIEMAIN_H
