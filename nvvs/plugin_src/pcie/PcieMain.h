/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
int main_init(BusGrind &bg,
              const dcgmDiagPluginEntityList_v1 &entityInfo,
              DcgmNs::DcgmOpsAnyPtr dcgmOps,
              dcgmHandle_t handle,
              CudaOpsAnyPtr cudaOps);

/*****************************************************************************/
int main_entry(BusGrind *bg,
               const dcgmDiagPluginEntityList_v1 &entityInfo,
               DcgmNs::DcgmOpsAnyPtr dcgmOps,
               dcgmHandle_t handle,
               CudaOpsAnyPtr cudaOps);

/*****************************************************************************/
bool pcie_gpu_id_in_list(unsigned int gpuId, const dcgmDiagPluginEntityList_v1 &entityInfo);

/**
 * Check whether the global PCIe stop flag is set.
 *
 * When the stop flag is set, records a DCGM_FR_ABORTED error and marks the PCIe test as skipped.
 *
 * @param[in,out] bg BusGrind context used to record the abort error and test result
 *
 * @return 0 if the stop flag is clear; 1 if the stop flag is set and the test was skipped
 */
int bg_should_stop(BusGrind *bg);

/**
 * Add a formatted latency measurement to a GPU's verbose test information.
 *
 * @param[in,out] bg      BusGrind context used to record the verbose information
 * @param[in]     gpu     GPU entity ID associated with the measurement
 * @param[in]     key     Transfer direction: "bidir", "d2h", or "h2d"
 * @param[in]     latency Latency in microseconds
 */
void addLatencyInfo(BusGrind *bg, unsigned int gpu, std::string key, double latency);

/**
 * Add a formatted bandwidth measurement to a GPU's verbose test information.
 *
 * @param[in,out] bg        BusGrind context used to record the verbose information
 * @param[in]     gpu       GPU entity ID associated with the measurement
 * @param[in]     key       Transfer direction: "bidir", "d2h", or "h2d"
 * @param[in]     bandwidth Bandwidth in gigabytes per second
 */
void addBandwidthInfo(BusGrind &bg, unsigned int gpu, const std::string &key, double bandwidth);

/*****************************************************************************/
void pcie_check_nvlink_status(BusGrind *bg,
                              const dcgmDiagPluginEntityList_v1 &entityInfo,
                              DcgmNs::DcgmOpsAnyPtr dcgmOps,
                              dcgmHandle_t handle);

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
                                               std::vector<unsigned int> const &gpuIds,
                                               DcgmNs::DcgmOpsAnyPtr dcgmOps,
                                               dcgmHandle_t handle);

std::string_view ExtractJson(std::string_view output);

/**
 * Output a pairs P2P bandwidth matrix for concurrent GPU transfers
 *
 * Computes and records bandwidth measurements for P2P transfers between GPU pairs,
 * with optional P2P enablement. Skips the test if fewer than 2 GPUs are available.
 *
 * @param[in] bg  BusGrind context containing GPUs, test parameters, and logging helpers
 * @param[in] cudaOps CUDA API interface
 * @param[in] p2p     Whether P2P is enabled for the test
 *
 * @return 0 when the subtest is skipped or completes successfully
 */
int outputConcurrentPairsP2PBandwidthMatrix(BusGrind *bg, CudaOpsAnyPtr cudaOps, bool p2p);

/**
 * Output a 1D exchange bandwidth matrix for concurrent neighbor GPU transfers
 *
 * Computes and records bandwidth measurements for 1D exchange transfers across
 * adjacent GPUs, with optional P2P enablement. Skips the test if fewer than 2 GPUs
 * are available.
 *
 * @param[in] bg  BusGrind context containing GPUs, test parameters, and logging helpers
 * @param[in] cudaOps CUDA API interface
 * @param[in] p2p     Whether P2P is enabled for the test
 *
 * @return 0 when the subtest is skipped or completes successfully
 */
int outputConcurrent1DExchangeBandwidthMatrix(BusGrind *bg, CudaOpsAnyPtr cudaOps, bool p2p);

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

/**
 * Check host-device PCIe bandwidth for all GPUs against the configured minimum
 *
 * Compares each GPU's measured host-device bandwidth (d2h, h2d, bidirectional)
 * against the configured minimum, recording the bandwidth as a stat and flagging
 * any GPU whose bandwidth falls below the threshold with a DCGM_FR_LOW_BANDWIDTH error.
 *
 * @param[in,out] bg              BusGrind context containing GPUs, stats, and error reporting
 * @param[in]     groupName       Subtest name used to read the min-bandwidth parameter and record stats
 * @param[in]     bandwidthMatrix Measured bandwidths, laid out as [labelIndex * gpuCount + gpuIndex]
 * @param[in]     pinned          Whether the bandwidths were measured with pinned host memory
 *
 * @return the number of GPUs that fell below the minimum bandwidth
 */
int pcie_check_bandwidth(BusGrind &bg,
                         std::string const &groupName,
                         std::vector<double> const &bandwidthMatrix,
                         bool pinned);

/**
 * Check concurrent host-device PCIe bandwidth for all GPUs against the configured minimum
 *
 * Compares each GPU's measured concurrent host-device bandwidth (h2d, d2h, bidirectional)
 * against the configured minimum, recording each bandwidth and the per-direction sum as
 * stats and flagging any GPU whose bandwidth falls below the threshold with a
 * DCGM_FR_LOW_BANDWIDTH error (one per direction below the minimum).
 *
 * @param[in,out] bg              BusGrind context containing GPUs, stats, and error reporting
 * @param[in]     groupName       Subtest name used to read the min-bandwidth parameter and record stats
 * @param[in]     bandwidthMatrix Measured bandwidths, laid out as [labelIndex * gpuCount + gpuIndex]
 *
 * @return the number of (GPU, direction) measurements that fell below the minimum bandwidth
 */
int pcie_check_concurrent_bandwidth(BusGrind &bg,
                                    std::string const &groupName,
                                    std::vector<double> const &bandwidthMatrix);

/**
 * Check host-device PCIe latency for all GPUs against the configured maximum
 *
 * Compares each GPU's measured host-device latency (d2h, h2d, bidirectional)
 * against the configured maximum, recording the latency as a stat and flagging
 * any GPU whose latency exceeds the threshold with a DCGM_FR_HIGH_LATENCY error.
 *
 * @param[in,out] bg            BusGrind context containing GPUs, stats, and error reporting
 * @param[in]     groupName     Subtest name used to read the max-latency parameter and record stats
 * @param[in]     latencyMatrix Measured latencies, laid out as [labelIndex * gpuCount + gpuIndex]
 * @param[in]     pinned        Whether the latencies were measured with pinned host memory
 *
 * @return the number of GPUs that exceeded the maximum latency
 */
int pcie_check_latency(BusGrind &bg,
                       std::string const &groupName,
                       std::vector<double> const &latencyMatrix,
                       bool pinned);

#endif // PCIEMAIN_H
