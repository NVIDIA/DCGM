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

#ifndef __BROKENP2P_H__
#define __BROKENP2P_H__

#include <vector>

#include <NvvsCommon.h>
#include <PluginDevice.h>

#include "Pcie.h"

struct TestTuple
{
    int d1;
    int d2;
    size_t size;
    bool success;
};

enum class P2pLink : std::uint8_t
{
    NvLink,
    Pcie,
    NoInfo
};

/*
 * Gets the P2P dcgmError_t codes based on the NVLINK_TX_BYTES field summary max value.
 * A zero or blank field summary value indicates that NVLink was not used during the test.
 *
 * @param bg - the plugin this subtest is a part of
 * @param startTime - the start time for the field summary request
 * @param gpuId - the GPU corresponding to the field summary request
 * @param memoryError - memory error code to be filled based on the field summary value
 * @param writerError - writer error code to be filled based on the field summary value
 */
void GetP2PError(BusGrind *bg,
                 timelib64_t const &startTime,
                 unsigned int const &gpuId,
                 dcgmError_t &memoryError,
                 dcgmError_t &writerError);

class Brokenp2p
{
public:
    /*****************************************************************************/
    /*
     * Initializes the subtest. It needs to have a pointer to the plugin, know the GPUs we
     * are testing, and the maximum size of the writes we are performing.
     *
     * @param bg   - the plugin this subtest is a part of
     * @param gpus - information about the GPUs in this test
     * @param size - the largest write that we are performing across the devices
     */
    Brokenp2p(BusGrind *bg, std::vector<PluginDevice *> &gpus, size_t size);

    /*****************************************************************************/
    /*
     * Runs the entire subtest; specifically, performs p2p writes for the different permutations
     * of devices and makes sure they are working correctly.
     * If there is an error, the details are recorded on the plugin
     *
     * @return NVVS_RESULT_FAIL if something went wrong, or NVVS_RESULT_PASS on success
     */
    nvvsPluginResult_t RunTest();

    /*****************************************************************************/
    /*
     * Performs basic checks to see if p2p writing is working between the two specified devices
     *
     * @param cudaId1 - the CUDA runtime id of device 1 (destination)
     * @param cudaId2 - the CUDA runtime id of device 2 (writer)
     * @param errstr  - populated with an error from the plugin if an error occurred.
     *
     * @retrun false if there was an error, true otherwise
     */
    bool CheckPairP2pWindow(int cudaId1, int cudaId2, std::string &errStr);

    /*****************************************************************************/
    /*
     * Resetse the specified cuda devices as part of cleanup
     *
     * @param cudaId1 - the CUDA runtime id of device 1 (destination)
     * @param cudaId2 - the CUDA runtime id of device 2 (writer)
     */
    void ResetCudaDevices(int cudaId1, int cudaId2);

private:
    /*****************************************************************************/
    /*
     * Copies the buffer from device to to device 1, then back to the host, and compares that result to the expected
     * buffer. Writes an error to error string if the buffer isn't as expected.
     *
     * NOTE: all buffers must have the specified size or greater or else this will segfault
     *
     * @param bufferd1 - the buffer on the destination device
     * @param cudaId1  - the cuda runtime id of device 1 (destination)
     * @param bufferd2 - the buffer on the source device
     * @param cudaId2  - the CUDA runtime id of device 2 (writer)
     * @param hostBuf  - the destination host buffer
     * @param expected - a buffer containing the expected results
     * @param size     - the size of the copy to be performed
     * @param errstr   - a string where any applicable error should be written.
     *
     * @retrun false if there was an error, true otherwise
     */
    bool PerformP2PWrite(void *bufferd1,
                         int cudaId1,
                         void *bufferd2,
                         int cudaId2,
                         void *hostBuf,
                         void *expected,
                         size_t size,
                         std::string &errStr);

    BusGrind *m_bg;
    std::vector<PluginDevice *> m_gpus;
    size_t m_size;
    CUfunction m_func;
};

#endif
