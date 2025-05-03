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
#pragma once

#include <DcgmError.h>
#include <DcgmMutex.h>
#include <DcgmRecorder.h>
#include <DcgmUtilities.h>
#include <Plugin.h>
#include <PluginCommon.h>
#include <PluginDevice.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <optional>

namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{
// Forward declaration
struct NVBandwidthResult;

class NVBandwidthPlugin : public Plugin
{
public:
    NVBandwidthPlugin(dcgmHandle_t handle);
    ~NVBandwidthPlugin();

    /*************************************************************************/
    /*
     * Run NVBandwidth Plugin
     *
     */
    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    dcgmReturn_t GetResults(std::string const &testName, dcgmDiagEntityResults_v2 *entityResults) override;

    /*************************************************************************/
    /*
     * Attempt to read the exit code from the nvbandwidth plugin
     *
     * @return true if an error is found, false otherwise
     */
    std::pair<bool, std::optional<NVBandwidthResult>> AttemptToReadOutput(std::string_view output);

    dcgmReturn_t Shutdown();
#ifndef DCGM_NVBANDWIDTH_TEST_TEST
private:
#endif
    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object, including memory and
     * file handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Confirm the location of the NVBandwidth binary
     */
    std::optional<std::string> FindExecutable();

    bool LaunchExecutable(std::string const &testName, std::vector<std::string> const &execArgv);

    /*************************************************************************/
    /*
     * Gets the CUDA driver version from DCGM and saves the major version in m_cudaDriverMajorVersion.
     */
    void SetCudaDriverMajorVersion();

    std::string GetNvBandwidthTestName() const;

    void appendExtraArgv(std::vector<std::string> &execArgv) const;

    /*************************************************************************/
    TestParameters m_testParameters; /* Parameters for this test, passed in from the framework.
                                                           Set when the go() method is called. DO NOT FREE */
    DcgmRecorder m_dcgmRecorder;
    dcgmHandle_t m_handle { 0 };
    bool m_dcgmRecorderInitialized { false };                  /* Has DcgmRecorder been initialized? */
    unsigned int m_cudaDriverMajorVersion;                     /* Cuda driver major version */
    std::unique_ptr<dcgmDiagPluginEntityList_v1> m_entityInfo; // The information about each GPU
    std::string m_nvbandwidthDir;
    std::string m_originalCudaVisibleDevices; // Store the original CUDA_VISIBLE_DEVICES value
    DcgmMutex m_envMutex { 0 };               // Mutex for protecting environment variable operations
};
} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth