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
#pragma once

#include <Plugin.h>
#include <PluginCommon.h>
#include <PluginStrings.h>

static constexpr auto NCCL_TESTS_DESCRIPTION                    = "This plugin will run the NCCL tests.";
static constexpr auto NCCL_TESTS_EXECUTABLE                     = "all_reduce_perf";
static constexpr auto DCGM_NCCL_TESTS_BIN_PATH_ENV              = "DCGM_NCCL_TESTS_BIN_PATH";
static constexpr auto DCGM_NCCL_TESTS_SKIP_BIN_PERMISSION_CHECK = "DCGM_NCCL_TESTS_SKIP_BIN_PERMISSION_CHECK";

namespace DcgmNs::Nvvs::Plugins::NcclTests
{

class NcclTestsPlugin : public Plugin
{
public:
    NcclTestsPlugin(dcgmHandle_t handle);
    ~NcclTestsPlugin();

    /*************************************************************************/
    /*
     * Run NcclTests Plugin
     */
    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    dcgmReturn_t Shutdown();

    friend class TestNcclTestsPlugin;

private:
    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object.
     */
    void Cleanup();

    std::string GetNcclTestsTestName() const;

    /*************************************************************************/
    TestParameters m_testParameters; /* Parameters for this test, passed in from the framework.
                                        Set when the go() method is called. DO NOT FREE */
    DcgmRecorder m_dcgmRecorder;
    bool m_dcgmRecorderInitialized { false };
    dcgmHandle_t m_handle { 0 };
    unsigned int m_cudaDriverMajorVersion;       /* Cuda driver major version */
    unsigned int m_cudaDriverMinorVersion { 0 }; /* Cuda driver minor version */
    std::string m_ncclTestsExecutable { NCCL_TESTS_EXECUTABLE };
};

} // namespace DcgmNs::Nvvs::Plugins::NcclTests
