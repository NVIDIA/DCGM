/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "MnDiagMpiRunner.h"

#include <string>
#include <string_view>
namespace DcgmNs::Mndiag::Nvbandwidth
{

/**
 * Populate an mnnvbandwidth diagnostic response from captured process output.
 *
 * The response is cleared and then populated from nvbandwidth JSON or recoverable text errors.
 *
 * @param[in] stdoutOutput Captured standard output from mnnvbandwidth.
 * @param[in] stderrOutput Captured standard error from mnnvbandwidth.
 * @param[out] response Response structure to initialize and populate.
 * @param[in] nodeInfo Per-host DCGM and driver version information.
 */
void PopulateResponseFromOutput(std::string_view stdoutOutput,
                                std::string_view stderrOutput,
                                dcgmMnDiagResponse_v2 &response,
                                nodeInfoMap_t const &nodeInfo);

} // namespace DcgmNs::Mndiag::Nvbandwidth

class MnDiagMpiNvbandwidthRunner : public MnDiagMpiRunner
{
public:
    MnDiagMpiNvbandwidthRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid);
    ~MnDiagMpiNvbandwidthRunner() override = default;

    /**
     * Resolve the mnnvbandwidth test binary path.
     *
     * @param[out] path Resolved binary path on success.
     * @return `DCGM_ST_OK` on success.
     * @return `DCGM_ST_BADPARAM` when an override or resolved path exceeds the supported length.
     * @return `DCGM_ST_NO_DATA` when no compatible default binary path can be resolved.
     */
    dcgmReturn_t GetTestBinaryPath(std::string &path) const override;

    /**
     * Populate a diagnostic response from the captured mnnvbandwidth output files.
     *
     * @param[in,out] responseStruct Pointer to a version 2 response that is cleared and populated on success.
     * @param[in] nodeInfo Per-host DCGM and driver version information.
     * @return `DCGM_ST_OK` when the response was populated.
     * @return `DCGM_ST_BADPARAM` when responseStruct is null or has an unsupported version.
     */
    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo) override;

    /** Execute the complete GPU-pair plan within the supplied overall timeout. */
    std::expected<void, dcgmReturn_t> RunBatch(dcgmRunMnDiag_t const &params,
                                               nodeInfoMap_t const &nodeInfo,
                                               dcgmMnDiagResponse_t &response,
                                               std::chrono::milliseconds timeout) override;

    /**
     * Get the parameter prefix accepted by this runner.
     *
     * @return The `mnnvbandwidth.` parameter prefix.
     */
    std::string_view GetTestPrefix() const override
    {
        return "mnnvbandwidth.";
    }

    /**
     * Get the prefix used for mnnvbandwidth log files.
     *
     * @return The mnnvbandwidth log file prefix.
     */
    std::string_view GetLogFilePrefix() const override
    {
        return "mndiag_mnnvbandwidth";
    }

    /**
     * Get the default mnnvbandwidth command-line parameters.
     *
     * @return A parameter map that selects the required multinode copy tests.
     */
    std::unordered_map<std::string, std::string> GetDefaultParametersMap() const override
    {
        return { { "testcase",
                   "multinode_device_to_device_memcpy_write_ce multinode_device_to_device_memcpy_read_ce" } };
    }

protected:
    void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) override;

    std::expected<void, dcgmReturn_t> UpdateParamsMapFromRunParameter(
        std::unordered_map<std::string, std::string> &paramsMap,
        std::string_view paramString) const override;

    bool IsMultiTokenParameter(std::string_view key) const override
    {
        return key == "testcase" || key == "testcasePrefixes";
    }

    bool ShouldSetNetLinkOrder() const override
    {
        return false;
    }

    void AdjustParamsMap(std::unordered_map<std::string, std::string> &paramsMap,
                         unsigned int processCount) const override;

private:
    std::expected<std::string, dcgmReturn_t> ResolveTestBinaryPath() const;
    int m_cudaVersion = 0;
};
