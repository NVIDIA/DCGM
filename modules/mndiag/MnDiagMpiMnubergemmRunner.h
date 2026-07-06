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

class MnDiagMpiMnubergemmRunner : public MnDiagMpiRunner
{
public:
    MnDiagMpiMnubergemmRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid);
    ~MnDiagMpiMnubergemmRunner() override = default;

    /**
     * Get the path to the test binary
     *
     * @param[out] path The path to the test binary
     * @return DCGM_ST_OK on success
     */
    dcgmReturn_t GetTestBinaryPath(std::string &path) const override;

    /**
     * Get the log file prefix
     *
     * @return std::string_view The log file prefix
     */
    std::string_view GetLogFilePrefix() const override
    {
        return "mndiag_mnubergemm";
    }

protected:
    /**
     * Parse the test output
     *
     * @param[in] fd The file descriptor to parse
     * @param[in] responseStruct The response structure to populate
     * @param[in] nodeInfo The node info map used to populate the response structure
     */
    void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) override;

    /**
     * Get the test prefix
     *
     * @return std::string_view The test prefix
     */
    std::string_view GetTestPrefix() const override
    {
        return "mnubergemm.";
    }

    /**
     * Get the default parameters map for mnubergemm
     *
     * @return std::unordered_map<std::string, std::string> Map of parameter name to value
     */
    std::unordered_map<std::string, std::string> GetDefaultParametersMap() const override
    {
        // Default workload parameters for mnubergemm (10kHz FP32 pulse configuration)
        std::unordered_map<std::string, std::string> params = { { "workload", "GC" },
                                                                { "time_to_run", "3600" },
                                                                { "max_workload", "65536" },
                                                                { "MM_max_workload", "65536" },
                                                                { "MM_sm_count", "144" },
                                                                { "CE_type", "H" },
                                                                { "MM_N", "0" },
                                                                { "CE_size", "200000" },
                                                                { "MM_type", "ST_ST_SSS" },
                                                                { "MM_M_per_sm", "32" },
                                                                { "freq", "10000" },
                                                                { "duty", "0.5" },
                                                                { "dynamic_adj", "" } }; // Flag (no value)

        return params;
    }

private:
    /**
     * Resolve the path to the test binary from an environment variable
     *
     * @return std::expected<std::string, dcgmReturn_t> The path to the test binary on success, or an error code on
     * failure
     */
    std::expected<std::string, dcgmReturn_t> ResolveTestBinaryPath() const;
    int m_cudaVersion = 0;
};
