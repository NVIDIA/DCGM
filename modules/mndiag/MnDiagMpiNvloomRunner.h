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

#include <set>
#include <string>
#include <string_view>

class MnDiagMpiNvloomRunner : public MnDiagMpiRunner
{
public:
    MnDiagMpiNvloomRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid);
    ~MnDiagMpiNvloomRunner() override = default;

    /** Construct the MPI command, adding --merge-stderr-to-stdout.
     *
     * nvloom_cli writes errors to stderr. This flag ensures they appear
     * in the merged stdout stream that the parser reads.
     *
     * @param[in] params Parameters used to construct the MPI command.
     * @return Empty on success.
     * @return `DCGM_ST_BADPARAM` when the input cannot form a valid MPI command.
     */
    std::expected<void, dcgmReturn_t> ConstructMpiCommand(void const *params) override;

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
        return "mndiag_nvloom";
    }

    /**
     * Get the test prefix
     *
     * @return std::string_view The test prefix
     */
    std::string_view GetTestPrefix() const override
    {
        return "nvloom.";
    }

    /**
     * Get the default parameters map for nvloom
     *
     * @return std::unordered_map<std::string, std::string> Map of parameter name to value
     */
    std::unordered_map<std::string, std::string> GetDefaultParametersMap() const override;

    /** Get the set of recognised nvloom parameter names (after prefix stripping).
     *
     * Used by AdjustParamsMap to reject unknown user-supplied parameters.
     *
     * @return An immutable reference to the allowed parameter name set.
     */
    static std::set<std::string> const &AllowedParamNames();

protected:
    /**
     * Adjust the merged parameter map for nvloom_cli.
     *
     * Renames DCGM-style keys to nvloom_cli CLI names and validates that
     * every key is a recognised nvloom parameter.
     *
     * @param[in,out] paramsMap     The merged parameter map.
     * @param[in]     processCount  Total MPI process count (unused by nvloom).
     */
    void AdjustParamsMap(std::unordered_map<std::string, std::string> &paramsMap,
                         unsigned int processCount) const override;

    /**
     * Parse the test output
     *
     * @param[in] fd The file descriptor to parse
     * @param[in] responseStruct The response structure to populate
     * @param[in] nodeInfo The node info map used to populate the response structure
     */
    void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) override;

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
