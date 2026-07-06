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

#ifndef MNDIAG_MPI_RUNNER_H
#define MNDIAG_MPI_RUNNER_H

#include "MpiRunner.h"
#include "dcgm_mndiag_structs.hpp"
#include <chrono>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace DcgmNs::Common::ProcessUtils
{
class CommandExecutor;
} // namespace DcgmNs::Common::ProcessUtils

/**
 * @brief Base MPI runner for mndiag diagnostics. Subclasses implement test-specific behavior.
 */
class MnDiagMpiRunner : public MpiRunner
{
public:
    /**
     * @brief Constructor
     *
     * @param coreProxy The core proxy to use for the ChildProcess management
     * @param effectiveUid The effective UID of the caller
     */
    MnDiagMpiRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid)
        : MpiRunner(coreProxy, effectiveUid)
    {}

    /**
     * @brief Destructor
     */
    virtual ~MnDiagMpiRunner() = default;

    /**
     * @brief Type alias for the routing interface resolver function.
     *
     * Takes a list of hostnames/IPs and returns a comma-separated string of
     * network interface names (or subnets) to pass to MPI's btl/oob_tcp_if_include.
     * Defaults to MnDiagMpiRunnerInternal::GetRoutingInterfacesForHosts.
     * Injectable via SetRoutingInterfaceResolver for unit testing.
     */
    using RoutingInterfaceResolver = std::function<std::string(std::vector<std::string> const &)>;

    /**
     * @brief Override the routing interface resolver (for unit testing only).
     *
     * @param resolver  Callable that returns the interface string for a given host list.
     */
    void SetRoutingInterfaceResolver(RoutingInterfaceResolver resolver)
    {
        m_routingInterfaceResolver = std::move(resolver);
    }

    /**
     * @brief Custom output callback handler for mnubergemm diagnostics
     *
     * This method processes output from the MPI process and populates
     * a dcgmMnDiagResponse_t structure with the results based on the version
     *
     * @param fd File descriptor to read MPI process output from
     * @param responseStruct Pointer to a dcgmMnDiagResponse_t structure to be updated
     * @param nodeInfo The node info map used to populate the response structure
     */
    dcgmReturn_t MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo);

    /**
     * @brief Construct an MPI command from input parameters and store it internally
     *
     * This method expects a pointer to dcgmRunMnDiag_t.
     *
     * @param params Pointer to dcgmRunMnDiag_t
     */
    void ConstructMpiCommand(void const *params) override;

    /**
     * @brief Get the path to the test binary
     *
     * @param path Output: the resolved binary path
     * @return DCGM_ST_OK on success
     */
    virtual dcgmReturn_t GetTestBinaryPath(std::string &path) const = 0;

    /**
     * @brief Get the test prefix used to identify test-specific parameters
     *
     * @return std::string_view The prefix (e.g. "mnubergemm.")
     */
    virtual std::string_view GetTestPrefix() const = 0;

    virtual std::string_view GetLogFilePrefix() const = 0;

    /**
     * @brief Get the default parameters map for this test type
     *
     * @return std::unordered_map<std::string, std::string> Map of parameter name to value
     */
    virtual std::unordered_map<std::string, std::string> GetDefaultParametersMap() const = 0;

    virtual void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) = 0;

    virtual std::expected<std::chrono::milliseconds, dcgmReturn_t> GetTestRunTime(dcgmRunMnDiag_t const &params) const;

protected:
    /**
     * @brief Get the path to the mpirun binary specific to mnubergemm
     *
     * @return std::string The path to the mpirun binary for mnubergemm
     */
    std::string GetMpiBinPath() const override;

private:
    /**
     * @brief Parse a dcgmRunMnDiag_ struct into an mpirun command for mnubergemm
     *
     * @param drmnd The dcgmRunMnDiag_v1 struct containing parameters for mnubergemm.
     *              The passthrough testParms are not validated here, same as the current behavior of EUD as of DCGM 4.2
     *              For those with security concerns, we already have the following security assumption built in:
     *              only a certain trusted user is allowed to successfully launch mndiag - the user previously vetted
     *              for remote ssh access
     */
    void ParseDcgmMnDiagToMpiCommand_v1(dcgmRunMnDiag_v1 const &drmnd);

    /**
     * Check that the mpirun binary and ompi_info report the same Open MPI version.
     *
     * Logs a warning when they differ, naming the exact env var to set as the fix.
     * Silently skips when either tool is absent or is not Open MPI.
     *
     * @param[in] mpirunPath Resolved path to the mpirun binary
     * @param[in] executor   Command runner to use; uses SystemCommandExecutor when nullptr.
     *                       Only pass a non-null value in tests to inject a fake executor —
     *                       production callers always omit this parameter.
     * @return false if a version mismatch was detected; true otherwise
     */
    bool CheckMpiVersionConsistency(std::string const &mpirunPath,
                                    DcgmNs::Common::ProcessUtils::CommandExecutor *executor = nullptr) const;

    unsigned int m_totalProcessCount { 0 };
    RoutingInterfaceResolver m_routingInterfaceResolver;

    friend class MnDiagMpiMnubergemmRunnerTests;

protected:
    mutable std::optional<std::expected<std::string, dcgmReturn_t>> m_testBinaryPath; //!< nullopt = not yet resolved
};

#endif // MNDIAG_MPI_RUNNER_H
