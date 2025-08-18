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

#ifndef MNDIAG_MPI_RUNNER_H
#define MNDIAG_MPI_RUNNER_H

#include "MpiRunner.h"
#include "dcgm_mndiag_structs.hpp"
#include <chrono>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Specialized MPI runner for mnubergemm diagnostics
 */
class MnDiagMpiRunner : public MpiRunner
{
public:
    /**
     * @brief Constructor
     */
    MnDiagMpiRunner(DcgmCoreProxyBase &coreProxy)
        : MpiRunner(coreProxy)
        , m_mnubergemmPath(MnDiagConstants::DEFAULT_MNUBERGEMM_PATH)
    {}

    /**
     * @brief Destructor
     */
    virtual ~MnDiagMpiRunner() = default;

    /**
     * @brief Custom output callback handler for mnubergemm diagnostics
     *
     * This method processes output from the MPI process and populates
     * a dcgmMnDiagResponse_t structure with the results based on the version
     *
     * @param dataStream The stream to parse
     * @param responseStruct Pointer to a dcgmMnDiagResponse_t structure to be updated
     * @param nodeInfo The node info map used to populate the response structure
     */
    dcgmReturn_t MnDiagOutputCallback(std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo);

    /**
     * @brief Construct an MPI command from input parameters and store it internally
     *
     * This method expects a pointer to dcgmRunMnDiag_t.
     *
     * @param params Pointer to dcgmRunMnDiag_t
     */
    void ConstructMpiCommand(void const *params) override;

    /**
     * @brief Check if MPI has launched enough processes
     *
     * This method monitors the MPI process output to determine if the expected
     * number of processes have been launched successfully. It handles cases where
     * the data stream producer is slow or intermittent.
     *
     * @return std::expected<bool, dcgmReturn_t> True if enough processes launched, error code on failure
     */
    std::expected<bool, dcgmReturn_t> HasMpiLaunchedEnoughProcesses();

    /**
     * @brief Set the mnubergemm path
     *
     * @param mnubergemmPath The path to the mnubergemm binary
     */
    void SetMnubergemmPath(std::string const &mnubergemmPath);

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
     * @brief Parse MPI output and populate dcgmMnDiagResponse_v1 structure
     *
     * @param dataStream The stream to parse
     * @param responseStruct Pointer to the response struct to populate
     * @param nodeInfo The node info map used to populate the response structure
     */
    void ParseMnUberGemmOutput_v1(std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo);

    /**
     * @brief Get the path to the mnubergemm binary
     *
     * @return std::string The path to the mnubergemm binary
     */
    std::string GetMnubergemmBinPath() const;

    unsigned int m_totalProcessCount { 0 };
    std::string m_mnubergemmPath;

    friend class MnDiagMpiRunnerTests;
};

#endif // MNDIAG_MPI_RUNNER_H