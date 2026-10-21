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


/*
Common helper functions and classes relating to DCGM Multi Node GPU Diagnostics
*/

#include "dcgm_mndiag_structs.hpp"
#include <DcgmUtilities.h>
#include <dcgm_structs.h>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

/** Per-host version metadata collected from each hostengine. */
typedef struct
{
    /** GPU identity discovered from one multinode host. */
    struct GpuInfo
    {
        unsigned int gpuId {}; //!< DCGM GPU ID on the host.
        std::string uuid {};   //!< GPU UUID.
    };

    std::string dcgmVersion {};   //!< DCGM version reported by the host.
    std::string driverVersion {}; //!< NVIDIA driver version reported by the host.
    std::vector<GpuInfo> gpus {}; //!< Active GPUs reported by the host.
    std::string mpiNodeName {};   //!< Node identifier stored in Open MPI rank assignments.
} nodeInfo_t;

/** Map from hostname (or IP) to per-host version metadata. */
typedef std::unordered_map<std::string, nodeInfo_t> nodeInfoMap_t;

/** Validate that hostList and runValue are non-empty; throws on failure.
 *
 * @param[in] hostListVec  Host list entries from --hostList.
 * @param[in] runValue     Test name from --run.
 */
void ValidateParameters(std::vector<std::string> const &hostListVec, std::string_view runValue);

/** Populate a dcgmRunMnDiag_v2 struct from parsed CLI arguments.
 *
 * @param[in,out] mndrd       The struct to populate.
 * @param[in]     hostList    Host list entries.
 * @param[in]     parameters  Semicolon-separated test parameters.
 * @param[in]     runValue    Test name.
 * @return DCGM_ST_OK on success.
 */
dcgmReturn_t PopulateRunMnDiag(dcgmRunMnDiag_v2 &mndrd,
                               std::vector<std::string> const &hostList,
                               std::string_view parameters,
                               std::string_view runValue);

/** Resolve the path of the currently running executable.
 *
 * @return The executable path on success, or an unexpected `std::system_error` when `/proc/self/exe` cannot be read
 *         or its path buffer cannot be expanded.
 */
std::expected<std::filesystem::path, std::system_error> GetExecutablePath();

/** Construct the path to an mndiag binary compatible with the CUDA driver version.
 *
 * @param[in] binaryName The binary filename to append to the selected CUDA plugin directory.
 * @param[in] cudaVersion The CUDA driver version encoded as reported by NVML.
 * @return The binary path on success. Returns an unexpected `std::system_error` with `ENOENT` when no compatible
 *         CUDA version exists, or the error from resolving the current executable path.
 */
std::expected<std::string, std::system_error> GetTestBinaryPath(std::string_view binaryName, int cudaVersion);

/** Resolve the default path for an mndiag test binary for a CUDA driver version.
 *
 * @param[in] testName Human-readable test name for log messages.
 * @param[in] binaryName Default binary filename.
 * @param[in] cudaVersion The CUDA driver version encoded as reported by NVML.
 * @return The resolved binary path on success, or `DCGM_ST_NO_DATA` when a compatible path cannot be resolved.
 */
DcgmResult<std::string> InferTestDefaultPath(std::string_view testName, std::string_view binaryName, int cudaVersion);

/**
 * Resolve a test binary path by checking an environment variable override first,
 * then falling back to the default path derived from the CUDA version.
 *
 * @param[in] testName     Human-readable test name for log messages (e.g. "nvloom").
 * @param[in] envVarName   Environment variable that may override the binary path.
 * @param[in] binaryName   Default binary filename (e.g. "nvloom_cli").
 * @param[in] cudaVersion  CUDA driver version used to select the cuda plugin directory.
 * @return The resolved binary path on success, or a dcgmReturn_t error code.
 */
DcgmResult<std::string> ResolveTestBinaryPath(std::string_view testName,
                                              std::string_view envVarName,
                                              std::string_view binaryName,
                                              int cudaVersion);

/** Determine whether mnnvbandwidth was requested in all-GPU pair mode.
 *
 * @param[in] params The diagnostic run parameters to inspect.
 * @return `true` when the parameters contain `mnnvbandwidth.gpuPairs=all`; otherwise `false`.
 */
bool IsGpuPairMode(dcgmRunMnDiag_t const &params);

/**
 * @brief Parse the time_to_run value from a testParms array.
 *
 * Scans params.testParms for an entry whose text is exactly `key=<value>`,
 * anchored at the start, and returns the validated integer value in seconds.
 * Semicolons are treated as delimiters: only the text between '=' and the
 * first ';' (if any) is interpreted as the value.
 *
 * @param[in] params  The run parameters struct whose testParms array is scanned.
 * @param[in] key     Full parameter key to match (e.g. "mnubergemm.time_to_run").
 * @return On success: 0 if the key is absent, or a positive integer (seconds) if found and valid.
 *         On error: std::unexpected(DCGM_ST_BADPARAM) if the entry is
 *         present but malformed or has a non-positive value.
 */
std::expected<int, dcgmReturn_t> ParseTimeToRunSeconds(dcgmRunMnDiag_t const &params, std::string_view key);

/** Resolve a hostname to an IPv4 address string via @c getaddrinfo.
 *
 * If @p host is already a numeric IP address it is returned unchanged
 * (detected via @c AI_NUMERICHOST, no DNS call). If DNS resolution fails,
 * the original @p host string is returned unmodified.
 *
 * @param[in] host  The hostname or numeric IP address to resolve.
 * @return The resolved IPv4 address as a dotted-decimal string. If the host
 *         is already a numeric IP it is returned unchanged. If DNS resolution
 *         fails, the original @p host string is returned unmodified.
 */
std::string ResolveHostnameToIp(std::string const &host);
