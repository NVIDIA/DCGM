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
#include "PluginCommon.h"
#include "DcgmError.h"
#include <DcgmLogging.h>
#include <NvvsCommon.h>
#include <dcgm_fields.h>

#include <dlfcn.h>
#include <filesystem>
#include <sstream>

namespace
{
/*****************************************************************************/
/**
 * @brief Get the executable in path
 *
 * @param path Path to search in
 * @param executableName Name of the executable to find
 * @return std::optional<std::string> Path to executable if found, nullopt otherwise
 */
std::optional<std::string> GetExecutableInPath(std::string_view path, std::string_view executableName)
{
    auto const filePath = std::filesystem::path(path) / executableName;
    std::error_code ec;

    if (std::filesystem::is_regular_file(filePath, ec) && !ec)
    {
        return filePath.string();
    }

    return std::nullopt;
}

/*****************************************************************************/
/**
 * @brief Get the location of the current shared module
 *
 * @return std::string Location of the current shared module or an '.' that represents the current working directory
 */
std::string GetCurrentModuleLocation()
{
    Dl_info info;
    if (0 == dladdr((void *)GetCurrentModuleLocation, &info))
    {
        log_error(
            "Failed to get the location of the current module via dladdr for function {}. Returning '.' as the current module location.",
            __func__);
        return "."; // Just fallback to current working directory
    }
    return std::filesystem::path { info.dli_fname }.parent_path().string();
}

/*****************************************************************************/
/**
 * @brief Get the NVVS bin path
 *
 * @param cudaDriverMajorVersion CUDA driver major version
 * @param updatedPath Updated path
 * @return std::string Path to the NVVS bin path
 */
std::string GetNvvsBinPath(unsigned int cudaDriverMajorVersion, std::string_view updatedPath)
{
    const char *nvvsBinPath = getenv("NVVS_BIN_PATH");
    if (nvvsBinPath != nullptr)
    {
        return fmt::format("{}/plugins/cuda{}{}", nvvsBinPath, cudaDriverMajorVersion, updatedPath);
    }

    return "";
}
} // namespace

/*****************************************************************************/
void CheckAndSetResult(Plugin *p,
                       std::string const &testName,
                       const std::vector<unsigned int> &gpuList,
                       size_t i,
                       bool passed,
                       const std::vector<DcgmError> &errorList,
                       bool &allPassed,
                       bool dcgmCommError)
{
    if (passed)
    {
        p->SetResultForGpu(testName, gpuList[i], NVVS_RESULT_PASS);
    }
    else
    {
        allPassed = false;
        p->SetResultForGpu(testName, gpuList[i], NVVS_RESULT_FAIL);
        for (size_t j = 0; j < errorList.size(); j++)
        {
            DcgmError d { errorList[j] };
            d.SetGpuId(gpuList[i]);
            p->AddError(testName, d);
        }
    }

    if (dcgmCommError)
    {
        for (size_t j = i; j < gpuList.size(); j++)
        {
            p->SetResultForGpu(testName, gpuList[j], NVVS_RESULT_FAIL);
        }
    }
}

/*****************************************************************************/
bool IsSmallFrameBufferModeSet(void)
{
    const char *str = getenv("__DCGM_DIAG_SMALL_FB_MODE");
    if (str != nullptr && str[0] == '1')
    {
        return true;
    }
    else
    {
        return false;
    }
}

/*****************************************************************************/
std::expected<std::pair<unsigned int, unsigned int>, dcgmReturn_t> GetCudaDriverVersions(DcgmRecorder &dcgmRecorder,
                                                                                         unsigned int gpuId)
{
    const unsigned int flags = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first
    dcgmFieldValue_v2 cudaDriverValue {};

    dcgmReturn_t ret = dcgmRecorder.GetCurrentFieldValue(gpuId, DCGM_FI_CUDA_DRIVER_VERSION, cudaDriverValue, flags);
    if (ret != DCGM_ST_OK)
    {
        return std::unexpected(ret);
    }

    unsigned int major = cudaDriverValue.value.i64 / 1000;
    unsigned int minor = (cudaDriverValue.value.i64 % 1000) / 10;
    return std::make_pair(major, minor);
}

/*****************************************************************************/
dcgmReturn_t SetCudaDriverVersions(DcgmRecorder &dcgmRecorder,
                                   unsigned int gpuId,
                                   unsigned int defaultMajorVersion,
                                   unsigned int defaultMinorVersion,
                                   unsigned int &cudaDriverMajorVersion,
                                   unsigned int &cudaDriverMinorVersion)
{
    auto versions = GetCudaDriverVersions(dcgmRecorder, gpuId);
    if (versions.has_value())
    {
        cudaDriverMajorVersion = versions->first;
        cudaDriverMinorVersion = versions->second;
        return DCGM_ST_OK;
    }
    else
    {
        log_error("Cannot read CUDA version from DCGM ({}). Assuming major version {} and minor version {}.",
                  errorString(versions.error()),
                  defaultMajorVersion,
                  defaultMinorVersion);
        cudaDriverMajorVersion = defaultMajorVersion;
        cudaDriverMinorVersion = defaultMinorVersion;
        return versions.error();
    }
}

/*****************************************************************************/
dcgmReturn_t SetCudaDriverMajorVersion(DcgmRecorder &dcgmRecorder,
                                       unsigned int gpuId,
                                       unsigned int defaultMajorVersion,
                                       unsigned int &cudaDriverMajorVersion)
{
    auto versions = GetCudaDriverVersions(dcgmRecorder, gpuId);
    if (versions.has_value())
    {
        cudaDriverMajorVersion = versions->first;
        return DCGM_ST_OK;
    }
    else
    {
        log_error("Cannot read CUDA version from DCGM ({}). Assuming major version {}.",
                  errorString(versions.error()),
                  defaultMajorVersion);
        cudaDriverMajorVersion = defaultMajorVersion;
        return versions.error();
    }
}

/*****************************************************************************/
std::expected<std::string, dcgmReturn_t> FindExecutable(std::string_view executableName,
                                                        std::vector<std::string> const &searchPaths,
                                                        std::string &executableDir)
{
    for (auto const &path : searchPaths)
    {
        auto filename = GetExecutableInPath(path, executableName);
        if (filename.has_value())
        {
            log_debug("Found {} in the path '{}'.", executableName, path);
            executableDir = path;
            return filename.value();
        }
        log_debug("{} was not present in the path '{}'.", executableName, path);
    }

    log_error(
        "Couldn't find the {} binary in the predefined search paths ({})", executableName, fmt::join(searchPaths, ":"));

    return std::unexpected(DCGM_ST_NO_DATA);
}

/*****************************************************************************/
std::vector<std::string> GetDefaultSearchPaths(unsigned int cudaDriverMajorVersion, bool useUpdatedPath)
{
    std::string const updatedPath = useUpdatedPath ? "/updated" : "";

    return { GetCurrentModuleLocation(),
             GetNvvsBinPath(cudaDriverMajorVersion, updatedPath),
             fmt::format("./apps/nvvs/plugins/cuda{}{}", cudaDriverMajorVersion, updatedPath),
             fmt::format(
                 "/usr/libexec/datacenter-gpu-manager-4/plugins/cuda{}{}", cudaDriverMajorVersion, updatedPath) };
}
