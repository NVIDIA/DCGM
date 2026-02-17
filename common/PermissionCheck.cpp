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

#include "PermissionCheck.hpp"

#include <boost/process.hpp>
#include <sys/stat.h>

#include <DcgmLogging.h>
#include <utility>


namespace
{

using namespace DcgmNs::Utils;

inline OwnershipResult CheckAllParentDirectoriesOwnedByRoot(boost::filesystem::path path)
{
    /* TODO(nkonyuchenko): Consider if parent directories permissions should be checked recursively.
     *  The requirement of root ownership and root-only write permissions for the all parent directories may be too
     *  restrictive for our needs. For EA builds we disable this */
#if 0
    auto tmpPath = path;
    while (tmpPath.has_parent_path())
    {
        tmpPath = tmpPath.parent_path();

        struct stat rawFileStat = {};
        if (-1 == stat(tmpPath.c_str(), &rawFileStat))
        {
            return OwnershipResultError { OwnershipErrorCode::SystemError,
                                          std::system_error(errno, std::system_category()) };
        }
        if (rawFileStat.st_uid != 0 || rawFileStat.st_gid != 0)
        {
            return OwnershipResultError { OwnershipErrorCode::NotOwnedByRoot, {} };
        }

        if ((rawFileStat.st_mode & S_IWOTH) != 0 || (rawFileStat.st_mode & S_IWGRP) != 0)
        {
            return OwnershipResultError { OwnershipErrorCode::WritableByNonRoot, {} };
        }
    }
#endif
    return OwnershipResultSuccess { std::move(path) };
}

} // namespace


namespace DcgmNs::Utils
{

OwnershipResult CheckExecutableAndOwnership(boost::filesystem::path filePath, bool bypassBinPermissionCheck)
{
    boost::system::error_code systemError;

    auto fileStats = status(filePath, systemError);
    if (systemError)
    {
        if (fileStats.type() == boost::filesystem::file_not_found)
        {
            return OwnershipResultError { OwnershipErrorCode::FileNotFound, std::system_error { systemError } };
        }
        return OwnershipResultError { OwnershipErrorCode::SystemError, std::system_error { systemError } };
    }

    filePath = canonical(filePath, systemError);
    if (systemError)
    {
        return OwnershipResultError { OwnershipErrorCode::SystemError, std::system_error { systemError } };
    }

    if (!is_regular_file(std::move(fileStats)))
    {
        return OwnershipResultError { OwnershipErrorCode::NotRegularFile, {} };
    }

    struct stat rawFileStat = {};
    if (-1 == stat(filePath.c_str(), &rawFileStat))
    {
        return OwnershipResultError { OwnershipErrorCode::SystemError,
                                      std::system_error(errno, std::system_category()) };
    }

    // Check that the owner and the group of the file is root
    if (!bypassBinPermissionCheck)
    {
        if (rawFileStat.st_uid != 0 || rawFileStat.st_gid != 0)
        {
            return OwnershipResultError { OwnershipErrorCode::NotOwnedByRoot, {} };
        }

        // Check that the file has the correct permissions
        // We only allow ??xr-xr-x (1|3|5|7)55 permissions for the file.
        // Write permission for anyone but the root user is not allowed.
        if ((rawFileStat.st_mode & S_IWOTH) != 0 || (rawFileStat.st_mode & S_IWGRP) != 0)
        {
            return OwnershipResultError { OwnershipErrorCode::WritableByNonRoot, {} };
        }
        if ((rawFileStat.st_mode & S_IXUSR) == 0)
        {
            return OwnershipResultError { OwnershipErrorCode::NotExecutable, {} };
        }
    }
    else
    {
        log_error("Skip checking the permission of binary [{}].", filePath.string());
    }

    return CheckAllParentDirectoriesOwnedByRoot(std::move(filePath));
}

fmt::string_view format_as(OwnershipErrorCode const &error)
{
    switch (error)
    {
        using enum OwnershipErrorCode;
        case FileNotFound:
            return "OwnershipErrorCode::FileNotFound";
        case NotRegularFile:
            return "OwnershipErrorCode::NotRegularFile";
        case NotExecutable:
            return "OwnershipErrorCode::NotExecutable";
        case NotOwnedByRoot:
            return "OwnershipErrorCode::NotOwnedByRoot";
        case WritableByNonRoot:
            return "OwnershipErrorCode::WritableByNonRoot";
        case SystemError:
            return "OwnershipErrorCode::SystemError";
    }

    throw std::runtime_error("Unknown OwnershipErrorCode");
}

DcgmError OwnershipErrorToDcgmError(OwnershipResultError const &error,
                                    std::string_view binaryName,
                                    std::string_view envVariable)
{
    DcgmError result { DcgmError::GpuIdTag::Unknown };
    result.SetCode(DCGM_FR_BINARY_PERMISSIONS);
    log_debug("OwnershipResultError: {}", error.errorCode);
    if (error.errorCode == OwnershipErrorCode::SystemError)
    {
        log_debug("System error: ({}) {}", error.systemError.code().value(), error.systemError.what());
    }
    switch (error.errorCode)
    {
        using enum OwnershipErrorCode;
        case FileNotFound:
            result.SetMessage(fmt::format("{} binary not found.", binaryName));
            result.SetNextSteps(fmt::format("Ensure the {} binary is installed into proper location or "
                                            "{} environment variable is set for the "
                                            "hostengine/nvvs process.",
                                            binaryName,
                                            envVariable));
            break;
        case NotRegularFile:
            result.SetMessage(fmt::format("{} binary is not a regular file.", binaryName));
            result.SetNextSteps(fmt::format("Ensure the {} binary is installed into proper location or "
                                            "{} environment variable points to the specializediag executable.",
                                            binaryName,
                                            envVariable));
            break;
        case NotExecutable:
            result.SetMessage(fmt::format("{} binary is not executable.", binaryName));
            result.SetNextSteps(fmt::format("Ensure the {} binary is installed into proper location or "
                                            "{} environment variable points to the specializediag executable.",
                                            binaryName,
                                            envVariable));
            break;
        case NotOwnedByRoot:
            result.SetMessage(fmt::format("{} binary is not owned by root.", binaryName));
            result.SetNextSteps(fmt::format("The {} binary and all its parent directories should be "
                                            "owned by the root:root user and group.",
                                            binaryName));
            break;
        case WritableByNonRoot:
            result.SetMessage(fmt::format("{} binary is writable by non-root user.", binaryName));
            result.SetNextSteps(
                fmt::format("Make sure the {} binary has 755 permissions and all the parent directories are not "
                            "writable by non-root user.",
                            binaryName));
            break;
        case SystemError:
            result.SetMessage(fmt::format("System error: {}.", error.systemError.what()));
            result.SetNextSteps(fmt::format("A system error (code {}) occurred while checking the {} binary ownership. "
                                            "Please check the system logs for more details.",
                                            error.systemError.code().value(),
                                            binaryName));
            break;
    }

    return result;
}

} // namespace DcgmNs::Utils
