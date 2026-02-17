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

#include "DcgmError.h"

#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <system_error>
#include <variant>


namespace DcgmNs::Utils
{

enum struct OwnershipErrorCode
{
    FileNotFound,      //!< The file does not exist.
    NotRegularFile,    //!< The file is not a regular file.
    NotExecutable,     //!< The file is not executable.
    NotOwnedByRoot,    //!< The file or its parent directories are not owned by root.
    WritableByNonRoot, //!< The file or its parent directories are writable by non-root.
    SystemError,       //!< A system error occurred.
};
fmt::string_view format_as(OwnershipErrorCode const &error);

struct OwnershipResultError
{
    OwnershipErrorCode errorCode;  //!< The error code.
    std::system_error systemError; //!< system_error if errorCode == SystemError
};

struct OwnershipResultSuccess
{
    boost::filesystem::path filePath;
};

/**
 * @brief Result of \c CheckExecutableAndOwnership function.
 */
using OwnershipResult = std::variant<OwnershipResultError, OwnershipResultSuccess>;

/**
 * @brief Checks that the given path is a directory and has the right permissions to be run as root.
 *        The file must be owned by the root user and should not provide permissions to write to either the group or
 *        other users.
 *        <s>(Disabled for now) All parent directories are also checked for the same permissions.</s>
 *        If the filePath is a symlink, it will be converted to canonical form before checking.
 * @param[in] filePath The path to an executable file to check.
 * @param[in] bypassBinPermissionCheck If true, the permission check will be skipped for the binary file.
 * @return \c OwnershipResultError if the file is not a regular file, or if it is not owned by the root user, or if
 *           it has permissions to write to either the group or other users.
 * @return \c OwnershipResultSuccess if the file is a regular file and is owned by the root user and has no
 *           permissions to write to either the group or other users. If the file is a symbolic link, the target is
 *           returned.
 * @see \c OwnershipResultError
 * @see \c OwnershipResultSuccess
 * @see \c CheckAllParentDirectoriesOwnedByRoot for details on the parent directories check.
 */
OwnershipResult CheckExecutableAndOwnership(boost::filesystem::path filePath, bool bypassBinPermissionCheck = false);

/**
 * @brief Converts an OwnershipResultError to a DcgmError
 * @param[in] error The OwnershipResultError to convert
 * @param[in] binaryName The name of the binary for error messages
 * @param[in] envVariable The environment variable name to mention in error messages
 * @return The converted DcgmError with the appropriate message and next steps
 */
DcgmError OwnershipErrorToDcgmError(OwnershipResultError const &error,
                                    std::string_view binaryName,
                                    std::string_view envVariable);

} // namespace DcgmNs::Utils
