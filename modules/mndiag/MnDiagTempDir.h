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

#include <DcgmLogging.h>
#include <DcgmUtilities.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace MnDiagDetail
{

/**
 * Production filesystem operations for ResolveMnDiagTempDir.
 *
 * Tests inject a fake that satisfies the same implicit interface.
 */
struct MnDiagFilesystemOps
{
    static uid_t GetProcessEuid()
    {
        return geteuid();
    }

    static bool IsDirectory(std::filesystem::path const &p)
    {
        std::error_code ec;
        return std::filesystem::is_directory(p, ec);
    }

    static int OpenOrCreateDirectory(std::filesystem::path const &p)
    {
        // mkdir atomically claims a missing leaf. If it already exists, open it
        // without following a symlink so later metadata changes affect only the
        // directory that was actually inspected.
        if (mkdir(p.c_str(), S_IRWXU) != 0 && errno != EEXIST)
        {
            return -1;
        }

        return open(p.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    }

    static std::optional<uid_t> GetOwner(int fd)
    {
        struct stat statBuf = {};
        if (fstat(fd, &statBuf) != 0)
        {
            return std::nullopt;
        }

        return statBuf.st_uid;
    }

    static bool SetOwner(int fd, uid_t owner)
    {
        return fchown(fd, owner, static_cast<gid_t>(-1)) == 0;
    }

    static bool SetPermissions(int fd)
    {
        return fchmod(fd, S_IRWXU) == 0;
    }

    static bool CloseFd(int fd)
    {
        return close(fd) == 0;
    }
};

/**
 * Resolve the mndiag temporary directory following the DCGM directory convention.
 *
 * - Root process + root session: /run/dcgm/
 * - Root process + non-root session (with /run/user/<uid> present): /run/user/<uid>/dcgm/
 * - Fallback: /tmp/dcgm_<uid>/
 *
 * Creates the directory with 0700 permissions if it does not exist. A root process assigns
 * the directory to a non-root session UID so that processes launched as that user can access
 * it.
 *
 * @tparam Ops Filesystem operations type (defaults to MnDiagFilesystemOps for production)
 * @param[in] sessionUid The UID of the API caller. If std::nullopt, falls back to
 *                       ops.GetProcessEuid(), keeping geteuid() mockable in tests.
 * @param[in,out] ops Filesystem operations instance
 * @return The resolved directory path, or empty path on failure (logged internally)
 */
template <typename Ops>
std::filesystem::path ResolveMnDiagTempDir(std::optional<uid_t> sessionUid, Ops &ops)
{
    uid_t uid = sessionUid.value_or(ops.GetProcessEuid());

    std::filesystem::path tmpDir;
    auto runPath = std::filesystem::path("/run");

    if (ops.GetProcessEuid() == 0 && ops.IsDirectory(runPath))
    {
        if (uid == 0)
        {
            tmpDir = runPath / "dcgm";
        }
        else if (auto userRunDir = runPath / "user" / std::to_string(uid); ops.IsDirectory(userRunDir))
        {
            tmpDir = userRunDir / "dcgm";
        }
    }
    if (tmpDir.empty())
    {
        tmpDir = std::filesystem::path("/tmp") / fmt::format("dcgm_{}", uid);
    }

    int const dirFd = ops.OpenOrCreateDirectory(tmpDir);
    if (dirFd < 0)
    {
        log_error("Failed to securely open or create mndiag temp directory {}: {}", tmpDir.string(), strerror(errno));
        return {};
    }

    auto const currentOwner = ops.GetOwner(dirFd);
    if (!currentOwner.has_value())
    {
        auto const savedErrno = errno;
        ops.CloseFd(dirFd);
        log_error("Failed to inspect owner of mndiag temp directory {}: {}", tmpDir.string(), strerror(savedErrno));
        return {};
    }

    // A directory created by this process is initially owned by its EUID. Refuse
    // an existing directory owned by another user instead of taking it over.
    auto const processEuid = ops.GetProcessEuid();
    if (*currentOwner != processEuid && *currentOwner != uid)
    {
        ops.CloseFd(dirFd);
        log_error("Refusing mndiag temp directory {} owned by unexpected UID {} (expected {} or {})",
                  tmpDir.string(),
                  *currentOwner,
                  processEuid,
                  uid);
        return {};
    }

    // Apply ownership and permissions through the verified descriptor. This
    // prevents a pathname substitution from redirecting chown or chmod.
    if (processEuid == 0 && uid != 0 && *currentOwner != uid && !ops.SetOwner(dirFd, uid))
    {
        auto const savedErrno = errno;
        ops.CloseFd(dirFd);
        log_error("Failed to set owner of mndiag temp directory {} to UID {}: {}",
                  tmpDir.string(),
                  uid,
                  strerror(savedErrno));
        return {};
    }
    if (!ops.SetPermissions(dirFd))
    {
        auto const savedErrno = errno;
        ops.CloseFd(dirFd);
        log_error("Failed to set permissions on mndiag temp directory {}: {}", tmpDir.string(), strerror(savedErrno));
        return {};
    }
    if (!ops.CloseFd(dirFd))
    {
        log_warning("Failed to close mndiag temp directory {}: {}", tmpDir.string(), strerror(errno));
    }

    return tmpDir;
}

inline std::filesystem::path ResolveMnDiagTempDir(std::optional<uid_t> sessionUid = std::nullopt)
{
    MnDiagFilesystemOps ops;
    return ResolveMnDiagTempDir(sessionUid, ops);
}

/**
 * Production file-creation operations for CreateMnDiagHostfile.
 *
 * Tests inject a fake that satisfies the same implicit interface.
 */
struct MnDiagHostfileOps
{
    static uid_t GetProcessEuid()
    {
        return geteuid();
    }

    static int MakeTempFile(std::string &nameTemplate)
    {
        return mkostemp(nameTemplate.data(), O_CLOEXEC);
    }

    static bool SetOwner(int fd, uid_t owner)
    {
        return fchown(fd, owner, static_cast<gid_t>(-1)) == 0;
    }

    static bool CloseFd(int fd)
    {
        return close(fd) == 0;
    }

    static bool WriteLines(int fd, std::vector<std::string> const &lines)
    {
        std::string contents;
        for (auto const &line : lines)
        {
            contents.append(line);
            contents.push_back('\n');
        }

        if (auto const error = DcgmNs::Utils::WriteAll(fd, contents.data(), contents.size()); error != 0)
        {
            errno = error;
            return false;
        }
        return true;
    }

    static void RemoveFile(std::filesystem::path const &path)
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/**
 * Create a temporary hostfile containing one hostname per line.
 *
 * Uses mkostemp for atomic creation and writes all entries from @p hostList through
 * the returned file descriptor.
 * On any failure, cleans up the file and returns an empty path.
 *
 * @tparam Ops File-creation operations type (defaults to MnDiagHostfileOps for production)
 * @param[in] tmpDir Directory in which to create the temporary file
 * @param[in] hostList Hostnames to write, one per line
 * @param[in] ownerUid UID that must be able to read the hostfile. A root creator assigns the
 *                     file to a non-root UID before closing it.
 * @param[in,out] ops File-creation operations instance
 * @return Path to the created hostfile, or empty path on failure (logged internally)
 */
template <typename Ops>
std::filesystem::path CreateMnDiagHostfile(std::filesystem::path const &tmpDir,
                                           std::vector<std::string> const &hostList,
                                           std::optional<uid_t> ownerUid,
                                           Ops &ops)
{
    auto tmpPath        = tmpDir / "dcgm_mndiag_hostfile_XXXXXX";
    std::string tmpName = tmpPath.string();

    int fd = ops.MakeTempFile(tmpName);
    if (fd < 0)
    {
        log_error("Failed to create temporary hostfile: {}", strerror(errno));
        return {};
    }

    // Keep using the descriptor returned by mkstemp until the file is complete
    // and assigned to the session user. Reopening the pathname would introduce
    // a race because the session user owns the containing directory.
    if (!ops.WriteLines(fd, hostList))
    {
        auto const savedErrno = errno;
        if (!ops.CloseFd(fd))
        {
            log_warning("Failed to close temporary hostfile fd after write error: {}", strerror(errno));
        }
        ops.RemoveFile(tmpName);
        log_error("Failed to write to temporary hostfile {}: {}", tmpName, strerror(savedErrno));
        return {};
    }
    if (ownerUid.has_value() && ops.GetProcessEuid() == 0 && *ownerUid != 0 && !ops.SetOwner(fd, *ownerUid))
    {
        auto const savedErrno = errno;
        if (!ops.CloseFd(fd))
        {
            log_warning("Failed to close temporary hostfile fd after ownership error: {}", strerror(errno));
        }
        ops.RemoveFile(tmpName);
        log_error(
            "Failed to set owner of temporary hostfile {} to UID {}: {}", tmpName, *ownerUid, strerror(savedErrno));
        return {};
    }
    if (!ops.CloseFd(fd))
    {
        log_error("Failed to close temporary hostfile fd: {}", strerror(errno));
        ops.RemoveFile(tmpName);
        return {};
    }

    return std::filesystem::path(tmpName);
}

inline std::filesystem::path CreateMnDiagHostfile(std::filesystem::path const &tmpDir,
                                                  std::vector<std::string> const &hostList,
                                                  std::optional<uid_t> ownerUid = std::nullopt)
{
    MnDiagHostfileOps ops;
    return CreateMnDiagHostfile(tmpDir, hostList, ownerUid, ops);
}

} // namespace MnDiagDetail
