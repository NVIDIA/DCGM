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

#include <MnDiagTempDir.h>
#include <ScopedFd.hpp>
#include <TempDir.hpp>
#include <catch2/catch_all.hpp>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace
{

struct FakeMnDiagFilesystemOps
{
    uid_t processEuid = 0;
    std::set<std::filesystem::path> existingDirs;
    std::set<std::filesystem::path> symlinkDirs;
    std::map<std::filesystem::path, uid_t> directoryOwners;
    std::vector<std::filesystem::path> createdDirs;
    std::vector<std::pair<std::filesystem::path, uid_t>> ownershipChanges;
    std::vector<std::filesystem::path> permissionChanges;
    std::filesystem::path openedDir;
    bool createFails   = false;
    bool closeFdFails  = false;
    bool setOwnerFails = false;
    bool isFdOpen      = false;

    uid_t GetProcessEuid()
    {
        return processEuid;
    }

    bool IsDirectory(std::filesystem::path const &p)
    {
        return existingDirs.contains(p);
    }

    int OpenOrCreateDirectory(std::filesystem::path const &p)
    {
        if (createFails || symlinkDirs.contains(p))
        {
            return -1;
        }
        if (!existingDirs.contains(p))
        {
            existingDirs.insert(p);
            directoryOwners[p] = processEuid;
            createdDirs.push_back(p);
        }
        openedDir = p;
        isFdOpen  = true;
        return 42;
    }

    std::optional<uid_t> GetOwner(int /*fd*/)
    {
        if (!isFdOpen)
        {
            return std::nullopt;
        }
        return directoryOwners.contains(openedDir) ? directoryOwners.at(openedDir) : processEuid;
    }

    bool SetOwner(int /*fd*/, uid_t owner)
    {
        ownershipChanges.emplace_back(openedDir, owner);
        if (setOwnerFails)
        {
            return false;
        }
        directoryOwners[openedDir] = owner;
        return true;
    }

    bool SetPermissions(int /*fd*/)
    {
        permissionChanges.push_back(openedDir);
        return isFdOpen;
    }

    bool CloseFd(int /*fd*/)
    {
        isFdOpen = false;
        return !closeFdFails;
    }
};

} // namespace

TEST_CASE("MnDiagFilesystemOps rejects a pre-existing directory symlink", "[mndiag][tempdir]")
{
    auto const testDir = TempDir::Random("mndiag-temp-dir-");
    auto const target  = testDir.Path() / "target";
    auto const symlink = testDir.Path() / "dcgm_1000";
    std::error_code ec;
    std::filesystem::create_directory(target, ec);
    REQUIRE_FALSE(ec);
    std::filesystem::create_directory_symlink(target, symlink, ec);
    REQUIRE_FALSE(ec);

    int const fd = MnDiagDetail::MnDiagFilesystemOps::OpenOrCreateDirectory(symlink);

    CHECK(fd == -1);
}

TEST_CASE("ResolveMnDiagTempDir", "[mndiag][tempdir]")
{
    SECTION("Root process with root session uses /run/dcgm")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 0 }, ops);

        CHECK(result == std::filesystem::path("/run/dcgm"));
        REQUIRE(ops.createdDirs.size() == 1);
        CHECK(ops.createdDirs[0] == std::filesystem::path("/run/dcgm"));
        CHECK(ops.ownershipChanges.empty());
    }

    SECTION("Root process with non-root session and /run/user/<uid> existing uses /run/user/<uid>/dcgm")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");
        ops.existingDirs.insert("/run/user/1000");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result == std::filesystem::path("/run/user/1000/dcgm"));
        REQUIRE(ops.createdDirs.size() == 1);
        CHECK(ops.createdDirs[0] == std::filesystem::path("/run/user/1000/dcgm"));
        REQUIRE(ops.ownershipChanges.size() == 1);
        CHECK(ops.ownershipChanges[0] == std::pair(std::filesystem::path("/run/user/1000/dcgm"), uid_t { 1000 }));
    }

    SECTION("Root process with non-root session without /run/user/<uid> falls back to /tmp/dcgm_<uid>")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result == std::filesystem::path("/tmp/dcgm_1000"));
        REQUIRE(ops.createdDirs.size() == 1);
        CHECK(ops.createdDirs[0] == std::filesystem::path("/tmp/dcgm_1000"));
        REQUIRE(ops.ownershipChanges.size() == 1);
        CHECK(ops.ownershipChanges[0] == std::pair(std::filesystem::path("/tmp/dcgm_1000"), uid_t { 1000 }));
    }

    SECTION("Non-root process falls back to /tmp/dcgm_<uid>")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 1000;
        ops.existingDirs.insert("/run");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result == std::filesystem::path("/tmp/dcgm_1000"));
        REQUIRE(ops.createdDirs.size() == 1);
        CHECK(ops.createdDirs[0] == std::filesystem::path("/tmp/dcgm_1000"));
        CHECK(ops.ownershipChanges.empty());
    }

    SECTION("OpenOrCreateDirectory failure returns empty path")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");
        ops.createFails = true;

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 0 }, ops);

        CHECK(result.empty());
    }

    SECTION("Pre-existing symlink fallback is rejected")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");
        ops.symlinkDirs.insert("/tmp/dcgm_1000");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result.empty());
        CHECK(ops.createdDirs.empty());
        CHECK(ops.ownershipChanges.empty());
        CHECK(ops.permissionChanges.empty());
    }

    SECTION("Pre-existing fallback owned by another user is rejected")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");
        ops.existingDirs.insert("/tmp/dcgm_1000");
        ops.directoryOwners["/tmp/dcgm_1000"] = 2000;

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result.empty());
        CHECK(ops.ownershipChanges.empty());
        CHECK(ops.permissionChanges.empty());
    }

    SECTION("Pre-existing fallback owned by the session user is reused")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 0;
        ops.existingDirs.insert("/run");
        ops.existingDirs.insert("/tmp/dcgm_1000");
        ops.directoryOwners["/tmp/dcgm_1000"] = 1000;

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result == std::filesystem::path("/tmp/dcgm_1000"));
        CHECK(ops.createdDirs.empty());
        CHECK(ops.ownershipChanges.empty());
        CHECK(ops.permissionChanges == std::vector<std::filesystem::path> { "/tmp/dcgm_1000" });
    }

    SECTION("SetOwner failure returns empty path")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid   = 0;
        ops.setOwnerFails = true;
        ops.existingDirs.insert("/run");
        ops.existingDirs.insert("/run/user/1000");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 1000 }, ops);

        CHECK(result.empty());
        REQUIRE(ops.ownershipChanges.size() == 1);
        CHECK(ops.ownershipChanges[0] == std::pair(std::filesystem::path("/run/user/1000/dcgm"), uid_t { 1000 }));
    }

    SECTION("CloseFd failure still returns the configured path")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid  = 0;
        ops.closeFdFails = true;
        ops.existingDirs.insert("/run");

        auto result = MnDiagDetail::ResolveMnDiagTempDir(uid_t { 0 }, ops);

        CHECK(result == std::filesystem::path("/run/dcgm"));
    }

    SECTION("std::nullopt falls back to GetProcessEuid for session UID")
    {
        FakeMnDiagFilesystemOps ops;
        ops.processEuid = 1000;

        auto result = MnDiagDetail::ResolveMnDiagTempDir(std::nullopt, ops);

        CHECK(result == std::filesystem::path("/tmp/dcgm_1000"));
        REQUIRE(ops.createdDirs.size() == 1);
        CHECK(ops.createdDirs[0] == std::filesystem::path("/tmp/dcgm_1000"));
        CHECK(ops.ownershipChanges.empty());
    }
}

namespace
{

struct FakeMnDiagHostfileOps
{
    bool makeTempFileFails = false;
    bool closeFdFails      = false;
    bool setOwnerFails     = false;
    bool writeLinesFails   = false;
    uid_t processEuid      = 0;
    bool isFdOpen          = false;
    int closeFdCallCount   = 0;
    std::vector<std::pair<int, uid_t>> ownershipChanges;
    std::string lastMakeTempTemplate;
    int lastWriteFd = -1;
    std::vector<std::string> lastWrittenLines;
    std::vector<std::filesystem::path> removedFiles;

    int MakeTempFile(std::string &nameTemplate)
    {
        lastMakeTempTemplate = nameTemplate;
        if (makeTempFileFails)
        {
            return -1;
        }
        nameTemplate.replace(nameTemplate.find("XXXXXX"), 6, "abc123");
        isFdOpen = true;
        return 42;
    }

    uid_t GetProcessEuid()
    {
        return processEuid;
    }

    bool SetOwner(int fd, uid_t owner)
    {
        if (!isFdOpen)
        {
            return false;
        }
        ownershipChanges.emplace_back(fd, owner);
        return !setOwnerFails;
    }

    bool CloseFd(int /*fd*/)
    {
        ++closeFdCallCount;
        isFdOpen = false;
        return !closeFdFails;
    }

    bool WriteLines(int fd, std::vector<std::string> const &lines)
    {
        if (!isFdOpen)
        {
            return false;
        }
        lastWriteFd      = fd;
        lastWrittenLines = lines;
        return !writeLinesFails;
    }

    void RemoveFile(std::filesystem::path const &path)
    {
        removedFiles.push_back(path);
    }
};

} // namespace

TEST_CASE("MnDiagHostfileOps creates a close-on-exec file descriptor", "[mndiag][hostfile]")
{
    auto const testDir       = TempDir::Random("mndiag-hostfile-");
    std::string nameTemplate = (testDir.Path() / "dcgm_mndiag_hostfile_XXXXXX").string();
    ScopedFd const fd(MnDiagDetail::MnDiagHostfileOps::MakeTempFile(nameTemplate));

    int const flags = fcntl(fd.Get(), F_GETFD);

    REQUIRE(flags != -1);
    CHECK((flags & FD_CLOEXEC) != 0);
}

TEST_CASE("CreateMnDiagHostfile", "[mndiag][hostfile]")
{
    std::filesystem::path const tmpDir("/tmp/dcgm_test");

    SECTION("Success creates file with correct hosts")
    {
        FakeMnDiagHostfileOps ops;
        std::vector<std::string> hosts = { "host1", "host2", "host3" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, std::nullopt, ops);

        CHECK(result == std::filesystem::path("/tmp/dcgm_test/dcgm_mndiag_hostfile_abc123"));
        CHECK(ops.closeFdCallCount == 1);
        CHECK(ops.lastWriteFd == 42);
        REQUIRE(ops.lastWrittenLines.size() == 3);
        CHECK(ops.lastWrittenLines[0] == "host1");
        CHECK(ops.lastWrittenLines[1] == "host2");
        CHECK(ops.lastWrittenLines[2] == "host3");
        CHECK(ops.ownershipChanges.empty());
        CHECK(ops.removedFiles.empty());
    }

    SECTION("Different owner is applied while the file descriptor is open")
    {
        FakeMnDiagHostfileOps ops;
        std::vector<std::string> hosts = { "host1", "host2" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, uid_t { 1000 }, ops);

        CHECK_FALSE(result.empty());
        REQUIRE(ops.ownershipChanges.size() == 1);
        CHECK(ops.ownershipChanges[0] == std::pair(42, uid_t { 1000 }));
        CHECK(ops.closeFdCallCount == 1);
        CHECK(ops.removedFiles.empty());
    }

    SECTION("Matching owner does not require an ownership change")
    {
        FakeMnDiagHostfileOps ops;
        ops.processEuid                = 1000;
        std::vector<std::string> hosts = { "host1" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, uid_t { 1000 }, ops);

        CHECK_FALSE(result.empty());
        CHECK(ops.ownershipChanges.empty());
        CHECK(ops.closeFdCallCount == 1);
    }

    SECTION("SetOwner failure closes and removes file")
    {
        FakeMnDiagHostfileOps ops;
        ops.setOwnerFails              = true;
        std::vector<std::string> hosts = { "host1" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, uid_t { 1000 }, ops);

        CHECK(result.empty());
        REQUIRE(ops.ownershipChanges.size() == 1);
        CHECK(ops.ownershipChanges[0] == std::pair(42, uid_t { 1000 }));
        CHECK(ops.closeFdCallCount == 1);
        CHECK(ops.lastWrittenLines == hosts);
        REQUIRE(ops.removedFiles.size() == 1);
        CHECK(ops.removedFiles[0] == std::filesystem::path("/tmp/dcgm_test/dcgm_mndiag_hostfile_abc123"));
    }

    SECTION("MakeTempFile failure returns empty path")
    {
        FakeMnDiagHostfileOps ops;
        ops.makeTempFileFails          = true;
        std::vector<std::string> hosts = { "host1" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, std::nullopt, ops);

        CHECK(result.empty());
        CHECK(ops.closeFdCallCount == 0);
        CHECK(ops.lastWrittenLines.empty());
        CHECK(ops.removedFiles.empty());
    }

    SECTION("CloseFd failure removes file and returns empty path")
    {
        FakeMnDiagHostfileOps ops;
        ops.closeFdFails               = true;
        std::vector<std::string> hosts = { "host1" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, std::nullopt, ops);

        CHECK(result.empty());
        CHECK(ops.closeFdCallCount == 1);
        CHECK(ops.lastWrittenLines == hosts);
        REQUIRE(ops.removedFiles.size() == 1);
        CHECK(ops.removedFiles[0] == std::filesystem::path("/tmp/dcgm_test/dcgm_mndiag_hostfile_abc123"));
    }

    SECTION("WriteLines failure removes file and returns empty path")
    {
        FakeMnDiagHostfileOps ops;
        ops.writeLinesFails            = true;
        std::vector<std::string> hosts = { "host1", "host2" };

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, std::nullopt, ops);

        CHECK(result.empty());
        CHECK(ops.closeFdCallCount == 1);
        REQUIRE(ops.removedFiles.size() == 1);
        CHECK(ops.removedFiles[0] == std::filesystem::path("/tmp/dcgm_test/dcgm_mndiag_hostfile_abc123"));
    }

    SECTION("Empty host list creates file with no content")
    {
        FakeMnDiagHostfileOps ops;
        std::vector<std::string> hosts;

        auto result = MnDiagDetail::CreateMnDiagHostfile(tmpDir, hosts, std::nullopt, ops);

        CHECK_FALSE(result.empty());
        CHECK(ops.lastWrittenLines.empty());
        CHECK(ops.removedFiles.empty());
    }
}
