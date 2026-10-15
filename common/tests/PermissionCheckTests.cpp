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

#include <catch2/catch_all.hpp>

#include <PermissionCheck.hpp>

#include <fstream>
#include <utility>

namespace
{
class PathCleanup
{
public:
    explicit PathCleanup(boost::filesystem::path path)
        : m_path(std::move(path))
    {}

    ~PathCleanup()
    {
        boost::system::error_code ignored;
        boost::filesystem::remove_all(m_path, ignored);
    }

private:
    boost::filesystem::path m_path;
};
} // namespace


TEST_CASE("PermissionCheck: Valid executable")
{
    using namespace DcgmNs::Utils;
    boost::filesystem::path filePath = boost::filesystem::path("/usr/bin/ls");

    auto result = CheckExecutableAndOwnership(std::move(filePath));

    REQUIRE(std::holds_alternative<OwnershipResultSuccess>(result));
}

TEST_CASE("PermissionCheck: File not exists")
{
    using namespace DcgmNs::Utils;
    boost::filesystem::path filePath = boost::filesystem::path("/usr/bin/ls_not_exists");

    auto result = CheckExecutableAndOwnership(std::move(filePath));

    REQUIRE(std::holds_alternative<OwnershipResultError>(result));
    REQUIRE(std::get<OwnershipResultError>(result).errorCode == OwnershipErrorCode::FileNotFound);
}

TEST_CASE("PermissionCheck: Not Executable")
{
    using namespace DcgmNs::Utils;
    boost::filesystem::path filePath = boost::filesystem::path("/etc/passwd");

    auto result = CheckExecutableAndOwnership(std::move(filePath));

    REQUIRE(std::holds_alternative<OwnershipResultError>(result));
    REQUIRE(std::get<OwnershipResultError>(result).errorCode == OwnershipErrorCode::NotExecutable);
}

TEST_CASE("PermissionCheck: Not owned by root", "[.]")
{
    using namespace DcgmNs::Utils;
    boost::filesystem::path filePath = boost::filesystem::path("/home/builder/.profile");

    auto result = CheckExecutableAndOwnership(std::move(filePath));

    REQUIRE(std::holds_alternative<OwnershipResultError>(result));
    REQUIRE(std::get<OwnershipResultError>(result).errorCode == OwnershipErrorCode::NotOwnedByRoot);
}

TEST_CASE("PermissionCheck: formatting")
{
    using DcgmNs::Utils::OwnershipErrorCode;
    REQUIRE("OwnershipErrorCode::FileNotFound" == fmt::format("{}", OwnershipErrorCode::FileNotFound));
    REQUIRE("OwnershipErrorCode::NotRegularFile" == fmt::format("{}", OwnershipErrorCode::NotRegularFile));
    REQUIRE("OwnershipErrorCode::NotExecutable" == fmt::format("{}", OwnershipErrorCode::NotExecutable));
    REQUIRE("OwnershipErrorCode::NotOwnedByRoot" == fmt::format("{}", OwnershipErrorCode::NotOwnedByRoot));
    REQUIRE("OwnershipErrorCode::WritableByNonRoot" == fmt::format("{}", OwnershipErrorCode::WritableByNonRoot));
    REQUIRE("OwnershipErrorCode::SystemError" == fmt::format("{}", OwnershipErrorCode::SystemError));
}


SCENARIO("PermissionCheck bypass mode validates only file existence and type")
{
    GIVEN("a regular temporary file without executable permissions")
    {
        using namespace DcgmNs::Utils;
        auto const filePath = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
        PathCleanup cleanup(filePath);
        {
            std::ofstream file(filePath.string());
            REQUIRE(file.good());
            file << "content";
        }
        boost::filesystem::permissions(filePath, boost::filesystem::owner_read | boost::filesystem::owner_write);

        WHEN("ownership and executable permission checks are bypassed")
        {
            auto result = CheckExecutableAndOwnership(filePath, true);

            THEN("the regular file is accepted and canonicalized")
            {
                REQUIRE(std::holds_alternative<OwnershipResultSuccess>(result));
                CHECK(std::get<OwnershipResultSuccess>(result).filePath == boost::filesystem::canonical(filePath));
            }
        }
    }

    GIVEN("a directory path")
    {
        using namespace DcgmNs::Utils;
        auto const directoryPath = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
        REQUIRE(boost::filesystem::create_directory(directoryPath));
        PathCleanup cleanup(directoryPath);

        WHEN("the path is checked")
        {
            auto result = CheckExecutableAndOwnership(directoryPath, true);

            THEN("the path is rejected as not being a regular file")
            {
                REQUIRE(std::holds_alternative<OwnershipResultError>(result));
                CHECK(std::get<OwnershipResultError>(result).errorCode == OwnershipErrorCode::NotRegularFile);
            }
        }
    }
}

SCENARIO("PermissionCheck converts ownership failures to DCGM errors")
{
    GIVEN("each ownership error code")
    {
        using namespace DcgmNs::Utils;
        auto errorCode = GENERATE(OwnershipErrorCode::FileNotFound,
                                  OwnershipErrorCode::NotRegularFile,
                                  OwnershipErrorCode::NotExecutable,
                                  OwnershipErrorCode::NotOwnedByRoot,
                                  OwnershipErrorCode::WritableByNonRoot,
                                  OwnershipErrorCode::SystemError);
        CAPTURE(errorCode);
        OwnershipResultError ownershipError { errorCode,
                                              std::system_error(std::make_error_code(std::errc::permission_denied)) };

        WHEN("the error is converted")
        {
            auto dcgmError = OwnershipErrorToDcgmError(ownershipError, "diag", "DIAG_PATH");

            THEN("the result uses the binary permission error code and provides actionable text")
            {
                CHECK(dcgmError.GetCode() == DCGM_FR_BINARY_PERMISSIONS);
                CHECK_FALSE(dcgmError.GetMessage().empty());
                auto const nextSteps      = std::string(dcgmError.GetNextSteps());
                bool const mentionsBinary = nextSteps.find("diag") != std::string::npos;
                bool const mentionsEnvVar = nextSteps.find("DIAG_PATH") != std::string::npos;
                CHECK((mentionsBinary || mentionsEnvVar));
            }
        }
    }
}


SCENARIO("PermissionCheck canonicalizes symlinked executables")
{
    GIVEN("a symlink that points at a regular file")
    {
        using namespace DcgmNs::Utils;
        auto const directoryPath = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
        REQUIRE(boost::filesystem::create_directory(directoryPath));
        PathCleanup cleanup(directoryPath);

        auto const targetPath  = directoryPath / "target-file";
        auto const symlinkPath = directoryPath / "symlink-file";
        {
            std::ofstream file(targetPath.string());
            REQUIRE(file.good());
            file << "content";
        }
        boost::filesystem::create_symlink(targetPath, symlinkPath);

        WHEN("the symlink is checked with binary permission checks bypassed")
        {
            auto result = CheckExecutableAndOwnership(symlinkPath, true);

            THEN("the target file is accepted and returned in canonical form")
            {
                REQUIRE(std::holds_alternative<OwnershipResultSuccess>(result));
                CHECK(std::get<OwnershipResultSuccess>(result).filePath == boost::filesystem::canonical(targetPath));
            }
        }
    }
}
