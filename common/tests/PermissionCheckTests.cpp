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
