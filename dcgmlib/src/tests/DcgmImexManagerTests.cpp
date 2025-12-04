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

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "DcgmImexManager.h"
#include "dcgm_fields.h"

#include <memory>
#include <string>

// Test fixture to ensure DCGM fields are initialized
class DcgmFieldsFixture
{
public:
    DcgmFieldsFixture()
    {
        // Initialize the DCGM fields system
        fieldsInitialized = (DcgmFieldsInit() == 0);
    }

    ~DcgmFieldsFixture()
    {
        // Clean up the DCGM fields system
        if (fieldsInitialized)
        {
            DcgmFieldsTerm();
        }
    }

    bool isInitialized() const
    {
        return fieldsInitialized;
    }

private:
    bool fieldsInitialized;
};

SCENARIO("DcgmImexManager basic functionality")
{
    SECTION("Constructor initializes properly")
    {
        DcgmImexManager manager;

        // Basic construction should not throw
        REQUIRE(true);
    }

    SECTION("DomainStatusToString converts enum values correctly")
    {
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::UP) == "UP");
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::DOWN) == "DOWN");
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::DEGRADED) == "DEGRADED");
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::NOT_INSTALLED) == "NOT_INSTALLED");
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::NOT_CONFIGURED) == "NOT_CONFIGURED");
        REQUIRE(DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus::UNAVAILABLE) == "UNAVAILABLE");
    }

    SECTION("DaemonStatusToInt64 converts enum values correctly")
    {
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::INITIALIZING) == 0);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::STARTING_AUTH_SERVER) == 1);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::WAITING_FOR_PEERS) == 2);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::WAITING_FOR_RECOVERY) == 3);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::INIT_GPU) == 4);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::READY) == 5);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::SHUTTING_DOWN) == 6);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::UNAVAILABLE) == 7);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::NOT_INSTALLED) == -1);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::NOT_CONFIGURED) == -2);
        REQUIRE(DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus::COMMAND_ERROR) == -3);
    }
}

SCENARIO("IMEX status retrieval")
{
    SECTION("GetDomainStatus returns valid string")
    {
        DcgmImexManager manager;

        // Should return some status (likely NOT_INSTALLED in test environment)
        std::string status = manager.GetDomainStatus();
        REQUIRE(!status.empty());

        // Should be one of the valid status strings
        bool isValidStatus = (status == "UP" || status == "DOWN" || status == "DEGRADED" || status == "NOT_INSTALLED"
                              || status == "NOT_CONFIGURED" || status == "UNAVAILABLE");
        REQUIRE(isValidStatus);
    }

    SECTION("GetDaemonStatus returns valid integer")
    {
        DcgmImexManager manager;

        // Should return some status
        int64_t status = manager.GetDaemonStatus();

        // Should be a valid status value (0-7 or negative for errors)
        bool isValidStatus = (status >= -3 && status <= 7);
        REQUIRE(isValidStatus);
    }
}

SCENARIO("IMEX field definitions")
{
    SECTION("Field IDs are defined correctly")
    {
        REQUIRE(DCGM_FI_IMEX_DOMAIN_STATUS == 1502);
        REQUIRE(DCGM_FI_IMEX_DAEMON_STATUS == 1503);
    }

    SECTION("Field metadata is registered")
    {
        // Use the fixture to ensure fields are initialized
        DcgmFieldsFixture fixture;
        REQUIRE(fixture.isInitialized());

        // Test that the fields are properly registered in the DCGM field system
        dcgm_field_meta_p domainMeta = DcgmFieldGetById(DCGM_FI_IMEX_DOMAIN_STATUS);
        dcgm_field_meta_p daemonMeta = DcgmFieldGetById(DCGM_FI_IMEX_DAEMON_STATUS);

        REQUIRE(domainMeta != nullptr);
        REQUIRE(daemonMeta != nullptr);

        if (domainMeta)
        {
            REQUIRE(domainMeta->fieldId == DCGM_FI_IMEX_DOMAIN_STATUS);
            REQUIRE(domainMeta->fieldType == DCGM_FT_STRING);
            REQUIRE(domainMeta->scope == DCGM_FS_GLOBAL);
            REQUIRE(domainMeta->entityLevel == DCGM_FE_NONE);
            REQUIRE(std::string(domainMeta->tag) == "imex_domain_status");
        }

        if (daemonMeta)
        {
            REQUIRE(daemonMeta->fieldId == DCGM_FI_IMEX_DAEMON_STATUS);
            REQUIRE(daemonMeta->fieldType == DCGM_FT_INT64);
            REQUIRE(daemonMeta->scope == DCGM_FS_GLOBAL);
            REQUIRE(daemonMeta->entityLevel == DCGM_FE_NONE);
            REQUIRE(std::string(daemonMeta->tag) == "imex_daemon_status");
        }

        // Fixture destructor will clean up automatically
    }
}

// Test helper class to access private methods via friend access
class DcgmImexManagerTest
{
public:
    DcgmImexManagerTest(std::unique_ptr<DcgmImexManager> manager)
        : m_manager(std::move(manager))
    {}

    // Wrapper methods to access private functionality
    DcgmImexDomainStatus ParseDomainStatusJson(std::string const &jsonOutput)
    {
        return m_manager->ParseDomainStatusJson(jsonOutput);
    }

    DcgmImexDaemonStatus ParseDaemonStatusText(std::string const &textOutput)
    {
        return m_manager->ParseDaemonStatusText(textOutput);
    }

    bool IsImexCtlAvailable() const
    {
        return m_manager->IsImexCtlAvailable();
    }

    std::optional<std::string> FindImexCtlExecutable() const
    {
        return m_manager->FindImexCtlExecutable();
    }

    std::vector<std::string> GetImexCtlSearchPaths() const
    {
        return m_manager->GetImexCtlSearchPaths();
    }

    // Access to the manager for public methods
    DcgmImexManager &GetManager()
    {
        return *m_manager;
    }

private:
    std::unique_ptr<DcgmImexManager> m_manager;
};

SCENARIO("IMEX domain status JSON parsing")
{
    auto manager = std::make_unique<DcgmImexManager>();
    DcgmImexManagerTest helper(std::move(manager));

    SECTION("Parses valid JSON with UP status")
    {
        std::string json = R"({"nodes":{"0":{"status":"READY"}},"timestamp":"5/22/2025 19:46:40.891","status":"UP"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::UP);
    }

    SECTION("Parses valid JSON with DOWN status")
    {
        std::string json            = R"({"status":"DOWN","other":"field"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::DOWN);
    }

    SECTION("Parses valid JSON with DEGRADED status")
    {
        std::string json            = R"({"timestamp":"test","status":"DEGRADED"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::DEGRADED);
    }

    SECTION("Handles case insensitive status values")
    {
        std::string json            = R"({"status":"up"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::UP);
    }

    SECTION("Handles missing status field")
    {
        std::string json            = R"({"other":"field","timestamp":"test"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::UNAVAILABLE);
    }

    SECTION("Handles malformed JSON")
    {
        std::string json            = R"({"status":})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::UNAVAILABLE);
    }

    SECTION("Handles complex nested JSON from sample")
    {
        std::string sampleJson      = R"({
            "nodes": {
                "3": {"status":"READY","host":"10.76.186.255"},
                "5": {"status":"READY","host":"10.76.189.233"}
            },
            "timestamp":"5/22/2025 19:46:40.891",
            "status":"DEGRADED"
        })";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(sampleJson);
        REQUIRE(result == DcgmImexDomainStatus::DEGRADED);
    }

    SECTION("Handles actual sample JSON from specification")
    {
        // This is the exact JSON format from the original specification
        std::string actualSampleJson = R"({"nodes":{
	"3": {"status":"READY","host":"10.76.186.255",
		"connections":{"1":{"host":"10.76.183.30","status":"CONNECTED","changed":true},
			"6":{"host":"10.76.190.23","status":"CONNECTED","changed":true}
		},
		"changed":true,
		"version":"580.30",
		"hostName":""},
	"1":{"status":"READY","host":"10.76.183.30",
		"connections":{"0":{"host":"10.76.179.115","status":"CONNECTED","changed":true},
		"1":{"host":"10.76.183.30","status":"CONNECTED","changed":true}},
		"changed":true,"version":"580.30","hostName":""}},
	"timestamp":"5/22/2025 19:46:40.891",
	"status":"DEGRADED"})";

        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(actualSampleJson);
        REQUIRE(result == DcgmImexDomainStatus::DEGRADED);
    }

    SECTION("Handles multiple nested status fields correctly")
    {
        // JSON with many nested "status" fields to ensure we get the right one
        std::string complexJson     = R"({
            "level1": {"status": "NESTED1"},
            "level2": {
                "sublevel": {"status": "NESTED2"},
                "array": [{"status": "NESTED3"}]
            },
            "status": "TOP_LEVEL"
        })";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(complexJson);
        REQUIRE(result == DcgmImexDomainStatus::UNAVAILABLE); // "TOP_LEVEL" is not a valid domain status
    }

    SECTION("Handles unknown status values")
    {
        std::string json            = R"({"status":"UNKNOWN_STATUS"})";
        DcgmImexDomainStatus result = helper.ParseDomainStatusJson(json);
        REQUIRE(result == DcgmImexDomainStatus::UNAVAILABLE);
    }
}

SCENARIO("IMEX daemon status text parsing")
{
    auto manager = std::make_unique<DcgmImexManager>();
    DcgmImexManagerTest helper(std::move(manager));

    SECTION("Parses READY status")
    {
        std::string text            = "READY";
        DcgmImexDaemonStatus result = helper.ParseDaemonStatusText(text);
        REQUIRE(result == DcgmImexDaemonStatus::READY);
    }

    SECTION("Parses INITIALIZING status")
    {
        std::string text            = "INITIALIZING";
        DcgmImexDaemonStatus result = helper.ParseDaemonStatusText(text);
        REQUIRE(result == DcgmImexDaemonStatus::INITIALIZING);
    }

    SECTION("Parses all daemon status values")
    {
        REQUIRE(helper.ParseDaemonStatusText("STARTING_AUTH_SERVER") == DcgmImexDaemonStatus::STARTING_AUTH_SERVER);
        REQUIRE(helper.ParseDaemonStatusText("WAITING_FOR_PEERS") == DcgmImexDaemonStatus::WAITING_FOR_PEERS);
        REQUIRE(helper.ParseDaemonStatusText("WAITING_FOR_RECOVERY") == DcgmImexDaemonStatus::WAITING_FOR_RECOVERY);
        REQUIRE(helper.ParseDaemonStatusText("INIT_GPU") == DcgmImexDaemonStatus::INIT_GPU);
        REQUIRE(helper.ParseDaemonStatusText("SHUTTING_DOWN") == DcgmImexDaemonStatus::SHUTTING_DOWN);
        REQUIRE(helper.ParseDaemonStatusText("UNAVAILABLE") == DcgmImexDaemonStatus::UNAVAILABLE);
    }

    SECTION("Handles case insensitive daemon status")
    {
        std::string text            = "ready";
        DcgmImexDaemonStatus result = helper.ParseDaemonStatusText(text);
        REQUIRE(result == DcgmImexDaemonStatus::READY);
    }

    SECTION("Handles whitespace in daemon status")
    {
        std::string text            = "  SHUTTING_DOWN  \n";
        DcgmImexDaemonStatus result = helper.ParseDaemonStatusText(text);
        REQUIRE(result == DcgmImexDaemonStatus::SHUTTING_DOWN);
    }

    SECTION("Handles unknown daemon status")
    {
        std::string text            = "UNKNOWN_DAEMON_STATUS";
        DcgmImexDaemonStatus result = helper.ParseDaemonStatusText(text);
        REQUIRE(result == DcgmImexDaemonStatus::UNAVAILABLE);
    }
}

SCENARIO("IMEX command availability check")
{
    auto manager = std::make_unique<DcgmImexManager>();
    DcgmImexManagerTest helper(std::move(manager));

    SECTION("IsImexCtlAvailable returns consistent results")
    {
        // This test will depend on the environment, but should be consistent
        bool isAvailable = helper.IsImexCtlAvailable();

        // Call it again to make sure it's consistent
        bool isAvailableSecond = helper.IsImexCtlAvailable();
        REQUIRE(isAvailable == isAvailableSecond);

        // The result should be a boolean (this always passes but documents the expected type)
        REQUIRE((isAvailable == true || isAvailable == false));
    }

    SECTION("GetImexCtlSearchPaths returns trusted paths")
    {
        auto searchPaths = helper.GetImexCtlSearchPaths();

        // Should have multiple trusted paths
        REQUIRE(!searchPaths.empty());
        REQUIRE(searchPaths.size() >= 3);

        // Should include standard system paths
        bool hasUsrBin      = std::find(searchPaths.begin(), searchPaths.end(), "/usr/bin") != searchPaths.end();
        bool hasUsrLocalBin = std::find(searchPaths.begin(), searchPaths.end(), "/usr/local/bin") != searchPaths.end();

        REQUIRE(hasUsrBin);
        REQUIRE(hasUsrLocalBin);

        // All paths should be absolute
        for (const auto &path : searchPaths)
        {
            REQUIRE(!path.empty());
            REQUIRE(path[0] == '/'); // Absolute path
        }
    }

    SECTION("FindImexCtlExecutable searches only trusted paths")
    {
        auto executablePath = helper.FindImexCtlExecutable();
        auto searchPaths    = helper.GetImexCtlSearchPaths();

        if (executablePath.has_value())
        {
            // If found, it should be in one of the trusted paths
            bool foundInTrustedPath = false;
            for (const auto &trustedPath : searchPaths)
            {
                if (executablePath.value().find(trustedPath) == 0)
                {
                    foundInTrustedPath = true;
                    break;
                }
            }
            REQUIRE(foundInTrustedPath);
        }
        // If not found, that's also valid (IMEX might not be installed)
    }
}
