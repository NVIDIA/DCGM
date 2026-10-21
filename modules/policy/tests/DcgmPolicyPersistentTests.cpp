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

#include <DcgmLogging.h>
#include <DcgmPolicyPersistent.h>
#include <DcgmPolicyRegistry.h>
#include <ScopedEnvVars.hpp>
#include <TempDir.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>
#include <sys/stat.h>

// Test helper: snapshot all policies from the registry.
static std::vector<DcgmPolicy> GetAllPoliciesOrFail(DcgmPolicyRegistry &registry)
{
    std::vector<DcgmPolicy> policies;
    (void)registry.GetAllPolicies(std::back_inserter(policies));
    std::sort(policies.begin(), policies.end(), [](DcgmPolicy const &a, DcgmPolicy const &b) {
        return a.GetFieldId() < b.GetFieldId();
    });
    return policies;
}

/**
 * Test fixture for DcgmPolicyPersistence tests
 */
class TestPolicyPersistence
{
public:
    // clang-format off
    static constexpr std::string_view VALID_POLICIES_YAML = R"(policies:
  - fieldId: 150
    name: "GPU Temperature Warning"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: [{ entityGroupId: 1, entityId: 0 }, { entityGroupId: 1, entityId: 1 }, { entityGroupId: 1, entityId: 2 }, { entityGroupId: 1, entityId: 3 }]
    channels: ["CONSOLE", "FILE"]
    status: "ENABLED"
    rateLimitSeconds: 60

  - fieldId: 155
    name: "Power Limit Exceeded"
    operator: "GREATER_THAN"
    threshold: 300.0
    entities: []
    channels: ["CONSOLE", "CALLBACK"]
    status: "ENABLED"
    rateLimitSeconds: 30
)";
    // clang-format on

    DcgmPolicyRegistry m_registry;

    TestPolicyPersistence()
        : m_tempDir(TempDir::Random("dcgm_policy_test_"))
        , m_testDir(m_tempDir.Path())
        , m_testFile(m_testDir / "test_policies.yaml")
    {
        // Permit the /tmp scratch area for tests via the allowed-directories override
        m_env.Add("DCGM_POLICY_ALLOWED_DIRS", "/tmp");
    }

    std::filesystem::path CreateValidYamlFile(std::filesystem::path const &path = {})
    {
        std::filesystem::path filePath = path.empty() ? m_testFile : path;

        std::ofstream file(filePath);
        file << VALID_POLICIES_YAML;
        file.close();
        return filePath;
    }

    std::filesystem::path CreateMalformedYamlFile()
    {
        std::ofstream file(m_testFile);
        file << "policies:\n  - fieldId: invalid_value\n";
        file.close();
        return m_testFile;
    }

    TempDir m_tempDir;
    std::filesystem::path m_testDir;
    std::filesystem::path m_testFile;
    ScopedEnvVars m_env;
};

TEST_CASE("DcgmPolicyPersistence")
{
    SECTION("Path resolution")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence defaultPersistence(fixture.m_registry);
        REQUIRE(defaultPersistence.GetConfigPath()
                == std::filesystem::path("/etc/datacenter-gpu-manager-4/policies.yaml"));

        DcgmPolicyPersistence constructorPersistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(constructorPersistence.GetConfigPath() == fixture.m_testFile);

        REQUIRE(defaultPersistence.SetConfigPath(fixture.m_testFile) == DCGM_ST_OK);
        REQUIRE(defaultPersistence.GetConfigPath() == fixture.m_testFile);

        fixture.m_env.Add("DCGM_POLICY_CONFIG_PATH", fixture.m_testFile.string());
        DcgmPolicyPersistence envPersistence(fixture.m_registry);
        REQUIRE(envPersistence.GetConfigPath() == fixture.m_testFile);
    }

    SECTION("Path validation - allowed directories")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence tmpPersistence(fixture.m_registry, "/tmp/test.yaml");
        REQUIRE(tmpPersistence.LoadPolicies() == DCGM_ST_FILE_IO_ERROR);
        // Paths under allowed directories should fail only because the files don't exist
        std::array<std::string_view, 2> const allowedDirs
            = { "/etc/datacenter-gpu-manager-4/", "/var/log/nvidia-dcgm/" };
        for (auto const &dir : allowedDirs)
        {
            DcgmPolicyPersistence persistence(fixture.m_registry, std::filesystem::path(dir) / "test.yaml");
            REQUIRE(persistence.LoadPolicies() == DCGM_ST_FILE_IO_ERROR);
        }
    }

    SECTION("Path validation - disallowed directories")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence etcPersistence(fixture.m_registry, "/etc/passwd");
        REQUIRE(etcPersistence.LoadPolicies() == DCGM_ST_BADPARAM);

        DcgmPolicyPersistence homePersistence(fixture.m_registry, "/home/user/test.yaml");
        REQUIRE(homePersistence.LoadPolicies() == DCGM_ST_BADPARAM);
    }

    SECTION("Path validation - directory traversal attacks")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence etcTraversal(fixture.m_registry, "/tmp/../etc/passwd");
        REQUIRE(etcTraversal.LoadPolicies() == DCGM_ST_BADPARAM);

        DcgmPolicyPersistence rootTraversal(fixture.m_registry, "/tmp/../../root/.ssh/id_rsa");
        REQUIRE(rootTraversal.LoadPolicies() == DCGM_ST_BADPARAM);
    }

    SECTION("Load policies successfully - verifies fields, operators and channels")
    {
        TestPolicyPersistence fixture;

        fixture.CreateValidYamlFile();
        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 2);
        REQUIRE(policies[0].GetFieldId() == 150);
        REQUIRE(policies[0].GetThreshold() == 85.0f);
        REQUIRE(policies[0].GetPolicyOperator() == DCGM_POLICY_OP_GT);
        REQUIRE(policies[1].GetFieldId() == 155);

        unsigned int channelMask = policies[0].GetChannelMask();
        REQUIRE((channelMask & DCGM_POLICY_CHANNEL_CONSOLE) != 0);
        REQUIRE((channelMask & DCGM_POLICY_CHANNEL_FILE) != 0);
        REQUIRE(channelMask == (DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE));
    }

    SECTION("Load policies rejects out-of-range field ID")
    {
        TestPolicyPersistence fixture;
        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 65536
    name: "Out of Range"
    operator: "GREATER_THAN"
    threshold: 70.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
    }

    SECTION("Load policies - file not found")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testDir / "nonexistent.yaml");
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_FILE_IO_ERROR);
    }

    SECTION("Load policies - malformed YAML")
    {
        TestPolicyPersistence fixture;

        fixture.CreateMalformedYamlFile();
        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
    }

    SECTION("Load policies - preserves double precision threshold")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Precise Threshold"
    operator: "GREATER_THAN"
    threshold: 300.123456789
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 1);
        REQUIRE(policies[0].GetThreshold() == Catch::Approx(300.123456789));
    }

    SECTION("Load policies - malformed policy node does not skip later valid nodes")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - name: "Missing Field ID"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60

  - fieldId: 155
    name: "Power Limit Exceeded"
    operator: "GREATER_THAN"
    threshold: 300.0
    entities: []
    channels: ["CONSOLE", "CALLBACK"]
    status: "ENABLED"
    rateLimitSeconds: 30
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 1);
        REQUIRE(policies[0].GetFieldId() == 155);
    }

    SECTION("Load policies - missing entities key means all GPUs")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "GPU Temperature Warning"
    operator: "GREATER_THAN"
    threshold: 85.0
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 1);
        REQUIRE(policies[0].GetEntities().empty());
    }

    SECTION("Load policies - symlink rejection")
    {
        TestPolicyPersistence fixture;

        fixture.CreateValidYamlFile();
        std::filesystem::path symlinkPath = fixture.m_testDir / "symlink.yaml";
        std::filesystem::create_symlink(fixture.m_testFile, symlinkPath);

        DcgmPolicyPersistence persistence(fixture.m_registry, symlinkPath);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_FILE_IO_ERROR);
    }

    SECTION("Save policies - directory creation and atomic write")
    {
        TestPolicyPersistence fixture;

        std::filesystem::path newFile = fixture.m_testDir / "nested/deep/policies.yaml";
        DcgmPolicyPersistence persistence(fixture.m_registry, newFile);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);
        REQUIRE(std::filesystem::exists(newFile));
        REQUIRE_FALSE(std::filesystem::exists(std::filesystem::path(newFile) += ".tmp"));
    }

    SECTION("Save policies - symlink attack prevention")
    {
        TestPolicyPersistence fixture;

        std::filesystem::path target = "/tmp/malicious_target.yaml";
        std::ofstream(target).close();
        std::filesystem::create_symlink(target, fixture.m_testFile);

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_FILE_IO_ERROR);

        std::filesystem::remove(fixture.m_testFile);
        std::filesystem::remove(target);
    }

    SECTION("Save policies - temp symlink precreation is rejected")
    {
        TestPolicyPersistence fixture;

        std::filesystem::path target = "/tmp/malicious_temp_target.yaml";
        std::filesystem::remove(target);
        std::ofstream(target) << "original";

        std::filesystem::path tempPath = fixture.m_testFile;
        tempPath += ".tmp";
        std::filesystem::create_symlink(target, tempPath);

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_FILE_IO_ERROR);

        std::ifstream targetFile(target);
        std::string targetContent;
        targetFile >> targetContent;
        REQUIRE(targetContent == "original");

        std::filesystem::remove(tempPath);
        std::filesystem::remove(target);
    }

    SECTION("Save policies - stale regular temp file is replaced")
    {
        TestPolicyPersistence fixture;

        std::filesystem::path tempPath = fixture.m_testFile;
        tempPath += ".tmp";
        std::ofstream(tempPath) << "stale";

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);
        REQUIRE(std::filesystem::exists(fixture.m_testFile));
        REQUIRE_FALSE(std::filesystem::exists(tempPath));
    }

    SECTION("Config file presence reflected by LoadPolicies result")
    {
        TestPolicyPersistence fixture;

        fixture.CreateValidYamlFile();
        DcgmPolicyPersistence persistence1(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence1.LoadPolicies() == DCGM_ST_OK);

        DcgmPolicyPersistence persistence2(fixture.m_registry, fixture.m_testDir / "nonexistent.yaml");
        REQUIRE(persistence2.LoadPolicies() == DCGM_ST_FILE_IO_ERROR);
    }

    SECTION("Invalid operator causes policy to be skipped")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Test"
    operator: "INVALID_OPERATOR"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // All entries invalid — should fail rather than silently wipe the registry
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).empty());
    }

    SECTION("Negative rate limit causes policy to be skipped")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: -1
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).empty());
    }

    SECTION("Missing rate limit defaults to 60 seconds")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);
        auto const policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 1);
        REQUIRE(policies[0].GetRateLimit() == std::chrono::seconds { 60 });
    }

    SECTION("Invalid channel causes policy to be skipped")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["INVALID_CHANNEL"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // All entries invalid — should fail rather than silently wipe the registry
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).empty());
    }

    SECTION("Empty channels causes policy to be skipped")
    {
        TestPolicyPersistence fixture;

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: []
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).empty());
    }

    SECTION("Empty policies save and load")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Save empty registry
        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);

        // Load back should work but registry stays empty
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).empty());
    }

    SECTION("All policy operators supported")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        std::vector<dcgmPolicyOperator_t> operators
            = { DCGM_POLICY_OP_EQ, DCGM_POLICY_OP_NE, DCGM_POLICY_OP_GT,     DCGM_POLICY_OP_LT,
                DCGM_POLICY_OP_GE, DCGM_POLICY_OP_LE, DCGM_POLICY_OP_CHANGED };

        int fieldId = 150;
        for (auto op : operators)
        {
            REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(static_cast<unsigned short>(fieldId++),
                                                            "Op Test",
                                                            0.0,
                                                            85.0,
                                                            op,
                                                            std::vector<dcgmGroupEntityPair_t> {},
                                                            DCGM_POLICY_CHANNEL_CONSOLE,
                                                            DCGM_POLICY_STATE_ENABLED,
                                                            60))
                    == DCGM_ST_OK);
        }

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);

        // Clear and reload — RemovePolicy now requires (fieldId, policyId). For a bulk wipe
        // we don't care about individual ids, so call RemoveAllPolicies.
        REQUIRE(fixture.m_registry.RemoveAllPolicies() == DCGM_ST_OK);

        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).size() == operators.size());
    }

    SECTION("All notification channels supported")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(fixture.m_registry.AddPolicy(
                    DcgmPolicy(200,
                               "All Channels",
                               0.0,
                               85.0,
                               DCGM_POLICY_OP_GT,
                               std::vector<dcgmGroupEntityPair_t> {},
                               DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE | DCGM_POLICY_CHANNEL_CALLBACK,
                               DCGM_POLICY_STATE_ENABLED,
                               60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);

        REQUIRE(fixture.m_registry.RemoveAllPolicies() == DCGM_ST_OK);
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto loaded = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE((loaded[0].GetChannelMask()
                 & (DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE | DCGM_POLICY_CHANNEL_CALLBACK))
                == (DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE | DCGM_POLICY_CHANNEL_CALLBACK));
    }

    SECTION("Save and hot reload - file is created and registry is replaced")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Add runtime policies to registry
        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(100,
                                                        "Runtime Policy 1",
                                                        0.0,
                                                        static_cast<double>(50.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        30))
                == DCGM_ST_OK);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(200,
                                                        "Runtime Policy 2",
                                                        0.0,
                                                        static_cast<double>(75.0),
                                                        DCGM_POLICY_OP_LT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_FILE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        45))
                == DCGM_ST_OK);

        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).size() == 2);

        // Save runtime policies to disk, then reload from a different YAML file
        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);
        REQUIRE(std::filesystem::exists(fixture.m_testFile));

        fixture.CreateValidYamlFile(); // Overwrites with policies fieldId 150 and 155
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        // Verify: Runtime policies (100, 200) are GONE, replaced by YAML policies (150, 155)
        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 2);
        REQUIRE(policies[0].GetFieldId() == 150);
        REQUIRE(policies[1].GetFieldId() == 155);

        // Verify old policies are not present
        REQUIRE(!fixture.m_registry.HasPolicy(100));
        REQUIRE(!fixture.m_registry.HasPolicy(200));
    }

    SECTION("Schema version - saved in YAML output")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Add policy and save
        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);

        // Read file and verify schema_version is present
        std::ifstream file(fixture.m_testFile);
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        REQUIRE(content.find("schema_version: 1") != std::string::npos);
    }

    SECTION("Schema version - loads files without version (defaults to 1)")
    {
        TestPolicyPersistence fixture;

        // Create YAML file without schema_version
        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "No Version Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Should load successfully (defaults to version 1)
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).size() == 1);
    }

    SECTION("Schema version - rejects future versions")
    {
        TestPolicyPersistence fixture;

        // Create YAML file with future schema version
        std::ofstream file(fixture.m_testFile);
        file << R"(schema_version: 999
policies:
  - fieldId: 150
    name: "Future Version Test"
    operator: "GREATER_THAN"
    threshold: 85.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
    rateLimitSeconds: 60
)";
        file.close();

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Should reject future schema version
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_BADPARAM);
    }

    SECTION("File permissions - saved with 0600 permissions")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Add policy and save
        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(150,
                                                        "Test",
                                                        0.0,
                                                        static_cast<double>(85.0),
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        REQUIRE(persistence.SavePolicies() == DCGM_ST_OK);

        // Verify file permissions are 0600 (owner read/write only)
        struct stat fileStat;
        REQUIRE(stat(fixture.m_testFile.c_str(), &fileStat) == 0); // NOLINT: stat needs c_str()

        // Check permissions bits (mask out file type bits)
        mode_t permissions = fileStat.st_mode & 0777;
        REQUIRE(permissions == 0600);
    }

    SECTION("LoadPolicies atomically replaces existing registry contents")
    {
        TestPolicyPersistence fixture;

        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        // Pre-populate registry with policies not in the YAML file
        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(999,
                                                        "Old Policy",
                                                        0.0,
                                                        1.0,
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).size() == 1);

        // Load YAML — should replace registry entirely
        fixture.CreateValidYamlFile(); // fieldId 150 and 155
        REQUIRE(persistence.LoadPolicies() == DCGM_ST_OK);

        auto policies = GetAllPoliciesOrFail(fixture.m_registry);
        REQUIRE(policies.size() == 2);

        // Old policy must be gone
        REQUIRE(!fixture.m_registry.HasPolicy(999));
        // New policies must be present
        REQUIRE(fixture.m_registry.HasPolicy(150));
        REQUIRE(fixture.m_registry.HasPolicy(155));
    }

    SECTION("LoadPolicies preserves existing registry when replacement is invalid")
    {
        TestPolicyPersistence fixture;
        DcgmPolicyPersistence persistence(fixture.m_registry, fixture.m_testFile);

        REQUIRE(fixture.m_registry.AddPolicy(DcgmPolicy(999,
                                                        "Existing Policy",
                                                        0.0,
                                                        1.0,
                                                        DCGM_POLICY_OP_GT,
                                                        std::vector<dcgmGroupEntityPair_t> {},
                                                        DCGM_POLICY_CHANNEL_CONSOLE,
                                                        DCGM_POLICY_STATE_ENABLED,
                                                        60))
                == DCGM_ST_OK);

        std::ofstream file(fixture.m_testFile);
        file << R"(policies:
  - fieldId: 150
    name: "First"
    operator: "GREATER_THAN"
    threshold: 70.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
  - fieldId: 150
    name: "Duplicate"
    operator: "GREATER_THAN"
    threshold: 80.0
    entities: []
    channels: ["CONSOLE"]
    status: "ENABLED"
)";
        file.close();

        REQUIRE(persistence.LoadPolicies() == DCGM_ST_DUPLICATE_KEY);
        REQUIRE(fixture.m_registry.HasPolicy(999));
        REQUIRE(GetAllPoliciesOrFail(fixture.m_registry).size() == 1);
    }
}
