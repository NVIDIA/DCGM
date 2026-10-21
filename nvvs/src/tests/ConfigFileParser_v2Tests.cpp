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
#include <ConfigFileParser_v2.h>
#include <NvvsCommon.h>
#include <TempFile.hpp>
#include <catch2/catch_all.hpp>

#include <memory>
#include <string_view>
#include <yaml-cpp/yaml.h>


namespace
{
struct ConfigFileParserTestFixture
{
    NvvsCommon nvvsCommon;
};
} // namespace

#define CONFIG_FILE_CONTENTS                \
    "%YAML 1.2\n"                           \
    "---\n"                                 \
                                            \
    "globals:\n"                            \
    "  logfile: caseSensitiveLogFile.txt\n" \
    "  logfile_type: biNARY\n"              \
    "  overrideMinMax: yes\n"               \
    "  scriptable: true\n"                  \
    "  serial_override: true\n"             \
    "  require_persistence_mode: false\n"   \
    "  throttle-mask: 40\n"                 \
                                            \
    "test_suite_name:\n"                    \
    "- test_class_name1:\n"                 \
    "    test_name1:\n"                     \
    "      key1: value\n"                   \
    "      key2: value\n"                   \
    "      subtests:\n"                     \
    "        subtest_name1:\n"              \
    "          key1: value\n"               \
    "          key2: value\n"               \
    "    test_name2:\n"                     \
    "      key1: value\n"                   \
    "      key2: value\n"                   \
    "-test_class_name2:\n"                  \
    "    test_name3:\n"                     \
    "      key1: value\n"                   \
    "      key2: value\n"                   \
                                            \
    "not_test_suite_name:\n"                \
    "- not_test_class_name1:\n"             \
    "    not_test_name1:\n"                 \
    "      not_key1: not_value\n"           \
    "      not_key2: not_value\n"           \
    "      subtests:\n"                     \
    "        not_subtest_name1:\n"          \
    "          not_key1: not_value\n"       \
    "          not_key2: not_value\n"       \
    "    not_test_name2:\n"                 \
    "      not_key1: not_value\n"           \
    "      not_key2: not_value\n"           \
    "-not_test_class_name2:\n"              \
    "    not_test_name3:\n"                 \
    "      not_key1: not_value\n"           \
    "      not_key2: not_value\n"           \
                                            \
    "long:\n"                               \
    "- integration:\n"                      \
    "    pcie:\n"                           \
    "      test_unpinned: false\n"          \
    "      subtests:\n"                     \
    "        h2d_d2h_single_pinned:\n"      \
    "          min_bandwidth: 20\n"         \
    "          min_pci_width: 16\n"         \
                                            \
    "gpus:\n"                               \
    "- gpuset: gpusetStr\n"                 \
    "  properties:\n"                       \
    "    name: gpuname\n"                   \
    "    brand: gpubrand\n"                 \
    "    busid: gpubusid\n"                 \
    "    uuid: gpuuuid\n"                   \
    "    index: 1,3,5\n"                    \
    "  tests:\n"                            \
    "    name: gputestname\n"

#define CHECK_CONFIG_FIELD_VALUE(fc, field, value)  \
    {                                               \
        NvvsFrameworkConfig config = fc.GetFWCFG(); \
        CHECK(config.field == value);               \
    }

using namespace DcgmNs::Nvvs;

SCENARIO_METHOD(ConfigFileParserTestFixture, "FrameworkConfig::FrameworkConfig()")
{
    FrameworkConfig fc;
    NvvsFrameworkConfig config = fc.GetFWCFG();

    CHECK(config.dataFile == "stats");
    CHECK(config.dataFileType == NVVS_LOGFILE_TYPE_JSON);
    CHECK(config.overrideSerial == false);
    CHECK(config.scriptable == false);
    CHECK(config.requirePersistence == true);
    CHECK(config.index.size() == 0);
    CHECK(config.brand == "");
    CHECK(config.name == "");
    CHECK(config.busid == "");
    CHECK(config.uuid == "");
    CHECK(config.testname == "Long");
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const T& value)")
{
    FrameworkConfig fc;
    bool res = false;

    // Valid inputs
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_OVERRIDEMINMAX, true);
    CHECK(res == false);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_OVERRIDESERIAL, true);
    CHECK_CONFIG_FIELD_VALUE(fc, overrideSerial, true);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_SCRIPTABLE, true);
    CHECK_CONFIG_FIELD_VALUE(fc, scriptable, true);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_PERSISTENCE, false);
    CHECK_CONFIG_FIELD_VALUE(fc, requirePersistence, false);
    CHECK(res == true);

    // Invalid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_DATAFILETYPE, false);
    CHECK_CONFIG_FIELD_VALUE(fc, dataFileType, NVVS_LOGFILE_TYPE_JSON);
    CHECK(res == false);
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const logFileType_enum& value)")
{
    FrameworkConfig fc;
    bool res = false;

    // Valid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_DATAFILETYPE, NVVS_LOGFILE_TYPE_TEXT);
    CHECK_CONFIG_FIELD_VALUE(fc, dataFileType, NVVS_LOGFILE_TYPE_TEXT);
    CHECK(res == true);

    // Invalid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_SCRIPTABLE, NVVS_LOGFILE_TYPE_TEXT);
    CHECK_CONFIG_FIELD_VALUE(fc, scriptable, false);
    CHECK(res == false);
}

SCENARIO_METHOD(
    ConfigFileParserTestFixture,
    "FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const std::vector<unsigned int>& value)")
{
    FrameworkConfig fc;
    bool res = false;

    const std::vector<unsigned int> v({ 1, 2, 3 });

    // Valid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPU_INDEX, v);
    CHECK_CONFIG_FIELD_VALUE(fc, index, v);
    CHECK(res == true);

    // Invalid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_SCRIPTABLE, v);
    CHECK_CONFIG_FIELD_VALUE(fc, scriptable, false);
    CHECK(res == false);
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const std::string& value)")
{
    FrameworkConfig fc;
    bool res = false;

    const std::string str("teststring");

    // Valid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_DATAFILE, str);
    CHECK_CONFIG_FIELD_VALUE(fc, dataFile, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPU_BRAND, str);
    CHECK_CONFIG_FIELD_VALUE(fc, brand, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPU_NAME, str);
    CHECK_CONFIG_FIELD_VALUE(fc, name, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPU_BUSID, str);
    CHECK_CONFIG_FIELD_VALUE(fc, busid, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPU_UUID, str);
    CHECK_CONFIG_FIELD_VALUE(fc, uuid, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_TEST_NAME, str);
    CHECK_CONFIG_FIELD_VALUE(fc, testname, str);
    CHECK(res == true);

    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GPUSET_NAME, str);
    CHECK_CONFIG_FIELD_VALUE(fc, gpuSetIdentifier, str);
    CHECK(res == true);

    // Invalid input
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_SCRIPTABLE, str);
    CHECK_CONFIG_FIELD_VALUE(fc, scriptable, false);
    CHECK(res == false);
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "Load Default Config")
{
    std::string error("We caught an exception parsing the default yaml!");
    FrameworkConfig fc;
    try
    {
        // We will throw an error if there's a problem with the default yaml
        ConfigFileParser_v2 cfp(&nvvsCommon, "", fc);
        cfp.Init();
    }
    catch (std::runtime_error &e)
    {
        CHECK(error == e.what());
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 merges user SKU YAML")
{
    GIVEN("a user config file with one SKU and nested test parameters")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: user-sku\n"
                                                               "  name: User SKU\n"
                                                               "  diagnostic:\n"
                                                               "    simple_param: enabled\n"
                                                               "    subtests:\n"
                                                               "      fast: 3\n",
                                                               "user-sku-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            bool const initialized = parser.Init();

            THEN("the SKU is available with diagnostic values copied")
            {
                REQUIRE(initialized);
                auto const &skus = parser.GetSkus();
                REQUIRE(skus.contains("user-sku"));
                CHECK(skus.at("user-sku")["diagnostic"]["simple_param"].as<std::string>() == "enabled");
                CHECK(skus.at("user-sku")["diagnostic"]["subtests"]["fast"].as<int>() == 3);
            }
        }
    }
}
SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 rejects unsupported user YAML node shapes")
{
    GIVEN("a user config with a sequence where a test parameter or subtest map is expected")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: bad-sku\n"
                                                               "  diagnostic:\n"
                                                               "    invalid:\n"
                                                               "    - list-value\n",
                                                               "bad-test-child-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            THEN("a runtime error identifies the invalid child node")
            {
                REQUIRE_THROWS_WITH(parser.Init(),
                                    Catch::Matchers::ContainsSubstring("A test's child node must be a param"));
            }
        }
    }

    GIVEN("a user config with a nested subtest value that is not scalar")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: bad-subtest-sku\n"
                                                               "  diagnostic:\n"
                                                               "    subtests:\n"
                                                               "      fast:\n"
                                                               "      - not-scalar\n",
                                                               "bad-subtest-child-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            THEN("a runtime error identifies the invalid subtest child")
            {
                REQUIRE_THROWS_WITH(parser.Init(),
                                    Catch::Matchers::ContainsSubstring("Subtest child must be a scalar"));
            }
        }
    }
}


SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 applies deterministic user SKU overrides")
{
    auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                           "---\n"
                                                           "skus:\n"
                                                           "- id: repeated-sku\n"
                                                           "  name: Original Name\n"
                                                           "  diagnostic:\n"
                                                           "    duration: 30\n"
                                                           "    subtests:\n"
                                                           "      fast: enabled\n"
                                                           "      slow: disabled\n"
                                                           "- id: repeated-sku\n"
                                                           "  name: Override Name\n"
                                                           "  diagnostic:\n"
                                                           "    duration: 45\n"
                                                           "    threshold: strict\n"
                                                           "    subtests:\n"
                                                           "      slow: enabled\n",
                                                           "merged-user-sku-yaml-");
    auto const configFileName = configFile->Path().string();

    FrameworkConfig fc;
    ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);
    REQUIRE(parser.Init());

    auto const &skus = parser.GetSkus();
    REQUIRE(skus.contains("repeated-sku"));
    YAML::Node const sku = skus.at("repeated-sku");
    CHECK(sku["diagnostic"]["duration"].as<int>() == 45);
    CHECK(sku["diagnostic"]["threshold"].as<std::string>() == "strict");
    CHECK(sku["diagnostic"]["subtests"]["fast"].as<std::string>() == "enabled");
    CHECK(sku["diagnostic"]["subtests"]["slow"].as<std::string>() == "enabled");
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 rejects invalid SKU scalar fields")
{
    SECTION("name must be scalar")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: bad-name\n"
                                                               "  name:\n"
                                                               "    nested: value\n",
                                                               "bad-sku-name-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);
        REQUIRE_THROWS_WITH(parser.Init(), Catch::Matchers::ContainsSubstring("Key must be scalar: name"));
    }

    SECTION("test children must be maps")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: bad-test\n"
                                                               "  diagnostic: scalar-not-map\n",
                                                               "bad-test-node-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);
        REQUIRE_THROWS_WITH(
            parser.Init(), Catch::Matchers::ContainsSubstring("A SKU's child that is not in {name, id} must be a map"));
    }
}


SCENARIO_METHOD(ConfigFileParserTestFixture,
                "ConfigFileParser_v2 propagates framework globals and ignores incomplete SKU entries")
{
    GIVEN("framework defaults overridden before parser construction")
    {
        nvvsCommon.parse = true;

        FrameworkConfig fc;
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_DATAFILE, std::string("custom-stats")));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_DATAFILETYPE, NVVS_LOGFILE_TYPE_TEXT));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_OVERRIDESERIAL, true));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_SCRIPTABLE, false));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_PERSISTENCE, false));
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("legacy globals are copied to the injected NvvsCommon")
        {
            parser.legacyGlobalStructHelper();

            THEN("all mapped values are copied and command-line parse mode is preserved")
            {
                CHECK(nvvsCommon.logFile == "custom-stats");
                CHECK(nvvsCommon.logFileType == NVVS_LOGFILE_TYPE_TEXT);
                CHECK(nvvsCommon.serialize == true);
                CHECK(nvvsCommon.parse == true);
                CHECK(nvvsCommon.requirePersistenceMode == false);
            }
        }
    }

    GIVEN("a user config containing malformed SKU entries that should be ignored")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- just-a-scalar\n"
                                                               "- name: Missing Id\n"
                                                               "  diagnostic:\n"
                                                               "    duration: 5\n"
                                                               "- id: valid-sku\n"
                                                               "  diagnostic:\n"
                                                               "    duration: 9\n",
                                                               "ignored-skus-yaml-");
        auto const configFileName = configFile->Path().string();

        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            bool const initialized = parser.Init();

            THEN("only the valid user SKU is merged")
            {
                REQUIRE(initialized);
                auto const &skus = parser.GetSkus();
                REQUIRE(skus.contains("valid-sku"));
                CHECK(skus.at("valid-sku")["diagnostic"]["duration"].as<int>() == 9);
                CHECK_FALSE(skus.contains("Missing Id"));
            }
        }
    }
}


SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 handles user file selection and parse errors")
{
    GIVEN("a parser constructed without a user config")
    {
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "- id: late-sku\n"
                                                               "  name: Late SKU\n"
                                                               "  diagnostic:\n"
                                                               "    duration: 11\n",
                                                               "late-user-config-yaml-");
        auto const configFileName = configFile->Path().string();

        WHEN("the user config is set before initialization")
        {
            parser.setConfigFile(configFileName);
            bool const initialized = parser.Init();

            THEN("the late-selected SKU is merged")
            {
                REQUIRE(initialized);
                auto const &skus = parser.GetSkus();
                REQUIRE(skus.contains("late-sku"));
                CHECK(skus.at("late-sku")["diagnostic"]["duration"].as<int>() == 11);
            }
        }
    }

    GIVEN("a parser pointed at a missing user config")
    {
        FrameworkConfig fc;
        auto missingBase         = TempFile::Random("", "missing-user-config-");
        auto const missingConfig = missingBase.Path().string() + ".missing";
        ConfigFileParser_v2 parser(&nvvsCommon, missingConfig, fc);

        WHEN("the parser is initialized")
        {
            THEN("the YAML file error is surfaced to the caller")
            {
                REQUIRE_THROWS_AS(parser.Init(), YAML::BadFile);
            }
        }
    }

    GIVEN("a parser pointed at syntactically invalid YAML")
    {
        auto configFile = TempFile::MakeUniqueRandom(
            "%YAML 1.2\n---\nskus:\n- id: broken\n  diagnostic: [unterminated\n", "invalid-yaml-");
        auto const configFileName = configFile->Path().string();
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            THEN("the YAML parser exception is not swallowed")
            {
                REQUIRE_THROWS_AS(parser.Init(), YAML::ParserException);
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "ConfigFileParser_v2 ignores missing top-level SKU sequences without discarding valid fallback data")
{
    GIVEN("a user config whose skus node is not a sequence")
    {
        auto configFile           = TempFile::MakeUniqueRandom("%YAML 1.2\n"
                                                               "---\n"
                                                               "skus:\n"
                                                               "  id: not-a-sequence\n",
                                                               "bad-skus-root-yaml-");
        auto const configFileName = configFile->Path().string();
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, configFileName, fc);

        WHEN("the parser is initialized")
        {
            bool const initialized = parser.Init();

            THEN("the invalid user node is ignored while fallback SKUs remain available")
            {
                REQUIRE(initialized);
                CHECK_FALSE(parser.GetSkus().empty());
                CHECK_FALSE(parser.GetSkus().contains("not-a-sequence"));
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "FrameworkConfig copies are independent after assignment")
{
    GIVEN("a framework config with several explicit values")
    {
        FrameworkConfig original;
        REQUIRE(original.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_DATAFILE, std::string("stats-a")));
        REQUIRE(original.SetFrameworkConfigValue(NVVS_FWCFG_GPU_INDEX, std::vector<unsigned int> { 1, 3 }));
        REQUIRE(original.SetFrameworkConfigValue(NVVS_FWCFG_TEST_NAME, std::string("diagnostic")));

        WHEN("the config is copied and the original is changed")
        {
            FrameworkConfig copy = original;
            REQUIRE(original.SetFrameworkConfigValue(NVVS_FWCFG_GLOBAL_DATAFILE, std::string("stats-b")));
            REQUIRE(original.SetFrameworkConfigValue(NVVS_FWCFG_GPU_INDEX, std::vector<unsigned int> { 9 }));

            THEN("the copy retains the values from the time it was copied")
            {
                auto copied = copy.GetFWCFG();
                CHECK(copied.dataFile == "stats-a");
                CHECK(copied.index == std::vector<unsigned int> { 1, 3 });
                CHECK(copied.testname == "diagnostic");

                auto changed = original.GetFWCFG();
                CHECK(changed.dataFile == "stats-b");
                CHECK(changed.index == std::vector<unsigned int> { 9 });
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 prepares GPU entity sets from command-line selectors")
{
    GIVEN("a parser with an explicit test name and GPU set name")
    {
        nvvsCommon.entityIds.clear();
        FrameworkConfig fc;
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_TEST_NAME, std::string("diagnostic")));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPUSET_NAME, std::string("selected-gpus")));

        WHEN("the legacy GPU index selector is present")
        {
            nvvsCommon.indexString = "1,3";
            ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

            parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0));

            THEN("a GPU entity set is created and the requested test is defaulted")
            {
                auto &sets = parser.GetEntitySets();
                REQUIRE(sets.size() == 1);
                CHECK(sets[0]->GetEntityGroup() == DCGM_FE_GPU);
                CHECK(sets[0]->GetName() == "selected-gpus");
                CHECK(sets[0]->GetEntityIds() == std::vector<dcgm_field_eid_t> { 1, 3 });
                CHECK(nvvsCommon.desiredTest.contains("diagnostic"));
            }
        }

        WHEN("fake GPU ids and regular GPU indexes are both present")
        {
            nvvsCommon.indexString    = "1,3";
            nvvsCommon.fakegpusString = "8,9";
            ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

            parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0));

            THEN("the fake GPU selector takes precedence")
            {
                auto &sets = parser.GetEntitySets();
                REQUIRE(sets.size() == 1);
                CHECK(sets[0]->GetEntityGroup() == DCGM_FE_GPU);
                CHECK(sets[0]->GetEntityIds() == std::vector<dcgm_field_eid_t> { 8, 9 });
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "ConfigFileParser_v2 rejects duplicate GPU ids while preparing entity sets")
{
    GIVEN("a parser with duplicate fake GPU ids")
    {
        nvvsCommon.entityIds.clear();
        nvvsCommon.fakegpusString = "2,2";
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            THEN("the duplicate list is rejected before a set is published")
            {
                REQUIRE_THROWS_MATCHES(
                    parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0)),
                    std::runtime_error,
                    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("duplicate IDs")));
                CHECK(parser.GetEntitySets().empty());
            }
        }
    }

    GIVEN("a parser with duplicate framework GPU indexes")
    {
        nvvsCommon.entityIds.clear();
        FrameworkConfig fc;
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPU_INDEX, std::vector<unsigned int> { 4, 4 }));
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            THEN("the duplicate framework list is rejected before a set is published")
            {
                REQUIRE_THROWS_MATCHES(
                    parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0)),
                    std::runtime_error,
                    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("duplicate IDs")));
                CHECK(parser.GetEntitySets().empty());
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture, "ConfigFileParser_v2 handles empty and malformed entity selectors")
{
    GIVEN("no command-line or framework GPU selectors")
    {
        nvvsCommon.desiredTest.clear();
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0));

            THEN("no empty GPU or CPU set is published and no default test is requested")
            {
                CHECK(parser.GetEntitySets().empty());
                CHECK(nvvsCommon.desiredTest.empty());
            }
        }
    }

    GIVEN("a malformed entity-id selector")
    {
        nvvsCommon.entityIds = "dcgm:0";
        FrameworkConfig fc;
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            THEN("the entity parser error is surfaced before publishing any sets")
            {
                REQUIRE_THROWS_MATCHES(
                    parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0)),
                    std::runtime_error,
                    Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("failed to parse entity ids")));
                CHECK(parser.GetEntitySets().empty());
            }
        }
    }
}

SCENARIO_METHOD(ConfigFileParserTestFixture,
                "ConfigFileParser_v2 prepares GPU entity sets from framework configuration")
{
    GIVEN("framework GPU indexes and metadata without command-line selectors")
    {
        nvvsCommon.entityIds.clear();
        nvvsCommon.indexString.clear();
        nvvsCommon.fakegpusString.clear();
        FrameworkConfig fc;
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPUSET_NAME, std::string("framework-gpus")));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_TEST_NAME, std::string("memtest")));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPU_INDEX, std::vector<unsigned int> { 2, 5 }));
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0));

            THEN("the framework GPU indexes publish a named GPU set and default desired test")
            {
                auto &sets = parser.GetEntitySets();
                REQUIRE(sets.size() == 1);
                CHECK(sets[0]->GetEntityGroup() == DCGM_FE_GPU);
                CHECK(sets[0]->GetName() == "framework-gpus");
                CHECK(sets[0]->GetEntityIds() == std::vector<dcgm_field_eid_t> { 2, 5 });
                CHECK(nvvsCommon.desiredTest.contains("memtest"));
            }
        }
    }

    GIVEN("framework GPU properties without explicit indexes")
    {
        nvvsCommon.entityIds.clear();
        FrameworkConfig fc;
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPU_BRAND, std::string("NVIDIA")));
        REQUIRE(fc.SetFrameworkConfigValue(NVVS_FWCFG_GPU_NAME, std::string("A100")));
        ConfigFileParser_v2 parser(&nvvsCommon, "", fc);

        WHEN("entity sets are prepared")
        {
            parser.PrepareEntitySets(static_cast<dcgmHandle_t>(0));

            THEN("a property-selected GPU set is published even when the concrete IDs are deferred")
            {
                auto &sets = parser.GetEntitySets();
                REQUIRE(sets.size() == 1);
                CHECK(sets[0]->GetEntityGroup() == DCGM_FE_GPU);
                CHECK(sets[0]->GetEntityIds().empty());
                CHECK(nvvsCommon.desiredTest.contains("Long"));
            }
        }
    }
}
