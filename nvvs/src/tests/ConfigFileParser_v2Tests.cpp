/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmDiagUnitTestCommon.h"
#include <ConfigFileParser_v2.h>
#include <catch2/catch.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

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

SCENARIO("FrameworkConfig::FrameworkConfig()")
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

SCENARIO("FrameworkConfig::SetFrameworkConfigValue(nvvs_fwcfg_enum field, const T& value)")
{
    FrameworkConfig fc;
    bool res = false;

    // Valid inputs
    res = fc.SetFrameworkConfigValue<>(NVVS_FWCFG_GLOBAL_OVERRIDEMINMAX, true);
    CHECK(res == true);

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

SCENARIO("FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const logFileType_enum& value)")
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

SCENARIO("FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const std::vector<unsigned int>& value)")
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

SCENARIO("FrameworkConfig::SetFrameworkConfigValue (nvvs_fwcfg_enum field, const std::string& value)")
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
