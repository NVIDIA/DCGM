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

#include <Defer.hpp>

#include <dcgmi_common.h>

#include <dcgm_fields.h>

#include <string>
#include <vector>

TEST_CASE("ComputeItemsPerLine basic computation")
{
    using namespace DcgmNs::Terminal;

    SECTION("Simple case: 100 width, 20 item width")
    {
        auto result = ComputeItemsPerLine(20, 100);
        REQUIRE(result == 5); // 100 / 20 = 5
    }

    SECTION("Rounds down: 100 width, 30 item width")
    {
        auto result = ComputeItemsPerLine(30, 100);
        REQUIRE(result == 3); // 100 / 30 = 3.33 -> 3
    }

    SECTION("Very narrow: 50 width, 20 item width")
    {
        auto result = ComputeItemsPerLine(20, 50);
        REQUIRE(result == 2); // 50 / 20 = 2.5 -> 2
    }
}

TEST_CASE("dcgmi common helper parsing")
{
    SECTION("special entity group ids")
    {
        GIVEN("the all_gpus special group name")
        {
            std::string groupIdStr = "all_gpus";
            dcgmGroupType_t groupType {};
            dcgmGpuGrp_t groupId {};

            WHEN("the group id is parsed")
            {
                bool result = dcgmi_entity_group_id_is_special(groupIdStr, &groupType, &groupId);

                THEN("the default GPU group is selected")
                {
                    CHECK(result);
                    CHECK(groupType == DCGM_GROUP_DEFAULT);
                    CHECK(groupId == static_cast<dcgmGpuGrp_t>(DCGM_GROUP_ALL_GPUS));
                }
            }
        }

        GIVEN("the all_nvswitches special group name")
        {
            std::string groupIdStr = "all_nvswitches";
            dcgmGroupType_t groupType {};
            dcgmGpuGrp_t groupId {};

            WHEN("the group id is parsed")
            {
                bool result = dcgmi_entity_group_id_is_special(groupIdStr, &groupType, &groupId);

                THEN("the default NvSwitch group is selected")
                {
                    CHECK(result);
                    CHECK(groupType == DCGM_GROUP_DEFAULT_NVSWITCHES);
                    CHECK(groupId == static_cast<dcgmGpuGrp_t>(DCGM_GROUP_ALL_NVSWITCHES));
                }
            }
        }

        GIVEN("a regular group id")
        {
            std::string groupIdStr    = "7";
            dcgmGroupType_t groupType = DCGM_GROUP_DEFAULT;
            dcgmGpuGrp_t groupId      = static_cast<dcgmGpuGrp_t>(99);

            WHEN("the group id is parsed")
            {
                bool result = dcgmi_entity_group_id_is_special(groupIdStr, &groupType, &groupId);

                THEN("the caller can create or look up an explicit group")
                {
                    CHECK_FALSE(result);
                    CHECK(groupType == DCGM_GROUP_EMPTY);
                    CHECK(groupId == static_cast<dcgmGpuGrp_t>(99));
                }
            }
        }
    }

    SECTION("field id list parsing")
    {
        GIVEN("a comma-separated list with validation disabled")
        {
            std::vector<unsigned short> fieldIds;

            WHEN("the list is parsed")
            {
                dcgmReturn_t result = dcgmi_parse_field_id_list_string("0,150,155", fieldIds, false);

                THEN("the numeric values are returned in order")
                {
                    REQUIRE(result == DCGM_ST_OK);
                    REQUIRE(fieldIds.size() == 3);
                    CHECK(fieldIds[0] == 0);
                    CHECK(fieldIds[1] == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                    CHECK(fieldIds[2] == DCGM_FI_DEV_BOARD_POWER_WATTS);
                }
            }
        }

        GIVEN("a valid field list with validation enabled")
        {
            REQUIRE(DcgmFieldsInit() == 0);
            DcgmNs::Defer cleanup([] { DcgmFieldsTerm(); });
            std::vector<unsigned short> fieldIds;

            WHEN("the list is parsed")
            {
                dcgmReturn_t result
                    = dcgmi_parse_field_id_list_string(std::to_string(DCGM_FI_DEV_GPU_TEMP_CELSIUS), fieldIds, true);

                THEN("the valid DCGM field is accepted")
                {
                    REQUIRE(result == DCGM_ST_OK);
                    REQUIRE(fieldIds.size() == 1);
                    CHECK(fieldIds[0] == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
                }
            }
        }

        GIVEN("a non-numeric field token")
        {
            std::vector<unsigned short> fieldIds { DCGM_FI_DEV_BOARD_POWER_WATTS };

            WHEN("the list is parsed")
            {
                dcgmReturn_t result = dcgmi_parse_field_id_list_string("12,bad", fieldIds, false);

                THEN("bad parameter is returned after preserving previously appended values")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    REQUIRE(fieldIds.size() == 2);
                    CHECK(fieldIds[0] == DCGM_FI_DEV_BOARD_POWER_WATTS);
                    CHECK(fieldIds[1] == 12);
                }
            }
        }

        GIVEN("an invalid DCGM field id with validation enabled")
        {
            REQUIRE(DcgmFieldsInit() == 0);
            DcgmNs::Defer cleanup([] { DcgmFieldsTerm(); });
            std::vector<unsigned short> fieldIds;

            WHEN("the list is parsed")
            {
                dcgmReturn_t result = dcgmi_parse_field_id_list_string("65535", fieldIds, true);

                THEN("bad parameter is returned")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(fieldIds.empty());
                }
            }
        }
    }

    SECTION("hostname parsing")
    {
        GIVEN("a regular host name")
        {
            bool isUnixSocket = true;

            WHEN("the hostname is parsed")
            {
                char const *result = dcgmi_parse_hostname_string("localhost", &isUnixSocket, false);

                THEN("the original hostname is returned")
                {
                    REQUIRE(result != nullptr);
                    CHECK(std::string(result) == "localhost");
                    CHECK_FALSE(isUnixSocket);
                }
            }
        }

        GIVEN("a unix socket hostname")
        {
            bool isUnixSocket = false;

            WHEN("the hostname is parsed")
            {
                char const *result = dcgmi_parse_hostname_string("unix:///tmp/dcgm.sock", &isUnixSocket, false);

                THEN("the unix socket path is returned")
                {
                    REQUIRE(result != nullptr);
                    CHECK(std::string(result) == "/tmp/dcgm.sock");
                    CHECK(isUnixSocket);
                }
            }
        }

        GIVEN("a unix socket prefix without a path")
        {
            bool isUnixSocket = false;

            WHEN("the hostname is parsed")
            {
                char const *result = dcgmi_parse_hostname_string("unix://", &isUnixSocket, false);

                THEN("parsing fails")
                {
                    CHECK(result == nullptr);
                }
            }
        }
    }

    SECTION("terminal helpers")
    {
        GIVEN("the test process stdout")
        {
            WHEN("terminal state is queried")
            {
                bool isTty      = DcgmNs::Terminal::IsTTY();
                auto dimensions = DcgmNs::Terminal::GetTermDimensions();

                THEN("dimensions are only returned for an interactive terminal")
                {
                    CHECK(dimensions.has_value() == isTty);
                }
            }
        }
    }
}

TEST_CASE("ComputeItemsPerLine respects min/max constraints")
{
    using namespace DcgmNs::Terminal;

    SECTION("Clamps to minimum when computed is too low")
    {
        auto result = ComputeItemsPerLine(50, 60, 2, 8);
        REQUIRE(result == 2); // 60/50 = 1, clamped to min=2
    }

    SECTION("Clamps to maximum when computed is too high")
    {
        auto result = ComputeItemsPerLine(10, 200, 2, 8);
        REQUIRE(result == 8); // 200/10 = 20, clamped to max=8
    }

    SECTION("Returns value within range unchanged")
    {
        auto result = ComputeItemsPerLine(20, 100, 2, 8);
        REQUIRE(result == 5); // 100/20 = 5, within [2,8]
    }
}

TEST_CASE("ComputeItemsPerLine edge cases")
{
    using namespace DcgmNs::Terminal;

    SECTION("Zero item width returns minimum")
    {
        auto result = ComputeItemsPerLine(0, 100, 2, 8);
        REQUIRE(result == 2);
    }

    SECTION("Zero available width returns minimum")
    {
        auto result = ComputeItemsPerLine(20, 0, 2, 8);
        REQUIRE(result == 2);
    }

    SECTION("Both zero returns minimum")
    {
        auto result = ComputeItemsPerLine(0, 0, 3, 8);
        REQUIRE(result == 3);
    }

    SECTION("Item width equals available width")
    {
        auto result = ComputeItemsPerLine(50, 50, 2, 8);
        REQUIRE(result == 2); // 50/50 = 1, clamped to min=2
    }

    SECTION("Item width exceeds available width")
    {
        auto result = ComputeItemsPerLine(100, 50, 2, 8);
        REQUIRE(result == 2); // 50/100 = 0, clamped to min=2
    }
}
