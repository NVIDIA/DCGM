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

#include <catch2/catch.hpp>

#include "CommandOutputController.h"
#include "MigTestsHelper.hpp"

#include <Query.h>

#include <DcgmiOutput.h>
#include <string>


TEST_CASE("DCGMI Output")
{
    SECTION("Tree Output for MIG hierarchy")
    {
        static const char expectedResult[]
            = R""(+-------------------+--------------------------------------------------------------------+
| Instance Hierarchy                                                                     |
+===================+====================================================================+
| GPU 0             | GPU GPU-f04cff15-f427-9612-3f5e-60a6e0cd2018 (EntityID: 0)         |
| -> I 0/0          | GPU Instance (EntityID: 0)                                         |
|    -> CI 0/0/0    | Compute Instance (EntityID: 0)                                     |
|    -> CI 0/0/1    | Compute Instance (EntityID: 1000)                                  |
| -> I 0/1          | GPU Instance (EntityID: 100)                                       |
|    -> CI 0/1/0    | Compute Instance (EntityID: 100)                                   |
|    -> CI 0/1/1    | Compute Instance (EntityID: 1100)                                  |
| -> I 0/2          | GPU Instance (EntityID: 200)                                       |
|    -> CI 0/2/0    | Compute Instance (EntityID: 200)                                   |
|    -> CI 0/2/1    | Compute Instance (EntityID: 1200)                                  |
| -> I 0/3          | GPU Instance (EntityID: 300)                                       |
|    -> CI 0/3/0    | Compute Instance (EntityID: 300)                                   |
|    -> CI 0/3/1    | Compute Instance (EntityID: 1300)                                  |
+-------------------+--------------------------------------------------------------------+
| GPU 1             | GPU GPU-1a7b269f-8591-c6a5-a274-6f4f1ad9f6b3 (EntityID: 10)        |
| -> I 1/0          | GPU Instance (EntityID: 10)                                        |
|    -> CI 1/0/0    | Compute Instance (EntityID: 10)                                    |
|    -> CI 1/0/1    | Compute Instance (EntityID: 1010)                                  |
| -> I 1/1          | GPU Instance (EntityID: 110)                                       |
|    -> CI 1/1/0    | Compute Instance (EntityID: 110)                                   |
|    -> CI 1/1/1    | Compute Instance (EntityID: 1110)                                  |
| -> I 1/2          | GPU Instance (EntityID: 210)                                       |
|    -> CI 1/2/0    | Compute Instance (EntityID: 210)                                   |
|    -> CI 1/2/1    | Compute Instance (EntityID: 1210)                                  |
| -> I 1/3          | GPU Instance (EntityID: 310)                                       |
|    -> CI 1/3/0    | Compute Instance (EntityID: 310)                                   |
|    -> CI 1/3/1    | Compute Instance (EntityID: 1310)                                  |
+-------------------+--------------------------------------------------------------------+
| GPU 2             | GPU GPU-4d75f16a-433f-d7d0-5a15-8348bc3661bf (EntityID: 20)        |
| -> I 2/0          | GPU Instance (EntityID: 20)                                        |
|    -> CI 2/0/0    | Compute Instance (EntityID: 20)                                    |
|    -> CI 2/0/1    | Compute Instance (EntityID: 1020)                                  |
| -> I 2/1          | GPU Instance (EntityID: 120)                                       |
|    -> CI 2/1/0    | Compute Instance (EntityID: 120)                                   |
|    -> CI 2/1/1    | Compute Instance (EntityID: 1120)                                  |
| -> I 2/2          | GPU Instance (EntityID: 220)                                       |
|    -> CI 2/2/0    | Compute Instance (EntityID: 220)                                   |
|    -> CI 2/2/1    | Compute Instance (EntityID: 1220)                                  |
| -> I 2/3          | GPU Instance (EntityID: 320)                                       |
|    -> CI 2/3/0    | Compute Instance (EntityID: 320)                                   |
|    -> CI 2/3/1    | Compute Instance (EntityID: 1320)                                  |
+-------------------+--------------------------------------------------------------------+
| GPU 3             | GPU GPU-78b71234-6259-326d-3308-79f2b69553ea (EntityID: 30)        |
| -> I 3/0          | GPU Instance (EntityID: 30)                                        |
|    -> CI 3/0/0    | Compute Instance (EntityID: 30)                                    |
|    -> CI 3/0/1    | Compute Instance (EntityID: 1030)                                  |
| -> I 3/1          | GPU Instance (EntityID: 130)                                       |
|    -> CI 3/1/0    | Compute Instance (EntityID: 130)                                   |
|    -> CI 3/1/1    | Compute Instance (EntityID: 1130)                                  |
| -> I 3/2          | GPU Instance (EntityID: 230)                                       |
|    -> CI 3/2/0    | Compute Instance (EntityID: 230)                                   |
|    -> CI 3/2/1    | Compute Instance (EntityID: 1230)                                  |
| -> I 3/3          | GPU Instance (EntityID: 330)                                       |
|    -> CI 3/3/0    | Compute Instance (EntityID: 330)                                   |
|    -> CI 3/3/1    | Compute Instance (EntityID: 1330)                                  |
+-------------------+--------------------------------------------------------------------+
)"";

        auto hierarchy = Create_4GPU_4_GI_2_CI();
        auto result    = FormatMigHierarchy(hierarchy);

        REQUIRE(result == expectedResult);
    }
}

TEST_CASE("DCGMI: CommandOutputController")
{
    SECTION("Tag Replacement")
    {
        static std::string const haystack = "| <GPUID > | <MSG       >|";

        static char const gpuIdTag[] = "<GPUID";
        static char const msgTag[]   = "<MSG";

        std::string test1 = haystack;
        CommandOutputController::ReplaceTag(test1, gpuIdTag, "%s", "1");
        CommandOutputController::ReplaceTag(test1, msgTag, "%s", "a");
        REQUIRE(test1 == "| 1        | a           |");

        std::string test2 = haystack;
        CommandOutputController::ReplaceTag(test2, gpuIdTag, "%s", "12345678");
        CommandOutputController::ReplaceTag(test2, msgTag, "%s", "a");
        REQUIRE(test2 == "| 12345678 | a           |");

        std::string test3 = haystack;
        CommandOutputController::ReplaceTag(test3, gpuIdTag, "%s", "123456789");
        CommandOutputController::ReplaceTag(test3, msgTag, "%s", "a");
        REQUIRE(test3 == "| 12345... | a           |");

        std::string test4 = haystack;
        CommandOutputController::ReplaceTag(test4, gpuIdTag, "%s", "123456789");
        CommandOutputController::ReplaceTag(test4, msgTag, "%s", "abcdefghijklmnopqrst");
        REQUIRE(test4 == "| 12345... | abcdefghi...|");
    }
}