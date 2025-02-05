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

#include "IgnoreErrorCodesHelper.h"

#include <catch2/catch_all.hpp>

TEST_CASE("Ignore error codes validation tests")
{
    std::vector<unsigned int> validGpuIndices        = { 0, 5 };
    std::unordered_set<unsigned int> validErrorCodes = { 23, 455 };
    gpuIgnoreErrorCodeMap_t expectedMap = { { { DCGM_FE_GPU, 0 }, { 23, 455 } }, { { DCGM_FE_GPU, 5 }, { 23, 455 } } };
    gpuIgnoreErrorCodeMap_t parsedMap;
    SECTION("Valid strings")
    {
        std::string inputStr = "*:*";
        auto errString       = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap.clear();
        parsedMap.clear();
        inputStr  = "";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 5 }, { 23, 455 } } });
        parsedMap.clear();
        inputStr  = "gpu5:*";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 23, 455 } } });
        parsedMap.clear();
        inputStr  = "gpu0:*";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap
            = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 23, 455 } }, { { DCGM_FE_GPU, 5 }, { 23, 455 } } });
        parsedMap.clear();
        inputStr  = "gpu0:*;gpu5:*";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 23 } }, { { DCGM_FE_GPU, 5 }, { 23 } } });
        parsedMap.clear();
        inputStr  = "*:23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap
            = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 23, 455 } }, { { DCGM_FE_GPU, 5 }, { 23, 455 } } });
        parsedMap.clear();
        inputStr  = "*:455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap
            = gpuIgnoreErrorCodeMap_t({ { { { DCGM_FE_GPU, 0 }, { 23, 455 } }, { { DCGM_FE_GPU, 5 }, { 23, 455 } } } });
        parsedMap.clear();
        inputStr  = "455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 455 } }, { { DCGM_FE_GPU, 5 }, { 455 } } });
        parsedMap.clear();
        inputStr  = "gpu0:455;gpu5:455";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { { DCGM_FE_GPU, 0 }, { 23, 455 } } } });
        parsedMap.clear();
        inputStr  = "gpu0:455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { { DCGM_FE_GPU, 0 }, { 23, 455 } } } });
        parsedMap.clear();
        inputStr  = "gpu0:23, 455";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);

        expectedMap = gpuIgnoreErrorCodeMap_t({ { { DCGM_FE_GPU, 0 }, { 455 } }, { { DCGM_FE_GPU, 5 }, { 455 } } });
        parsedMap.clear();
        inputStr  = "gpu0:455[gpu5:455";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes, '[');
        CHECK(errString.empty());
        CHECK(parsedMap == expectedMap);
    }

    SECTION("Invalid strings")
    {
        gpuIgnoreErrorCodeMap_t parsedMap;
        std::string inputStr = "garbag232.e";
        auto errString       = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = ":;";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = ";";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = ":,";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "gpu0";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices);
        CHECK(!errString.empty());
    }
    SECTION("Invalid gpu index")
    {
        gpuIgnoreErrorCodeMap_t parsedMap;
        std::string inputStr = "gpu3:455,23";
        auto errString       = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "gpu1:455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "gpu0:455;gpu3:455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "gpu0:455,gpu0:455,23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());
    }
    SECTION("Invalid error code")
    {
        gpuIgnoreErrorCodeMap_t parsedMap;
        std::string inputStr = "gpu0:455;gpu5:88";
        auto errString       = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "*:88";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "*:23,88";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());
    }
    SECTION("Invalid use of wildcard")
    {
        gpuIgnoreErrorCodeMap_t parsedMap;
        std::string inputStr = "gpu0:455,*:*";
        auto errString       = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "*:455,gpu5:23";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());

        parsedMap.clear();
        inputStr  = "*";
        errString = ParseIgnoreErrorCodesString(inputStr, parsedMap, validGpuIndices, validErrorCodes);
        CHECK(!errString.empty());
    }
}
