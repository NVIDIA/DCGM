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

#include <InjectionArgument.h>

#include <DcgmStringHelpers.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
template <typename T, typename Mutator>
void CheckGeneratedStructCopyAndCompare(Mutator mutator)
{
    T source {};
    T different {};
    mutator(source, different);
    T target {};

    InjectionArgument sourceArg(source);
    InjectionArgument differentArg(different);
    InjectionArgument targetArg(&target);

    auto const ret = targetArg.SetValueFrom(sourceArg);

    REQUIRE(ret == NVML_SUCCESS);
    CHECK(InjectionArgument(target).Compare(sourceArg) == 0);
    CHECK(sourceArg.Compare(differentArg) != 0);
    CHECK(differentArg.Compare(sourceArg) != 0);
}
} // namespace

SCENARIO("InjectionArgument: copies and compares generated memory and utilization structs")
{
    GIVEN("a generated NVML memory or utilization struct with a populated field")
    {
        SECTION("nvmlMemory_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlMemory_t>([](auto &source, auto &different) {
                source.total    = 1024ULL;
                source.used     = 256ULL;
                different.total = 1024ULL;
                different.used  = 512ULL;
            });
        }

        SECTION("nvmlBAR1Memory_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlBAR1Memory_t>([](auto &source, auto &different) {
                source.bar1Total   = 4096ULL;
                source.bar1Used    = 64ULL;
                different          = source;
                different.bar1Used = 128ULL;
            });
        }

        SECTION("nvmlUtilization_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlUtilization_t>([](auto &source, auto &different) {
                source.gpu       = 10U;
                source.memory    = 20U;
                different        = source;
                different.memory = 30U;
            });
        }
    }
}

SCENARIO("InjectionArgument: copies and compares generated process and accounting structs")
{
    GIVEN("a generated NVML process or accounting struct with a populated field")
    {
        SECTION("nvmlProcessInfo_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlProcessInfo_t>([](auto &source, auto &different) {
                source.pid              = 1234U;
                source.usedGpuMemory    = 2048ULL;
                different               = source;
                different.usedGpuMemory = 4096ULL;
            });
        }

        SECTION("nvmlAccountingStats_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlAccountingStats_t>([](auto &source, auto &different) {
                source.gpuUtilization    = 70U;
                source.memoryUtilization = 80U;
                different                = source;
                different.maxMemoryUsage = 99ULL;
            });
        }
    }
}

SCENARIO("InjectionArgument: copies generated structs with character arrays")
{
    GIVEN("a generated NVML struct with character-array or nested fields")
    {
        SECTION("nvmlPciInfo_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlPciInfo_t>([](auto &source, auto &different) {
                SafeCopyTo(source.busId, "0000:01:00.0");
                SafeCopyTo(different.busId, "0000:02:00.0");
            });
        }

        SECTION("nvmlBridgeChipInfo_t")
        {
            CheckGeneratedStructCopyAndCompare<nvmlBridgeChipInfo_t>([](auto &source, auto &different) {
                source.type         = NVML_BRIDGE_CHIP_PLX;
                source.fwVersion    = 1U;
                different           = source;
                different.fwVersion = 2U;
            });
        }
    }
}
