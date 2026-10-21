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

#include <catch2/catch_test_macros.hpp>

#include <array>

namespace
{
void SetMemory(nvmlMemory_t &value, unsigned long long base)
{
    value.total = base;
    value.free  = base + 1;
    value.used  = base + 2;
}

void SetBar1Memory(nvmlBAR1Memory_t &value, unsigned long long base)
{
    value.bar1Total = base;
    value.bar1Free  = base + 1;
    value.bar1Used  = base + 2;
}

void SetUtilization(nvmlUtilization_t &value, unsigned int base)
{
    value.gpu    = base;
    value.memory = base + 1;
}

void SetProcessInfo(nvmlProcessInfo_t &value, unsigned int pid, unsigned long long memory)
{
    value.pid           = pid;
    value.usedGpuMemory = memory;
}

void FillPair(std::array<nvmlMemory_t, 2> &values)
{
    SetMemory(values[0], 10);
    SetMemory(values[1], 20);
}

void FillPair(std::array<nvmlBAR1Memory_t, 2> &values)
{
    SetBar1Memory(values[0], 10);
    SetBar1Memory(values[1], 20);
}

void FillPair(std::array<nvmlUtilization_t, 2> &values)
{
    SetUtilization(values[0], 10);
    SetUtilization(values[1], 20);
}
} //namespace

SCENARIO("InjectionArgument: copies generated structure arrays")
{
    GIVEN("an array-backed nvmlMemory_t argument")
    {
        std::array<nvmlMemory_t, 2> source {};
        std::array<nvmlMemory_t, 2> target {};
        FillPair(source);

        InjectionArgument sourceArg(source.data(), static_cast<unsigned int>(source.size()));
        InjectionArgument targetArg(target.data(), static_cast<unsigned int>(target.size()));

        WHEN("the value is copied into another array-backed argument")
        {
            auto ret = targetArg.SetValueFrom(sourceArg);

            THEN("each generated structure element is copied")
            {
                CHECK(ret == NVML_SUCCESS);
                CHECK(target[0].total == 10);
                CHECK(target[0].free == 11);
                CHECK(target[1].total == 20);
                CHECK(target[1].used == 22);
            }
        }
    }

    GIVEN("an array-backed nvmlBAR1Memory_t argument")
    {
        std::array<nvmlBAR1Memory_t, 2> source {};
        std::array<nvmlBAR1Memory_t, 2> target {};
        FillPair(source);

        InjectionArgument sourceArg(source.data(), static_cast<unsigned int>(source.size()));
        InjectionArgument targetArg(target.data(), static_cast<unsigned int>(target.size()));

        WHEN("the value is copied into another array-backed argument")
        {
            auto ret = targetArg.SetValueFrom(sourceArg);

            THEN("the BAR1 memory entries are preserved")
            {
                CHECK(ret == NVML_SUCCESS);
                CHECK(target[0].bar1Total == 10);
                CHECK(target[0].bar1Free == 11);
                CHECK(target[1].bar1Total == 20);
                CHECK(target[1].bar1Used == 22);
            }
        }
    }

    GIVEN("an array-backed nvmlUtilization_t argument")
    {
        std::array<nvmlUtilization_t, 2> source {};
        std::array<nvmlUtilization_t, 2> target {};
        FillPair(source);

        InjectionArgument sourceArg(source.data(), static_cast<unsigned int>(source.size()));
        InjectionArgument targetArg(target.data(), static_cast<unsigned int>(target.size()));

        WHEN("the value is copied into another array-backed argument")
        {
            auto ret = targetArg.SetValueFrom(sourceArg);

            THEN("the utilization entries are preserved")
            {
                CHECK(ret == NVML_SUCCESS);
                CHECK(target[0].gpu == 10);
                CHECK(target[0].memory == 11);
                CHECK(target[1].gpu == 20);
                CHECK(target[1].memory == 21);
            }
        }
    }
}

SCENARIO("InjectionArgument: compares equal generated structure arrays")
{
    GIVEN("two equal generated structure arrays")
    {
        std::array<nvmlProcessInfo_t, 2> lower {};
        std::array<nvmlProcessInfo_t, 2> equal {};
        SetProcessInfo(lower[0], 100, 1024);
        SetProcessInfo(lower[1], 101, 2048);
        equal = lower;

        InjectionArgument lowerArg(lower.data(), static_cast<unsigned int>(lower.size()));
        InjectionArgument equalArg(equal.data(), static_cast<unsigned int>(equal.size()));

        WHEN("the generated arrays are compared")
        {
            auto equalResult = lowerArg.Compare(equalArg);

            THEN("equal arrays compare equal")
            {
                CHECK(equalResult == 0);
            }
        }
    }
}

SCENARIO("InjectionArgument: copies fixed-array fields inside generated structures")
{
    GIVEN("a vGPU scheduler log with populated log entries")
    {
        nvmlVgpuSchedulerLog_t source {};
        nvmlVgpuSchedulerLog_t target {};
        source.engineId                      = 1;
        source.schedulerPolicy               = 2;
        source.arrMode                       = 3;
        source.entriesCount                  = 2;
        source.logEntries[0].timestamp       = 10;
        source.logEntries[0].timeRunTotal    = 11;
        source.logEntries[0].targetTimeSlice = 12;
        source.logEntries[1].timestamp       = 20;
        source.logEntries[1].timeRunTotal    = 21;
        source.logEntries[1].targetTimeSlice = 22;

        InjectionArgument sourceArg(source);
        InjectionArgument targetArg(&target);

        WHEN("the generated structure is copied")
        {
            auto ret = targetArg.SetValueFrom(sourceArg);

            THEN("the embedded array entries are preserved")
            {
                CHECK(ret == NVML_SUCCESS);
                CHECK(target.engineId == 1);
                CHECK(target.entriesCount == 2);
                CHECK(target.logEntries[0].timestamp == 10);
                CHECK(target.logEntries[0].targetTimeSlice == 12);
                CHECK(target.logEntries[1].timestamp == 20);
                CHECK(target.logEntries[1].timeRunTotal == 21);
            }
        }
    }
}
