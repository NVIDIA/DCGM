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
#include <catch2/catch_test_macros.hpp>

#include <DcgmEntityTypes.hpp>

#include <cstdint>
#include <functional>
#include <sstream>
#include <type_traits>
#include <unordered_set>

SCENARIO("DcgmEntityTypes: entity ID storage types remain lightweight")
{
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Cpu::CpuId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Cpu::CoreId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Mig::GpuInstanceId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Mig::ComputeInstanceId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Mig::GpuInstanceProfileId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Mig::Nvml::GpuInstanceId>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<DcgmNs::Mig::Nvml::ComputeInstanceId>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Cpu::CpuId::IdType, std::uint64_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Cpu::CoreId::IdType, std::uint64_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Mig::GpuInstanceId::IdType, std::uint64_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Mig::ComputeInstanceId::IdType, std::uint32_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Mig::GpuInstanceProfileId::IdType, std::uint32_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Mig::Nvml::GpuInstanceId::IdType, std::uint32_t>);
    STATIC_REQUIRE(std::is_same_v<DcgmNs::Mig::Nvml::ComputeInstanceId::IdType, std::uint32_t>);
}

SCENARIO("DcgmEntityTypes: entity IDs expose value semantics")
{
    GIVEN("CPU IDs with equal and different values")
    {
        DcgmNs::Cpu::CpuId const cpuA { 12 };
        DcgmNs::Cpu::CpuId const cpuB { 12 };
        DcgmNs::Cpu::CpuId const cpuC { 13 };

        WHEN("they are compared and hashed")
        {
            std::unordered_set<DcgmNs::Cpu::CpuId> ids;
            ids.insert(cpuA);
            ids.insert(cpuB);
            ids.insert(cpuC);

            THEN("the wrapped ID value defines identity")
            {
                CHECK(cpuA == cpuB);
                CHECK_FALSE(cpuA == cpuC);
                CHECK(std::hash<DcgmNs::Cpu::CpuId> {}(cpuA) == std::hash<DcgmNs::Cpu::CpuId> {}(cpuB));
                CHECK(ids.size() == 2);
            }
        }
    }

    GIVEN("MIG IDs with equal and different values")
    {
        DcgmNs::Mig::GpuInstanceId const instanceA { 42 };
        DcgmNs::Mig::GpuInstanceId const instanceB { 42 };
        DcgmNs::Mig::GpuInstanceId const instanceC { 43 };
        DcgmNs::Mig::ComputeInstanceId const computeA { 7 };
        DcgmNs::Mig::ComputeInstanceId const computeB { 7 };
        DcgmNs::Mig::GpuInstanceProfileId const profileA { 2 };
        DcgmNs::Mig::GpuInstanceProfileId const profileB { 2 };
        DcgmNs::Mig::GpuInstanceProfileId const profileC { 3 };

        WHEN("they are compared and hashed")
        {
            std::unordered_set<DcgmNs::Mig::GpuInstanceId> gpuInstances;
            std::unordered_set<DcgmNs::Mig::ComputeInstanceId> computeInstances;
            gpuInstances.insert(instanceA);
            gpuInstances.insert(instanceB);
            gpuInstances.insert(instanceC);
            computeInstances.insert(computeA);
            computeInstances.insert(computeB);

            THEN("each ID type hashes its own wrapped value")
            {
                CHECK(instanceA == instanceB);
                CHECK_FALSE(instanceA == instanceC);
                CHECK(computeA == computeB);
                CHECK(profileA == profileB);
                CHECK_FALSE(profileA == profileC);
                CHECK(gpuInstances.size() == 2);
                CHECK(computeInstances.size() == 1);
            }
        }
    }

    GIVEN("NVML MIG IDs with equal values")
    {
        DcgmNs::Mig::Nvml::GpuInstanceId const gpuInstanceA { 3 };
        DcgmNs::Mig::Nvml::GpuInstanceId const gpuInstanceB { 3 };
        DcgmNs::Mig::Nvml::ComputeInstanceId const computeInstanceA { 4 };
        DcgmNs::Mig::Nvml::ComputeInstanceId const computeInstanceB { 4 };

        WHEN("they are compared and hashed")
        {
            THEN("NVML IDs use their wrapped NVML values")
            {
                CHECK(gpuInstanceA == gpuInstanceB);
                CHECK(computeInstanceA == computeInstanceB);
                CHECK(std::hash<DcgmNs::Mig::Nvml::GpuInstanceId> {}(gpuInstanceA)
                      == std::hash<DcgmNs::Mig::Nvml::GpuInstanceId> {}(gpuInstanceB));
                CHECK(std::hash<DcgmNs::Mig::Nvml::ComputeInstanceId> {}(computeInstanceA)
                      == std::hash<DcgmNs::Mig::Nvml::ComputeInstanceId> {}(computeInstanceB));
            }
        }
    }
}

SCENARIO("DcgmEntityTypes: entity IDs format for streams")
{
    GIVEN("CPU entity IDs")
    {
        std::ostringstream cpu;
        std::ostringstream core;

        WHEN("they are streamed")
        {
            cpu << DcgmNs::Cpu::CpuId { 1 };
            core << DcgmNs::Cpu::CoreId { 2 };

            THEN("the stream identifies the CPU entity kind and value")
            {
                CHECK(cpu.str() == "CPU(1)");
                CHECK(core.str() == "CPU_CORE(2)");
            }
        }
    }

    GIVEN("DCGM MIG entity IDs")
    {
        std::ostringstream gpuInstance;
        std::ostringstream computeInstance;

        WHEN("they are streamed")
        {
            gpuInstance << DcgmNs::Mig::GpuInstanceId { 3 };
            computeInstance << DcgmNs::Mig::ComputeInstanceId { 4 };

            THEN("the stream identifies the DCGM MIG entity kind and value")
            {
                CHECK(gpuInstance.str() == "GPU_I(3)");
                CHECK(computeInstance.str() == "GPU_CI(4)");
            }
        }
    }

    GIVEN("NVML MIG entity IDs")
    {
        std::ostringstream gpuInstance;
        std::ostringstream computeInstance;

        WHEN("they are streamed")
        {
            gpuInstance << DcgmNs::Mig::Nvml::GpuInstanceId { 5 };
            computeInstance << DcgmNs::Mig::Nvml::ComputeInstanceId { 6 };

            THEN("the stream identifies the NVML MIG entity kind and value")
            {
                CHECK(gpuInstance.str() == "NVML_GI(5)");
                CHECK(computeInstance.str() == "NVML_CI(6)");
            }
        }
    }
}
