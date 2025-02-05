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
#include <catch2/catch_all.hpp>

#include <DcgmCpuManager.h>
#include <DcgmEntityTypes.hpp>

using namespace DcgmNs::Cpu;

TEST_CASE("DcgmCpuManager basics")
{
    DcgmCpuManager cpus;
    CHECK(cpus.IsEmpty() == true);
    CpuId fakeCpuId = cpus.AddFakeCpu();
    CHECK(fakeCpuId.id == 0);
    for (unsigned int i = 0; i < 64; i++)
    {
        auto cpuId = CpuId { i };
        CHECK(cpus.AddFakeCore(fakeCpuId) == CoreId { i });
        if (i == 0)
        {
            CHECK(cpus.IsValidCpuId(cpuId) == true);
        }
        else
        {
            CHECK(cpus.IsValidCpuId(cpuId) == false);
        }
        CHECK(cpus.IsValidCoreId(CoreId { i }) == true);
        CHECK(cpus.IsValidCoreId(CoreId { i + 64 }) == false);
        CHECK(cpus.IsEmpty() == false);
    }

    fakeCpuId = cpus.AddFakeCpu();
    CHECK(fakeCpuId == CoreId { 1 });
    for (unsigned int i = 64; i < 128; i++)
    {
        CHECK(cpus.AddFakeCore(CpuId { fakeCpuId }) == CoreId { i });
    }

    for (unsigned int i = 0; i < 128; i++)
    {
        auto cpuId = CpuId { i };
        if (i < 2)
        {
            CHECK(cpus.IsValidCpuId(cpuId) == true);
        }
        else
        {
            CHECK(cpus.IsValidCpuId(cpuId) == false);
        }
        CHECK(cpus.IsValidCoreId(CoreId { i }) == true);
        CHECK(cpus.IsValidCoreId(CoreId { i + 128 }) == false);
    }

    dcgm_sysmon_msg_get_cpus_t msg;
    cpus.GetCpus(msg);
    CHECK(msg.cpuCount == 2);
    CHECK(msg.cpus[0].cpuId == 0);
    CHECK(msg.cpus[1].cpuId == 1);

    std::vector<unsigned int> coreIds = cpus.GetCoreIdList(0);
    for (unsigned int i = 0; i < coreIds.size(); i++)
    {
        CHECK(i == coreIds[i]);
        CHECK((msg.cpus[0].ownedCores.bitmask[0] & 1ULL << i) != 0);
    }

    std::vector<unsigned int> coreIds2 = cpus.GetCoreIdList(1);
    for (unsigned int i = 0; i < coreIds2.size(); i++)
    {
        CHECK(i + 64 == coreIds2[i]);
        CHECK((msg.cpus[1].ownedCores.bitmask[1] & 1ULL << i) != 0);
    }
}

TEST_CASE("DcgmCpuManager AddCpu")
{
    DcgmCpuManager cpus;

    dcgm_sysmon_cpu_t cpu {};
    cpu.cpuId     = 4;
    cpu.coreCount = 6;
    for (unsigned int i = 24; i < 30; i++)
    {
        cpu.ownedCores.bitmask[0] |= 1ULL << i;
    }

    cpus.AddCpu(cpu);
    CHECK(cpus.IsValidCpuId(CpuId { cpu.cpuId }));
    for (unsigned int i = 24; i < 30; i++)
    {
        CHECK(cpus.IsValidCoreId(CoreId { i }));
    }
}
