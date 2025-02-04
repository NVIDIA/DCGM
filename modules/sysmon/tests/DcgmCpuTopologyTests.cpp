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
#include <fmt/format.h>

#define DCGM_SYSMON_TEST
#include <DcgmCpuTopology.h>

using namespace DcgmNs;

std::string lscpuShortInvalidJson = R""(
{
    "lscpu" : [
      {
         "field": "CPU(s):",
         "data": "20"
      },{
         "field": "Socket(s):",
         "data": "3"
      },{
         "field": "Core(s) per socket:",
         "data": "2"
      }
}
)"";

std::string lscpuBadSyntaxJson = "bob don't understand jason or whatever";

std::string lscpuShortValidJson = R""(
{
    "lscpu" : [
      {
         "field": "CPU(s):",
         "data": "12"
      },{
         "field": "Socket(s):",
         "data": "3"
      },{
         "field": "Core(s) per socket:",
         "data": "4"
      },{
         "field": "NUMA node(s):",
         "data": "6"
      },{
         "field": "NUMA node0 CPU(s):",
         "data": "0-1"
      },{
         "field": "NUMA node1 CPU(s):",
         "data": "2-3"
      },{
         "field": "NUMA node2 CPU(s):",
         "data": "4-5"
      },{
         "field": "NUMA node3 CPU(s):",
         "data": "6-7"
      },{
         "field": "NUMA node4 CPU(s):",
         "data": "8-9"
      },{
         "field": "NUMA node5 CPU(s):",
         "data": "10-11"
      }
    ]
})"";

std::string lscpuSkinnyJoeFullJson = R""(
{
   "lscpu": [
      {
         "field": "Architecture:",
         "data": "aarch64"
      },{
         "field": "CPU op-mode(s):",
         "data": "64-bit"
      },{
         "field": "Byte Order:",
         "data": "Little Endian"
      },{
         "field": "CPU(s):",
         "data": "288"
      },{
         "field": "On-line CPU(s) list:",
         "data": "0-287"
      },{
         "field": "Vendor ID:",
         "data": "ARM"
      },{
         "field": "Model:",
         "data": "0"
      },{
         "field": "Thread(s) per core:",
         "data": "1"
      },{
         "field": "Core(s) per socket:",
         "data": "72"
      },{
         "field": "Socket(s):",
         "data": "4"
      },{
         "field": "Stepping:",
         "data": "r0p0"
      },{
         "field": "Frequency boost:",
         "data": "disabled"
      },{
         "field": "CPU max MHz:",
         "data": "3420.0000"
      },{
         "field": "CPU min MHz:",
         "data": "81.0000"
      },{
         "field": "BogoMIPS:",
         "data": "2000.00"
      },{
         "field": "Flags:",
         "data": "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti"
      },{
         "field": "L1d cache:",
         "data": "18 MiB (288 instances)"
      },{
         "field": "L1i cache:",
         "data": "18 MiB (288 instances)"
      },{
         "field": "L2 cache:",
         "data": "288 MiB (288 instances)"
      },{
         "field": "L3 cache:",
         "data": "456 MiB (4 instances)"
      },{
         "field": "NUMA node(s):",
         "data": "36"
      },{
         "field": "NUMA node0 CPU(s):",
         "data": "0-71"
      },{
         "field": "NUMA node1 CPU(s):",
         "data": "72-143"
      },{
         "field": "NUMA node2 CPU(s):",
         "data": "144-215"
      },{
         "field": "NUMA node3 CPU(s):",
         "data": "216-287"
      },{
         "field": "NUMA node4 CPU(s):",
         "data": null
      },{
         "field": "NUMA node5 CPU(s):",
         "data": null
      },{
         "field": "NUMA node6 CPU(s):",
         "data": null
      },{
         "field": "NUMA node7 CPU(s):",
         "data": null
      },{
         "field": "NUMA node8 CPU(s):",
         "data": null
      },{
         "field": "NUMA node9 CPU(s):",
         "data": null
      },{
         "field": "NUMA node10 CPU(s):",
         "data": null
      },{
         "field": "NUMA node11 CPU(s):",
         "data": null
      },{
         "field": "NUMA node12 CPU(s):",
         "data": null
      },{
         "field": "NUMA node13 CPU(s):",
         "data": null
      },{
         "field": "NUMA node14 CPU(s):",
         "data": null
      },{
         "field": "NUMA node15 CPU(s):",
         "data": null
      },{
         "field": "NUMA node16 CPU(s):",
         "data": null
      },{
         "field": "NUMA node17 CPU(s):",
         "data": null
      },{
         "field": "NUMA node18 CPU(s):",
         "data": null
      },{
         "field": "NUMA node19 CPU(s):",
         "data": null
      },{
         "field": "NUMA node20 CPU(s):",
         "data": null
      },{
         "field": "NUMA node21 CPU(s):",
         "data": null
      },{
         "field": "NUMA node22 CPU(s):",
         "data": null
      },{
         "field": "NUMA node23 CPU(s):",
         "data": null
      },{
         "field": "NUMA node24 CPU(s):",
         "data": null
      },{
         "field": "NUMA node25 CPU(s):",
         "data": null
      },{
         "field": "NUMA node26 CPU(s):",
         "data": null
      },{
         "field": "NUMA node27 CPU(s):",
         "data": null
      },{
         "field": "NUMA node28 CPU(s):",
         "data": null
      },{
         "field": "NUMA node29 CPU(s):",
         "data": null
      },{
         "field": "NUMA node30 CPU(s):",
         "data": null
      },{
         "field": "NUMA node31 CPU(s):",
         "data": null
      },{
         "field": "NUMA node32 CPU(s):",
         "data": null
      },{
         "field": "NUMA node33 CPU(s):",
         "data": null
      },{
         "field": "NUMA node34 CPU(s):",
         "data": null
      },{
         "field": "NUMA node35 CPU(s):",
         "data": null
      },{
         "field": "Vulnerability Itlb multihit:",
         "data": "Not affected"
      },{
         "field": "Vulnerability L1tf:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Mds:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Meltdown:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Mmio stale data:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Retbleed:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Spec store bypass:",
         "data": "Mitigation; Speculative Store Bypass disabled via prctl"
      },{
         "field": "Vulnerability Spectre v1:",
         "data": "Mitigation; __user pointer sanitization"
      },{
         "field": "Vulnerability Spectre v2:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Srbds:",
         "data": "Not affected"
      },{
         "field": "Vulnerability Tsx async abort:",
         "data": "Not affected"
      }
   ]
}
)"";

std::string lscpuShortValidJsonWithInvalidFieldType = R""(
{
    "lscpu" : [
      {
         "field": "CPU(s):",
         "data": "12"
      },{
         "field": 2,
         "data": "3"
      }
   ]
}
)"";

TEST_CASE("DcgmCpuTopology::ParseLscpuOutputAndReadValues")
{
    DcgmCpuTopology dct;
    CHECK(dct.ParseLscpuOutputAndReadValues(lscpuBadSyntaxJson) != DCGM_ST_OK);
    CHECK(dct.ParseLscpuOutputAndReadValues(lscpuShortInvalidJson) != DCGM_ST_OK);

    CHECK(dct.ParseLscpuOutputAndReadValues(lscpuShortValidJsonWithInvalidFieldType) != DCGM_ST_OK);

    CHECK(dct.ParseLscpuOutputAndReadValues(lscpuShortValidJson) == DCGM_ST_OK);
    CHECK(dct.m_initialized == false);

    CHECK(dct.m_socketCount == 3);
    CHECK(dct.m_coresPerSocket == 4);
    CHECK(dct.m_coreCount == 12);
    CHECK(dct.m_numaNodeToCoreRange.size() == 6);
    for (unsigned int index = 0; index < dct.m_numaNodeToCoreRange.size(); index++)
    {
        unsigned int firstCore = index * 2;
        CHECK(dct.m_numaNodeToCoreRange[index] == fmt::format("{}-{}", firstCore, firstCore + 1));
    }

    DcgmCpuTopology dct2;
    CHECK(dct2.ParseLscpuOutputAndReadValues(lscpuSkinnyJoeFullJson) == DCGM_ST_OK);
    CHECK(dct2.m_socketCount == 4);
    CHECK(dct2.m_coresPerSocket == 72);
    CHECK(dct2.m_coreCount == 288);
    CHECK(dct2.m_numaNodeToCoreRange.size() == 4);

    for (unsigned int index = 0; index < dct2.m_numaNodeToCoreRange.size(); index++)
    {
        unsigned int firstCore = index * 72;
        CHECK(dct2.m_numaNodeToCoreRange[index] == fmt::format("{}-{}", firstCore, firstCore + 71));
    }
}

TEST_CASE("DcgmCpuTopology::BitmaskIsSubset")
{
    dcgmCpuHierarchyOwnedCores_t one {};
    dcgmCpuHierarchyOwnedCores_t two {};

    one.version = dcgmCpuHierarchyOwnedCores_version1;
    CHECK(DcgmCpuTopology::BitmaskIsSubset(one, two) == false); // should fail version check
    two.version = dcgmCpuHierarchyOwnedCores_version1;
    // both are empty, is subset
    CHECK(DcgmCpuTopology::BitmaskIsSubset(one, two) == true);
    CHECK(DcgmCpuTopology::BitmaskIsSubset(two, one) == true);
    for (unsigned int i = 0; i < DCGM_CPU_CORE_BITMASK_COUNT_V1; i++)
    {
        two.bitmask[i] = 0xab;
        CHECK(DcgmCpuTopology::BitmaskIsSubset(one, two) == false);
        one.bitmask[i] = 0xbab;
        CHECK(DcgmCpuTopology::BitmaskIsSubset(one, two) == true);
    }
}
