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
#include <dcgm_structs_internal.h>


dcgmGpuTopologyLevel_t GetSlowestPath(dcgmTopology_t &topology);


TEST_CASE("DcgmApi: Test GetSlowestPath")
{
    dcgmTopology_t topology;
    topology.numElements     = 0;
    topology.element[0].path = DCGM_TOPOLOGY_NVLINK6;
    topology.element[1].path = DCGM_TOPOLOGY_NVLINK5;

    // Make sure num elements gates what is evaluated
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_UNINITIALIZED);

    topology.numElements = 2;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_NVLINK5);

    topology.element[2].path = DCGM_TOPOLOGY_MULTIPLE;
    topology.numElements     = 3;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_MULTIPLE);

    topology.element[3].path = DCGM_TOPOLOGY_HOSTBRIDGE;
    topology.numElements     = 4;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_HOSTBRIDGE);

    topology.element[4].path = DCGM_TOPOLOGY_BOARD;
    topology.numElements     = 5;
    REQUIRE(GetSlowestPath(topology) == DCGM_TOPOLOGY_HOSTBRIDGE);
}
