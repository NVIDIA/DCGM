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

#include "../Arguments.h"
#include "../PhysicalGpu.h"

namespace DcgmNs::ProfTester
{

// Global test configuration for PhysicalGpu stub behavior
static bool g_testIsMIG        = false;
static std::string g_testBusId = "0000:01:00.0";

/**
 * Set test configuration for PhysicalGpu stub
 */
void SetPhysicalGpuTestConfig(bool isMIG, std::string const &busId)
{
    g_testIsMIG = isMIG;
    g_testBusId = busId;
}

/**
 * Minimal stub implementation of PhysicalGpu for unit testing
 * This provides just the constructor and destructor needed for inheritance
 * Also provides stub implementations of methods that return configurable test values
 */
PhysicalGpu::PhysicalGpu(std::shared_ptr<DcgmProfTester> /* tester */,
                         unsigned int /* gpuId */,
                         const Arguments_t::Parameters & /* parameters */)
{
    // Minimal stub - do nothing
}

PhysicalGpu::~PhysicalGpu()
{
    // Minimal stub - do nothing
}

// Stub implementations of methods used by DistributedCudaContext
bool PhysicalGpu::IsMIG() const
{
    return g_testIsMIG; // Use test configuration
}

std::string PhysicalGpu::GetGpuBusId() const
{
    return g_testBusId; // Use test configuration
}

dcgmHandle_t PhysicalGpu::GetHandle() const
{
    return 0; // Mock handle
}

dcgmFieldGrp_t PhysicalGpu::GetFieldGroupId() const
{
    return 0; // Mock field group ID
}

bool PhysicalGpu::GetDcgmValidation() const
{
    return false; // Disable validation for tests
}

unsigned int PhysicalGpu::GetGpuId() const
{
    return 0; // Mock GPU ID
}

bool PhysicalGpu::IsSynchronous() const
{
    return false; // Disable synchronous mode for tests
}

bool PhysicalGpu::UseCublas() const
{
    return false; // Disable CUBLAS for unit tests
}

} // namespace DcgmNs::ProfTester
