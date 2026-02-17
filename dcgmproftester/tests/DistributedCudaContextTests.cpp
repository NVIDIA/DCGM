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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DistributedCudaContext.h"
#include "PhysicalGpu.h"

#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <unistd.h>

using namespace DcgmNs::ProfTester;
using namespace Catch::Matchers;

namespace DcgmNs::ProfTester
{

// Function to configure PhysicalGpu stub behavior for testing
void SetPhysicalGpuTestConfig(bool isMIG, std::string const &busId = "0000:01:00.0");

/**
 * Friend test class for testing DistributedCudaContext private methods
 */
class DistributedCudaContextTests
{
public:
    DistributedCudaContextTests(DistributedCudaContext &context)
        : m_context(context)
    {}

    /**
     * Test the HandleCudaVisibleDevices helper method
     */
    dcgmReturn_t HandleCudaVisibleDevices()
    {
        return m_context.HandleCudaVisibleDevices();
    }

    /**
     * Get the message stream content for verification
     */
    std::string GetMessageString() const
    {
        return m_context.m_message.str();
    }

    /**
     * Get the error stream content for verification
     */
    std::string GetErrorString() const
    {
        return m_context.m_error.str();
    }

    /**
     * Set the test field ID for testing different behaviors
     */
    void SetTestFieldId(unsigned int fieldId)
    {
        m_context.m_testFieldId = fieldId;
    }

private:
    DistributedCudaContext &m_context;
};

TEST_CASE("DistributedCudaContext::HandleCudaVisibleDevices")
{
    // Environment cleanup helper
    auto cleanupEnv = [](const char *originalValue) {
        if (originalValue != nullptr)
        {
            setenv("CUDA_VISIBLE_DEVICES", originalValue, 1);
        }
        else
        {
            unsetenv("CUDA_VISIBLE_DEVICES");
        }
    };

    const char *originalCvd = getenv("CUDA_VISIBLE_DEVICES");

    SECTION("No existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        unsetenv("CUDA_VISIBLE_DEVICES");

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_ACTIVE);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));

        cleanupEnv(originalCvd);
    }

    SECTION("Empty existing CUDA_VISIBLE_DEVICES - should set new value")
    {
        setenv("CUDA_VISIBLE_DEVICES", "", 1);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_ACTIVE);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Set CUDA_VISIBLE_DEVICES to 1"));

        cleanupEnv(originalCvd);
    }

    SECTION("Whitespace-only CUDA_VISIBLE_DEVICES - should respect existing value")
    {
        setenv("CUDA_VISIBLE_DEVICES", "   ", 1);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_ACTIVE);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "   "); // Should remain unchanged
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES:    "));

        cleanupEnv(originalCvd);
    }

    SECTION("MIG GPU - should always override existing CUDA_VISIBLE_DEVICES")
    {
        setenv("CUDA_VISIBLE_DEVICES", "0,1", 1);

        SetPhysicalGpuTestConfig(true); // Configure stub for MIG
        auto mockMigGpu              = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockMigGpu, entities, entity, "MIG-12345678-1234-1234-1234-123456789abc");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_ACTIVE);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "MIG-12345678-1234-1234-1234-123456789abc");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("for MIG partition"));

        cleanupEnv(originalCvd);
    }

    SECTION("NvLink tests - should override existing CUDA_VISIBLE_DEVICES")
    {
        setenv("CUDA_VISIBLE_DEVICES", "1", 1);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "0,1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_NVLINK_RX_BYTES);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "") == "0,1");
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Overriding CUDA_VISIBLE_DEVICES"));
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("for NvLink test"));

        cleanupEnv(originalCvd);
    }

    SECTION("Other tests - should respect existing CUDA_VISIBLE_DEVICES")
    {
        setenv("CUDA_VISIBLE_DEVICES", "2", 1);

        SetPhysicalGpuTestConfig(false); // Configure stub for non-MIG
        auto mockGpu                 = std::make_shared<PhysicalGpu>(nullptr, 0, Arguments_t::Parameters {});
        auto entities                = std::make_shared<std::map<dcgm_field_entity_group_t, dcgm_field_eid_t>>();
        (*entities)[DCGM_FE_GPU]     = 0;
        dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, 0 };

        DistributedCudaContext context(mockGpu, entities, entity, "0,1");
        DistributedCudaContextTests tester(context);
        tester.SetTestFieldId(DCGM_FI_PROF_SM_ACTIVE);

        dcgmReturn_t result = tester.HandleCudaVisibleDevices();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "")
                == "2"); // Should remain unchanged
        REQUIRE_THAT(tester.GetMessageString(), ContainsSubstring("Respecting existing CUDA_VISIBLE_DEVICES: 2"));

        cleanupEnv(originalCvd);
    }
}

} // namespace DcgmNs::ProfTester
