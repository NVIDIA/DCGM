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
/*
 * File:   TestHealthMonitor.h
 */

#ifndef TESTHEALTHMONITOR_H
#define TESTHEALTHMONITOR_H

#include "TestDcgmModule.h"
#include "dcgm_agent.h"
#include "dcgm_errors.h"
#include "dcgm_structs.h"
#include <TimeLib.hpp>
#include <memory>
#include <string>
#include <vector>

class TestHealthMonitor : public TestDcgmModule
{
public:
    TestHealthMonitor();
    virtual ~TestHealthMonitor();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(const TestDcgmModuleInitParams &initParams) override;
    int Run() override;
    int Cleanup() override;
    std::string GetTag() override;

private:
    // Test data structure for XIDs
    struct XidTestInfo
    {
        uint64_t xid;
        char const *desc;
        dcgmHealthWatchResults_t expectedStatus;
        dcgmHealthSystems_t subsystem;
        dcgmError_t expectedError;
    };

    // Context structure for NVLink subtest methods
    struct NVLinkTestContext;

    // Test methods
    int TestHMSet();
    int TestHMCheckPCIe();
    int TestHMCheckMemSbe();
    int TestHMCheckMemDbe();
    int TestHMCheckMemUnrepairableFlag();
    int TestHMCheckImex();
    int TestHMCheckInforom();
    int TestHMCheckThermal();
    int TestHMCheckPower();
    int TestHMCheckNVLink();
    int TestHMCheckNVLinkStates();
    int SubtestHMCheckNVLinkStatesAllLinksUp(NVLinkTestContext &context);
    int SubtestHMCheckNVLinkStatesLinkDown(NVLinkTestContext &context);
    int SubtestHMCheckNVLinkStatesOther(NVLinkTestContext &context);
    int TestHMCheckXids();
    int TestDevastatingXids();
    int TestSubsystemXids();
    int TestXidSeverityLevels();
    int TestHMCheckFabricManagerStatus();

    /**
     * Test that residual unhealthy state from previous tests is cleared
     * This test enables all health watches and fails if any report non-PASS status
     * @note XID-related incidents are excluded because XID history cannot be cleared through public API
     */
    int TestHMCheckResidual();

    dcgmReturn_t TestSingleXid(unsigned int const gpuId,
                               uint64_t const xid,
                               char const *const xidDesc,
                               dcgmHealthWatchResults_t const expectedStatus,
                               dcgmError_t const expectedError,
                               auto const timestamp,
                               std::unique_ptr<dcgmHealthResponse_t> &response,
                               dcgmHealthSystems_t const currentSubsystem) const;

    /**
     * Helper function which tests the health of the IMEX domain
     * @param gpuId The GPU ID
     * @param domainStatus The status of the domain
     * @param daemonStatus The status of the daemon
     * @param expectedHealth The expected health
     * @param checkIncidents Whether to check incidents
     * @param timestamp The timestamp
     * @param description The description
     * @return The result of the test (DCGM_ST_OK if successful, DCGM_ST_GENERIC_ERROR if failed)
     */
    dcgmReturn_t ValidateImexHealth(unsigned int gpuId,
                                    std::string_view domainStatus,
                                    int64_t daemonStatus,
                                    dcgmHealthWatchResults_t expectedHealth,
                                    bool checkIncidents,
                                    DcgmNs::Timelib::TimePoint const timestamp,
                                    std::string_view description);

    std::vector<unsigned int> m_gpus; /* List of GPUs to run on, copied in Init() */
    dcgmGpuGrp_t m_gpuGroup;          /* Group consisting of the members of m_gpus */

    std::vector<XidTestInfo> m_devastatingXids;
    std::vector<XidTestInfo> m_subsystemXids;
};

#endif /* HM */
