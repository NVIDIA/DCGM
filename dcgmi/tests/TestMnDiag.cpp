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
#include <fstream>
#include <iostream>
#include <sstream>

#include <DcgmStringHelpers.h>
#include <MnDiag.h>
#include <NvvsJsonStrings.h> // For JSON constants
#include <dcgm_errors.h>     // For error handling
#include <dcgm_structs.h>
#include <json/json.h> // For JSON parsing

// Friend class for testing MnDiag private methods
class MnDiagTester
{
public:
    MnDiagTester(MnDiag &mnDiag)
        : m_mnDiag(mnDiag)
    {}

    dcgmReturn_t HelperDisplayAsCli(dcgmMnDiagResponse_t const &response, bool mndiagFailed)
    {
        return m_mnDiag.HelperDisplayAsCli(response, mndiagFailed);
    }

    // JSON testing methods
    dcgmReturn_t HelperDisplayAsJson(dcgmMnDiagResponse_v1 const &response)
    {
        return m_mnDiag.HelperDisplayAsJson(response);
    }

    void HelperJsonAddMetadata(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
    {
        m_mnDiag.HelperJsonAddMetadata(output, response);
    }

    void HelperJsonAddHosts(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
    {
        m_mnDiag.HelperJsonAddHosts(output, response);
    }

    void HelperJsonAddTest(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
    {
        m_mnDiag.HelperJsonAddTest(output, response);
    }

    void HelperJsonAddErrors(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
    {
        m_mnDiag.HelperJsonAddErrors(output, response);
    }

    void HelperDisplayFailureMessage(std::string_view errMsg,
                                     dcgmReturn_t result,
                                     dcgmMnDiagResponse_v1 const &response)
    {
        m_mnDiag.HelperDisplayFailureMessage(errMsg, result, response);
    }

private:
    MnDiag &m_mnDiag;
};

// Redirection class to capture stdout for testing
class StdoutRedirect
{
public:
    StdoutRedirect()
        : m_oldBuffer(std::cout.rdbuf(m_buffer.rdbuf()))
    {}

    ~StdoutRedirect()
    {
        std::cout.rdbuf(m_oldBuffer);
    }

    std::string GetOutput() const
    {
        return m_buffer.str();
    }

private:
    std::stringstream m_buffer;
    std::streambuf *m_oldBuffer;
};

// Helper function to create a test response
dcgmMnDiagResponse_t CreateTestResponse(dcgmDiagResult_t overallResult, unsigned int numHosts, unsigned int numErrors)
{
    dcgmMnDiagResponse_t response = {};
    response.version              = dcgmMnDiagResponse_version1;

    // Set up test data - first, constrain the number of hosts
    numHosts          = std::min(numHosts, (unsigned int)DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
    response.numHosts = static_cast<unsigned char>(numHosts);

    // Set the test fields
    response.numTests        = 1;
    response.tests[0].result = overallResult;
    SafeCopyTo(response.tests[0].name, "MNUBERGEMM");

    // Set up host information
    if (numHosts > 0)
    {
        SafeCopyTo(response.hosts[0].hostname, "test-host-0");
        SafeCopyTo(response.hosts[0].driverVersion, "570.32.17");
        SafeCopyTo(response.hosts[0].dcgmVersion, "4.3.0");
    }

    if (numHosts > 1)
    {
        SafeCopyTo(response.hosts[1].hostname, "test-host-1");
        SafeCopyTo(response.hosts[1].driverVersion, "570.32.17");
        SafeCopyTo(response.hosts[1].dcgmVersion, "4.3.0");
    }

    if (numHosts > 2)
    {
        SafeCopyTo(response.hosts[2].hostname, "test-host-2");
        SafeCopyTo(response.hosts[2].driverVersion, "570.32.17");
        SafeCopyTo(response.hosts[2].dcgmVersion, "4.3.0");
    }

    if (numHosts > 3)
    {
        SafeCopyTo(response.hosts[3].hostname, "test-host-3");
        SafeCopyTo(response.hosts[3].driverVersion, "570.32.17");
        SafeCopyTo(response.hosts[3].dcgmVersion, "4.3.0");
    }

    if (numHosts > 4)
    {
        SafeCopyTo(response.hosts[4].hostname, "test-host-4");
        SafeCopyTo(response.hosts[4].driverVersion, "570.32.17");
        SafeCopyTo(response.hosts[4].dcgmVersion, "4.3.0");
    }

    // Set up entity results with different failure patterns:
    // - Host 0: GPU 0 fails
    // - Host 1: GPU 1 fails
    // - Host 2: GPU 0 fails
    // - Host 3: GPU 1 fails
    // - Host 4: passes
    response.numResults = 0;

    if (numHosts > 0)
    {
        // Host 0, GPU 0 - FAIL
        response.results[response.numResults].hostId               = 0;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 0;
        response.results[response.numResults].result               = overallResult;
        response.results[response.numResults].testId               = 0;
        response.numResults++;

        // Host 0, GPU 1 - PASS
        response.results[response.numResults].hostId               = 0;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 1;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;
    }

    if (numHosts > 1)
    {
        // Host 1, GPU 0 - PASS
        response.results[response.numResults].hostId               = 1;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 0;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;

        // Host 1, GPU 1 - FAIL
        response.results[response.numResults].hostId               = 1;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 1;
        response.results[response.numResults].result               = overallResult;
        response.results[response.numResults].testId               = 0;
        response.numResults++;
    }

    if (numHosts > 2)
    {
        // Host 2, GPU 0 - FAIL
        response.results[response.numResults].hostId               = 2;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 0;
        response.results[response.numResults].result               = overallResult;
        response.results[response.numResults].testId               = 0;
        response.numResults++;

        // Host 2, GPU 1 - PASS
        response.results[response.numResults].hostId               = 2;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 1;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;
    }

    if (numHosts > 3)
    {
        // Host 3, GPU 0 - PASS
        response.results[response.numResults].hostId               = 3;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 0;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;

        // Host 3, GPU 1 - FAIL
        response.results[response.numResults].hostId               = 3;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 1;
        response.results[response.numResults].result               = overallResult;
        response.results[response.numResults].testId               = 0;
        response.numResults++;
    }

    // Add Host 4 results (always passes)
    if (numHosts > 4)
    {
        // Host 4, GPU 0 - Always PASS
        response.results[response.numResults].hostId               = 4;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 0;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;

        // Host 4, GPU 1 - Always PASS
        response.results[response.numResults].hostId               = 4;
        response.results[response.numResults].entity.entityGroupId = DCGM_FE_GPU;
        response.results[response.numResults].entity.entityId      = 1;
        response.results[response.numResults].result               = DCGM_DIAG_RESULT_PASS;
        response.results[response.numResults].testId               = 0;
        response.numResults++;
    }

    // Create one error message for each failing GPU, up to numErrors
    response.numErrors          = 0;
    unsigned int errorsToCreate = std::min(numErrors, (unsigned int)DCGM_MN_DIAG_RESPONSE_ERRORS_MAX);

    // Host 0, GPU 0 error
    if (numHosts > 0 && response.numErrors < errorsToCreate)
    {
        response.errors[response.numErrors].hostId               = 0;
        response.errors[response.numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response.errors[response.numErrors].entity.entityId      = 0;
        response.errors[response.numErrors].testId               = 0;
        response.errors[response.numErrors].code                 = 100;
        response.errors[response.numErrors].category             = 1;
        response.errors[response.numErrors].severity             = 2;
        SafeCopyTo(response.errors[response.numErrors].msg, "Test error message for host 0, GPU 0");
        response.numErrors++;
    }

    // Host 1, GPU 1 error
    if (numHosts > 1 && response.numErrors < errorsToCreate)
    {
        response.errors[response.numErrors].hostId               = 1;
        response.errors[response.numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response.errors[response.numErrors].entity.entityId      = 1;
        response.errors[response.numErrors].testId               = 0;
        response.errors[response.numErrors].code                 = 101;
        response.errors[response.numErrors].category             = 1;
        response.errors[response.numErrors].severity             = 2;
        SafeCopyTo(response.errors[response.numErrors].msg, "Test error message for host 1, GPU 1");
        response.numErrors++;
    }

    // Host 2, GPU 0 error
    if (numHosts > 2 && response.numErrors < errorsToCreate)
    {
        response.errors[response.numErrors].hostId               = 2;
        response.errors[response.numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response.errors[response.numErrors].entity.entityId      = 0;
        response.errors[response.numErrors].testId               = 0;
        response.errors[response.numErrors].code                 = 102;
        response.errors[response.numErrors].category             = 1;
        response.errors[response.numErrors].severity             = 2;
        SafeCopyTo(response.errors[response.numErrors].msg, "Test error message for host 2, GPU 0");
        response.numErrors++;
    }

    // Host 3, GPU 1 error
    if (numHosts > 3 && response.numErrors < errorsToCreate)
    {
        response.errors[response.numErrors].hostId               = 3;
        response.errors[response.numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response.errors[response.numErrors].entity.entityId      = 1;
        response.errors[response.numErrors].testId               = 0;
        response.errors[response.numErrors].code                 = 103;
        response.errors[response.numErrors].category             = 1;
        response.errors[response.numErrors].severity             = 2;
        SafeCopyTo(response.errors[response.numErrors].msg, "Test error message for host 3, GPU 1");
        response.numErrors++;
    }

    return response;
}

SCENARIO("MnDiag::HelperDisplayAsCli output formatting")
{
    SECTION("Test with passing result")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 2, 0);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Check header is present
        REQUIRE(output.contains("Successfully ran multi-node diagnostic"));
        REQUIRE(output.contains("+---------------------------+------------------------------------------------+"));
        REQUIRE(output.contains("| Diagnostic                | Result                                         |"));

        // Check content
        REQUIRE(output.contains("MNUBERGEMM Test"));
        REQUIRE(output.contains("Pass"));
        REQUIRE(output.contains("Hosts Found"));
        REQUIRE(output.contains("Hosts With Errors"));

        // Check for version information
        REQUIRE(output.contains("Driver Version"));
        REQUIRE(output.contains("570.32.17"));
        REQUIRE(output.contains("DCGM Version"));
        REQUIRE(output.contains("4.3.0"));

        // Check host summary information
        REQUIRE(output.contains("Host List"));
        REQUIRE(output.contains("0: test-host-0"));
        REQUIRE(output.contains("1: test-host-1"));
        REQUIRE(output.contains("Total GPUs"));
        REQUIRE(output.contains("4")); // 2 hosts × 2 GPUs each = 4 GPUs

        // Check detailed host information - new format
        REQUIRE(output.contains("Host 0")); // Host info with just index
        REQUIRE(output.contains("Host 1")); // Host info with just index

        // Check host statuses
        REQUIRE(output.contains("Host 0                    | Pass"));
        REQUIRE(output.contains("Host 1                    | Pass"));

        // Check for GPU information with summary line for all-passing GPUs
        REQUIRE(output.contains("  GPUs: 0, 1"));
        REQUIRE(output.contains("Pass"));

        // Should not have errors section
        REQUIRE(!output.contains("Error Summary"));
    }

    SECTION("Test with failing result and errors")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 2);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Check content
        REQUIRE(output.contains("MNUBERGEMM Test"));
        REQUIRE(output.contains("Fail"));
        REQUIRE(output.contains("Hosts Found"));

        // Check that we're limited to DCGM_MN_DIAG_RESPONSE_HOSTS_MAX hosts
        unsigned int expectedHosts = std::min(5u, (unsigned int)DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        REQUIRE(output.contains(fmt::format("{}", expectedHosts)));

        // Check host list summary with truncation
        REQUIRE(output.contains("Host List"));
        REQUIRE(output.contains("0: test-host-0"));
        REQUIRE(output.contains("1: test-host-1"));
        REQUIRE(output.contains("2: test-host-2"));
        // If we have more than 3 hosts, check for the rest
        if (expectedHosts > 3)
        {
            REQUIRE(output.contains("3: test-host-3"));
        }
        if (expectedHosts > 4)
        {
            REQUIRE(output.contains("4: test-host-4"));
        }

        // Check host status for alternating GPU failures
        REQUIRE(output.contains("Host 0                    | Fail")); // Host 0 GPU 0 fails
        REQUIRE(output.contains("Host 1                    | Fail")); // Host 1 GPU 1 fails

        // Check that errors are displayed for failing GPUs
        REQUIRE(output.contains("Error: Test error message for host 0, GPU 0"));
        REQUIRE(output.contains("Error: Test error message for host 1, GPU 1"));

        // Check error summary section
        REQUIRE(output.contains("Error Summary"));
    }

    SECTION("Test with other result types")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_WARN, 1, 1);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Check content shows proper status string
        REQUIRE(output.contains("Warn"));

        // Check host format with colon and status
        REQUIRE(output.contains("Host 0"));
        REQUIRE(output.contains("Host 0                    | Warn")); // Host has Warn status

        // Check version information
        REQUIRE(output.contains("Driver Version"));
        REQUIRE(output.contains("570.32.17"));
        REQUIRE(output.contains("DCGM Version"));
        REQUIRE(output.contains("4.3.0"));

        // Check GPU status is displayed without indentation
        REQUIRE(output.contains("GPU 0"));
        REQUIRE(output.contains("Warn"));
        REQUIRE(output.contains("GPU 1"));
        REQUIRE(output.contains("Pass"));
    }

    SECTION("Test with no hosts")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 0, 0);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Should not have host details section with data
        REQUIRE(!output.contains("Host 0                    |"));

        // Version information should not be displayed when there are no hosts
        REQUIRE(!output.contains("Driver Version"));
        REQUIRE(!output.contains("DCGM Version"));
    }

    SECTION("Test GPU-specific error display")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        // Create test response with multiple hosts and all possible errors
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 4, 4);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Verify host format and status
        REQUIRE(output.contains("Host 0")); // Host info with just index
        REQUIRE(output.contains("Host 1")); // Host info with just index
        REQUIRE(output.contains("Host 2")); // Host info with just index
        REQUIRE(output.contains("Host 3")); // Host info with just index

        // Verify specific GPU failures on alternating GPUs without indentation
        REQUIRE(output.contains("  GPU 0                   | Fail")); // Host 0 GPU 0 fails
        REQUIRE(output.contains("  GPU 1                   | Fail")); // Host 1 GPU 1 fails

        // Verify error messages appear for the failing GPUs
        REQUIRE(output.contains("Error: Test error message for host 0, GPU 0"));
        REQUIRE(output.contains("Error: Test error message for host 1, GPU 1"));
        REQUIRE(output.contains("Error: Test error message for host 2, GPU 0"));
        REQUIRE(output.contains("Error: Test error message for host 3, GPU 1"));

        // Verify error summary section contains all expected errors
        REQUIRE(output.contains("Error Summary"));
        REQUIRE(output.contains("Host 0 (GPU 0)"));
        REQUIRE(output.contains("Host 1 (GPU 1)"));
        REQUIRE(output.contains("Host 2 (GPU 0)"));
        REQUIRE(output.contains("Host 3 (GPU 1)"));
    }

    SECTION("Test host with all passing GPUs")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        // Create test response with 5 hosts, including one with all passing GPUs
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 4);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Verify that Host 4 has a Pass status since all GPUs pass
        REQUIRE(output.contains("Host 4")); // Host info with just index
        REQUIRE(output.contains("| Pass"));

        // Verify that Host 4's GPUs are shown with a summary format
        REQUIRE(output.contains("  GPUs: 0, 1"));

        // Verify that Host 4 doesn't appear in the Error Summary section
        REQUIRE(!output.contains("Host 4 (GPU"));

        // Verify that the "Hosts With Errors" count is correct (should be 4, not 5)
        REQUIRE(output.contains("| Hosts With Errors         | 4"));
    }

    SECTION("Test host details disabled (mndiagFailed = true)")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        // Create test response with multiple hosts
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 1, 1);
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, true);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Verify that host details section is NOT present
        REQUIRE(!output.contains("Host Details"));

        // Verify that individual host information is NOT displayed
        REQUIRE(!output.contains("Host 0                    |"));

        // Verify that error summary section is still present
        REQUIRE(output.contains("Error Summary"));
        REQUIRE(output.contains("Host 0 (GPU 0)"));

        // Verify that metadata section is still present
        REQUIRE(output.contains("Total GPUs"));
        REQUIRE(output.contains("Hosts With Errors"));
    }
}

SCENARIO("MnDiag CLI Display Documentation", "[.documentation]")
{
    SECTION("Save sample output to file")
    {
        StdoutRedirect redirect;
        MnDiag mnDiag("testhost");
        MnDiagTester tester(mnDiag);

        // Create a comprehensive test response
        auto response = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 4);

        // Execute the display method
        dcgmReturn_t result = tester.HelperDisplayAsCli(response, true);
        REQUIRE(result == DCGM_ST_OK);

        // Get the captured output
        std::string output = redirect.GetOutput();

        // Write the output to a file
        std::ofstream outFile("mndiag_sample_output.txt");
        REQUIRE(outFile.is_open());

        outFile << output;
        outFile.close();

        // Basic verifications
        REQUIRE(result == DCGM_ST_OK);

        // Verify that all error messages appear in the output
        REQUIRE(output.contains("Error: Test error message for host 0, GPU 0"));
        REQUIRE(output.contains("Error: Test error message for host 1, GPU 1"));
        REQUIRE(output.contains("Error: Test error message for host 2, GPU 0"));
        REQUIRE(output.contains("Error: Test error message for host 3, GPU 1"));
    }
}

// Helper function to create a test response with comprehensive JSON data
dcgmMnDiagResponse_t CreateJsonTestResponse()
{
    dcgmMnDiagResponse_t response = {};
    response.version              = dcgmMnDiagResponse_version1;

    // Set up test information
    response.numTests = 1;
    SafeCopyTo(response.tests[0].name, "MNUBERGEMM");
    SafeCopyTo(response.tests[0].pluginName, "MNUBERGEMM");
    response.tests[0].result = DCGM_DIAG_RESULT_FAIL;

    // Set up hosts
    response.numHosts = 3;

    // Host 0
    SafeCopyTo(response.hosts[0].hostname, "host-01");
    SafeCopyTo(response.hosts[0].dcgmVersion, "4.3.0");
    SafeCopyTo(response.hosts[0].driverVersion, "570.32.17");
    response.hosts[0].numEntities      = 2;
    response.hosts[0].entityIndices[0] = 0;
    response.hosts[0].entityIndices[1] = 1;

    // Host 1
    SafeCopyTo(response.hosts[1].hostname, "host-02");
    SafeCopyTo(response.hosts[1].dcgmVersion, "4.3.0");
    SafeCopyTo(response.hosts[1].driverVersion, "570.32.17");
    response.hosts[1].numEntities      = 2;
    response.hosts[1].entityIndices[0] = 2;
    response.hosts[1].entityIndices[1] = 3;

    // Host 2
    SafeCopyTo(response.hosts[2].hostname, "host-03");
    SafeCopyTo(response.hosts[2].dcgmVersion, "4.3.0");
    SafeCopyTo(response.hosts[2].driverVersion, "570.32.17");
    response.hosts[2].numEntities      = 1;
    response.hosts[2].entityIndices[0] = 4;

    // Set up entities
    response.numEntities = 5;

    // Host 0 entities
    response.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    response.entities[0].entity.entityId      = 0;
    SafeCopyTo(response.entities[0].serialNum, "0324522050123");
    SafeCopyTo(response.entities[0].skuDeviceId, "20F1");

    response.entities[1].entity.entityGroupId = DCGM_FE_GPU;
    response.entities[1].entity.entityId      = 1;
    SafeCopyTo(response.entities[1].serialNum, "0324522050124");
    SafeCopyTo(response.entities[1].skuDeviceId, "20F1");

    // Host 1 entities
    response.entities[2].entity.entityGroupId = DCGM_FE_GPU;
    response.entities[2].entity.entityId      = 0;
    SafeCopyTo(response.entities[2].serialNum, "0324522050125");
    SafeCopyTo(response.entities[2].skuDeviceId, "20F1");

    response.entities[3].entity.entityGroupId = DCGM_FE_GPU;
    response.entities[3].entity.entityId      = 1;
    SafeCopyTo(response.entities[3].serialNum, "0324522050126");
    SafeCopyTo(response.entities[3].skuDeviceId, "20F1");

    // Host 2 entities
    response.entities[4].entity.entityGroupId = DCGM_FE_GPU;
    response.entities[4].entity.entityId      = 0;
    SafeCopyTo(response.entities[4].serialNum, "0324522050127");
    SafeCopyTo(response.entities[4].skuDeviceId, "20F1");

    // Set up results
    response.numResults = 5;

    // Host 0 results
    response.results[0].hostId               = 0;
    response.results[0].entity.entityGroupId = DCGM_FE_GPU;
    response.results[0].entity.entityId      = 0;
    response.results[0].result               = DCGM_DIAG_RESULT_PASS;
    response.results[0].testId               = 0;

    response.results[1].hostId               = 0;
    response.results[1].entity.entityGroupId = DCGM_FE_GPU;
    response.results[1].entity.entityId      = 1;
    response.results[1].result               = DCGM_DIAG_RESULT_FAIL;
    response.results[1].testId               = 0;

    // Host 1 results
    response.results[2].hostId               = 1;
    response.results[2].entity.entityGroupId = DCGM_FE_GPU;
    response.results[2].entity.entityId      = 0;
    response.results[2].result               = DCGM_DIAG_RESULT_PASS;
    response.results[2].testId               = 0;

    response.results[3].hostId               = 1;
    response.results[3].entity.entityGroupId = DCGM_FE_GPU;
    response.results[3].entity.entityId      = 1;
    response.results[3].result               = DCGM_DIAG_RESULT_FAIL;
    response.results[3].testId               = 0;

    // Host 2 results
    response.results[4].hostId               = 2;
    response.results[4].entity.entityGroupId = DCGM_FE_GPU;
    response.results[4].entity.entityId      = 0;
    response.results[4].result               = DCGM_DIAG_RESULT_PASS;
    response.results[4].testId               = 0;

    // Set up errors
    response.numErrors = 2;

    // Error for Host 0, GPU 1
    response.errors[0].hostId               = 0;
    response.errors[0].testId               = 0;
    response.errors[0].entity.entityGroupId = DCGM_FE_GPU;
    response.errors[0].entity.entityId      = 1;
    response.errors[0].code                 = DCGM_FR_UNKNOWN;
    response.errors[0].category             = DCGM_FR_EC_HARDWARE_OTHER;
    response.errors[0].severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(response.errors[0].msg, "CUDA memory allocation failed");

    // Error for Host 1, GPU 1
    response.errors[1].hostId               = 1;
    response.errors[1].testId               = 0;
    response.errors[1].entity.entityGroupId = DCGM_FE_GPU;
    response.errors[1].entity.entityId      = 1;
    response.errors[1].code                 = DCGM_FR_UNKNOWN;
    response.errors[1].category             = DCGM_FR_EC_HARDWARE_OTHER;
    response.errors[1].severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(response.errors[1].msg, "NCCL communication timeout");

    // Set up test indices for JSON output
    response.tests[0].numResults = response.numResults;
    for (unsigned int i = 0; i < response.numResults; i++)
    {
        response.tests[0].resultIndices[i] = i;
    }

    response.tests[0].numErrors = response.numErrors;
    for (unsigned int i = 0; i < response.numErrors; i++)
    {
        response.tests[0].errorIndices[i] = i;
    }

    return response;
}

// Helper function to convert dcgmDiagResult_t to string (similar to Diag::HelperDisplayDiagResult)
std::string DiagResultToString(dcgmDiagResult_t result)
{
    switch (result)
    {
        case DCGM_DIAG_RESULT_PASS:
            return "Pass";
        case DCGM_DIAG_RESULT_FAIL:
            return "Fail";
        case DCGM_DIAG_RESULT_WARN:
            return "Warn";
        case DCGM_DIAG_RESULT_SKIP:
            return "Skip";
        case DCGM_DIAG_RESULT_NOT_RUN:
        default:
            return "";
    }
}

TEST_CASE("MnDiag JSON Output Methods")
{
    SECTION("HelperJsonAddMetadata - Basic Metadata")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddMetadata(output, response);

        // Verify basic structure
        REQUIRE(output.isMember(NVVS_NAME));
        REQUIRE(output[NVVS_NAME].asString() == "DCGM Multi-Node Diagnostic");

        // Verify metadata section
        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(output[NVVS_METADATA].isMember("num_hosts"));
        REQUIRE(output[NVVS_METADATA]["num_hosts"].asUInt() == 3);
        REQUIRE(output[NVVS_METADATA].isMember("num_entities"));
        REQUIRE(output[NVVS_METADATA]["num_entities"].asUInt() == 5);
    }

    SECTION("HelperJsonAddHosts - Complete Host Information")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddHosts(output, response);

        // Verify hosts array exists and has correct number of entries
        REQUIRE(output.isMember("hosts"));
        REQUIRE(output["hosts"].isArray());
        REQUIRE(output["hosts"].size() == 3);

        // Verify first host
        Json::Value host0 = output["hosts"][0];
        REQUIRE(host0["host_id"].asUInt() == 0);
        REQUIRE(host0["hostname"].asString() == "host-01");
        REQUIRE(host0["dcgm_version"].asString() == "4.3.0");
        REQUIRE(host0["driver_version"].asString() == "570.32.17");
        REQUIRE(host0["num_entities"].asUInt() == 2);

        // Verify host0 entities
        REQUIRE(host0.isMember(NVVS_ENTITIES));
        REQUIRE(host0[NVVS_ENTITIES].isArray());
        REQUIRE(host0[NVVS_ENTITIES].size() == 2);

        Json::Value entity0 = host0[NVVS_ENTITIES][0];
        REQUIRE(entity0[NVVS_ENTITY_GRP_ID].asUInt() == DCGM_FE_GPU);
        REQUIRE(entity0[NVVS_ENTITY_GRP].asString() == "GPU");
        REQUIRE(entity0[NVVS_ENTITY_ID].asUInt() == 0);
        REQUIRE(entity0[NVVS_ENTITY_SERIAL].asString() == "0324522050123");
        REQUIRE(entity0[NVVS_ENTITY_DEVICE_ID].asString() == "20F1");

        // Verify second host
        Json::Value host1 = output["hosts"][1];
        REQUIRE(host1["host_id"].asUInt() == 1);
        REQUIRE(host1["hostname"].asString() == "host-02");
        REQUIRE(host1["num_entities"].asUInt() == 2);

        // Verify third host
        Json::Value host2 = output["hosts"][2];
        REQUIRE(host2["host_id"].asUInt() == 2);
        REQUIRE(host2["hostname"].asString() == "host-03");
        REQUIRE(host2["num_entities"].asUInt() == 1);
    }

    SECTION("HelperJsonAddTest - Test Structure and Results")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddTest(output, response);

        // Verify test structure
        REQUIRE(output.isMember(NVVS_HEADERS));
        REQUIRE(output[NVVS_HEADERS].isArray());
        REQUIRE(output[NVVS_HEADERS].size() == 1);

        Json::Value testCategory = output[NVVS_HEADERS][0];
        REQUIRE(testCategory[NVVS_HEADER].asString() == "Multi-Node Tests");
        REQUIRE(testCategory.isMember(NVVS_TESTS));
        REQUIRE(testCategory[NVVS_TESTS].isArray());
        REQUIRE(testCategory[NVVS_TESTS].size() == 1);

        Json::Value test = testCategory[NVVS_TESTS][0];
        REQUIRE(test[NVVS_TEST_NAME].asString() == "MNUBERGEMM");
        REQUIRE(test[NVVS_STATUS].asString() == "Fail");

        // Verify test summary
        REQUIRE(test.isMember(NVVS_TEST_SUMMARY));
        REQUIRE(test[NVVS_TEST_SUMMARY][NVVS_STATUS].asString() == "Fail");

        // Verify results array
        REQUIRE(test.isMember(NVVS_RESULTS));
        REQUIRE(test[NVVS_RESULTS].isArray());
        REQUIRE(test[NVVS_RESULTS].size() == 5);

        // Check first result (Host 0, GPU 0 - PASS)
        Json::Value result0 = test[NVVS_RESULTS][0];
        REQUIRE(result0["host_id"].asUInt() == 0);
        REQUIRE(result0[NVVS_ENTITY_GRP_ID].asUInt() == DCGM_FE_GPU);
        REQUIRE(result0[NVVS_ENTITY_GRP].asString() == "GPU");
        REQUIRE(result0[NVVS_ENTITY_ID].asUInt() == 0);
        REQUIRE(result0[NVVS_STATUS].asString() == "Pass");

        // Check second result (Host 0, GPU 1 - FAIL with error)
        Json::Value result1 = test[NVVS_RESULTS][1];
        REQUIRE(result1["host_id"].asUInt() == 0);
        REQUIRE(result1[NVVS_ENTITY_ID].asUInt() == 1);
        REQUIRE(result1[NVVS_STATUS].asString() == "Fail");
        REQUIRE(result1.isMember(NVVS_WARNINGS));
        REQUIRE(result1[NVVS_WARNINGS].isArray());
        REQUIRE(result1[NVVS_WARNINGS].size() == 1);

        Json::Value warning = result1[NVVS_WARNINGS][0];
        REQUIRE(warning[NVVS_WARNING].asString() == "CUDA memory allocation failed");
        REQUIRE(warning[NVVS_ERROR_ID].asUInt() == DCGM_FR_UNKNOWN);
    }

    SECTION("HelperJsonAddErrors - Global Errors")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = CreateJsonTestResponse();

        // Add a global error (not associated with specific entity)
        response.errors[response.numErrors].hostId               = 0;
        response.errors[response.numErrors].testId               = 0;
        response.errors[response.numErrors].entity.entityGroupId = DCGM_FE_NONE;
        response.errors[response.numErrors].entity.entityId      = 0;
        response.errors[response.numErrors].code                 = DCGM_FR_UNKNOWN;
        response.errors[response.numErrors].category             = DCGM_FR_EC_HARDWARE_OTHER;
        response.errors[response.numErrors].severity             = DCGM_ERROR_TRIAGE;
        SafeCopyTo(response.errors[response.numErrors].msg, "MPI initialization failed");
        response.numErrors++;

        Json::Value output;

        tester.HelperJsonAddErrors(output, response);

        // Verify global errors
        REQUIRE(output.isMember("global_errors"));
        REQUIRE(output["global_errors"].isArray());
        REQUIRE(output["global_errors"].size() == 1);

        Json::Value globalError = output["global_errors"][0];
        REQUIRE(globalError[NVVS_WARNING].asString() == "MPI initialization failed");
        REQUIRE(globalError[NVVS_ERROR_ID].asUInt() == DCGM_FR_UNKNOWN);
        REQUIRE(globalError[NVVS_ERROR_CATEGORY].asUInt() == DCGM_FR_EC_HARDWARE_OTHER);
        REQUIRE(globalError[NVVS_ERROR_SEVERITY].asUInt() == DCGM_ERROR_TRIAGE);
        REQUIRE(globalError["host_id"].asUInt() == 0);
    }

    SECTION("HelperDisplayAsJson - Complete JSON Output")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = CreateJsonTestResponse();

        // Capture stdout to verify JSON output
        StdoutRedirect redirect;

        dcgmReturn_t result = tester.HelperDisplayAsJson(response);

        REQUIRE(result == DCGM_ST_OK);

        std::string output = redirect.GetOutput();
        REQUIRE(!output.empty());

        // Parse the JSON output to verify structure
        Json::Value jsonOutput;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse(output, jsonOutput);
        REQUIRE(parsingSuccessful);

        // Verify main structure
        REQUIRE(jsonOutput.isMember(NVVS_NAME));
        REQUIRE(jsonOutput.isMember(NVVS_METADATA));
        REQUIRE(jsonOutput.isMember("hosts"));
        REQUIRE(jsonOutput.isMember(NVVS_HEADERS));

        // Verify it contains all expected data
        REQUIRE(jsonOutput["hosts"].size() == 3);
        REQUIRE(jsonOutput[NVVS_HEADERS].size() == 1);
        REQUIRE(jsonOutput[NVVS_HEADERS][0][NVVS_TESTS].size() == 1);
        REQUIRE(jsonOutput[NVVS_HEADERS][0][NVVS_TESTS][0][NVVS_RESULTS].size() == 5);
    }

    SECTION("HelperDisplayFailureMessage - JSON Error Format")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        // Create a dummy response with version 1
        dcgmMnDiagResponse_t dummyResponse = {};
        dummyResponse.version              = dcgmMnDiagResponse_version1;

        // Capture stdout to verify JSON error output
        StdoutRedirect redirect;

        std::string errorMessage = "Multi-node diagnostic failed with timeout";
        tester.HelperDisplayFailureMessage(errorMessage, DCGM_ST_TIMEOUT, dummyResponse);

        std::string output = redirect.GetOutput();
        REQUIRE(!output.empty());

        // Parse the JSON error output
        Json::Value jsonOutput;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse(output, jsonOutput);
        REQUIRE(parsingSuccessful);

        // Verify error structure
        REQUIRE(jsonOutput.isMember(NVVS_NAME));
        REQUIRE(jsonOutput[NVVS_NAME].asString() == "DCGM Multi-Node Diagnostic");
        REQUIRE(jsonOutput.isMember(NVVS_RUNTIME_ERROR));
        REQUIRE(jsonOutput[NVVS_RUNTIME_ERROR].asString() == errorMessage);
    }

    SECTION("Empty Response - JSON Output")
    {
        MnDiag mnDiag("localhost");
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        dcgmMnDiagResponse_t response = {};
        response.version              = dcgmMnDiagResponse_version1;
        response.numTests             = 0;
        response.numHosts             = 0;
        response.numResults           = 0;
        response.numErrors            = 0;
        response.numEntities          = 0;

        // Capture stdout to verify JSON output
        StdoutRedirect redirect;

        dcgmReturn_t result = tester.HelperDisplayAsJson(response);

        REQUIRE(result == DCGM_ST_OK);

        std::string output = redirect.GetOutput();
        REQUIRE(!output.empty());

        // Parse the JSON output
        Json::Value jsonOutput;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse(output, jsonOutput);
        REQUIRE(parsingSuccessful);

        // Verify basic structure exists even with empty data
        REQUIRE(jsonOutput.isMember(NVVS_NAME));
        REQUIRE(jsonOutput.isMember(NVVS_METADATA));
        REQUIRE(jsonOutput[NVVS_METADATA]["num_hosts"].asUInt() == 0);
        REQUIRE(jsonOutput[NVVS_METADATA]["num_entities"].asUInt() == 0);
    }
}