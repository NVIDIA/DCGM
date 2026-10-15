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

#include <catch2/catch_all.hpp>

#include <CoutCapture.hpp>
#include <TempFile.hpp>

#include <filesystem>
#include <memory>

#include <DcgmStringHelpers.h>
#include <MnDiag.h>
#include <MockDcgmOps.h>
#include <NvvsJsonStrings.h> // For JSON constants
#include <UniquePtrUtil.h>
#include <dcgm_errors.h> // For error handling
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
    dcgmReturn_t HelperDisplayAsJson(dcgmMnDiagResponse_v2 const &response)
    {
        return m_mnDiag.HelperDisplayAsJson(response);
    }

    void HelperJsonAddMetadata(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
    {
        m_mnDiag.HelperJsonAddMetadata(output, response);
    }

    void HelperJsonAddHosts(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
    {
        m_mnDiag.HelperJsonAddHosts(output, response);
    }

    void HelperJsonAddTest(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
    {
        m_mnDiag.HelperJsonAddTest(output, response);
    }

    void HelperJsonAddErrors(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
    {
        m_mnDiag.HelperJsonAddErrors(output, response);
    }

    void HelperDisplayFailureMessage(std::string_view errMsg,
                                     dcgmReturn_t result,
                                     dcgmMnDiagResponse_v2 const &response)
    {
        m_mnDiag.HelperDisplayFailureMessage(errMsg, result, response);
    }

private:
    MnDiag &m_mnDiag;
};

namespace
{
struct MnDiagTestFixture
{
    MnDiagTestFixture()
        : ownedMockDcgmOps(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(*ownedMockDcgmOps)
        , dcgmOps(&mockDcgmOps)
    {}

    std::unique_ptr<DcgmNs::MockDcgmOps> ownedMockDcgmOps;
    DcgmNs::MockDcgmOps &mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
};

class TestStartMnDiag : public StartMnDiag
{
public:
    using StartMnDiag::StartMnDiag;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestAbortMnDiag : public AbortMnDiag
{
public:
    using AbortMnDiag::AbortMnDiag;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};
} // namespace

// Helper function to create a test response
std::unique_ptr<dcgmMnDiagResponse_t> CreateTestResponse(dcgmDiagResult_t overallResult,
                                                         unsigned int numHosts,
                                                         unsigned int numErrors)
{
    auto response     = MakeUniqueZero<dcgmMnDiagResponse_t>();
    response->version = dcgmMnDiagResponse_version2;

    // Set up test data - first, constrain the number of hosts
    numHosts           = std::min(numHosts, (unsigned int)DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
    response->numHosts = static_cast<unsigned short>(numHosts);

    // Set the test fields
    response->numTests        = 1;
    response->tests[0].result = overallResult;
    SafeCopyTo(response->tests[0].name, "MNUBERGEMM");

    // Set up host information
    if (numHosts > 0)
    {
        SafeCopyTo(response->hosts[0].hostname, "test-host-0");
        SafeCopyTo(response->hosts[0].driverVersion, "570.32.17");
        SafeCopyTo(response->hosts[0].dcgmVersion, "4.3.0");
    }

    if (numHosts > 1)
    {
        SafeCopyTo(response->hosts[1].hostname, "test-host-1");
        SafeCopyTo(response->hosts[1].driverVersion, "570.32.17");
        SafeCopyTo(response->hosts[1].dcgmVersion, "4.3.0");
    }

    if (numHosts > 2)
    {
        SafeCopyTo(response->hosts[2].hostname, "test-host-2");
        SafeCopyTo(response->hosts[2].driverVersion, "570.32.17");
        SafeCopyTo(response->hosts[2].dcgmVersion, "4.3.0");
    }

    if (numHosts > 3)
    {
        SafeCopyTo(response->hosts[3].hostname, "test-host-3");
        SafeCopyTo(response->hosts[3].driverVersion, "570.32.17");
        SafeCopyTo(response->hosts[3].dcgmVersion, "4.3.0");
    }

    if (numHosts > 4)
    {
        SafeCopyTo(response->hosts[4].hostname, "test-host-4");
        SafeCopyTo(response->hosts[4].driverVersion, "570.32.17");
        SafeCopyTo(response->hosts[4].dcgmVersion, "4.3.0");
    }

    // Set up entity results with different failure patterns:
    // - Host 0: GPU 0 fails
    // - Host 1: GPU 1 fails
    // - Host 2: GPU 0 fails
    // - Host 3: GPU 1 fails
    // - Host 4: passes
    response->numResults = 0;

    if (numHosts > 0)
    {
        // Host 0, GPU 0 - FAIL
        response->results[response->numResults].hostId               = 0;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 0;
        response->results[response->numResults].result               = overallResult;
        response->results[response->numResults].testId               = 0;
        response->numResults++;

        // Host 0, GPU 1 - PASS
        response->results[response->numResults].hostId               = 0;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 1;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;
    }

    if (numHosts > 1)
    {
        // Host 1, GPU 0 - PASS
        response->results[response->numResults].hostId               = 1;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 0;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;

        // Host 1, GPU 1 - FAIL
        response->results[response->numResults].hostId               = 1;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 1;
        response->results[response->numResults].result               = overallResult;
        response->results[response->numResults].testId               = 0;
        response->numResults++;
    }

    if (numHosts > 2)
    {
        // Host 2, GPU 0 - FAIL
        response->results[response->numResults].hostId               = 2;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 0;
        response->results[response->numResults].result               = overallResult;
        response->results[response->numResults].testId               = 0;
        response->numResults++;

        // Host 2, GPU 1 - PASS
        response->results[response->numResults].hostId               = 2;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 1;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;
    }

    if (numHosts > 3)
    {
        // Host 3, GPU 0 - PASS
        response->results[response->numResults].hostId               = 3;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 0;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;

        // Host 3, GPU 1 - FAIL
        response->results[response->numResults].hostId               = 3;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 1;
        response->results[response->numResults].result               = overallResult;
        response->results[response->numResults].testId               = 0;
        response->numResults++;
    }

    // Add Host 4 results (always passes)
    if (numHosts > 4)
    {
        // Host 4, GPU 0 - Always PASS
        response->results[response->numResults].hostId               = 4;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 0;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;

        // Host 4, GPU 1 - Always PASS
        response->results[response->numResults].hostId               = 4;
        response->results[response->numResults].entity.entityGroupId = DCGM_FE_GPU;
        response->results[response->numResults].entity.entityId      = 1;
        response->results[response->numResults].result               = DCGM_DIAG_RESULT_PASS;
        response->results[response->numResults].testId               = 0;
        response->numResults++;
    }

    // Create one error message for each failing GPU, up to numErrors
    response->numErrors         = 0;
    unsigned int errorsToCreate = std::min(numErrors, (unsigned int)DCGM_MN_DIAG_RESPONSE_ERRORS_MAX);

    // Host 0, GPU 0 error
    if (numHosts > 0 && response->numErrors < errorsToCreate)
    {
        response->errors[response->numErrors].hostId               = 0;
        response->errors[response->numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response->errors[response->numErrors].entity.entityId      = 0;
        response->errors[response->numErrors].testId               = 0;
        response->errors[response->numErrors].code                 = 100;
        response->errors[response->numErrors].category             = 1;
        response->errors[response->numErrors].severity             = 2;
        SafeCopyTo(response->errors[response->numErrors].msg, "Test error message for host 0, GPU 0");
        response->numErrors++;
    }

    // Host 1, GPU 1 error
    if (numHosts > 1 && response->numErrors < errorsToCreate)
    {
        response->errors[response->numErrors].hostId               = 1;
        response->errors[response->numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response->errors[response->numErrors].entity.entityId      = 1;
        response->errors[response->numErrors].testId               = 0;
        response->errors[response->numErrors].code                 = 101;
        response->errors[response->numErrors].category             = 1;
        response->errors[response->numErrors].severity             = 2;
        SafeCopyTo(response->errors[response->numErrors].msg, "Test error message for host 1, GPU 1");
        response->numErrors++;
    }

    // Host 2, GPU 0 error
    if (numHosts > 2 && response->numErrors < errorsToCreate)
    {
        response->errors[response->numErrors].hostId               = 2;
        response->errors[response->numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response->errors[response->numErrors].entity.entityId      = 0;
        response->errors[response->numErrors].testId               = 0;
        response->errors[response->numErrors].code                 = 102;
        response->errors[response->numErrors].category             = 1;
        response->errors[response->numErrors].severity             = 2;
        SafeCopyTo(response->errors[response->numErrors].msg, "Test error message for host 2, GPU 0");
        response->numErrors++;
    }

    // Host 3, GPU 1 error
    if (numHosts > 3 && response->numErrors < errorsToCreate)
    {
        response->errors[response->numErrors].hostId               = 3;
        response->errors[response->numErrors].entity.entityGroupId = DCGM_FE_GPU;
        response->errors[response->numErrors].entity.entityId      = 1;
        response->errors[response->numErrors].testId               = 0;
        response->errors[response->numErrors].code                 = 103;
        response->errors[response->numErrors].category             = 1;
        response->errors[response->numErrors].severity             = 2;
        SafeCopyTo(response->errors[response->numErrors].msg, "Test error message for host 3, GPU 1");
        response->numErrors++;
    }

    return response;
}

SCENARIO_METHOD(MnDiagTestFixture, "MnDiag::HelperDisplayAsCli output formatting")
{
    SECTION("Test with passing result")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 2, 0);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

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
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 2);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

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
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_WARN, 1, 1);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

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

    SECTION("Test with skipped result")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response               = CreateTestResponse(DCGM_DIAG_RESULT_SKIP, 1, 0);
        response->results[1].result = DCGM_DIAG_RESULT_SKIP;
        dcgmReturn_t result         = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        CHECK(result == DCGM_ST_OK);
        CHECK(output.contains("Multi-node diagnostic skipped"));
        CHECK(output.contains("Host 0                    | Skip"));
        CHECK(output.contains("  GPU 0                   | Skip"));
        CHECK_FALSE(output.contains("Multi-node diagnostic failed"));
        CHECK_FALSE(output.contains("Error Summary"));
        CHECK_FALSE(output.contains("MPI Error"));
    }

    SECTION("Test with passing and skipped GPU results")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response               = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 1, 0);
        response->results[1].result = DCGM_DIAG_RESULT_SKIP;
        dcgmReturn_t result         = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        CHECK(result == DCGM_ST_OK);
        CHECK(output.contains("Host 0                    | Pass"));
        CHECK(output.contains("  GPU 0                   | Pass"));
        CHECK(output.contains("  GPU 1                   | Skip"));
    }

    SECTION("Test with skipped result and no entity results")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response        = CreateTestResponse(DCGM_DIAG_RESULT_SKIP, 1, 0);
        response->numResults = 0;
        dcgmReturn_t result  = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        CHECK(result == DCGM_ST_OK);
        CHECK(output.contains("Host 0                    | Skip"));
        CHECK_FALSE(output.contains("Host 0                    | Pass"));
        CHECK(output.contains("No GPU results available"));
    }

    SECTION("Test with warning result and no errors")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response = CreateTestResponse(DCGM_DIAG_RESULT_WARN, 1, 0);
        SafeCopyTo(response->tests[0].auxData.data, "MPI returned a warning");
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        CHECK(result == DCGM_ST_OK);
        CHECK(output.contains("Error Summary"));
        CHECK(output.contains("MPI Error"));
        CHECK(output.contains("MPI returned a warning"));
    }

    SECTION("Test with no hosts")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 0, 0);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // Should not have host details section with data
        REQUIRE(!output.contains("Host 0                    |"));

        // Version information should not be displayed when there are no hosts
        REQUIRE(!output.contains("Driver Version"));
        REQUIRE(!output.contains("DCGM Version"));
    }

    SECTION("Test name comes from the diagnostic response")
    {
        auto const testName = GENERATE("MNUBERGEMM", "RANDOM_TEST_NAME");
        CAPTURE(testName);

        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response = CreateTestResponse(DCGM_DIAG_RESULT_PASS, 1, 0);
        SafeCopyTo(response->tests[0].name, testName);

        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);
        std::string output  = redirect.GetOutput();
        auto const expected = fmt::format("| {:<26}| {:<47}|", fmt::format("{} Test", testName), "Pass");
        auto const pos      = output.find(expected);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(pos != std::string::npos);
        CHECK(output.substr(pos, expected.size()) == expected);
    }

    SECTION("Test GPU-specific error display")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        // Create test response with multiple hosts and all possible errors
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 4, 4);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

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

    SECTION("Test mixed GPU-specific and global error display")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response              = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 1, 1);
        auto &error                = response->errors[response->numErrors++];
        error.entity.entityGroupId = DCGM_FE_NONE;
        error.entity.entityId      = 0;
        error.hostId               = 0;
        error.testId               = 0;
        SafeCopyTo(error.msg, "MPI_ABORT was invoked on rank 0");

        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);
        std::string output  = redirect.GetOutput();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(output.contains("Host 0 (GPU 0)"));
        REQUIRE(output.contains("Test error message for host 0, GPU 0"));
        REQUIRE(output.contains("Global Error"));
        REQUIRE(output.contains("MPI_ABORT was invoked on rank 0"));
    }

    SECTION("Test host with all passing GPUs")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        // Create test response with 5 hosts, including one with all passing GPUs
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 4);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        // Check result code
        REQUIRE(result == DCGM_ST_OK);

        // This host has passing GPU results despite failures elsewhere in the test.
        REQUIRE(output.contains("Host 4")); // Host info with just index
        REQUIRE(output.contains("Host 4                    | Pass"));

        // Verify that Host 4's GPUs are shown with a summary format
        REQUIRE(output.contains("  GPUs: 0, 1"));

        // Verify that Host 4 doesn't appear in the Error Summary section
        REQUIRE(!output.contains("Host 4 (GPU"));

        // Verify that the "Hosts With Errors" count remains specific to failed GPU results.
        REQUIRE(output.contains("| Hosts With Errors         | 4"));
    }

    SECTION("Test failed diagnostic with no host result")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        auto response        = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 2, 0);
        response->numResults = 0;
        dcgmReturn_t result  = tester.HelperDisplayAsCli(*response, false);

        std::string output = redirect.GetOutput();

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(output.contains("| Hosts With Errors         | 0"));
        REQUIRE(output.contains("Host 0                    | No host result available"));
        REQUIRE(output.contains("Host 1                    | No host result available"));
        REQUIRE(!output.contains("  GPUs                    | No GPU results available"));
    }

    SECTION("Test host details disabled (mndiagFailed = true)")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        // Create test response with multiple hosts
        auto response       = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 1, 1);
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, true);

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

SCENARIO_METHOD(MnDiagTestFixture, "MnDiag CLI Display Documentation", "[.documentation]")
{
    SECTION("Save sample output to file")
    {
        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        // Create a comprehensive test response
        auto response = CreateTestResponse(DCGM_DIAG_RESULT_FAIL, 5, 4);

        // Execute the display method
        dcgmReturn_t result = tester.HelperDisplayAsCli(*response, true);
        REQUIRE(result == DCGM_ST_OK);

        // Get the captured output
        std::string output = redirect.GetOutput();

        // Write the output to a file
        auto outputFile = TempFile::Random(output, "mndiag-sample-output-");
        REQUIRE(std::filesystem::file_size(outputFile.Path()) == output.size());

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
std::unique_ptr<dcgmMnDiagResponse_t> CreateJsonTestResponse()
{
    auto response     = MakeUniqueZero<dcgmMnDiagResponse_t>();
    response->version = dcgmMnDiagResponse_version2;

    // Set up test information
    response->numTests = 1;
    SafeCopyTo(response->tests[0].name, "MNUBERGEMM");
    SafeCopyTo(response->tests[0].pluginName, "MNUBERGEMM");
    response->tests[0].result = DCGM_DIAG_RESULT_FAIL;

    // Set up hosts
    response->numHosts = 3;

    // Host 0
    SafeCopyTo(response->hosts[0].hostname, "host-01");
    SafeCopyTo(response->hosts[0].dcgmVersion, "4.3.0");
    SafeCopyTo(response->hosts[0].driverVersion, "570.32.17");
    response->hosts[0].numEntities      = 2;
    response->hosts[0].entityIndices[0] = 0;
    response->hosts[0].entityIndices[1] = 1;

    // Host 1
    SafeCopyTo(response->hosts[1].hostname, "host-02");
    SafeCopyTo(response->hosts[1].dcgmVersion, "4.3.0");
    SafeCopyTo(response->hosts[1].driverVersion, "570.32.17");
    response->hosts[1].numEntities      = 2;
    response->hosts[1].entityIndices[0] = 2;
    response->hosts[1].entityIndices[1] = 3;

    // Host 2
    SafeCopyTo(response->hosts[2].hostname, "host-03");
    SafeCopyTo(response->hosts[2].dcgmVersion, "4.3.0");
    SafeCopyTo(response->hosts[2].driverVersion, "570.32.17");
    response->hosts[2].numEntities      = 1;
    response->hosts[2].entityIndices[0] = 4;

    // Set up entities
    response->numEntities = 5;

    // Host 0 entities
    response->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    response->entities[0].entity.entityId      = 0;
    SafeCopyTo(response->entities[0].serialNum, "0324522050123");
    SafeCopyTo(response->entities[0].skuDeviceId, "20F1");

    response->entities[1].entity.entityGroupId = DCGM_FE_GPU;
    response->entities[1].entity.entityId      = 1;
    SafeCopyTo(response->entities[1].serialNum, "0324522050124");
    SafeCopyTo(response->entities[1].skuDeviceId, "20F1");

    // Host 1 entities
    response->entities[2].entity.entityGroupId = DCGM_FE_GPU;
    response->entities[2].entity.entityId      = 0;
    SafeCopyTo(response->entities[2].serialNum, "0324522050125");
    SafeCopyTo(response->entities[2].skuDeviceId, "20F1");

    response->entities[3].entity.entityGroupId = DCGM_FE_GPU;
    response->entities[3].entity.entityId      = 1;
    SafeCopyTo(response->entities[3].serialNum, "0324522050126");
    SafeCopyTo(response->entities[3].skuDeviceId, "20F1");

    // Host 2 entities
    response->entities[4].entity.entityGroupId = DCGM_FE_GPU;
    response->entities[4].entity.entityId      = 0;
    SafeCopyTo(response->entities[4].serialNum, "0324522050127");
    SafeCopyTo(response->entities[4].skuDeviceId, "20F1");

    // Set up results
    response->numResults = 5;

    // Host 0 results
    response->results[0].hostId               = 0;
    response->results[0].entity.entityGroupId = DCGM_FE_GPU;
    response->results[0].entity.entityId      = 0;
    response->results[0].result               = DCGM_DIAG_RESULT_PASS;
    response->results[0].testId               = 0;

    response->results[1].hostId               = 0;
    response->results[1].entity.entityGroupId = DCGM_FE_GPU;
    response->results[1].entity.entityId      = 1;
    response->results[1].result               = DCGM_DIAG_RESULT_FAIL;
    response->results[1].testId               = 0;

    // Host 1 results
    response->results[2].hostId               = 1;
    response->results[2].entity.entityGroupId = DCGM_FE_GPU;
    response->results[2].entity.entityId      = 0;
    response->results[2].result               = DCGM_DIAG_RESULT_PASS;
    response->results[2].testId               = 0;

    response->results[3].hostId               = 1;
    response->results[3].entity.entityGroupId = DCGM_FE_GPU;
    response->results[3].entity.entityId      = 1;
    response->results[3].result               = DCGM_DIAG_RESULT_FAIL;
    response->results[3].testId               = 0;

    // Host 2 results
    response->results[4].hostId               = 2;
    response->results[4].entity.entityGroupId = DCGM_FE_GPU;
    response->results[4].entity.entityId      = 0;
    response->results[4].result               = DCGM_DIAG_RESULT_PASS;
    response->results[4].testId               = 0;

    // Set up errors
    response->numErrors = 2;

    // Error for Host 0, GPU 1
    response->errors[0].hostId               = 0;
    response->errors[0].testId               = 0;
    response->errors[0].entity.entityGroupId = DCGM_FE_GPU;
    response->errors[0].entity.entityId      = 1;
    response->errors[0].code                 = DCGM_FR_UNKNOWN;
    response->errors[0].category             = DCGM_FR_EC_HARDWARE_OTHER;
    response->errors[0].severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(response->errors[0].msg, "CUDA memory allocation failed");

    // Error for Host 1, GPU 1
    response->errors[1].hostId               = 1;
    response->errors[1].testId               = 0;
    response->errors[1].entity.entityGroupId = DCGM_FE_GPU;
    response->errors[1].entity.entityId      = 1;
    response->errors[1].code                 = DCGM_FR_UNKNOWN;
    response->errors[1].category             = DCGM_FR_EC_HARDWARE_OTHER;
    response->errors[1].severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(response->errors[1].msg, "NCCL communication timeout");

    // Set up test indices for JSON output
    response->tests[0].numResults = response->numResults;
    for (unsigned int i = 0; i < response->numResults; i++)
    {
        response->tests[0].resultIndices[i] = i;
    }

    response->tests[0].numErrors = response->numErrors;
    for (unsigned int i = 0; i < response->numErrors; i++)
    {
        response->tests[0].errorIndices[i] = i;
    }

    return response;
}

SCENARIO_METHOD(MnDiagTestFixture, "MnDiag: GPU pair mode CLI presentation")
{
    GIVEN("a pair-mode response with forwarded log paths")
    {
        auto response      = CreateJsonTestResponse();
        response->numHosts = 2;
        SafeCopyTo(response->tests[0].auxData.data,
                   "MNNVBANDWIDTH_SUMMARY planned=6;passed=5;failed=1;stdout=/var/log/mndiag.stdout;"
                   "stderr=/var/log/mndiag.stderr");
        response->errors[0].entity.entityGroupId = DCGM_FE_NONE;
        response->errors[1].entity.entityGroupId = DCGM_FE_NONE;
        SafeCopyTo(response->errors[0].msg,
                   "GPU pair first-hostname-that-is-long GPU 0 <-> second-hostname-that-is-long GPU 1 failed: "
                   "[CUDA_ERROR_NOT_READY] device not ready in expression cuMemImportFromShareableHandle");
        SafeCopyTo(response->errors[1].msg, "2 additional GPU pair failures omitted");

        CoutCapture redirect;
        MnDiag mnDiag("testhost", dcgmOps);
        MnDiagTester tester(mnDiag);

        WHEN("the response is displayed as text")
        {
            dcgmReturn_t const result = tester.HelperDisplayAsCli(*response, false);
            std::string const output  = redirect.GetOutput();

            THEN("the pair metadata and forwarded log paths are shown without a GPU list")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(output.contains("| Pairs Planned             | 6"));
                CHECK(output.contains("| Pairs Passed              | 5"));
                CHECK(output.contains("| Pairs Failed              | 1"));
                CHECK_FALSE(output.contains("GPU List"));
                CHECK(output.contains("Log Files"));
                CHECK(output.contains("/var/log/mndiag.stdout"));
                CHECK(output.contains("/var/log/mndiag.stderr"));
                CHECK(output.contains("Error Summary"));
                CHECK(output.contains("first-hostname-that-is-lo"));
                CHECK(output.contains("ng GPU 0"));
                CHECK(output.contains("<-> second-hostname-that-"));
                CHECK(output.contains("is-long GPU 1"));
                CHECK(output.contains("Error: [CUDA_ERROR_NOT_READY] device not read"));
                CHECK(output.contains("y in expression cuMemImportFromShareableHandl"));
                CHECK_FALSE(output.contains("MPI exited with code"));
                CHECK_FALSE(output.contains("Error: GPU pair"));
                CHECK_FALSE(output.contains("No GPU-specific errors reported"));
            }
        }
    }
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

TEST_CASE_METHOD(MnDiagTestFixture, "MnDiag JSON Output Methods")
{
    SECTION("HelperJsonAddMetadata - Basic Metadata")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddMetadata(output, *response);

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
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddHosts(output, *response);

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
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response = CreateJsonTestResponse();
        Json::Value output;

        tester.HelperJsonAddTest(output, *response);

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

    SECTION("HelperJsonAddTest - Failed test without errors reports an MPI error")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        MnDiagTester tester(mnDiag);
        auto response                = CreateJsonTestResponse();
        response->numErrors          = 0;
        response->tests[0].numErrors = 0;
        Json::Value output;

        tester.HelperJsonAddTest(output, *response);

        Json::Value const &test = output[NVVS_HEADERS][0][NVVS_TESTS][0];
        CHECK(test[NVVS_STATUS].asString() == "Fail");
        CHECK(test.isMember("mpi_error"));
    }

    SECTION("HelperJsonAddTest - Skipped test without errors does not report an MPI error")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        MnDiagTester tester(mnDiag);
        auto response                = CreateJsonTestResponse();
        response->numErrors          = 0;
        response->tests[0].result    = DCGM_DIAG_RESULT_SKIP;
        response->tests[0].numErrors = 0;
        Json::Value output;

        tester.HelperJsonAddTest(output, *response);

        Json::Value const &test = output[NVVS_HEADERS][0][NVVS_TESTS][0];
        CHECK(test[NVVS_STATUS].asString() == "Skip");
        CHECK_FALSE(test.isMember("mpi_error"));
    }

    SECTION("HelperJsonAddTest - Warning without errors reports an MPI error")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        MnDiagTester tester(mnDiag);
        auto response                = CreateJsonTestResponse();
        response->numErrors          = 0;
        response->tests[0].result    = DCGM_DIAG_RESULT_WARN;
        response->tests[0].numErrors = 0;
        Json::Value output;

        tester.HelperJsonAddTest(output, *response);

        Json::Value const &test = output[NVVS_HEADERS][0][NVVS_TESTS][0];
        CHECK(test[NVVS_STATUS].asString() == "Warn");
        CHECK(test.isMember("mpi_error"));
    }

    SECTION("HelperJsonAddErrors - Skipped test without errors does not synthesize a global error")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        MnDiagTester tester(mnDiag);
        auto response                = CreateJsonTestResponse();
        response->numErrors          = 0;
        response->tests[0].result    = DCGM_DIAG_RESULT_SKIP;
        response->tests[0].numErrors = 0;
        Json::Value output;

        tester.HelperJsonAddErrors(output, *response);

        CHECK_FALSE(output.isMember("global_errors"));
    }

    SECTION("HelperJsonAddErrors - Warning without errors synthesizes a global error")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        MnDiagTester tester(mnDiag);
        auto response                = CreateJsonTestResponse();
        response->numErrors          = 0;
        response->tests[0].result    = DCGM_DIAG_RESULT_WARN;
        response->tests[0].numErrors = 0;
        Json::Value output;

        tester.HelperJsonAddErrors(output, *response);

        REQUIRE(output["global_errors"].size() == 1);
        CHECK(output["global_errors"][0][NVVS_ERROR_CATEGORY].asString() == "MPI");
    }

    SECTION("HelperJsonAddErrors - Global Errors")
    {
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response = CreateJsonTestResponse();

        // Add a global error (not associated with specific entity)
        response->errors[response->numErrors].hostId               = 0;
        response->errors[response->numErrors].testId               = 0;
        response->errors[response->numErrors].entity.entityGroupId = DCGM_FE_NONE;
        response->errors[response->numErrors].entity.entityId      = 0;
        response->errors[response->numErrors].code                 = DCGM_FR_UNKNOWN;
        response->errors[response->numErrors].category             = DCGM_FR_EC_HARDWARE_OTHER;
        response->errors[response->numErrors].severity             = DCGM_ERROR_TRIAGE;
        SafeCopyTo(response->errors[response->numErrors].msg, "MPI initialization failed");
        response->numErrors++;

        Json::Value output;

        tester.HelperJsonAddErrors(output, *response);

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
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response = CreateJsonTestResponse();

        // Capture stdout to verify JSON output
        CoutCapture redirect;

        dcgmReturn_t result = tester.HelperDisplayAsJson(*response);

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
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        // Create a dummy response with version 1
        auto dummyResponse     = MakeUniqueZero<dcgmMnDiagResponse_t>();
        dummyResponse->version = dcgmMnDiagResponse_version2;

        // Capture stdout to verify JSON error output
        CoutCapture redirect;

        std::string errorMessage = "Multi-node diagnostic failed with timeout";
        tester.HelperDisplayFailureMessage(errorMessage, DCGM_ST_TIMEOUT, *dummyResponse);

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
        MnDiag mnDiag("localhost", dcgmOps);
        mnDiag.SetJsonOutput(true);
        MnDiagTester tester(mnDiag);

        auto response         = MakeUniqueZero<dcgmMnDiagResponse_t>();
        response->version     = dcgmMnDiagResponse_version2;
        response->numTests    = 0;
        response->numHosts    = 0;
        response->numResults  = 0;
        response->numErrors   = 0;
        response->numEntities = 0;

        // Capture stdout to verify JSON output
        CoutCapture redirect;

        dcgmReturn_t result = tester.HelperDisplayAsJson(*response);

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

TEST_CASE_METHOD(MnDiagTestFixture, "MnDiag command execution uses DCGM run and stop APIs")
{
    auto handle = static_cast<dcgmHandle_t>(0x6d);

    GIVEN("a StartMnDiag command with a successful mocked DCGM response")
    {
        auto request     = std::make_unique<dcgmRunMnDiag_v2>();
        request->version = dcgmRunMnDiag_version2;
        TestStartMnDiag command("mn-host", *request, false, dcgmOps);

        WHEN("the command runs against an existing handle")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.RunWithHandle(handle);
            auto output         = redirect.GetOutput();

            THEN("the run request is forwarded and CLI output is rendered")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_mnDiagState.runCallCount == 1);
                CHECK(mockDcgmOps.m_mnDiagState.lastRunHandle == handle);
                CHECK(mockDcgmOps.m_mnDiagState.lastRunVersion == dcgmRunMnDiag_version2);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 0);
                CHECK(output.contains("Successfully ran multi-node diagnostic"));
                CHECK(output.contains("MNUBERGEMM Test"));
            }
        }
    }

    GIVEN("a StartMnDiag command with JSON output enabled")
    {
        auto request     = std::make_unique<dcgmRunMnDiag_v2>();
        request->version = dcgmRunMnDiag_version2;
        TestStartMnDiag command("mn-host", *request, true, dcgmOps);

        WHEN("the mocked diagnostic succeeds")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.RunWithHandle(handle);
            auto output         = redirect.GetOutput();

            THEN("the JSON result is printed instead of the CLI table")
            {
                Json::Value jsonOutput;
                Json::Reader reader;

                CHECK(result == DCGM_ST_OK);
                REQUIRE(reader.parse(output, jsonOutput));
                CHECK(jsonOutput[NVVS_NAME].asString() == "DCGM Multi-Node Diagnostic");
                CHECK(jsonOutput[NVVS_METADATA]["num_hosts"].asUInt() == 0);
                CHECK(mockDcgmOps.m_mnDiagState.runCallCount == 1);
            }
        }
    }

    GIVEN("a StartMnDiag command whose server run times out")
    {
        mockDcgmOps.m_mnDiagState.runReturn  = DCGM_ST_TIMEOUT;
        mockDcgmOps.m_mnDiagState.stopReturn = DCGM_ST_OK;
        auto request                         = std::make_unique<dcgmRunMnDiag_v2>();
        request->version                     = dcgmRunMnDiag_version2;
        TestStartMnDiag command("mn-host", *request, false, dcgmOps);

        WHEN("the command handles the timeout")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.RunWithHandle(handle);
            auto output         = redirect.GetOutput();

            THEN("the diagnostic is stopped and the timeout message is shown")
            {
                CHECK(result == DCGM_ST_TIMEOUT);
                CHECK(mockDcgmOps.m_mnDiagState.runCallCount == 1);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 1);
                CHECK(mockDcgmOps.m_mnDiagState.lastStopHandle == handle);
                CHECK(output.contains("Unable to complete multi-node diagnostic"));
            }
        }
    }

    GIVEN("a StartMnDiag command whose timeout cleanup fails")
    {
        mockDcgmOps.m_mnDiagState.runReturn  = DCGM_ST_TIMEOUT;
        mockDcgmOps.m_mnDiagState.stopReturn = DCGM_ST_BADPARAM;
        auto request                         = std::make_unique<dcgmRunMnDiag_v2>();
        request->version                     = dcgmRunMnDiag_version2;
        TestStartMnDiag command("mn-host", *request, false, dcgmOps);

        WHEN("the command handles the timeout")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.RunWithHandle(handle);
            auto output         = redirect.GetOutput();

            THEN("the cleanup failure is included in the user-facing output")
            {
                CHECK(result == DCGM_ST_TIMEOUT);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 1);
                CHECK(output.contains("Could not stop the launched multi-node diagnostic"));
            }
        }
    }

    GIVEN("an AbortMnDiag command")
    {
        mockDcgmOps.m_mnDiagState.stopReturn = DCGM_ST_NOT_SUPPORTED;
        TestAbortMnDiag command("mn-host", dcgmOps);

        WHEN("the command runs against an existing handle")
        {
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the stop request return code is propagated")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 1);
                CHECK(mockDcgmOps.m_mnDiagState.lastStopHandle == handle);
            }
        }
    }

    GIVEN("a StartMnDiag command using the default hostengine address that cannot connect")
    {
        mockDcgmOps.SetConnectV3Result(DCGM_ST_CONNECTION_NOT_VALID);
        auto request     = std::make_unique<dcgmRunMnDiag_v2>();
        request->version = dcgmRunMnDiag_version2;
        TestStartMnDiag command("localhost", *request, false, dcgmOps);

        WHEN("the command is executed")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.Execute();
            auto output         = redirect.GetOutput();

            THEN("the connection failure is displayed and returned")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(output.contains("unable to establish a connection to the specified host: localhost"));
                CHECK(mockDcgmOps.GetConnectV3CallCount() == 1);
                CHECK(mockDcgmOps.m_mnDiagState.runCallCount == 0);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 0);
            }
        }
    }

    GIVEN("a StartMnDiag command using an explicit hostengine address that cannot connect")
    {
        mockDcgmOps.SetConnectV3Result(DCGM_ST_CONNECTION_NOT_VALID);
        auto request     = std::make_unique<dcgmRunMnDiag_v2>();
        request->version = dcgmRunMnDiag_version2;
        TestStartMnDiag command("mn-host", *request, false, dcgmOps);

        WHEN("the command is executed")
        {
            CoutCapture redirect;
            dcgmReturn_t result = command.Execute();
            auto output         = redirect.GetOutput();

            THEN("the explicit host connection failure is displayed and returned")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(output.contains("unable to establish a connection to the specified host: mn-host"));
                CHECK(mockDcgmOps.GetConnectV3CallCount() == 1);
                CHECK(mockDcgmOps.m_mnDiagState.runCallCount == 0);
                CHECK(mockDcgmOps.m_mnDiagState.stopCallCount == 0);
            }
        }
    }
}
