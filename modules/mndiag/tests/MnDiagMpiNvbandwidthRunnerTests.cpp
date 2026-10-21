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

#include <CommandLineArgUtils.hpp>
#include <DcgmStringHelpers.h>
#include <MnDiagMpiNvbandwidthRunner.h>
#include <MnDiagMpiRunnerFactory.h>
#include <ScopedEnvVars.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <UniquePtrUtil.h>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <cstdint>
#include <dcgm_structs.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "MockDcgmCoreProxy.h"

static constexpr uid_t MNNVBANDWIDTH_EFFECTIVE_UID = 0;

namespace DcgmNs::Mndiag::Nvbandwidth::detail
{
bool ReplaceArgValue(std::vector<std::string> &args, std::string_view key, std::string value);
bool ReplaceArg(std::vector<std::string> &args,
                std::string_view key,
                std::string replacementKey,
                std::string replacementValue);
} // namespace DcgmNs::Mndiag::Nvbandwidth::detail

namespace
{

std::string ReadFileContents(std::filesystem::path const &path)
{
    std::ifstream input(path);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

dcgmRunMnDiag_v2 CreateNvbandwidthDiagStruct(std::initializer_list<std::string> params = {}, unsigned int hostCount = 2)
{
    dcgmRunMnDiag_v2 diagStruct {};
    diagStruct.version = dcgmRunMnDiag_version2;
    SafeCopyTo(diagStruct.testName, "mnnvbandwidth");

    REQUIRE(hostCount <= std::size(diagStruct.hostList));
    for (unsigned int index = 0; index < hostCount; ++index)
    {
        auto const hostname = "host" + std::to_string(index + 1);
        SafeCopyTo(diagStruct.hostList[index], hostname.c_str());
    }

    REQUIRE(params.size() <= std::size(diagStruct.testParms));
    size_t index = 0;
    for (auto const &param : params)
    {
        SafeCopyTo(diagStruct.testParms[index], param.c_str());
        ++index;
    }

    return diagStruct;
}

class TestableNvbandwidthRunner : public MnDiagMpiNvbandwidthRunner
{
public:
    explicit TestableNvbandwidthRunner(DcgmCoreProxyBase &coreProxy)
        : MnDiagMpiNvbandwidthRunner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID)
    {
        m_testBinaryPath.emplace("/fake/test/mnnvbandwidth");
    }

    std::vector<std::string> const &GetLastCommandArgs() const
    {
        return m_lastCommand;
    }

    void SetTestBinaryPath(std::string path)
    {
        m_testBinaryPath.emplace(std::move(path));
    }

    void SetOutputFilePaths(std::string stdoutPath, std::string stderrPath)
    {
        m_stdoutFilePath = std::move(stdoutPath);
        m_stderrFilePath = std::move(stderrPath);
    }

    std::unordered_map<std::string, std::string> BuildInternalPairModeParams(
        std::initializer_list<std::string> parameters = {})
    {
        std::unordered_map<std::string, std::string> paramsMap;
        for (auto const &parameter : parameters)
        {
            REQUIRE(UpdateParamsMapFromRunParameter(paramsMap, parameter).has_value());
        }
        paramsMap["gpuPairs"] = "all";
        AdjustParamsMap(paramsMap, 2);
        return paramsMap;
    }
};

class PairExecutionNvbandwidthRunner final : public TestableNvbandwidthRunner
{
public:
    explicit PairExecutionNvbandwidthRunner(DcgmCoreProxyBase &coreProxy)
        : TestableNvbandwidthRunner(coreProxy)
    {}

    dcgmReturn_t LaunchMpiProcess() override
    {
        commands.push_back(GetLastCommand());
        auto const rankfileOption = std::ranges::find(m_lastCommand, "--rankfile");
        if (rankfileOption == m_lastCommand.end() || std::next(rankfileOption) == m_lastCommand.end())
        {
            return DCGM_ST_BADPARAM;
        }
        std::filesystem::path const rankfilePath(*std::next(rankfileOption));
        rankfilePaths.push_back(rankfilePath);
        rankfileExistsAtLaunch.push_back(std::filesystem::exists(rankfilePath));
        std::ifstream rankfile(rankfilePath);
        rankfileContents.emplace_back(std::istreambuf_iterator<char>(rankfile), std::istreambuf_iterator<char>());
        ++launches;
        if (m_logFileNames.has_value())
        {
            std::ofstream stdoutFile(m_logFileNames->first);
            std::ofstream stderrFile(m_logFileNames->second);
            if (!emptyPairOutput)
            {
                stdoutFile << "stdout pair " << launches;
                stderrFile << "stderr pair " << launches;
            }
        }
        return launches == 1 && failFirstLaunch ? DCGM_ST_GENERIC_ERROR : DCGM_ST_OK;
    }

    dcgmReturn_t Wait(int timeoutSec = -1) override
    {
        ++waits;
        waitTimeouts.push_back(timeoutSec);
        return launches == waitFailurePair ? DCGM_ST_GENERIC_ERROR : DCGM_ST_OK;
    }

    std::optional<pid_t> GetMpiProcessPid() const override
    {
        if (launches == missingPidPair)
        {
            return std::nullopt;
        }
        return static_cast<pid_t>(1000 + launches);
    }

    std::optional<int> GetMpiProcessExitCode() override
    {
        return launches == 4 && nonzeroFourthPair ? std::optional<int> { 9 } : std::optional<int> { 0 };
    }

    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const & /*nodeInfo*/) override
    {
        ++populatedResponses;
        if (launches == 3 && failThirdPairParsing)
        {
            return DCGM_ST_DIAG_BAD_JSON;
        }
        bool const hasFourthPairError = launches == 4 && !fourthPairError.empty();
        auto &response                = *static_cast<dcgmMnDiagResponse_v2 *>(responseStruct);
        response.tests[0].result
            = failAllPairTestcases || (launches == 5 && failFifthPairTestcase) || hasFourthPairError
                  ? DCGM_DIAG_RESULT_FAIL
                  : DCGM_DIAG_RESULT_PASS;
        if (hasFourthPairError)
        {
            response.numErrors = 1;
            SafeCopyTo(response.errors[0].msg, fourthPairError.c_str());
        }
        return DCGM_ST_OK;
    }

    dcgmReturn_t StopMpiProcess() override
    {
        ++stops;
        return launches == cleanupFailurePair ? DCGM_ST_GENERIC_ERROR : DCGM_ST_OK;
    }

    std::vector<std::string> commands;
    std::vector<int> waitTimeouts;
    std::vector<std::filesystem::path> rankfilePaths;
    std::vector<bool> rankfileExistsAtLaunch;
    std::vector<std::string> rankfileContents;
    unsigned int launches { 0 };
    unsigned int waits { 0 };
    unsigned int stops { 0 };
    unsigned int populatedResponses { 0 };
    unsigned int cleanupFailurePair { 0 };
    unsigned int waitFailurePair { 0 };
    unsigned int missingPidPair { 0 };
    bool failFirstLaunch { false };
    bool failThirdPairParsing { false };
    bool nonzeroFourthPair { false };
    bool failFifthPairTestcase { false };
    bool failAllPairTestcases { false };
    bool emptyPairOutput { false };
    std::string fourthPairError;
};

class ConfigurableGpuCountCoreProxy : public MockDcgmCoreProxy
{
public:
    explicit ConfigurableGpuCountCoreProxy(unsigned int gpuCount)
        : m_gpuCount(gpuCount)
    {}

    unsigned int GetGpuCount(GpuTypes /* activeOnly */ = GpuTypes::All) override
    {
        return m_gpuCount;
    }

private:
    unsigned int m_gpuCount;
};

std::string BuildTextWithDeviceList(unsigned int deviceCount, bool uniqueHosts)
{
    std::ostringstream output;
    for (unsigned int index = 0; index < deviceCount; ++index)
    {
        output << "Process " << index << " (" << (uniqueHosts ? "host-" + std::to_string(index) : "GB200-NVL4-TS2-114")
               << "): device " << index << ": NVIDIA GB200 (00000008:06:00)\n";
    }
    output << "Running device_to_device_memcpy_read_ce.\n"
           << "SUM device_to_device_memcpy_read_ce 42.0\n";

    return output.str();
}

std::string BuildJsonWithDeviceList(unsigned int deviceCount, bool uniqueHosts)
{
    std::ostringstream output;
    output << R"json({
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
)json";

    for (unsigned int index = 0; index < deviceCount; ++index)
    {
        if (index != 0)
        {
            output << ",\n";
        }
        output << "      \"" << index << ": NVIDIA GB200 (00000008:06:00): ("
               << (uniqueHosts ? "host-" + std::to_string(index) : "test-host-0") << ")\"";
    }

    output << R"json(
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
})json";

    return output.str();
}

std::string PrefixLines(std::string_view input, std::string_view prefix)
{
    std::istringstream lines { std::string(input) };
    std::ostringstream output;
    std::string line;
    while (std::getline(lines, line))
    {
        output << prefix << line << '\n';
    }
    return output.str();
}

} // namespace

SCENARIO("MnDiagMpiNvbandwidthRunner: replaces command-line option values")
{
    GIVEN("an option followed by a value")
    {
        std::vector<std::string> args { "-np", "8", "--map-by", "ppr:4:node" };

        WHEN("the option value is replaced")
        {
            auto const replaced = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArgValue(args, "-np", "2");

            THEN("the helper reports success and replaces only the value")
            {
                CHECK(replaced);
                CHECK(args == std::vector<std::string> { "-np", "2", "--map-by", "ppr:4:node" });
            }
        }
    }

    GIVEN("an argument vector without the requested option")
    {
        std::vector<std::string> args { "--map-by", "ppr:4:node" };

        WHEN("replacement is attempted")
        {
            auto const replaced = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArgValue(args, "-np", "2");

            THEN("the helper reports failure and leaves the arguments unchanged")
            {
                CHECK_FALSE(replaced);
                CHECK(args == std::vector<std::string> { "--map-by", "ppr:4:node" });
            }
        }
    }

    GIVEN("an option without a following value")
    {
        std::vector<std::string> args { "--map-by" };

        WHEN("replacement is attempted")
        {
            auto const replaced
                = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArgValue(args, "--map-by", "/tmp/rankfile");

            THEN("the helper reports failure and leaves the option unchanged")
            {
                CHECK_FALSE(replaced);
                CHECK(args == std::vector<std::string> { "--map-by" });
            }
        }
    }

    GIVEN("an option whose key and value must both change")
    {
        std::vector<std::string> args { "-np", "2", "--map-by", "ppr:4:node" };

        WHEN("the option is replaced")
        {
            auto const replaced
                = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArg(args, "--map-by", "--rankfile", "/tmp/rankfile");

            THEN("the helper replaces the key and value together")
            {
                CHECK(replaced);
                CHECK(args == std::vector<std::string> { "-np", "2", "--rankfile", "/tmp/rankfile" });
            }
        }
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner identity")
{
    MockDcgmCoreProxy coreProxy;
    MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

    CHECK(runner.GetTestPrefix() == "mnnvbandwidth.");
    CHECK(runner.GetLogFilePrefix() == "mndiag_mnnvbandwidth");

    auto const defaultParams = runner.GetDefaultParametersMap();
    REQUIRE(defaultParams.size() == 1);
    auto const testcase = defaultParams.find("testcase");
    REQUIRE(testcase != defaultParams.end());
    CHECK(testcase->second == "multinode_device_to_device_memcpy_write_ce multinode_device_to_device_memcpy_read_ce");
}

TEST_CASE("MnDiagMpiNvbandwidthRunner batch all-pairs execution [mndiag]")
{
    MockDcgmCoreProxy coreProxy;
    auto params = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.gpuPairs=all" });
    dcgmMnDiagResponse_v2 response {};
    response.version            = dcgmMnDiagResponse_version2;
    constexpr auto batchTimeout = std::chrono::hours(1);

    SECTION("Rejects incomplete inventory before launching a pair")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "" }, { 1, "uuid-1" } } } },
        };

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_NO_DATA);
        CHECK(runner.launches == 0);
    }

    SECTION("Rejects inventory containing fewer than two GPUs")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "uuid-0" } } } },
        };

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_NO_DATA);
        CHECK(runner.launches == 0);
    }

    SECTION("Rejects duplicate GPU IDs on a host")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 0, "GPU-1" } } } },
        };

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_BADPARAM);
        CHECK(runner.launches == 0);
    }

    SECTION("Rejects duplicate GPU UUIDs across hosts")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-shared" } } } },
            { "host2", { .gpus = { { 0, "GPU-shared" } } } },
        };

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_BADPARAM);
        CHECK(runner.launches == 0);
    }

    SECTION("Reports persistent log creation failure before launching a pair")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" } } } },
        };
        auto const nonDirectory = TempFile::Random("not a directory", "mndiag-pair-log-parent-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, nonDirectory.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_FILE_IO_ERROR);
        CHECK(runner.launches == 0);
    }

    SECTION("Executes every pair and aggregates each process failure mode")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.failFirstLaunch       = true;
        runner.failThirdPairParsing  = true;
        runner.nonzeroFourthPair     = true;
        runner.failFifthPairTestcase = true;
        nodeInfoMap_t nodeInfo {
            { "host2",
              { .dcgmVersion   = "4.0",
                .driverVersion = "600",
                .gpus          = { { 2, "GPU-remote-2" }, { 3, "GPU-remote-3" } } } },
            { "host1",
              { .dcgmVersion   = "4.0",
                .driverVersion = "600",
                .gpus          = { { 0, "GPU-local-0" }, { 1, "GPU-local-1" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-runner-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE(result.has_value());
        REQUIRE(runner.commands.size() == 6);
        CHECK(runner.launches == 6);
        CHECK(runner.waits == 5);
        CHECK(runner.stops == 6);
        CHECK(runner.populatedResponses == 5);
        CHECK(std::ranges::all_of(runner.commands, [](std::string const &command) {
            return command.find("-np 2") != std::string::npos && command.find("--rankfile") != std::string::npos
                   && command.find("--map-by") == std::string::npos && command.find("--pair") != std::string::npos;
        }));
        REQUIRE(runner.rankfileContents.size() == 6);
        CHECK(std::ranges::all_of(runner.rankfileExistsAtLaunch, std::identity {}));
        CHECK(runner.rankfileContents[0] == "rank 0=host1 slot=0\nrank 1=host1 slot=1\n");
        CHECK(runner.rankfileContents[1] == "rank 0=host1 slot=0\nrank 1=host2 slot=0\n");
        CHECK(std::ranges::none_of(runner.rankfilePaths,
                                   [](std::filesystem::path const &path) { return std::filesystem::exists(path); }));
        CHECK(runner.commands.front().find("--pair GPU-local-0 GPU-local-1") != std::string::npos);
        CHECK(response.tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response.numEntities == 4);
        CHECK(std::string_view(response.tests[0].auxData.data).contains("planned=6;passed=2;failed=4"));
        CHECK(std::ranges::all_of(std::span(response.entities, response.numEntities),
                                  [](auto const &entity) { return std::string_view(entity.serialNum).empty(); }));
        REQUIRE(response.numErrors == 4);
        CHECK(std::ranges::all_of(std::span(response.results, response.numResults),
                                  [](auto const &result) { return result.result == DCGM_DIAG_RESULT_FAIL; }));
        CHECK(std::ranges::all_of(std::span(response.errors, response.numErrors), [](auto const &error) {
            std::string_view const message(error.msg);
            return error.entity.entityGroupId == DCGM_FE_NONE && message.starts_with("GPU pair ")
                   && message.contains(" <-> ");
        }));
        CHECK(std::ranges::count_if(std::span(response.errors, response.numErrors),
                                    [](auto const &error) {
                                        return std::string_view(error.msg).contains("GPU pair MPI exited with code 9");
                                    })
              == 1);

        auto const logPaths       = DcgmNs::Utils::GetLogFilePath(runner.GetLogFilePrefix());
        auto const stdoutContents = ReadFileContents(logPaths.stdoutFileName);
        auto const stderrContents = ReadFileContents(logPaths.stderrFileName);
        CHECK(stdoutContents.contains("first_uuid=GPU-"));
        CHECK(stdoutContents.contains("command="));
        CHECK(stdoutContents.contains("status=PASS"));
        CHECK(stdoutContents.contains("status=FAIL"));
        for (size_t pairIndex = 1; pairIndex <= runner.commands.size(); ++pairIndex)
        {
            CAPTURE(pairIndex);
            CHECK(stdoutContents.contains(fmt::format("stdout pair {}", pairIndex)));
            CHECK(stderrContents.contains(fmt::format("stderr pair {}", pairIndex)));
            CHECK(stdoutContents.contains(fmt::format("PAIR sequence={} total=6", pairIndex)));
            CHECK(stderrContents.contains(fmt::format("PAIR sequence={} total=6", pairIndex)));
            CHECK_FALSE(
                std::filesystem::exists(fmt::format("{}.pair-{}.stdout", logPaths.stdoutFileName.string(), pairIndex)));
            CHECK_FALSE(
                std::filesystem::exists(fmt::format("{}.pair-{}.stderr", logPaths.stderrFileName.string(), pairIndex)));
        }
    }

    SECTION("Reports pair-error truncation without losing GPU failure results")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.failAllPairTestcases = true;
        nodeInfoMap_t nodeInfo { { "host1", {} } };
        for (unsigned int gpuId = 0; gpuId < 9; ++gpuId)
        {
            nodeInfo["host1"].gpus.push_back({ gpuId, fmt::format("GPU-{}", gpuId) });
        }
        auto const logDir = TempDir::Random("mndiag-pair-error-capacity-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE(result.has_value());
        CHECK(response.numResults == 9);
        CHECK(std::ranges::all_of(std::span(response.results, response.numResults), [](auto const &entityResult) {
            return entityResult.result == DCGM_DIAG_RESULT_FAIL;
        }));
        REQUIRE(response.numErrors == DCGM_MN_DIAG_RESPONSE_ERRORS_MAX);
        auto const errors = std::span(response.errors, response.numErrors);
        CHECK(std::ranges::count_if(
                  errors, [](auto const &error) { return std::string_view(error.msg).starts_with("GPU pair "); })
              == DCGM_MN_DIAG_RESPONSE_ERRORS_MAX - 1);
        CHECK(std::ranges::count_if(errors,
                                    [](auto const &error) {
                                        return std::string_view(error.msg)
                                               == "5 additional GPU pair failures omitted; see stderr log";
                                    })
              == 1);
    }

    SECTION("Retains pair headers when a successful pair emits no output")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.emptyPairOutput = true;
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-empty-output-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE(result.has_value());
        CHECK(response.tests[0].result == DCGM_DIAG_RESULT_PASS);
        auto const logPaths = DcgmNs::Utils::GetLogFilePath(runner.GetLogFilePrefix());
        CHECK(ReadFileContents(logPaths.stdoutFileName).contains("PAIR sequence=1 total=1"));
        CHECK(ReadFileContents(logPaths.stderrFileName).contains("PAIR sequence=1 total=1"));
    }

    SECTION("Prefers the parsed root cause over a nonzero MPI exit code")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.nonzeroFourthPair = true;
        runner.fourthPairError
            = "[CUDA_ERROR_NOT_READY] device not ready in expression cuMemImportFromShareableHandle on host1, "
              "rank = 1";
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" }, { 2, "GPU-2" }, { 3, "GPU-3" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-root-cause-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE(result.has_value());
        REQUIRE(response.numErrors == 1);
        std::string_view const error(response.errors[0].msg);
        CHECK(error.contains("[CUDA_ERROR_NOT_READY] device not ready"));
        CHECK(error.contains("rank = 1"));
        CHECK_FALSE(error.contains("MPI exited with code"));
    }

    SECTION("Aborts after a wait failure and starts no later pair")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.waitFailurePair = 2;
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" }, { 2, "GPU-2" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-wait-failure-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_GENERIC_ERROR);
        CHECK(runner.launches == 2);
        CHECK(runner.waits == 2);
        CHECK(runner.stops == 2);
        CHECK(response.numTests == 0);

        auto const logPaths       = DcgmNs::Utils::GetLogFilePath(runner.GetLogFilePrefix());
        auto const stdoutContents = ReadFileContents(logPaths.stdoutFileName);
        auto const stderrContents = ReadFileContents(logPaths.stderrFileName);
        for (size_t pairIndex = 1; pairIndex <= runner.launches; ++pairIndex)
        {
            CAPTURE(pairIndex);
            CHECK(stdoutContents.contains(fmt::format("stdout pair {}", pairIndex)));
            CHECK(stderrContents.contains(fmt::format("stderr pair {}", pairIndex)));
            CHECK_FALSE(
                std::filesystem::exists(fmt::format("{}.pair-{}.stdout", logPaths.stdoutFileName.string(), pairIndex)));
            CHECK_FALSE(
                std::filesystem::exists(fmt::format("{}.pair-{}.stderr", logPaths.stderrFileName.string(), pairIndex)));
        }
    }

    SECTION("Uses the Open MPI hostname for an endpoint selected by local IP address")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "192.0.2.10", { .gpus = { { 0, "GPU-remote" } }, .mpiNodeName = "192.0.2.10" } },
            { "192.0.2.20", { .gpus = { { 0, "GPU-local" } }, .mpiNodeName = "local-mpi-node.example.test" } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-mpi-hostname-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE(result.has_value());
        REQUIRE(runner.rankfileContents.size() == 1);
        CHECK(runner.rankfileContents[0] == "rank 0=192.0.2.10 slot=0\nrank 1=local-mpi-node.example.test slot=0\n");
        CHECK(runner.commands[0].find("--pair GPU-remote GPU-local") != std::string::npos);
    }

    SECTION("Aborts and cleans up when a successful launch has no PID")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.missingPidPair = 1;
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-missing-pid-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_CHILD_SPAWN_FAILED);
        CHECK(runner.launches == 1);
        CHECK(runner.waits == 0);
        CHECK(runner.stops == 1);
        CHECK(response.numTests == 0);

        auto const logPaths = DcgmNs::Utils::GetLogFilePath(runner.GetLogFilePrefix());
        CHECK(ReadFileContents(logPaths.stdoutFileName).empty());
        CHECK(ReadFileContents(logPaths.stderrFileName).empty());
    }

    SECTION("An expired overall timeout prevents pair launches")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-deadline-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, std::chrono::milliseconds::zero());

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_TIMEOUT);
        CHECK(runner.launches == 0);
    }

    SECTION("Aborts the batch on a cleanup failure")
    {
        PairExecutionNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });
        runner.cleanupFailurePair = 1;
        nodeInfoMap_t nodeInfo {
            { "host1", { .gpus = { { 0, "GPU-0" }, { 1, "GPU-1" } } } },
        };
        auto const logDir = TempDir::Random("mndiag-pair-cleanup-logs-");
        ScopedEnvVars const logDirEnv(DCGM_HOME_DIR_VAR_NAME, logDir.Path().string());

        auto const result = runner.RunBatch(params, nodeInfo, response, batchTimeout);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_GENERIC_ERROR);
        CHECK(response.numTests == 0);

        auto const logPaths = DcgmNs::Utils::GetLogFilePath(runner.GetLogFilePrefix());
        CHECK(ReadFileContents(logPaths.stdoutFileName).contains("stdout pair 1"));
        CHECK(ReadFileContents(logPaths.stderrFileName).contains("stderr pair 1"));
        CHECK_FALSE(std::filesystem::exists(fmt::format("{}.pair-1.stdout", logPaths.stdoutFileName.string())));
        CHECK_FALSE(std::filesystem::exists(fmt::format("{}.pair-1.stderr", logPaths.stderrFileName.string())));
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner command construction")
{
    SECTION("Builds default mpirun command with nvbandwidth plain-text output")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);
        runner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });

        auto diagStruct = CreateNvbandwidthDiagStruct();
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        REQUIRE_FALSE(cmdArgs.empty());
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--hostfile"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "6");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--map-by") == "ppr:3:node");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "/fake/test/mnnvbandwidth"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--json"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--NET_link_order"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--targetNumPairs") == "-1");

        auto mcaIt    = std::find(cmdArgs.begin(), cmdArgs.end(), "-mca");
        auto binaryIt = std::find(cmdArgs.begin(), cmdArgs.end(), "/fake/test/mnnvbandwidth");
        REQUIRE(mcaIt != cmdArgs.end());
        REQUIRE(binaryIt != cmdArgs.end());
        CHECK(mcaIt < binaryIt);

        auto testcaseIt = std::find(cmdArgs.begin(), cmdArgs.end(), "--testcase");
        REQUIRE(testcaseIt != cmdArgs.end());
        REQUIRE(std::distance(testcaseIt, cmdArgs.end()) >= 3);
        CHECK(*std::next(testcaseIt) == "multinode_device_to_device_memcpy_write_ce");
        CHECK(*std::next(testcaseIt, 2) == "multinode_device_to_device_memcpy_read_ce");
    }

    SECTION("Forwards only allowlisted nvbandwidth binary parameters")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.bufferSize=64",
                                                        "mnnvbandwidth.testcase=0",
                                                        "mnnvbandwidth.skipVerification",
                                                        "mnnvbandwidth.disableAffinity",
                                                        "mnnvbandwidth.useHugePages",
                                                        "mnnvbandwidth.useMean",
                                                        "mnnvbandwidth.testSamples=5",
                                                        "mnnvbandwidth.targetNumPairs=2",
                                                        "openmpi.detect_interfaces=0" });
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        REQUIRE_FALSE(cmdArgs.empty());
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--bufferSize") == "64");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testcase") == "0");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "multinode_device_to_device_memcpy_write_ce"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "multinode_device_to_device_memcpy_read_ce"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--skipVerification"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--disableAffinity"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--useHugePages"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--useMean"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testSamples") == "5");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--targetNumPairs") == "2");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--detect_interfaces"));
    }

    SECTION("Uses all pairs for an 8-GPU run")
    {
        ConfigurableGpuCountCoreProxy coreProxy(8);
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct();
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "16");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--targetNumPairs") == "-1");
    }

    SECTION("Applies pair-mode defaults without forwarding the DCGM-only parameter")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.gpuPairs=all" });
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--gpuPairs"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--bufferSize") == "1");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--testSamples"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--targetNumPairs"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testcase") == "multinode_device_to_device_memcpy_read_sm");
    }

    SECTION("Preserves explicit pair-mode testcase and buffer size but omits sweep controls")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.gpuPairs=all",
                                                        "mnnvbandwidth.testcase=multinode_custom_test",
                                                        "mnnvbandwidth.bufferSize=64",
                                                        "mnnvbandwidth.testSamples=7" });
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testcase") == "multinode_custom_test");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--bufferSize") == "64");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--testSamples"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--targetNumPairs"));
    }

    SECTION("Applies internal pair-mode defaults and preserves explicit parameters")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto const defaults = runner.BuildInternalPairModeParams();
        CHECK(defaults.at("testcase") == "multinode_device_to_device_memcpy_read_sm");
        CHECK(defaults.at("bufferSize") == "1");
        CHECK_FALSE(defaults.contains("testSamples"));
        CHECK_FALSE(defaults.contains("targetNumPairs"));
        CHECK_FALSE(defaults.contains("gpuPairs"));

        auto const explicitParams = runner.BuildInternalPairModeParams({ "mnnvbandwidth.testcase=multinode_custom_test",
                                                                         "mnnvbandwidth.bufferSize=64",
                                                                         "mnnvbandwidth.testSamples=7" });
        CHECK(explicitParams.at("testcase") == "multinode_custom_test");
        CHECK(explicitParams.at("bufferSize") == "64");
        CHECK_FALSE(explicitParams.contains("testSamples"));
        CHECK_FALSE(explicitParams.contains("targetNumPairs"));
    }

    SECTION("Uses the GPU count as the target pair count for runs larger than 16 GPUs")
    {
        ConfigurableGpuCountCoreProxy coreProxy(17);
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({}, 1);
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "17");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--targetNumPairs") == "17");
    }

    SECTION("Preserves an explicit target pair count for runs larger than 16 GPUs")
    {
        ConfigurableGpuCountCoreProxy coreProxy(17);
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.targetNumPairs=2" }, 1);
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "17");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--targetNumPairs") == "2");
    }

    SECTION("Expands whitespace-separated testcase values into separate arguments")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.testcase=0 1 host_to_device_memcpy_ce" });
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        auto option         = std::find(cmdArgs.begin(), cmdArgs.end(), "--testcase");
        REQUIRE(option != cmdArgs.end());
        REQUIRE(std::distance(option, cmdArgs.end()) >= 4);
        CHECK(*std::next(option) == "0");
        CHECK(*std::next(option, 2) == "1");
        CHECK(*std::next(option, 3) == "host_to_device_memcpy_ce");
    }

    SECTION("Expands whitespace-separated testcasePrefixes values into separate arguments")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.testcasePrefixes=host multinode" });
        runner.ConstructMpiCommand(&diagStruct);

        auto const &cmdArgs = runner.GetLastCommandArgs();
        auto option         = std::find(cmdArgs.begin(), cmdArgs.end(), "--testcasePrefixes");
        REQUIRE(option != cmdArgs.end());
        REQUIRE(std::distance(option, cmdArgs.end()) >= 3);
        CHECK(*std::next(option) == "host");
        CHECK(*std::next(option, 2) == "multinode");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--testcase"));
    }

    SECTION("Rejects testcase and testcasePrefixes specified together")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct
            = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.testcase=0", "mnnvbandwidth.testcasePrefixes=multinode" });
        runner.ConstructMpiCommand(&diagStruct);

        CHECK(runner.GetLastCommandArgs().empty());
    }

    SECTION("Accepts supported targetNumPairs boundary values")
    {
        auto targetNumPairs
            = GENERATE(std::string("-1"), std::string("0"), std::string("1"), std::string("9223372036854775807"));
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.targetNumPairs=" + targetNumPairs });
        runner.ConstructMpiCommand(&diagStruct);

        CHECK(CommandLineArgUtils::ArgAfter(runner.GetLastCommandArgs(), "--targetNumPairs") == targetNumPairs);
    }

    SECTION("Accepts upstream bufferSize boundary values")
    {
        auto bufferSize = GENERATE(std::string("0"), std::string("1"), std::string("18446744073709551615"));
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.bufferSize=" + bufferSize });
        runner.ConstructMpiCommand(&diagStruct);

        CHECK(CommandLineArgUtils::ArgAfter(runner.GetLastCommandArgs(), "--bufferSize") == bufferSize);
    }

    SECTION("Accepts time_to_run as a DCGM-only budget and omits it from nvbandwidth arguments")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvbandwidth.time_to_run=300" });
        runner.ConstructMpiCommand(&diagStruct);
        auto const runTime = runner.GetTestRunTime(diagStruct);

        REQUIRE(runTime.has_value());
        CHECK(*runTime == std::chrono::seconds(300));
        CHECK_FALSE(runner.GetLastCommandArgs().empty());
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(runner.GetLastCommandArgs(), "--time_to_run"));
    }

    SECTION("Invalid nvbandwidth parameters clear the command before launch")
    {
        auto invalidParam = GENERATE(std::string("mnnvbandwidth.bufferSize"),
                                     std::string("mnnvbandwidth.bufferSize=-1"),
                                     std::string("mnnvbandwidth.bufferSize=+1"),
                                     std::string("mnnvbandwidth.bufferSize=abc"),
                                     std::string("mnnvbandwidth.bufferSize=18446744073709551616"),
                                     std::string("mnnvbandwidth.testSamples=0"),
                                     std::string("mnnvbandwidth.testSamples=-1"),
                                     std::string("mnnvbandwidth.testSamples=+1"),
                                     std::string("mnnvbandwidth.testSamples=abc"),
                                     std::string("mnnvbandwidth.testSamples=2147483648"),
                                     std::string("mnnvbandwidth.targetNumPairs"),
                                     std::string("mnnvbandwidth.targetNumPairs=+1"),
                                     std::string("mnnvbandwidth.targetNumPairs=abc"),
                                     std::string("mnnvbandwidth.targetNumPairs=-2"),
                                     std::string("mnnvbandwidth.targetNumPairs=-9223372036854775808"),
                                     std::string("mnnvbandwidth.targetNumPairs=9223372036854775808"),
                                     std::string("mnnvbandwidth.gpuPairs"),
                                     std::string("mnnvbandwidth.gpuPairs=none"),
                                     std::string("mnnvbandwidth.gpuPairs=ALL"),
                                     std::string("mnnvbandwidth.gpuPairs=all "),
                                     std::string("mnnvbandwidth.testcase=0,1"),
                                     std::string("mnnvbandwidth.testcase=0;rm"),
                                     std::string("mnnvbandwidth.testcase=0 --json"),
                                     std::string("mnnvbandwidth.testcasePrefixes=host,multinode"),
                                     std::string("mnnvbandwidth.testcasePrefixes=host -h"),
                                     std::string("mnnvbandwidth.skipVerification=true"),
                                     std::string("mnnvbandwidth.useHugePages=true"),
                                     std::string("mnnvbandwidth.useMean=true"),
                                     std::string("mnnvbandwidth.unknown=1"),
                                     std::string("mnnvbandwidth.time_to_run"),
                                     std::string("mnnvbandwidth.time_to_run=0"),
                                     std::string("mnnvbandwidth.time_to_run=-1"),
                                     std::string("mnnvbandwidth.time_to_run=+1"),
                                     std::string("mnnvbandwidth.time_to_run=abc"),
                                     std::string("mnnvbandwidth.time_to_run=2147483648"));
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ invalidParam });
        runner.ConstructMpiCommand(&diagStruct);

        CHECK(runner.GetLastCommandArgs().empty());
    }

    SECTION("Parameters with an invalid prefix clear the command before launch")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        auto diagStruct = CreateNvbandwidthDiagStruct({ "mnnvabandwidth.testcase=0" });
        auto result     = runner.ConstructMpiCommand(&diagStruct);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_BADPARAM);
        CHECK(runner.GetLastCommandArgs().empty());
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner JSON response parsing")
{
    SECTION("Populates successful JSON response")
    {
        std::string output = R"json(
Open MPI binding noise
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)",
      "1: NVIDIA GB200 (00000009:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A", "10.0"], ["11.0", "N/A"]]
      }
    ]
    }
}
trailing text
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        nodeInfoMap_t nodeInfo {
            { "test-host-0",
              { .dcgmVersion = "4.6.0", .driverVersion = "580.00", .gpus = { { 0, "uuid-0" }, { 1, "uuid-1" } } } },
        };
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, nodeInfo);

        CHECK(response->version == dcgmMnDiagResponse_version2);
        REQUIRE(response->numTests == 1);
        CHECK(std::string(response->tests[0].name) == "MNNVBANDWIDTH");
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numHosts == 1);
        CHECK(response->numEntities == 2);
        CHECK(response->numResults == 2);
        CHECK(response->tests[0].numResults == 2);
        CHECK(response->numErrors == 0);
        CHECK(std::string(response->hosts[0].dcgmVersion) == "4.6.0");
    }

    SECTION("Finds nvbandwidth JSON after unrelated objects and unmatched launcher noise")
    {
        std::string output = R"json(
{"launcher":{"state":"ready"}}
warning { unmatched launcher diagnostic
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "warning": "literal braces { and } do not delimit the payload",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed"
      }
    ]
  }
}
trailing text
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        nodeInfoMap_t nodeInfo {
            { "test-host", { .dcgmVersion = "4.6.0", .driverVersion = "580.00" } },
        };

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, nodeInfo);

        REQUIRE(response->numTests == 1);
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numHosts == 1);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
        CHECK(std::string(response->hosts[0].hostname) == "test-host");
    }

    SECTION("Preserves DCGM node info when nvbandwidth reports a different driver version")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "999.0",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        nodeInfoMap_t nodeInfo {
            { "test-host-0", { .dcgmVersion = "4.6.0", .driverVersion = "580.00" } },
        };
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, nodeInfo);

        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].dcgmVersion) == "4.6.0");
        CHECK(std::string(response->hosts[0].driverVersion) == "580.00");
    }

    SECTION("Does not create entity results from JSON device list when no testcase results exist with top-level error")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "error": "setup failed before testcases ran",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)",
      "1: NVIDIA GB200 (00000009:06:00): (test-host-0)"
    ],
    "testcases": []
  }
}
)json";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(response->hosts[0].numEntities == 2);
        CHECK(response->numEntities == 2);
        CHECK(response->numResults == 0);
        CHECK(response->tests[0].numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == "setup failed before testcases ran");
    }

    SECTION("Preserves top-level JSON error when success schema is incomplete")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "error": "nvbandwidth failed before writing metadata"
  }
}
)json";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == "nvbandwidth failed before writing metadata");
    }

    SECTION("Fails JSON response with empty testcase results when no top-level error is present")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": []
  }
}
)json";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 0);
        CHECK(response->tests[0].numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == "mnnvbandwidth JSON output did not contain any testcase results");
    }

    SECTION("Fails JSON response with testcase results but no host information")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        CHECK(response->tests[0].numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == "mnnvbandwidth output did not contain any host information");
    }

    SECTION("Does not extract entities from malformed JSON output")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        nodeInfoMap_t nodeInfo {
            { "test-host-0", { .dcgmVersion = "4.6.0", .driverVersion = "580.00" } },
        };
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, nodeInfo);

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(std::string(response->errors[0].msg) == "Failed to parse mnnvbandwidth JSON output");
    }

    SECTION("Does not parse JSON from stderr")
    {
        std::string stderrOutput = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("\"nvbandwidth\"") != std::string::npos);
    }

    SECTION("Parses stderr text when stdout JSON is malformed")
    {
        std::string stdoutOutput = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json";
        std::string stderrOutput = "MPI_ABORT was invoked on rank 3 in communicator MPI_COMM_WORLD\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(stdoutOutput, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(std::string(response->errors[0].msg).find("MPI_ABORT") != std::string::npos);
    }

    SECTION("Populates response from GB200 sample output with MPI binding noise")
    {
        std::string output = R"sample(
[test-host-0:1845592] MCW rank 2 bound to socket 0[core 2[hwt 0]]: [././B/./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././.
/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[test-host-0:1845592] MCW rank 3 bound to socket 0[core 3[hwt 0]]: [./././B/./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././.
/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[test-host-0:1845592] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././.
/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
[test-host-0:1845592] MCW rank 1 bound to socket 0[core 1[hwt 0]]: [./B/./././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.][./././././././././.
/././././././././././././././././././././././././././././././././././././././././././././././././././././././././././././.]
{
        "nvbandwidth" :
        {
                "CUDA Runtime Version" : 13000,
                "Driver Version" : "580.169",
                "GPU Device list" :
                [
                        "0: NVIDIA GB200 (00000008:06:00): (test-host-0)",
                        "1: NVIDIA GB200 (00000009:06:00): (test-host-0)",
                        "2: NVIDIA GB200 (00000018:06:00): (test-host-0)",
                        "3: NVIDIA GB200 (00000019:06:00): (test-host-0)"
                ],
                "git_version" : "",
                "testcases" :
                [
                        {
                                "bandwidth_description" : "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)",
                                "bandwidth_matrix" :
                                [
                                        [
                                                "N/A",
                                                "3232.94",
                                                "3232.94",
                                                "3232.94"
                                        ],
                                        [
                                                "3231.73",
                                                "N/A",
                                                "3232.94",
                                                "3232.94"
                                        ],
                                        [
                                                "3231.73",
                                                "3232.94",
                                                "N/A",
                                                "3232.94"
                                        ],
                                        [
                                                "3234.16",
                                                "3234.16",
                                                "3234.16",
                                                "N/A"
                                        ]
                                ],
                                "name" : "multinode_device_to_device_memcpy_read_ce",
                                "status" : "Passed",
                                "sum" : 38796.557822041003
                        }
                ],
                "version" : "v0.8"
        }
}
)sample";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->version == dcgmMnDiagResponse_version2);
        REQUIRE(response->numTests == 1);
        CHECK(std::string(response->tests[0].name) == "MNNVBANDWIDTH");
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numErrors == 0);
        CHECK(response->tests[0].numErrors == 0);

        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(std::string(response->hosts[0].dcgmVersion) == "unknown");
        CHECK(std::string(response->hosts[0].driverVersion) == "See nvidia-smi");
        REQUIRE(response->hosts[0].numEntities == 4);

        REQUIRE(response->numEntities == 4);
        REQUIRE(response->numResults == 4);
        REQUIRE(response->tests[0].numResults == 4);

        for (unsigned int gpuId = 0; gpuId < 4; ++gpuId)
        {
            CHECK(response->hosts[0].entityIndices[gpuId] == gpuId);
            CHECK(response->tests[0].resultIndices[gpuId] == gpuId);

            CHECK(response->entities[gpuId].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->entities[gpuId].entity.entityId == gpuId);
            CHECK(response->entities[gpuId].hostId == 0);

            CHECK(response->results[gpuId].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->results[gpuId].entity.entityId == gpuId);
            CHECK(response->results[gpuId].hostId == 0);
            CHECK(response->results[gpuId].testId == 0);
            CHECK(response->results[gpuId].result == DCGM_DIAG_RESULT_PASS);
        }
    }

    SECTION("Records JSON testcase failures as global errors")
    {
        std::string output = R"({
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "testcases": [
      { "name": "all_to_all", "status": "ERROR" }
    ]
  }
})";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("all_to_all") != std::string::npos);
        CHECK(std::string(response->errors[0].msg).find("error") != std::string::npos);
    }

    SECTION("Marks JSON response skipped when all testcases are skipped")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      { "name": "waived_test_0", "status": "Waived" },
      { "name": "missing_test_0", "status": "Not Found" }
    ]
  }
}
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_SKIP);
        REQUIRE(response->numHosts == 1);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 0);
        CHECK(response->tests[0].numResults == 0);
        CHECK(response->numErrors == 0);
        CHECK(response->tests[0].numErrors == 0);
    }

    SECTION("Marks JSON response skipped when all testcases are skipped without a GPU device list")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "testcases": [
      { "name": "waived_test_0", "status": "Waived" },
      { "name": "missing_test_0", "status": "Not Found" }
    ]
  }
}
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_SKIP);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        CHECK(response->tests[0].numResults == 0);
        CHECK(response->numErrors == 0);
        CHECK(response->tests[0].numErrors == 0);
    }

    SECTION("Marks JSON response passed when passed testcases are mixed with skipped statuses")
    {
        std::string output = R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      { "name": "passed_test", "status": "Passed", "bandwidth_matrix": [["N/A"]] },
      { "name": "waived_test", "status": "Waived" },
      { "name": "missing_test", "status": "Not Found" }
    ]
  }
}
)json";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numHosts == 1);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->tests[0].numResults == 1);
        CHECK(response->results[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numErrors == 0);
        CHECK(response->tests[0].numErrors == 0);
    }

    SECTION("Caps JSON response hosts when device list exceeds response host capacity")
    {
        auto output = BuildJsonWithDeviceList(DCGM_MN_DIAG_RESPONSE_HOSTS_MAX + 1, true);

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numHosts == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->numEntities == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->numResults == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->tests[0].numResults == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->numErrors == 0);
    }

    SECTION("Caps JSON response entities when device list exceeds response entity capacity")
    {
        auto output = BuildJsonWithDeviceList(DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX + 1, false);

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numHosts == 1);
        CHECK(response->hosts[0].numEntities == DCGM_MN_DIAG_RESPONSE_ENTITIES_PER_HOST_MAX);
        CHECK(response->numEntities == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX);
        CHECK(response->numResults == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX);
        CHECK(response->tests[0].numResults == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX);
        CHECK(response->numErrors == 0);
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner plain-text response parsing")
{
    SECTION("Populates a passing multinode response and reuses node versions")
    {
        std::string output = R"(
nvbandwidth Version: v0.7
Process 0 (node-a): device 0: NVIDIA H100
Process 1 (node-a): device 1: NVIDIA H100
Process 2 (node-b): device 0: NVIDIA H100
Process 3 (node-b): device 1: NVIDIA H100
Running multinode_device_to_device_memcpy_read_ce.
SUM multinode_device_to_device_memcpy_read_ce 123.5
)";
        nodeInfoMap_t nodeInfo {
            { "node-a", { .dcgmVersion = "4.6.0", .driverVersion = "580.00" } },
            { "node-b", { .dcgmVersion = "4.6.1", .driverVersion = "580.01" } },
        };

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, nodeInfo);

        CHECK(response->version == dcgmMnDiagResponse_version2);
        REQUIRE(response->numTests == 1);
        CHECK(std::string(response->tests[0].name) == "MNNVBANDWIDTH");
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numHosts == 2);
        CHECK(std::string(response->hosts[0].dcgmVersion) == "4.6.0");
        CHECK(std::string(response->hosts[1].dcgmVersion) == "4.6.1");
        CHECK(response->numEntities == 4);
        CHECK(response->numResults == 4);
        CHECK(response->tests[0].numResults == 4);
        CHECK(response->numErrors == 0);
    }

    SECTION("Ignores matrix lines while parsing a completed multinode testcase")
    {
        std::string output = R"(
MPI version: Open MPI v4.1.4, package: Debian OpenMPI, ident: 4.1.4, May 26, 2022
CUDA Runtime Version: 12080
CUDA Driver Version: 12080
Driver Version: 570.124.06

Process 0 (nvbandwidth-test-worker-0): device 0: HGX GB200 (00000008:01:00)
Process 1 (nvbandwidth-test-worker-0): device 1: HGX GB200 (00000009:01:00)
Process 2 (nvbandwidth-test-worker-0): device 2: HGX GB200 (00000018:01:00)
Process 3 (nvbandwidth-test-worker-0): device 3: HGX GB200 (00000019:01:00)
Process 4 (nvbandwidth-test-worker-1): device 0: HGX GB200 (00000008:01:00)
Process 5 (nvbandwidth-test-worker-1): device 1: HGX GB200 (00000009:01:00)
Process 6 (nvbandwidth-test-worker-1): device 2: HGX GB200 (00000018:01:00)
Process 7 (nvbandwidth-test-worker-1): device 3: HGX GB200 (00000019:01:00)

Running multinode_device_to_device_memcpy_read_ce.

memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
    0      1      2      3      4      5      6      7
0  N/A   798.02 798.25 798.02 798.02 797.88 797.73 797.95
1  798.10 N/A   797.80 798.02 798.02 798.25 797.88 798.02
2  797.95 797.95 N/A   797.73 797.80 797.95 797.95 797.65
3  798.10 798.02 797.95 N/A   798.02 798.10 797.88 797.73
4  797.80 798.02 798.02 798.02 N/A   797.95 797.80 798.02
5  797.80 797.95 798.10 798.10 797.95 N/A   797.95 797.88
6  797.73 797.95 798.10 798.02 797.95 797.88 N/A   797.80
7  797.88 798.02 797.95 798.02 797.88 797.95 798.02 N/A

SUM multinode_device_to_device_memcpy_read_ce 44685.29
NOTE: The reported results may not reflect the full capabilities of the platform.
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numHosts == 2);
        CHECK(std::string(response->hosts[0].hostname) == "nvbandwidth-test-worker-0");
        CHECK(std::string(response->hosts[1].hostname) == "nvbandwidth-test-worker-1");
        CHECK(response->hosts[0].numEntities == 4);
        CHECK(response->hosts[1].numEntities == 4);
        REQUIRE(response->numEntities == 8);
        REQUIRE(response->numResults == 8);
        REQUIRE(response->tests[0].numResults == 8);
        CHECK(response->numErrors == 0);

        for (unsigned int index = 0; index < 8; ++index)
        {
            unsigned int const expectedHostId = index / 4;
            unsigned int const expectedGpuId  = index % 4;

            CHECK(response->entities[index].hostId == expectedHostId);
            CHECK(response->entities[index].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->entities[index].entity.entityId == expectedGpuId);
            CHECK(response->results[index].hostId == expectedHostId);
            CHECK(response->results[index].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->results[index].entity.entityId == expectedGpuId);
            CHECK(response->results[index].result == DCGM_DIAG_RESULT_PASS);
            CHECK(response->tests[0].resultIndices[index] == index);
        }
    }

    SECTION("Accepts multiple completed testcases and suffixed matrix sums")
    {
        std::string output = R"(
Process 0 (node-a): device 0: NVIDIA H100
Running device_to_device_bidirectional_memcpy_read_ce.
SUM device_to_device_bidirectional_memcpy_read_ce_read1 10
SUM device_to_device_bidirectional_memcpy_read_ce_read2 11
SUM device_to_device_bidirectional_memcpy_read_ce_total 21
Running host_to_device_memcpy_ce.
SUM host_to_device_memcpy_ce 12.5
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
    }

    SECTION("Treats completed and waived testcases as a pass")
    {
        std::string output = R"(
Process 0 (node-a): device 0: NVIDIA H100
Running host_to_device_memcpy_ce.
SUM host_to_device_memcpy_ce 12.5
Waived:
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
    }

    SECTION("Treats only waived testcases as skipped")
    {
        std::string output = R"(
Waived:
Waived:
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_SKIP);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        CHECK(response->numErrors == 0);
    }

    SECTION("Rejects incomplete, mismatched, orphan, and malformed SUM output")
    {
        std::string output
            = GENERATE("Process 0 (node-a): device 0: NVIDIA H100\nRunning host_to_device_memcpy_ce.\n",
                       "Process 0 (node-a): device 0: NVIDIA H100\nRunning host_to_device_memcpy_ce.\n"
                       "SUM device_to_host_memcpy_ce 1\n",
                       "Process 0 (node-a): device 0: NVIDIA H100\nSUM host_to_device_memcpy_ce 1\n",
                       "Process 0 (node-a): device 0: NVIDIA H100\nRunning host_to_device_memcpy_ce.\nSUM\n",
                       "Process 0 (node-a): device 0: NVIDIA H100\nRunning host_to_device_memcpy_ce.\n"
                       "SUM host_to_device_memcpy_ce not-a-number\n");

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numResults == 0);
        CHECK(response->numErrors >= 1);
    }

    SECTION("Rejects empty and unrelated output")
    {
        std::string output = GENERATE(std::string {}, std::string { "Open MPI startup noise only\n" });

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numResults == 0);
        CHECK(response->numErrors >= 1);
    }

    SECTION("Lets a later explicit failure override successful SUM blocks")
    {
        std::string output = R"(
Process 0 (node-a): device 0: NVIDIA H100
Running host_to_device_memcpy_ce.
SUM host_to_device_memcpy_ce 12.5
Running device_to_host_memcpy_ce.
SUM device_to_host_memcpy_ce 13.5
ASSERT in expression h_errorFlag == 0 on node-a, rank = 0 in compare() : memcpy.cpp:165
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numResults == 1);
        CHECK(response->results[0].hostId == 0);
        CHECK(response->results[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->results[0].entity.entityId == 0);
        CHECK(response->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->errors[0].entity.entityId == 0);
    }

    SECTION("Rejects an invalid testcase selection")
    {
        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(
            {}, "ERROR: Testcase missing_test not found!\n", *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(std::string(response->errors[0].msg).find("not found") != std::string::npos);
    }

    SECTION("Recognizes each upstream explicit failure form")
    {
        std::string output = GENERATE(
            std::string { "Error: testcase failed\n" },
            std::string { "[cudaErrorInvalidDevice] invalid device in expression cudaSetDevice(1)\n" },
            std::string { "[CUDA_ERROR_INVALID_DEVICE] invalid device in expression cuCtxSetCurrent(ctx)\n" },
            std::string { "NVML_ERROR: [Not Supported] in expression nvmlInit()\n" },
            std::string { "ASSERT in expression h_errorFlag == 0 in compare()\n" },
            std::string { "CUDA runtime sanity check failed: cudaMalloc on device 0 returned initialization error\n" },
            std::string { "You cannot specify both testcase and testcasePrefix options at the same time\n" },
            std::string { "Specified list of testcase prefixes did not match any testcases\n" });

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, output, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK_FALSE(std::string(response->errors[0].msg).empty());
    }

    SECTION("Caps passing hosts and entities at response capacity")
    {
        auto hostOutput   = BuildTextWithDeviceList(DCGM_MN_DIAG_RESPONSE_HOSTS_MAX + 1, true);
        auto hostResponse = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(hostOutput, {}, *hostResponse, {});

        CHECK(hostResponse->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(hostResponse->numHosts == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(hostResponse->numEntities == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(hostResponse->numResults == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);

        auto entityOutput   = BuildTextWithDeviceList(DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX + 1, false);
        auto entityResponse = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(entityOutput, {}, *entityResponse, {});

        CHECK(entityResponse->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(entityResponse->numEntities == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX);
        CHECK(entityResponse->numResults == DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX);
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner text failure parsing")
{
    SECTION("Preserves unrecognized stdout failure as a global error")
    {
        std::string stdoutOutput = "failed to initialize CUDA";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(stdoutOutput, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == stdoutOutput);
    }

    SECTION("Preserves unrecognized stderr failure as a global error")
    {
        std::string stderrOutput = "failed to initialize CUDA";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg) == stderrOutput);
    }

    SECTION("Maps CUDA rank failure to host GPU when process mapping is present")
    {
        std::string stdoutOutput = R"(
Process 0 (test-host-0): device 0: NVIDIA GB200 (00000008:06:00)
Process 1 (test-host-0): device 1: NVIDIA GB200 (00000009:06:00)
)";
        std::string stderrOutput = "[CUDA_ERROR_SYSTEM_NOT_READY] failure on test-host-0, rank = 1\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(stdoutOutput, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->errors[0].entity.entityId == 1);
        CHECK(response->errors[0].hostId == 0);
        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(response->hosts[0].numEntities == 2);
        CHECK(response->numEntities == 2);
        CHECK(response->numResults == 1);
        CHECK(response->tests[0].numResults == 1);
        CHECK(response->results[0].hostId == 0);
        CHECK(response->results[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->results[0].entity.entityId == 1);
        CHECK(response->results[0].result == DCGM_DIAG_RESULT_FAIL);
    }

    SECTION("Maps V0.8 CUDA rank failures across hosts when process mapping is present")
    {
        std::string output = R"(
[test-host-0:325263] MCW rank 3 bound to socket 0[core 3[hwt 0]]: [./././B]
[test-host-0:325263] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././.]
[test-host-0:325263] MCW rank 1 bound to socket 0[core 1[hwt 0]]: [./B/./.]
[test-host-0:325263] MCW rank 2 bound to socket 0[core 2[hwt 0]]: [././B/.]
[test-host-1:1847261] MCW rank 7 bound to socket 0[core 3[hwt 0]]: [./././B]
[test-host-1:1847261] MCW rank 4 bound to socket 0[core 0[hwt 0]]: [B/././.]
[test-host-1:1847261] MCW rank 5 bound to socket 0[core 1[hwt 0]]: [./B/./.]
[test-host-1:1847261] MCW rank 6 bound to socket 0[core 2[hwt 0]]: [././B/.]
nvbandwidth Version: v0.8
Built from Git version:

MPI version: Open MPI v4.1.6, package: Debian OpenMPI, ident: 4.1.6, repo rev: v4.1.6, Sep 30, 2023
CUDA Runtime Version: 13000
CUDA Driver Version: 13000
Driver Version: 580.169

Process 0 (test-host-0): device 0: NVIDIA GB200 (00000008:06:00)
Process 1 (test-host-0): device 1: NVIDIA GB200 (00000009:06:00)
Process 2 (test-host-0): device 2: NVIDIA GB200 (00000018:06:00)
Process 3 (test-host-0): device 3: NVIDIA GB200 (00000019:06:00)
Process 4 (test-host-1): device 0: NVIDIA GB200 (00000008:06:00)
Process 5 (test-host-1): device 1: NVIDIA GB200 (00000009:06:00)
Process 6 (test-host-1): device 2: NVIDIA GB200 (00000018:06:00)
Process 7 (test-host-1): device 3: NVIDIA GB200 (00000019:06:00)

Running multinode_device_to_device_memcpy_read_ce.
[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 ) on test-host-1, rank = 5 in MultinodeMemoryAllocationUnicast::MultinodeMemoryAllocationUnicast(size_t, int)() : /usr/local/src/nvbandwidth_src/multinode_memcpy.cpp:76

[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 ) on test-host-1, rank = 6 in MultinodeMemoryAllocationUnicast::MultinodeMemoryAllocationUnicast(size_t, int)() : /usr/local/src/nvbandwidth_src/multinode_memcpy.cpp:76

[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 ) on test-host-1, rank = 7 in MultinodeMemoryAllocationUnicast::MultinodeMemoryAllocationUnicast(size_t, int)() : /usr/local/src/nvbandwidth_src/multinode_memcpy.cpp:76

[CUDA_ERROR_ILLEGAL_STATE] the operation cannot be performed in the present state in expression cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 ) on test-host-1, rank = 4 in MultinodeMemoryAllocationUnicast::MultinodeMemoryAllocationUnicast(size_t, int)() : /usr/local/src/nvbandwidth_src/multinode_memcpy.cpp:76

--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 7 in communicator MPI_COMM_WORLD
with errorcode 1.

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
--------------------------------------------------------------------------
[test-host-0:325263] 3 more processes have sent help message help-mpi-api.txt / mpi-abort
[test-host-0:325263] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[test-host-0:325263] 3 more processes have sent help message help-mpi-btl-tcp.txt / peer hung up
)";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numHosts == 2);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(std::string(response->hosts[1].hostname) == "test-host-1");
        CHECK(response->numEntities == 8);
        CHECK(response->hosts[0].numEntities == 4);
        CHECK(response->hosts[1].numEntities == 4);
        CHECK(response->numResults == 4);
        CHECK(response->tests[0].numResults == 4);
        REQUIRE(response->numErrors == 5);

        for (unsigned int entityIndex = 0; entityIndex < 4; ++entityIndex)
        {
            CHECK(response->entities[entityIndex].hostId == 0);
            CHECK(response->entities[entityIndex].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->entities[entityIndex].entity.entityId == entityIndex);
            CHECK(response->entities[entityIndex + 4].hostId == 1);
            CHECK(response->entities[entityIndex + 4].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->entities[entityIndex + 4].entity.entityId == entityIndex);

            CHECK(response->results[entityIndex].hostId == 1);
            CHECK(response->results[entityIndex].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->results[entityIndex].entity.entityId == entityIndex);
            CHECK(response->results[entityIndex].result == DCGM_DIAG_RESULT_FAIL);
        }

        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->errors[0].entity.entityId == 0);
        CHECK(response->errors[0].hostId == 1);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_ILLEGAL_STATE") != std::string::npos);
        CHECK(std::string(response->errors[0].msg).find("rank = 4") != std::string::npos);

        for (unsigned int errorIndex = 1; errorIndex < 4; ++errorIndex)
        {
            CHECK(response->errors[errorIndex].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(response->errors[errorIndex].entity.entityId == errorIndex);
            CHECK(response->errors[errorIndex].hostId == 1);
            CHECK(std::string(response->errors[errorIndex].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
            CHECK(std::string(response->errors[errorIndex].msg).find("rank = " + std::to_string(errorIndex + 4))
                  != std::string::npos);
        }

        CHECK(response->errors[4].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[4].msg).find("MPI_ABORT") != std::string::npos);
        CHECK(std::string(response->errors[4].msg).find("NOTE") == std::string::npos);
    }

    SECTION("Records CUDA rank errors as global when process mapping is unavailable")
    {
        std::string stderrOutput
            = "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess(...) on "
              "test-host-0, rank = 3 in SomeFunction() : multinode_memcpy.cpp:76\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
        CHECK(std::string(response->errors[0].msg).find("rank = 3") != std::string::npos);
    }

    SECTION("Records CUDA rank errors as global when process rank overflows unsigned int")
    {
        std::string stderrOutput
            = "Process 4294967296 (test-host-0): device 0: NVIDIA GB200 (00000008:06:00)\n"
              "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess(...) on "
              "test-host-0, rank = 0 in SomeFunction() : multinode_memcpy.cpp:76\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
    }

    SECTION("Records CUDA rank errors as global when process GPU id overflows unsigned int")
    {
        std::string stderrOutput
            = "Process 0 (test-host-0): device 4294967296: NVIDIA GB200 (00000008:06:00)\n"
              "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess(...) on "
              "test-host-0, rank = 0 in SomeFunction() : multinode_memcpy.cpp:76\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
    }

    SECTION("Records CUDA rank errors as global when CUDA error rank overflows unsigned int")
    {
        std::string stderrOutput
            = "Process 0 (test-host-0): device 0: NVIDIA GB200 (00000008:06:00)\n"
              "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess(...) on "
              "test-host-0, rank = 4294967296 in SomeFunction() : multinode_memcpy.cpp:76\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numHosts == 1);
        CHECK(response->numEntities == 1);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
    }

    SECTION("Fails text fallback when only errored entity exceeds host capacity")
    {
        std::ostringstream stderrOutput;
        for (unsigned int hostIndex = 0; hostIndex < DCGM_MN_DIAG_RESPONSE_HOSTS_MAX; ++hostIndex)
        {
            stderrOutput << "Process " << hostIndex << " (host-" << hostIndex
                         << "): device 0: NVIDIA GB200 (00000008:06:00)\n";
        }

        unsigned int const overflowRank = DCGM_MN_DIAG_RESPONSE_HOSTS_MAX;
        stderrOutput << "Process " << overflowRank << " (overflow-host): device 0: NVIDIA GB200 (00000008:06:00)\n"
                     << "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal in expression cuMemSetAccess(...) on "
                     << "overflow-host, rank = " << overflowRank << " in SomeFunction() : multinode_memcpy.cpp:76\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput.str(), *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->numEntities == DCGM_MN_DIAG_RESPONSE_HOSTS_MAX);
        CHECK(response->numErrors == 0);
    }

    SECTION("Caps text fallback global errors at response capacity")
    {
        std::ostringstream stderrOutput;
        for (unsigned int index = 0; index < DCGM_MN_DIAG_RESPONSE_ERRORS_MAX + 1; ++index)
        {
            stderrOutput << "MPI_ABORT was invoked on rank " << index << " in communicator MPI_COMM_WORLD\n";
        }

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput.str(), *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(response->numHosts == 0);
        CHECK(response->numEntities == 0);
        CHECK(response->numResults == 0);
        REQUIRE(response->numErrors == DCGM_MN_DIAG_RESPONSE_ERRORS_MAX);
        CHECK(response->tests[0].numErrors == DCGM_MN_DIAG_RESPONSE_ERRORS_MAX);
        CHECK(response->tests[0].errorIndices[0] == 0);
        CHECK(response->tests[0].errorIndices[DCGM_MN_DIAG_RESPONSE_ERRORS_MAX - 1]
              == DCGM_MN_DIAG_RESPONSE_ERRORS_MAX - 1);
        CHECK(std::string(response->errors[DCGM_MN_DIAG_RESPONSE_ERRORS_MAX - 1].msg)
                  .find("rank " + std::to_string(DCGM_MN_DIAG_RESPONSE_ERRORS_MAX - 1))
              != std::string::npos);
    }

    SECTION("Records global MPI/rank errors when entity mapping is unavailable")
    {
        std::string stderrOutput = "MPI_ABORT was invoked on rank 3 in communicator MPI_COMM_WORLD\n";

        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(std::string(response->errors[0].msg).find("MPI_ABORT") != std::string::npos);
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner strips MPI timestamps")
{
    SECTION("Parses multiline JSON prefixed by OpenMPI 4")
    {
        auto output   = PrefixLines(BuildJsonWithDeviceList(2, false), "[2026-07-28 21:44:38.123] ");
        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numHosts == 1);
        CHECK(response->numEntities == 2);
        CHECK(response->numResults == 2);
        CHECK(response->numErrors == 0);
    }

    SECTION("Maps a rank error with interleaved OpenMPI 5 timestamps")
    {
        std::string output = "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (node-a): device "
                             "Mon Jul 20 09:42:51 2026<stdout>:0: NVIDIA H100\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:[CUDA_ERROR_ILLEGAL_ADDRESS] "
                             "failure on node-a, rank = 0\n";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->errors[0].entity.entityId == 0);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_ILLEGAL_ADDRESS") != std::string::npos);
    }

    SECTION("Parses bracketed OpenMPI 5 stdout timestamps")
    {
        auto output   = PrefixLines(BuildTextWithDeviceList(1, false), "[Thu Aug  6 21:22:35 2026]<stdout>:");
        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
    }

    SECTION("Parses bracketed OpenMPI 5 stderr timestamps")
    {
        auto output   = PrefixLines(BuildTextWithDeviceList(1, false), "[Thu Aug  6 21:22:35 2026]<stderr>:");
        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
    }

    SECTION("Recognizes an OpenMPI 4 timestamped stderr failure")
    {
        std::string stderrOutput
            = "[2026-07-28 21:44:39.456] MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n";
        auto response = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput({}, stderrOutput, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(std::string(response->errors[0].msg).find("MPI_ABORT") != std::string::npos);
    }

    SECTION("Parses Process line when timestamp splits rank from hostname parenthesis")
    {
        std::string output = "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (node-a): device 0: NVIDIA H100\n"
                             "Mon Jul 20 09:42:51 2026<stdout>:Process 1Mon Jul 20 09:42:51 2026<stdout>:"
                             " (node-a): device 1: NVIDIA H100\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:Running host_device_memcpy_ce.\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:SUM host_device_memcpy_ce 100.0\n";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numEntities == 2);
        REQUIRE(response->numResults == 2);
        CHECK(response->numErrors == 0);
    }

    SECTION("Parses Process line when timestamp splits device keyword from index")
    {
        std::string output = "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (node-a): device Mon Jul 20 09:42:51 "
                             "2026<stdout>:0: NVIDIA H100\n"
                             "Mon Jul 20 09:42:51 2026<stdout>:Process 1 (node-a): device 1: NVIDIA H100\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:Running host_device_memcpy_ce.\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:SUM host_device_memcpy_ce 100.0\n";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numEntities == 2);
        REQUIRE(response->numResults == 2);
        CHECK(response->numErrors == 0);
    }

    SECTION("Parses Process line when timestamp splits close-paren from colon")
    {
        std::string output = "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (node-a)Mon Jul 20 09:42:51 "
                             "2026<stdout>:: device 0: NVIDIA H100\n"
                             "Mon Jul 20 09:42:51 2026<stdout>:Process 1 (node-aMon Jul 20 09:42:51 "
                             "2026<stdout>:): device 1: NVIDIA H100\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:Running host_device_memcpy_ce.\n"
                             "Mon Jul 20 09:42:52 2026<stdout>:SUM host_device_memcpy_ce 100.0\n";
        auto response      = MakeUniqueZero<dcgmMnDiagResponse_v2>();

        DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(output, {}, *response, {});

        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numEntities == 2);
        REQUIRE(response->numResults == 2);
        CHECK(response->numErrors == 0);
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner PopulateResponse")
{
    SECTION("Rejects null response")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);

        CHECK(runner.PopulateResponse(nullptr, {}) == DCGM_ST_BADPARAM);
    }

    SECTION("Rejects unsupported response version")
    {
        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);
        auto response     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        response->version = dcgmMnDiagResponse_version2 + 1;

        CHECK(runner.PopulateResponse(response.get(), {}) == DCGM_ST_BADPARAM);
    }

    SECTION("Reads stdout and stderr files before populating response")
    {
        auto stdoutFile = TempFile::Random(R"json(
{
  "nvbandwidth": {
    "CUDA Runtime Version": 13000,
    "Driver Version": "580.00",
    "git_version": "abc123",
    "GPU Device list": [
      "0: NVIDIA GB200 (00000008:06:00): (test-host-0)"
    ],
    "testcases": [
      {
        "name": "device_to_device_memcpy_read_ce",
        "status": "Passed",
        "bandwidth_matrix": [["N/A"]]
      }
    ]
  }
}
)json",
                                           "mnnvbandwidth-stdout-");
        auto stderrFile = TempFile::Random("stderr text that should be ignored when stdout JSON is valid\n",
                                           "mnnvbandwidth-stderr-");

        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);
        runner.SetOutputFilePaths(stdoutFile.Path().string(), stderrFile.Path().string());
        auto response     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        response->version = dcgmMnDiagResponse_version2;

        CHECK(runner.PopulateResponse(response.get(), {}) == DCGM_ST_OK);
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        CHECK(response->numEntities == 1);
        CHECK(response->numResults == 1);
        CHECK(response->numErrors == 0);
    }

    SECTION("Attributes a stderr CUDA error using a stdout process mapping read from files")
    {
        auto stdoutFile = TempFile::Random("Process 3 (test-host-0): device 1: NVIDIA GB200 (00000009:06:00)\n",
                                           "mnnvbandwidth-stdout-");
        auto stderrFile = TempFile::Random(
            "[CUDA_ERROR_INVALID_DEVICE] invalid device ordinal on test-host-0, rank = 3\n", "mnnvbandwidth-stderr-");

        MockDcgmCoreProxy coreProxy;
        TestableNvbandwidthRunner runner(coreProxy);
        runner.SetOutputFilePaths(stdoutFile.Path().string(), stderrFile.Path().string());
        auto response     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
        response->version = dcgmMnDiagResponse_version2;

        CHECK(runner.PopulateResponse(response.get(), {}) == DCGM_ST_OK);
        CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numHosts == 1);
        CHECK(std::string(response->hosts[0].hostname) == "test-host-0");
        REQUIRE(response->numEntities == 1);
        CHECK(response->entities[0].hostId == 0);
        CHECK(response->entities[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->entities[0].entity.entityId == 1);
        CHECK(response->numResults == 1);
        CHECK(response->results[0].hostId == 0);
        CHECK(response->results[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->results[0].entity.entityId == 1);
        CHECK(response->results[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(response->numErrors == 1);
        CHECK(response->errors[0].hostId == 0);
        CHECK(response->errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(response->errors[0].entity.entityId == 1);
        CHECK(std::string(response->errors[0].msg).find("CUDA_ERROR_INVALID_DEVICE") != std::string::npos);
        CHECK(std::string(response->errors[0].msg).find("rank = 3") != std::string::npos);
    }
}

TEST_CASE("MnDiagMpiNvbandwidthRunner ResolveTestBinaryPath")
{
    SECTION("Uses default mnnvbandwidth path when no environment variable is set")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNNVBANDWIDTH_PATH.data(), std::nullopt);

        MockDcgmCoreProxy coreProxy;
        coreProxy.SetMockCudaVersion(12000);
        MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

        std::string path;
        auto result = runner.GetTestBinaryPath(path);

        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnnvbandwidth") != std::string::npos);
    }

    SECTION("Uses custom path from DCGM_MNDIAG_MNNVBANDWIDTH_PATH")
    {
        std::string customPath = "/bin/true";
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNNVBANDWIDTH_PATH.data(), customPath);

        MockDcgmCoreProxy coreProxy;
        coreProxy.SetMockCudaVersion(12000);
        MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

        std::string path;
        auto result = runner.GetTestBinaryPath(path);

        CHECK(result == DCGM_ST_OK);
        CHECK(path == customPath);
    }

    SECTION("Ignores mnubergemm environment variable")
    {
        ScopedEnvVars nvbandwidthEnv(MnDiagConstants::ENV_MNNVBANDWIDTH_PATH.data(), std::nullopt);
        ScopedEnvVars mnubergemmEnv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), "/bin/true");

        MockDcgmCoreProxy coreProxy;
        coreProxy.SetMockCudaVersion(12000);
        MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

        std::string path;
        auto result = runner.GetTestBinaryPath(path);

        CHECK(result == DCGM_ST_OK);
        CHECK(path != "/bin/true");
        CHECK(path.find("mnnvbandwidth") != std::string::npos);
    }

    SECTION("Rejects oversized custom path before launch")
    {
        std::string longPath(DCGM_MAX_STR_LENGTH, 'x');
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNNVBANDWIDTH_PATH.data(), longPath);

        MockDcgmCoreProxy coreProxy;
        coreProxy.SetMockCudaVersion(12000);
        MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

        std::string path;
        auto result = runner.GetTestBinaryPath(path);

        CHECK(result == DCGM_ST_BADPARAM);
        CHECK(path.empty());
    }

    SECTION("Unsupported CUDA version fails default path resolution")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNNVBANDWIDTH_PATH.data(), std::nullopt);

        MockDcgmCoreProxy coreProxy;
        coreProxy.SetMockCudaVersion(0);
        MnDiagMpiNvbandwidthRunner runner(coreProxy, MNNVBANDWIDTH_EFFECTIVE_UID);

        std::string path;
        auto result = runner.GetTestBinaryPath(path);

        CHECK(result == DCGM_ST_NO_DATA);
        CHECK(path.empty());
    }
}

TEST_CASE("MnDiagMpiRunnerFactory creates runners for supported mndiag tests")
{
    MockDcgmCoreProxy coreProxy;
    MnDiagMpiRunnerFactory factory;

    SECTION("Creates mnubergemm runner")
    {
        auto runner
            = factory.CreateMpiRunner(coreProxy, dcgmMultinodeTestType_t::mnubergemm, MNNVBANDWIDTH_EFFECTIVE_UID);

        REQUIRE(runner != nullptr);
        CHECK(runner->GetTestPrefix() == "mnubergemm.");
    }

    SECTION("Creates mnnvbandwidth runner")
    {
        auto runner
            = factory.CreateMpiRunner(coreProxy, dcgmMultinodeTestType_t::mnnvbandwidth, MNNVBANDWIDTH_EFFECTIVE_UID);

        REQUIRE(runner != nullptr);
        CHECK(runner->GetTestPrefix() == "mnnvbandwidth.");
    }

    SECTION("Rejects unknown test type")
    {
        auto runner
            = factory.CreateMpiRunner(coreProxy, static_cast<dcgmMultinodeTestType_t>(77), MNNVBANDWIDTH_EFFECTIVE_UID);

        CHECK(runner == nullptr);
    }
}
