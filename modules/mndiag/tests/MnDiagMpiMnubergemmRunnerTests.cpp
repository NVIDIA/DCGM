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

#include <CommandLineArgUtils.hpp>
#include <DcgmStringHelpers.h>
#include <MnDiagMpiMnubergemmRunner.h>
#include <MnDiagMpiRunner.h>
#include <MnDiagProcessUtils.h>
#include <ScopedEnvVars.hpp>
#include <ScopedFd.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <dcgm_errors.h>
#include <dcgm_structs.h>
#include <fmt/format.h>
#include <fstream>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>

#include "MockDcgmCoreProxy.h"

// Force a UID of 0, corresponding to the root user. This is so that there will
// always be a username associated with the UID
static constexpr uid_t MNUBERGEMM_EFFECTIVE_UID = 0;

// Forward-declare the internal helpers so tests can call them directly
// without a separate header file.
namespace MnDiagMpiRunnerInternal
{
std::string ParseRoutingInterface(std::string const &ipRouteOutput);
std::string ResolveHostnameToIp(std::string const &host);
std::string FindTrustedIpCommand();
std::string GetRoutingInterfacesForHosts(std::vector<std::string> const &hostList,
                                         DcgmNs::Common::ProcessUtils::CommandExecutor *executor = nullptr,
                                         std::string const &ipCommandPath                        = {});

bool IsInterfaceDetectionEnabled(dcgmRunMnDiag_v2 const &drmnd);
void LogMpiInterfaceEnvVars();
std::vector<std::string> ResolveMpiInterfaceArgs(dcgmRunMnDiag_v2 const &drmnd,
                                                 std::vector<std::string> const &hostList,
                                                 MnDiagMpiRunner::RoutingInterfaceResolver const &interfaceResolver);
} // namespace MnDiagMpiRunnerInternal

namespace
{

/**
 * @brief Mock CommandExecutor that returns pre-configured responses for
 *        `ip route get <host>` commands, used to unit test routing interface
 *        detection without requiring actual network access.
 */
class MnubergemmMockCommandExecutor : public DcgmNs::Common::ProcessUtils::CommandExecutor
{
public:
    /**
     * Register a canned response for a specific command string.
     * Any command not registered returns an empty string.
     */
    void SetResponse(std::string const &cmd, std::string response)
    {
        m_responses[cmd] = std::move(response);
    }

    std::string ExecuteCommand(std::string const &cmd) override
    {
        auto it = m_responses.find(cmd);
        if (it != m_responses.end())
        {
            return it->second;
        }
        return {};
    }

private:
    std::unordered_map<std::string, std::string> m_responses;
};

/**
 * @brief Helper function to create a memory-backed file descriptor from a string.
 * @param data The string to write to the file descriptor.
 * @return The file descriptor.
 */
ScopedFd CreateMemFdFromString(std::string const &data)
{
    ScopedFd fd(memfd_create("test_data", 0));
    REQUIRE(write(fd.Get(), data.data(), data.size()) == static_cast<ssize_t>(data.size()));
    REQUIRE(lseek(fd.Get(), 0, SEEK_SET) != -1);
    return fd;
}

/**
 * Helper function to create a populated dcgmRunMnDiag_v2 struct for testing
 */
dcgmRunMnDiag_v2 CreateTestDiagStruct()
{
    dcgmRunMnDiag_v2 diagStruct = {};

    // Set version
    diagStruct.version = dcgmRunMnDiag_version2;

    // Add some test hosts
    SafeCopyTo(diagStruct.hostList[0], "host1");
    SafeCopyTo(diagStruct.hostList[1], "host2");
    SafeCopyTo(diagStruct.hostList[2], "host3");

    // Add some test parameters
    SafeCopyTo(diagStruct.testParms[0], "mnubergemm.workload=GN");
    SafeCopyTo(diagStruct.testParms[1], "mnubergemm.time_to_run=300");
    SafeCopyTo(diagStruct.testParms[2], "mnubergemm.dynamic_adj");

    return diagStruct;
}

/**
 * Find the value of a specific -mca key in a command argument list.
 * Returns the value string, or empty if the key is not found.
 */
std::string FindMcaValue(std::vector<std::string> const &cmdArgs, std::string const &key)
{
    for (size_t i = 0; i + 2 < cmdArgs.size(); i++)
    {
        if (cmdArgs[i] == "-mca" && cmdArgs[i + 1] == key)
        {
            return cmdArgs[i + 2];
        }
    }
    return {};
}

/**
 * Helper function to create a populated dcgmMnDiagResponse_v2 struct for testing
 */
std::unique_ptr<dcgmMnDiagResponse_v2> CreateTestResponseStruct()
{
    auto responseUptr     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    responseUptr->version = dcgmMnDiagResponse_version2;
    return responseUptr;
}

/**
 * Fake executor for version-check tests.
 * Records every command string received so tests can assert on what was run.
 * Dispatches responses by checking whether the command contains "ompi_info".
 */
class FakeMpiVersionExecutor : public DcgmNs::Common::ProcessUtils::CommandExecutor
{
public:
    std::string mpirunOutput;
    std::string ompiInfoOutput;
    bool shouldThrow = false;
    std::vector<std::string> receivedCmds;

    std::string ExecuteCommand(std::string const &cmd) override
    {
        receivedCmds.push_back(cmd);
        if (shouldThrow)
        {
            throw std::runtime_error("Simulated executor failure");
        }
        if (cmd.find("ompi_info") != std::string::npos)
        {
            return ompiInfoOutput;
        }
        return mpirunOutput;
    }
};

// Test string with MNUB output containing both success and error messages
const std::string test_string = R"(
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  INFO  hosthash=13067628369524022839 B0I=632386609
MNUB [I] orig resources type 1 152 sms
MNUB [I] split for G groups 1 152 -> 120 32
[*** LOG ERROR #0001 ***] [2025-05-07 13:06:30] [mnubergemm_logger] string pointer is null
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  INFO  hosthash=8605271815604617176 B0I=632386609
MNUB [I] orig resources type 1 152 sms
MNUB [I] split for G groups 1 152 -> 120 32
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  OPENMPI/NCCL GEMM PULSE
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  MPI_VERSION=3 MPI_SUBVERSION=1 MPI_Get_version=3 MPI_Get_subversion=1
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  NCCL_VERSION_CODE=22501 ncclGetVersion=22602
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  INFO  hosthash=13067628369524022839 B0I=632386609
MNUB [I] orig resources type 1 152 sms
MNUB [I] split for G groups 1 152 -> 120 32
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  INFO  hosthash=8605271815604617176 B0I=632386609
MNUB [I] orig resources type 1 152 sms
MNUB [I] split for G groups 1 152 -> 120 32
MNUB [I] split for N groups 1 32 -> 32 0
MNUB [I] split for N groups 1 32 -> 32 0
MNUB [I] split for N groups 1 32 -> 32 0
MNUB [I] split for N groups 1 32 -> 32 0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 2 -> 3 -> 1 -> 0 -> ...
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=2 freq:20.1Hz duty:0.0242 P:49.8 S1:0.00218 G:1.21 S2:0.00266 I:48.6 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=3 freq:15.1Hz duty:0.0239 P:66.4 S1:0.0029 G:1.59 S2:0.00306 I:64.8 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=4 freq:13.9Hz duty:0.0238 P:71.9 S1:0.00313 G:1.71 S2:0.00321 I:70.2 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=6 freq:13.6Hz duty:0.0238 P:73.8 S1:0.00322 G:1.76 S2:0.00326 I:72 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=8 freq:13.4Hz duty:0.0238 P:74.4 S1:0.00325 G:1.77 S2:0.00327 I:72.6 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=12 freq:13.4Hz duty:0.0238 P:74.6 S1:0.00325 G:1.78 S2:0.00328 I:72.8 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=16 freq:13.4Hz duty:0.0238 P:74.7 S1:0.00326 G:1.78 S2:0.00328 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=24 freq:13.4Hz duty:0.0238 P:74.7 S1:0.00326 G:1.78 S2:0.00327 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=32 freq:13.4Hz duty:0.0238 P:74.7 S1:0.00325 G:1.78 S2:0.00328 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=48 freq:13.4Hz duty:0.0238 P:74.7 S1:0.00326 G:1.78 S2:0.00327 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=64 freq:13.4Hz duty:0.0238 P:74.7 S1:0.00326 G:1.78 S2:0.00327 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=96 freq:13.4Hz duty:0.0239 P:74.7 S1:0.00324 G:1.78 S2:0.00328 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=192 freq:13.4Hz duty:0.0253 P:74.8 S1:0.00327 G:1.89 S2:0.00327 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=768 freq:13.2Hz duty:0.0343 P:75.5 S1:0.00325 G:2.59 S2:0.00327 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=1024 freq:13.2Hz duty:0.04 P:76 S1:0.00325 G:3.04 S2:0.00329 I:72.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=1536 freq:12.9Hz duty:0.0499 P:77.3 S1:0.00325 G:3.86 S2:0.00328 I:73.5 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=2048 freq:12.8Hz duty:0.0613 P:78.4 S1:0.00326 G:4.81 S2:0.00328 I:73.6 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=3072 freq:12.5Hz duty:0.0794 P:80.3 S1:0.00324 G:6.37 S2:0.00327 I:73.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=4096 freq:12.2Hz duty:0.1 P:82.1 S1:0.00326 G:8.24 S2:0.00328 I:73.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=6144 freq:11.8Hz duty:0.135 P:84.7 S1:0.00326 G:11.5 S2:0.00326 I:73.2 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=8192 freq:11.4Hz duty:0.171 P:88.1 S1:0.00325 G:15 S2:0.00327 I:73 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=12288 freq:10.4Hz duty:0.223 P:96.6 S1:0.00326 G:21.5 S2:0.00331 I:75.1 GFlops:0
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:G estimated L value 14935
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:G estimated L value 14935
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:G estimated L value 13932
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=16384 freq:8.97Hz duty:0.265 P:112 S1:0.00326 G:29.6 S2:0.00329 I:81.9 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G estimated L value 14057
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:G estimated IDLE CYCLES value 32962350
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:G skipping fine tuning
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:G FINAL values L:14935 IDLE:32962350
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:G estimated IDLE CYCLES value 32816076
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:G skipping fine tuning
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:G FINAL values L:14935 IDLE:32816076
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:G estimated IDLE CYCLES value 33554440
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:G skipping fine tuning
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:G FINAL values L:13932 IDLE:33554440
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=14057 freq:9.32Hz duty:0.269 P:107 S1:0.00351 G:28.9 S2:0.00326 I:78.4 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G estimated IDLE CYCLES value 31878952
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G skipping fine tuning
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G FINAL values L:14057 IDLE:31878952
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=2 freq:57.1Hz duty:0.114 P:17.5 S1:0.00218 G:2 S2:0.00219 I:15.5 GB/s:128.16
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=4 freq:41.7Hz duty:0.137 P:24 S1:0.0029 G:3.29 S2:0.00292 I:20.7 GB/s:155.4
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=4 freq:38.2Hz duty:0.143 P:26.2 S1:0.00314 G:3.74 S2:0.00317 I:22.4 GB/s:136.76
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=6 freq:36Hz duty:0.172 P:27.8 S1:0.00339 G:4.79 S2:0.00323 I:23 GB/s:160.41
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=8 freq:34.2Hz duty:0.208 P:29.3 S1:0.00329 G:6.08 S2:0.00323 I:23.2 GB/s:168.37
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=12 freq:31.6Hz duty:0.266 P:31.6 S1:0.00326 G:8.4 S2:0.00326 I:23.2 GB/s:182.92
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=16 freq:28.7Hz duty:0.333 P:34.8 S1:0.00326 G:11.6 S2:0.00326 I:23.2 GB/s:176.81
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=24 freq:25.5Hz duty:0.407 P:39.2 S1:0.00324 G:16 S2:0.00325 I:23.2 GB/s:192.12
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=32 freq:22.4Hz duty:0.479 P:44.6 S1:0.00322 G:21.4 S2:0.00324 I:23.2 GB/s:191.58
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=48 freq:18.4Hz duty:0.572 P:54.4 S1:0.00322 G:31.1 S2:0.00322 I:23.3 GB/s:197.35
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=64 freq:15.5Hz duty:0.64 P:64.6 S1:0.00321 G:41.4 S2:0.00325 I:23.2 GB/s:197.95
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=96 freq:11.9Hz duty:0.722 P:83.7 S1:0.00326 G:60.5 S2:0.00327 I:23.2 GB/s:203.26
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=128 freq:9.42Hz duty:0.781 P:106 S1:0.00322 G:82.9 S2:0.0033 I:23.2 GB/s:197.67
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=192 freq:6.83Hz duty:0.841 P:146 S1:0.0032 G:123 S2:0.0033 I:23.2 GB/s:199.59
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=256 freq:5.16Hz duty:0.88 P:194 S1:0.00319 G:170 S2:0.00331 I:23.2 GB/s:192.28
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=384 freq:3.58Hz duty:0.917 P:279 S1:0.00317 G:256 S2:0.00331 I:23.2 GB/s:191.91
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=512 freq:2.64Hz duty:0.939 P:378 S1:0.00318 G:355 S2:0.00329 I:23.2 GB/s:184.62
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N estimated L value 37
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:N estimated L value 38
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:N estimated L value 39
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:N estimated L value 39
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=38 freq:6.23Hz duty:0.855 P:161 S1:0.00319 G:137 S2:0.00326 I:23.2 GB/s:35.403
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N estimated IDLE CYCLES value 34300960
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N skipping fine tuning
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N FINAL values L:38 IDLE:34300960
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  done tuning loop
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:N estimated IDLE CYCLES value 34312696
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:N skipping fine tuning
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:N FINAL values L:38 IDLE:34312696
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  done tuning loop
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:N estimated IDLE CYCLES value 34349864
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:N skipping fine tuning
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:N FINAL values L:38 IDLE:34349864
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  done tuning loop
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:N estimated IDLE CYCLES value 34347776
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:N skipping fine tuning
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:N FINAL values L:38 IDLE:34347776
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  done tuning loop
MNUB [I] done initial sync
MNUB [I] starting....
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=14057 freq:19.1Hz duty:0.425 P:52.3 S1:2.44 G:22.2 S2:5.34 I:22.3 GFlops:0
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=38 freq:18.2Hz duty:0.583 P:55.1 S1:2.65 G:32.1 S2:0.994 I:19.3 GB/s:151.58
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 2 -> 0 -> 3 -> 1 -> ...
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=14496 freq:20Hz duty:0.46 P:49.9 S1:2.41 G:23 S2:3.28 I:21.2 GFlops:2.7304e+06
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=36 freq:20Hz duty:0.51 P:50.1 S1:2.59 G:25.5 S2:0.84 I:21.1 GB/s:180.5
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 3 -> 0 -> 1 -> 2 -> ...
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=14814 freq:15.9Hz duty:0.367 P:63 S1:3.28 G:23.1 S2:14.4 I:22.1 GFlops:2.7304e+06
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=36 freq:15.9Hz duty:0.579 P:63 S1:3.4 G:36.4 S2:1 I:22.1 GB/s:126.48
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 2 -> 3 -> 1 -> 0 -> ...
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=15114 freq:18.1Hz duty:0.439 P:55.1 S1:2.67 G:24.2 S2:2.13 I:26.1 GFlops:2.7572e+06
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=34 freq:18.1Hz duty:0.463 P:55.1 S1:2.76 G:25.5 S2:0.745 I:26.1 GB/s:170.47
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 2 -> 3 -> 1 -> 0 -> ...
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:G L=15235 freq:18.3Hz duty:0.44 P:54.8 S1:4.81 G:24.1 S2:2.16 I:23.7 GFlops:2.6982e+06
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N L=34 freq:18.2Hz duty:0.454 P:54.8 S1:5.14 G:24.9 S2:1.08 I:23.7 GB/s:175.04
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N New Link order ... -> 3 -> 1 -> 0 -> 2 -> ...
MNUB [I] Ending main loop Time 5.7174077 >= 5
MNUB [I] Ending main loop Time 5.717587 >= 5
MNUB [I] Ending main loop Time 5.7175894 >= 5
MNUB [I] Ending main loop Time 5.7177653 >= 5
MNUB [I] Ending main loop Time 5.729109 >= 5
MNUB [I] Ending main loop Time 5.7292085 >= 5
MNUB [I] Ending main loop Time 5.716496 >= 5
MNUB [I] Ending main loop Time 5.716742 >= 5
MNUB [I] 1 is exiting
MNUB [I] GLOBAL PERF G : 2265061.5 GFlops stddev 1138556.6
MNUB [I] GLOBAL PERF N : 160.54855 GB/s stddev 19.520765
MNUB [I] RANK PERF G : RANK 0 : 2183234.5 GFlops stddev 1091777.5
MNUB [I] RANK PERF G : RANK 1 : 2356335.5 GFlops stddev 1179427.5
MNUB [I] RANK PERF G : RANK 2 : 2156287 GFlops stddev 1079421
MNUB [I] RANK PERF G : RANK 3 : 2364389.5 GFlops stddev 1183360.9
MNUB [I] RANK PERF N : RANK 0 : 160.81259 GB/s stddev 19.73412
MNUB [I] RANK PERF N : RANK 1 : 160.74977 GB/s stddev 19.816277
MNUB [I] RANK PERF N : RANK 2 : 160.20183 GB/s stddev 19.113583
MNUB [I] RANK PERF N : RANK 3 : 160.43005 GB/s stddev 19.404829
MNUB [I] 0 is exiting
MNUB [I] 3 is exiting
MNUB [I] 2 is exiting
MNUB [I] G: 3 b07-p1-dgx-07-c13 L: 1  T:N FINISHED
MNUB [I] G: 1 b07-p1-dgx-07-c12 L: 1  T:N FINISHED
MNUB [I] G: 2 b07-p1-dgx-07-c13 L: 0  T:N FINISHED
MNUB [I] G: 0 b07-p1-dgx-07-c12 L: 0  T:N FINISHED
MNUB [I] G: 4 b07-p1-dgx-07-c17 L: 0  INFO  hosthash=8605271815604617176 B0I=632386609
MNUB [I] G: 5 b07-p1-dgx-07-c17 L: 1  INFO  hosthash=8605271815604617176 B0I=632386610
MNUB [I] G: 6 b07-p1-dgx-07-c18 L: 0  INFO  hosthash=8605271815604617177 B0I=632386611
MNUB [I] G: 7 b07-p1-dgx-07-c18 L: 1  INFO  hosthash=8605271815604617177 B0I=632386612
MNUB [E] G: 4 b07-p1-dgx-07-c17 L: 0  T:N CUDA Error : cudaStreamQuery returned an illegal memory access was encountered
MNUB [E] G: 4 b07-p1-dgx-07-c17 L: 0  T:N cuBLAS Error : init failed
MNUB [E] G: 6 b07-p1-dgx-07-c18 L: 0  T:N NCCL Error : not initialized
MNUB [E] G: 7 b07-p1-dgx-07-c18 L: 1  T:N IMEX Error : status is DOWN
)";
} //namespace

class MnDiagMpiMnubergemmRunnerTests
{
public:
    explicit MnDiagMpiMnubergemmRunnerTests(uid_t uid = MNUBERGEMM_EFFECTIVE_UID)
        : m_mockCoreProxy(std::make_unique<MockDcgmCoreProxy>())
        , m_runner(*m_mockCoreProxy, uid)
    {
        // Set a default test binary path so command construction works without a real binary
        m_runner.m_testBinaryPath.emplace("/fake/test/mnubergemm");
    }

    // Expose methods for testing
    dcgmReturn_t MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
    {
        return m_runner.MnDiagOutputCallback(fd, responseStruct, nodeInfo);
    }

    std::expected<void, dcgmReturn_t> ConstructMpiCommand(void const *params)
    {
        return m_runner.ConstructMpiCommand(params);
    }

    void ParseDcgmMnDiagToMpiCommand(dcgmRunMnDiag_v2 const &drmnd)
    {
        m_runner.ParseDcgmMnDiagToMpiCommand(drmnd);
    }

    std::string GetMpiBinPath() const
    {
        return m_runner.GetMpiBinPath();
    }

    std::string GetLastCommand() const
    {
        return m_runner.GetLastCommand();
    }

    std::vector<std::string> GetLastCommandArgs() const
    {
        return m_runner.m_lastCommand;
    }

    void SetTestBinaryPath(std::string const &path)
    {
        m_runner.m_testBinaryPath.emplace(path);
    }

    bool CheckMpiVersionConsistency(std::string const &mpirunPath,
                                    DcgmNs::Common::ProcessUtils::CommandExecutor *executor = nullptr) const
    {
        return m_runner.CheckMpiVersionConsistency(mpirunPath, executor);
    }


    void SetRoutingInterfaceResolver(MnDiagMpiRunner::RoutingInterfaceResolver resolver)
    {
        m_runner.SetRoutingInterfaceResolver(std::move(resolver));
    }

    void SetMockCudaVersion(int version)
    {
        m_mockCoreProxy->SetMockCudaVersion(version);
    }

    int GetMockCudaVersion() const
    {
        int cudaVersion = 0;
        m_mockCoreProxy->GetCudaVersion(cudaVersion);
        return cudaVersion;
    }

private:
    std::unique_ptr<MockDcgmCoreProxy> m_mockCoreProxy;
    MnDiagMpiMnubergemmRunner m_runner;
};

TEST_CASE("MnDiagMpiRunner command construction")
{
    SECTION("ConstructMpiCommand with valid parameters")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set environment variable for mpirun path
        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        // Construct the command
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify the command is not empty
        REQUIRE_FALSE(cmdArgs.empty());

        // Verify expected parts are present in the command arguments
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--hostfile"));
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "9");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--map-by") == "ppr:3:node");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--time_to_run") == "300");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--workload") == "GN");
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--dynamic_adj"));
        // With 9 processes (odd), NET_link_order should be set to snake
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--NET_link_order") == "snake");
    }

    SECTION("ConstructMpiCommand with null parameters")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        // This should not crash
        mpiRunner.ConstructMpiCommand(nullptr);

        // Verify the command is empty
        auto const &command = mpiRunner.GetLastCommand();
        REQUIRE(command.empty());
    }

    SECTION("ConstructMpiCommand with invalid version")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set an invalid version
        diagStruct.version = 9999;

        // This should not crash
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Verify the command is empty
        auto const &command = mpiRunner.GetLastCommand();
        REQUIRE(command.empty());
    }

    SECTION("ConstructMpiCommand with an invalid parameter prefix produces no command")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();
        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=300");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");
        auto result = mpiRunner.ConstructMpiCommand(&diagStruct);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_BADPARAM);
        CHECK(mpiRunner.GetLastCommand().empty());
    }
}

TEST_CASE("MnDiagMpiRunner binary paths")
{
    std::string customMnubergemmPath = "/custom/path/to/mnubergemm";
    std::string customMpirunPath     = "/custom/path/to/mpirun";

    SECTION("GetMpiBinPath with custom environment")
    {
        // Set environment variable
        ScopedEnvVars envVars(MnDiagConstants::ENV_MPIRUN_PATH.data(), customMpirunPath);

        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto mpiPath = mpiRunner.GetMpiBinPath();

        REQUIRE(mpiPath == customMpirunPath);
    }

    SECTION("GetMpiBinPath with default environment")
    {
        // Ensure the environment variable is not set
        ScopedEnvVars envVars(MnDiagConstants::ENV_MPIRUN_PATH.data(), std::nullopt);

        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto mpiPath = mpiRunner.GetMpiBinPath();

        REQUIRE(mpiPath == MnDiagConstants::DEFAULT_MPIRUN_PATH);
    }

    SECTION("Custom mnubergemm path affects command construction")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), std::nullopt);

        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set the binary path directly via test harness
        mpiRunner.SetTestBinaryPath(customMnubergemmPath);

        // Construct the command
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Get the command as a string
        std::string commandStr = mpiRunner.GetLastCommand();

        // The custom mnubergemm path should be in the command
        REQUIRE(commandStr.find(customMnubergemmPath) != std::string::npos);
    }
}

TEST_CASE("MnDiagMpiRunner output parsing")
{
    SECTION("MnDiagOutputCallback with null response struct")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        // This should not crash
        auto dataFd = CreateMemFdFromString("test output");
        mpiRunner.MnDiagOutputCallback(dataFd.Get(), nullptr, nodeInfoMap_t());
    }

    SECTION("MnDiagOutputCallback with stdout output")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Call the callback with sample output
        auto dataFd = CreateMemFdFromString("Sample mnubergemm output");
        mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfoMap_t());

        // Version should be maintained
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);
    }

    SECTION("MnDiagOutputCallback with invalid version")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Set an invalid version
        responseStruct->version = 9999;

        // This should not crash
        auto dataFd = CreateMemFdFromString("test output");
        mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfoMap_t());

        // Version should be maintained
        REQUIRE(responseStruct->version == 9999);
    }
}

TEST_CASE("MnDiagMpiRunner ParseDcgmMnDiagToMpiCommand")
{
    SECTION("Parse dcgmRunMnDiag_v2 with standard parameters")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set mpirun path environment variable
        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        // Call the parse function directly
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify the hostfile argument is present and points to a readable file
        {
            std::ifstream hf(CommandLineArgUtils::ArgAfter(cmdArgs, "--hostfile"));
            REQUIRE(hf.is_open());
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(hf, line))
            {
                lines.push_back(line);
            }
            REQUIRE(lines.size() == 3);
            CHECK(lines[0] == "host1");
            CHECK(lines[1] == "host2");
            CHECK(lines[2] == "host3");
        }

        // Verify process count (should be hosts * gpus per node, which is 3*3=9)
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "9"); // 3 hosts * 3 GPUs per node

        // Verify --map-by is present and configured correctly
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--map-by") == "ppr:3:node"); // 3 GPUs per node

        // Verify workload parameter
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--workload") == "GN");

        // Verify time_to_run parameter
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--time_to_run") == "300");

        // Verify dynamic_adj flag is present
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--dynamic_adj"));

        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));
    }

    SECTION("Parse dcgmRunMnDiag_v2 with empty host list produces no command")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Clear host list
        for (int i = 0; i < DCGM_MNDIAG_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        CHECK(cmdArgs.empty());
    }

    SECTION("Parse dcgmRunMnDiag_v2 with hosts containing port numbers")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set hosts with port numbers
        SafeCopyTo(diagStruct.hostList[0], "host1:5555");
        SafeCopyTo(diagStruct.hostList[1], "host2:6666");
        SafeCopyTo(diagStruct.hostList[2], "host3:7777");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify the hostfile argument is present and contains stripped hostnames
        {
            std::ifstream hf(CommandLineArgUtils::ArgAfter(cmdArgs, "--hostfile"));
            REQUIRE(hf.is_open());
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(hf, line))
            {
                lines.push_back(line);
            }
            REQUIRE(lines.size() == 3);
            CHECK(lines[0] == "host1");
            CHECK(lines[1] == "host2");
            CHECK(lines[2] == "host3");
        }

        // Verify process count is hosts * gpus per node (3*3=9)
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "9");

        // Verify --map-by is present with correct format
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--map-by") == "ppr:3:node");

        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));
    }

    SECTION("Verify default parameters when not specified")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Clear all test parameters
        for (int i = 0; i < DCGM_MAX_TEST_PARMS; i++)
        {
            diagStruct.testParms[i][0] = '\0';
        }

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));

        // Check all default parameters from GetDefaultMnuberGemmParametersMap_v1
        // Updated for 10kHz FP32 pulse configuration
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--workload") == "GC");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--time_to_run") == "3600");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--max_workload") == "65536");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--MM_max_workload") == "65536");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--MM_sm_count") == "144");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--CE_type") == "H");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--MM_N") == "0");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--CE_size") == "200000");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--MM_type") == "ST_ST_SSS");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--MM_M_per_sm") == "32");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--freq") == "10000");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--duty") == "0.5");

        // Check flags (parameters with no values)
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--dynamic_adj"));
    }

    SECTION("Verify NET_link_order set to snake with odd number of processes")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // With 3 GPUs per node from MockDcgmCoreProxy, we can create an odd process count easily

        // Clear host list first
        for (int i = 0; i < DCGM_MNDIAG_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        // Add a single host to get exactly 3 processes (odd number)
        SafeCopyTo(diagStruct.hostList[0], "single-host");

        // Override default NET_link_order to verify it gets changed
        SafeCopyTo(diagStruct.testParms[3], "mnubergemm.NET_link_order=pair");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify we have 3 processes total (odd number)
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "3"); // 1 host * 3 GPUs = 3 processes

        // Verify NET_link_order is set to snake for odd number of processes
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--NET_link_order") == "snake"); // Should be overridden to snake

        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));
    }

    SECTION("Verify NET_link_order stays pair with even number of processes")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // With 3 GPUs per node from MockDcgmCoreProxy, we need 2 hosts to get 6 processes (even)

        // Clear host list first
        for (int i = 0; i < DCGM_MNDIAG_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        // Add exactly 2 hosts to get 6 processes (even number)
        SafeCopyTo(diagStruct.hostList[0], "host1");
        SafeCopyTo(diagStruct.hostList[1], "host2");

        // Set NET_link_order to verify it doesn't get changed
        SafeCopyTo(diagStruct.testParms[3], "mnubergemm.NET_link_order=pair");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify we have 6 processes total (even number)
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "6"); // 2 hosts * 3 GPUs = 6 processes

        // Verify NET_link_order remains pair for even number of processes
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--NET_link_order") == "pair"); // Should not be changed

        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));
    }
}

TEST_CASE("MnDiagMpiRunner user info via constructor")
{
    SECTION("Non-root uid does not add --allow-run-as-root")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner(1000);
        auto diagStruct = CreateTestDiagStruct();

        ScopedEnvVars envVars(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data(), "1");
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto args = mpiRunner.GetLastCommandArgs();

        REQUIRE_FALSE(args.empty());
        REQUIRE_FALSE(CommandLineArgUtils::ContainsArg(args, "--allow-run-as-root"));
    }

    SECTION("Root uid without env var does not add --allow-run-as-root")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner(0);
        auto diagStruct = CreateTestDiagStruct();

        ScopedEnvVars envVars(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data(), std::nullopt);
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto args = mpiRunner.GetLastCommandArgs();

        REQUIRE_FALSE(args.empty());
        REQUIRE_FALSE(CommandLineArgUtils::ContainsArg(args, "--allow-run-as-root"));
    }

    SECTION("Root uid with env var adds --allow-run-as-root")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner(0);
        auto diagStruct = CreateTestDiagStruct();

        ScopedEnvVars envVars(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data(), "1");
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto args = mpiRunner.GetLastCommandArgs();

        REQUIRE_FALSE(args.empty());
        REQUIRE(CommandLineArgUtils::ContainsArg(args, "--allow-run-as-root"));
    }
}

TEST_CASE("MnDiagMpiRunner ParseMnUberGemmOutput_v1")
{
    SECTION("Parse MNUB output with mixed success and error messages")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create input stream from the test_string
        auto dataFd = CreateMemFdFromString(test_string);

        // Call the parser
        std::array<std::string, 4> hostnames
            = { "b07-p1-dgx-07-c12", "b07-p1-dgx-07-c13", "b07-p1-dgx-07-c17", "b07-p1-dgx-07-c18" };
        std::array<std::string, 4> dcgmVersions   = { "3.1.1", "3.1.2", "3.3", "3.4.12" };
        std::array<std::string, 4> driverVersions = { "570.15", "590", "585.15.1", "520.0" };
        nodeInfoMap_t nodeInfo;
        for (unsigned int i = 0; i < hostnames.size(); i++)
        {
            nodeInfo[hostnames[i]].dcgmVersion   = dcgmVersions[i];
            nodeInfo[hostnames[i]].driverVersion = driverVersions[i];
        }
        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);

        // Verify test information
        REQUIRE(responseStruct->numTests == 1);
        REQUIRE(std::string(responseStruct->tests[0].name) == "MNUBERGEMM");
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL); // Should fail due to errors

        // Verify host information - should find 4 unique hosts
        REQUIRE(responseStruct->numHosts == 4);

        // Collect hostnames for verification
        std::set<std::string> foundHostnames;
        for (unsigned int i = 0; i < responseStruct->numHosts; i++)
        {
            foundHostnames.insert(responseStruct->hosts[i].hostname);
        }

        // Verify expected hostnames are present
        for (unsigned int i = 0; i < hostnames.size(); i++)
        {
            REQUIRE(foundHostnames.contains(hostnames[i]));
        }

        // Verify entity results - should have results for all GPUs found
        REQUIRE(responseStruct->numResults > 0);
        REQUIRE(responseStruct->numEntities == responseStruct->numResults);

        // Verify entities array is properly populated
        for (unsigned int i = 0; i < responseStruct->numEntities; i++)
        {
            dcgmMnDiagEntity_v1 const &entity = responseStruct->entities[i];

            // Verify entity properties
            REQUIRE(entity.entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(entity.hostId < responseStruct->numHosts);
            REQUIRE(std::string(entity.serialNum) == "");
            REQUIRE(std::string(entity.skuDeviceId) == "");

            // Verify entityId is valid (should be 0 or 1 based on the test data)
            REQUIRE((entity.entity.entityId == 0 || entity.entity.entityId == 1));
        }

        // Verify test result indices are properly populated
        REQUIRE(responseStruct->tests[0].numResults == responseStruct->numResults);
        for (unsigned int i = 0; i < responseStruct->tests[0].numResults; i++)
        {
            REQUIRE(responseStruct->tests[0].resultIndices[i] == i);
        }

        // Verify test error indices are properly populated
        REQUIRE(responseStruct->tests[0].numErrors == responseStruct->numErrors);
        for (unsigned int i = 0; i < responseStruct->tests[0].numErrors; i++)
        {
            REQUIRE(responseStruct->tests[0].errorIndices[i] == i);
        }

        // Count entities per host and verify they have the correct entity IDs
        std::unordered_map<std::string, std::set<unsigned int>> hostToEntities;
        for (unsigned int i = 0; i < responseStruct->numResults; i++)
        {
            dcgmMnDiagEntityResult_v1 const &result = responseStruct->results[i];
            REQUIRE(result.entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(result.testId == 0);

            // Get hostname for this result
            REQUIRE(result.hostId < responseStruct->numHosts);
            std::string hostname = responseStruct->hosts[result.hostId].hostname;
            hostToEntities[hostname].insert(result.entity.entityId);
        }

        // Verify that entities array matches the results array
        for (unsigned int i = 0; i < responseStruct->numEntities; i++)
        {
            dcgmMnDiagEntity_v1 const &entity       = responseStruct->entities[i];
            dcgmMnDiagEntityResult_v1 const &result = responseStruct->results[i];

            // Entity and corresponding result should have matching properties
            REQUIRE(entity.entity.entityGroupId == result.entity.entityGroupId);
            REQUIRE(entity.entity.entityId == result.entity.entityId);
            REQUIRE(entity.hostId == result.hostId);
        }

        // Verify expected entities per host
        for (unsigned int i = 0; i < hostnames.size(); i++)
        {
            REQUIRE(hostToEntities[hostnames[i]].contains(0));
            REQUIRE(hostToEntities[hostnames[i]].contains(1));
        }

        // Verify errors - should have 3 errors (one per unique entity with errors)
        REQUIRE(responseStruct->numErrors == 3);

        // Collect error information for verification
        std::map<std::string, std::vector<std::string>> entityErrors;
        for (unsigned int i = 0; i < responseStruct->numErrors; i++)
        {
            dcgmMnDiagError_v1 const &error = responseStruct->errors[i];
            REQUIRE(error.entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(error.testId == 0);
            REQUIRE(error.code == DCGM_FR_UNKNOWN);
            REQUIRE(error.category == DCGM_FR_EC_HARDWARE_OTHER);
            REQUIRE(error.severity == DCGM_ERROR_TRIAGE);

            // Get hostname for this error
            REQUIRE(error.hostId < responseStruct->numHosts);
            std::string hostname  = responseStruct->hosts[error.hostId].hostname;
            std::string entityKey = hostname + ":" + std::to_string(error.entity.entityId);
            entityErrors[entityKey].push_back(error.msg);
        }

        // Verify specific errors are captured (3 unique entities with errors)
        REQUIRE(entityErrors.contains("b07-p1-dgx-07-c17:0")); // G:4 L:0
        REQUIRE(entityErrors.contains("b07-p1-dgx-07-c18:0")); // G:6 L:0
        REQUIRE(entityErrors.contains("b07-p1-dgx-07-c18:1")); // G:7 L:1

        // Verify error messages contain expected content
        std::string c17_0_errors = entityErrors["b07-p1-dgx-07-c17:0"][0];
        REQUIRE(c17_0_errors.contains("CUDA Error"));
        REQUIRE(c17_0_errors.contains("cuBLAS Error"));

        std::string c18_0_errors = entityErrors["b07-p1-dgx-07-c18:0"][0];
        REQUIRE(c18_0_errors.contains("NCCL Error"));

        std::string c18_1_errors = entityErrors["b07-p1-dgx-07-c18:1"][0];
        REQUIRE(c18_1_errors.contains("IMEX Error"));

        // Verify entities without errors have PASS result
        for (unsigned int i = 0; i < responseStruct->numResults; i++)
        {
            dcgmMnDiagEntityResult_v1 const &result = responseStruct->results[i];
            std::string hostname                    = responseStruct->hosts[result.hostId].hostname;
            std::string entityKey                   = hostname + ":" + std::to_string(result.entity.entityId);

            if (!entityErrors.contains(entityKey))
            {
                REQUIRE(result.result == DCGM_DIAG_RESULT_PASS);
            }
            else
            {
                REQUIRE(result.result == DCGM_DIAG_RESULT_FAIL);
            }
        }

        // Verify host entities are populated correctly
        for (unsigned int i = 0; i < responseStruct->numHosts; i++)
        {
            dcgmMnDiagHosts_v1 const &host = responseStruct->hosts[i];
            REQUIRE(host.numEntities > 0);
            REQUIRE(host.dcgmVersion == dcgmVersions[i]);
            REQUIRE(host.driverVersion == driverVersions[i]);

            // Verify entity indices are valid global indices into entities[]
            // and that the referenced entity belongs to this host
            std::string hostname = host.hostname;
            for (unsigned int j = 0; j < host.numEntities; j++)
            {
                REQUIRE(host.entityIndices[j] < responseStruct->numEntities);
                REQUIRE(responseStruct->entities[host.entityIndices[j]].hostId == i);
            }
        }
    }

    SECTION("Parse empty MNUB output")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create empty input stream
        auto dataFd = CreateMemFdFromString("");

        // Call the parser
        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfoMap_t());

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);

        // With no output, should have default test setup but no entities or errors
        REQUIRE(responseStruct->numTests == 1);
        REQUIRE(std::string(responseStruct->tests[0].name) == "MNUBERGEMM");
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS); // No errors = PASS

        // Verify test indices are properly set to zero for empty output
        REQUIRE(responseStruct->tests[0].numResults == 0);
        REQUIRE(responseStruct->tests[0].numErrors == 0);

        REQUIRE(responseStruct->numHosts == 0);
        REQUIRE(responseStruct->numResults == 0);
        REQUIRE(responseStruct->numEntities == 0);
        REQUIRE(responseStruct->numErrors == 0);

        // Verify entities array is empty when there are no results
        REQUIRE(responseStruct->numEntities == 0);
    }

    SECTION("Parse empty MNUB output with expected hosts should FAIL")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        auto dataFd = CreateMemFdFromString("");

        nodeInfoMap_t nodeInfo;
        nodeInfo["expected-host-1"] = { .dcgmVersion = "3.1.1", .driverVersion = "570.15" };
        nodeInfo["expected-host-2"] = { .dcgmVersion = "3.1.2", .driverVersion = "570.16" };

        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(responseStruct->numErrors == 1);
        REQUIRE(std::string_view(responseStruct->errors[0].msg).contains("No hosts detected"));
        REQUIRE(responseStruct->errors[0].code == DCGM_FR_INTERNAL);
        REQUIRE(responseStruct->numHosts == 0);
        REQUIRE(responseStruct->numResults == 0);
    }

    SECTION("Parse invalid MNUB output (OCL errors) with expected hosts should FAIL")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // OCL errors without valid host data - simulates real-world early failure
        auto dataFd = CreateMemFdFromString(
            "[2025-11-13 05:12:34.274] MNUB [E] OCL failed <-clGetPlatformIDs_dl(0, nullptr, &num_plt)\n"
            "[2025-11-13 05:12:34.374] MNUB [E] OCL failed <-clGetPlatformIDs_dl(0, nullptr, &num_plt)\n");

        nodeInfoMap_t nodeInfo;
        nodeInfo["expected-host"] = { .dcgmVersion = "3.1.1", .driverVersion = "570.15" };

        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        REQUIRE(responseStruct->numErrors == 1);
        REQUIRE(std::string_view(responseStruct->errors[0].msg).contains("No hosts detected"));
        REQUIRE(responseStruct->numHosts == 0);
        REQUIRE(responseStruct->numResults == 0);
    }

    SECTION("Parse MNUB output with only INFO messages")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create input stream with only INFO messages (no errors)
        std::string infoOnlyString = "MNUB [I] G: 0 test-host L: 0  INFO  hosthash=123456789 B0I=632386609\n"
                                     "MNUB [I] G: 1 test-host L: 1  INFO  hosthash=123456789 B0I=632386610\n"
                                     "MNUB [I] G: 0 test-host L: 0  T:G FINAL values L:14057 IDLE:31878952\n"
                                     "MNUB [I] G: 1 test-host L: 1  T:N FINISHED\n";
        auto dataFd                = CreateMemFdFromString(infoOnlyString);

        // Call the parser
        nodeInfoMap_t nodeInfo;
        std::string hostname             = "test-host";
        std::string dcgmVersion          = "3.1.1";
        std::string driverVersion        = "570.15";
        nodeInfo[hostname].dcgmVersion   = dcgmVersion;
        nodeInfo[hostname].driverVersion = driverVersion;
        dcgmReturn_t result              = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);

        // Should have one host with two entities, no errors
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
        REQUIRE(responseStruct->numHosts == 1);
        REQUIRE(std::string(responseStruct->hosts[0].hostname) == hostname);
        REQUIRE(std::string(responseStruct->hosts[0].dcgmVersion) == dcgmVersion);
        REQUIRE(std::string(responseStruct->hosts[0].driverVersion) == driverVersion);
        REQUIRE(responseStruct->numResults == 2);
        REQUIRE(responseStruct->numErrors == 0);

        // Verify entities array is properly populated for INFO-only case
        REQUIRE(responseStruct->numEntities == 2);
        for (unsigned int i = 0; i < responseStruct->numEntities; i++)
        {
            dcgmMnDiagEntity_v1 const &entity = responseStruct->entities[i];

            // Verify entity properties
            REQUIRE(entity.entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(entity.entity.entityId == i); // Should be 0 and 1
            REQUIRE(entity.hostId == 0);          // Only one host, so hostId should be 0
            REQUIRE(std::string(entity.serialNum) == "");
            REQUIRE(std::string(entity.skuDeviceId) == "");
        }

        // Verify test result indices are properly populated (2 results, no errors)
        REQUIRE(responseStruct->tests[0].numResults == 2);
        REQUIRE(responseStruct->tests[0].resultIndices[0] == 0);
        REQUIRE(responseStruct->tests[0].resultIndices[1] == 1);

        // Verify no error indices since there are no errors
        REQUIRE(responseStruct->tests[0].numErrors == 0);

        // Both entities should have PASS result
        for (unsigned int i = 0; i < responseStruct->numResults; i++)
        {
            REQUIRE(responseStruct->results[i].result == DCGM_DIAG_RESULT_PASS);
        }
    }

    SECTION("Parse MNUB output with arbitrary prefixes before MNUB marker")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // The parser finds "MNUB [" anywhere in the line and strips everything before it,
        // so arbitrary prefixes (timestamps, tags, etc.) do not affect parsing.
        std::string prefixEdgeCases =
            // Bracketed prefix with MNUB message
            "[2025-01-15 10:30:45.123] MNUB [I] G: 0 host1 L: 0  INFO test message\n"
            // Prefix with no space before MNUB
            "[2025-01-15 10:30:45.456]MNUB [E] G: 1 host1 L: 1  T:N Error: test error\n"
            // Prefix with multiple spaces/tabs before MNUB
            "[2025-01-15 10:30:45.789]   \t  MNUB [I] G: 2 host2 L: 0  INFO another test\n"
            // Line with no MNUB marker (skipped)
            "[2025-01-15 10:30:45.012]   \t   \n"
            // Line with no MNUB marker (skipped)
            "[2025-01-15 10:30:45.345]\n"
            // Normal MNUB message without any prefix
            "MNUB [I] G: 3 host2 L: 1  INFO no prefix message\n"
            // OpenMPI-style timestamp prefix
            "Wed Mar  4 17:25:03 2026<stdout>:MNUB [E] G: 4 host3 L: 0  T:N openmpi timestamp\n"
            // Arbitrary text prefix
            "some-prefix: MNUB [I] G: 5 host3 L: 1  INFO arbitrary prefix\n"
            // ANSI escape codes with prefix
            "\x1B[31m[2025-01-15 10:30:45.999] MNUB [E] G: 6 host4 L: 0  T:N colored error\x1B[0m\n";

        auto dataFd = CreateMemFdFromString(prefixEdgeCases);

        // Set up node info for all hosts
        nodeInfoMap_t nodeInfo;
        std::vector<std::string> hostnames = { "host1", "host2", "host3", "host4" };
        for (const auto &hostname : hostnames)
        {
            nodeInfo[hostname].dcgmVersion   = "3.1.1";
            nodeInfo[hostname].driverVersion = "570.15";
        }

        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);

        // Verify test information
        REQUIRE(responseStruct->numTests == 1);
        REQUIRE(std::string(responseStruct->tests[0].name) == "MNUBERGEMM");
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL); // Should fail due to errors

        // Should find 4 unique hosts
        REQUIRE(responseStruct->numHosts == 4);

        // 7 lines contain "MNUB [" and are parsed; 2 lines without it are skipped
        REQUIRE(responseStruct->numResults == 7);

        // Verify errors are detected (3 error messages total)
        REQUIRE(responseStruct->numErrors == 3);

        // Collect error information
        std::map<std::string, std::vector<std::string>> entityErrors;
        for (unsigned int i = 0; i < responseStruct->numErrors; i++)
        {
            dcgmMnDiagError_v1 const &error = responseStruct->errors[i];
            std::string hostname            = responseStruct->hosts[error.hostId].hostname;
            std::string entityKey           = hostname + ":" + std::to_string(error.entity.entityId);
            entityErrors[entityKey].push_back(error.msg);
        }

        // Verify specific errors are captured
        REQUIRE(entityErrors.contains("host1:1")); // G:1 L:1 from bracketed prefix error
        REQUIRE(entityErrors.contains("host3:0")); // G:4 L:0 from OpenMPI timestamp prefix
        REQUIRE(entityErrors.contains("host4:0")); // G:6 L:0 from ANSI escape line

        // Verify error messages contain expected content
        REQUIRE(entityErrors["host1:1"][0].contains("test error"));
        REQUIRE(entityErrors["host3:0"][0].contains("openmpi timestamp"));
        REQUIRE(entityErrors["host4:0"][0].contains("colored error"));

        // Verify entities without errors have PASS result
        unsigned int passCount = 0;
        unsigned int failCount = 0;
        for (unsigned int i = 0; i < responseStruct->numResults; i++)
        {
            dcgmMnDiagEntityResult_v1 const &result = responseStruct->results[i];
            if (result.result == DCGM_DIAG_RESULT_PASS)
            {
                passCount++;
            }
            else
            {
                failCount++;
            }
        }

        // Should have 4 PASS results and 3 FAIL results
        REQUIRE(passCount == 4);
        REQUIRE(failCount == 3);
    }

    SECTION("Parse interleaved MNUB output from concurrent MPI ranks on same line")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // When multiple MPI ranks write to stdout concurrently, their output
        // can be interleaved so that several MNUB messages appear on a single
        // line without a newline separator.
        std::string interleavedOutput =
            // Normal single-message line
            "MNUB [I] G: 0 node1 L: 0  INFO normal line\n"
            // Two info messages from different ranks merged on one line
            "Mon Mar  9 14:43:25 2026<stdout>:MNUB [I] G: 1 node1 L: 1  INFO first rankMon Mar  9 14:43:25 2026<stdout>:MNUB [I] G: 2 node2 L: 0  INFO second rank\n"
            // A trace message (skipped) interleaved with an error message (parsed)
            "MNUB [T] G: 0 node1 L: 0  trace msgMNUB [E] G: 3 node2 L: 1  T:N interleaved error\n"
            // An error message followed by a debug message (skipped) on same line
            "MNUB [E] G: 4 node3 L: 0  T:N first errorMNUB [D] G: 0 node1 L: 0  debug msg\n"
            // Three messages on one line: skip, parse, skip
            "MNUB [D] G: 0 node1 L: 0  debugMNUB [I] G: 5 node3 L: 1  INFO middleMNUB [T] G: 0 node1 L: 0  trace\n";

        auto dataFd = CreateMemFdFromString(interleavedOutput);

        nodeInfoMap_t nodeInfo;
        for (auto const &hostname : { "node1", "node2", "node3" })
        {
            nodeInfo[hostname].dcgmVersion   = "3.1.1";
            nodeInfo[hostname].driverVersion = "570.15";
        }

        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataFd.Get(), responseStruct.get(), nodeInfo);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->numTests == 1);
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);

        REQUIRE(responseStruct->numHosts == 3);

        // 6 parseable messages: G:0/L:0, G:1/L:1, G:2/L:0, G:3/L:1, G:4/L:0, G:5/L:1
        REQUIRE(responseStruct->numResults == 6);

        REQUIRE(responseStruct->numErrors == 2);

        std::map<std::string, std::vector<std::string>> entityErrors;
        for (unsigned int i = 0; i < responseStruct->numErrors; i++)
        {
            dcgmMnDiagError_v1 const &error = responseStruct->errors[i];
            std::string hostname            = responseStruct->hosts[error.hostId].hostname;
            std::string entityKey           = hostname + ":" + std::to_string(error.entity.entityId);
            entityErrors[entityKey].push_back(error.msg);
        }

        REQUIRE(entityErrors.contains("node2:1"));
        REQUIRE(entityErrors.contains("node3:0"));

        REQUIRE(entityErrors["node2:1"][0].contains("interleaved error"));
        REQUIRE(entityErrors["node3:0"][0].contains("first error"));

        unsigned int passCount = 0;
        unsigned int failCount = 0;
        for (unsigned int i = 0; i < responseStruct->numResults; i++)
        {
            if (responseStruct->results[i].result == DCGM_DIAG_RESULT_PASS)
            {
                passCount++;
            }
            else
            {
                failCount++;
            }
        }

        REQUIRE(passCount == 4);
        REQUIRE(failCount == 2);
    }
}

SCENARIO("MnDiagMpiRunner MPI version consistency check")
{
    MnDiagMpiMnubergemmRunnerTests mpiRunner;

    SECTION("Matching versions — returns true")
    {
        FakeMpiVersionExecutor fakeExec;
        fakeExec.mpirunOutput   = "mpirun (Open MPI) 4.1.9a1\n";
        fakeExec.ompiInfoOutput = "Open MPI v4.1.9a1\n";

        REQUIRE(mpiRunner.CheckMpiVersionConsistency("/usr/bin/mpirun", &fakeExec) == true);
    }

    SECTION("Mismatched versions — returns false")
    {
        FakeMpiVersionExecutor fakeExec;
        fakeExec.mpirunOutput   = "mpirun (Open MPI) 4.1.9a1\n";
        fakeExec.ompiInfoOutput = "Open MPI v4.1.6\n";

        REQUIRE(mpiRunner.CheckMpiVersionConsistency("/usr/bin/mpirun", &fakeExec) == false);
    }

    SECTION("ompi_info derived from the same bin/ as mpirun")
    {
        FakeMpiVersionExecutor fakeExec;
        fakeExec.mpirunOutput   = "mpirun (Open MPI) 4.1.9a1\n";
        fakeExec.ompiInfoOutput = "Open MPI v4.1.9a1\n";

        bool result = mpiRunner.CheckMpiVersionConsistency("/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun", &fakeExec);

        REQUIRE(fakeExec.receivedCmds[0].find("/usr/mpi/gcc/openmpi-4.1.9a1/bin/mpirun") != std::string::npos);
        REQUIRE(fakeExec.receivedCmds[1].find("/usr/mpi/gcc/openmpi-4.1.9a1/bin/ompi_info") != std::string::npos);
        REQUIRE(result == true);
    }

    SECTION("Non-OpenMPI output — skipped, returns true")
    {
        FakeMpiVersionExecutor fakeExec;
        fakeExec.mpirunOutput   = "mpirun: Intel(R) MPI Library 2021.10\n";
        fakeExec.ompiInfoOutput = "";

        REQUIRE(mpiRunner.CheckMpiVersionConsistency("/usr/bin/mpirun", &fakeExec) == true);
    }

    SECTION("Executor throws — skipped, returns true")
    {
        FakeMpiVersionExecutor fakeExec;
        fakeExec.shouldThrow = true;

        REQUIRE(mpiRunner.CheckMpiVersionConsistency("/usr/bin/mpirun", &fakeExec) == true);
        REQUIRE(fakeExec.receivedCmds.size() == 1); // aborts after first call throws
    }
}

// ---------------------------------------------------------------------------
// Routing-interface helper tests (GetRoutingInterfacesForHosts)
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiRunnerInternal::GetRoutingInterfacesForHosts")
{
    using MnDiagMpiRunnerInternal::GetRoutingInterfacesForHosts;
    using MnDiagMpiRunnerInternal::ParseRoutingInterface;

    // Use a fake path so tests don't depend on the host having `ip` installed.
    std::string const ipCmd = "/test/ip";

    SECTION("ParseRoutingInterface returns interface name for normal route")
    {
        std::string output = "10.0.1.10 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n";
        REQUIRE(ParseRoutingInterface(output) == "enP5p9s0");
    }

    SECTION("ParseRoutingInterface returns empty for loopback route")
    {
        std::string output = "local 10.0.0.5 dev lo src 10.0.0.5 uid 1000\n    cache <local>\n";
        REQUIRE(ParseRoutingInterface(output).empty());
    }

    SECTION("ParseRoutingInterface returns empty for unparseable output")
    {
        REQUIRE(ParseRoutingInterface("").empty());
        REQUIRE(ParseRoutingInterface("some garbage output").empty());
    }

    SECTION("All hosts resolve to same interface")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.10", ipCmd),
                                 "10.0.0.10 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.11", ipCmd),
                                 "10.0.0.11 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.12", ipCmd),
                                 "10.0.0.12 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n");

        auto result = GetRoutingInterfacesForHosts({ "10.0.0.10", "10.0.0.11", "10.0.0.12" }, &mockExecutor, ipCmd);
        REQUIRE(result == "enP5p9s0");
    }

    SECTION("Hosts route via different interfaces — both returned")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        mockExecutor.SetResponse(fmt::format("{} route get 192.168.1.10", ipCmd),
                                 "192.168.1.10 via 192.168.1.1 dev eth0 src 192.168.1.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.10", ipCmd),
                                 "10.0.0.10 via 10.0.0.1 dev eth1 src 10.0.0.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.11", ipCmd),
                                 "10.0.0.11 via 10.0.0.1 dev eth1 src 10.0.0.5 uid 1000\n    cache\n");

        auto result = GetRoutingInterfacesForHosts({ "192.168.1.10", "10.0.0.10", "10.0.0.11" }, &mockExecutor, ipCmd);
        auto parts  = DcgmNs::Split(result, ',');
        std::unordered_set<std::string> ifaces;
        for (auto const &p : parts)
        {
            ifaces.emplace(p);
        }
        REQUIRE(ifaces == std::unordered_set<std::string> { "eth0", "eth1" });
    }

    SECTION("Local host (loopback route) is skipped; only remote interfaces returned")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.5", ipCmd),
                                 "local 10.0.0.5 dev lo src 10.0.0.5 uid 1000\n    cache <local>\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.1.10", ipCmd),
                                 "10.0.1.10 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.1.11", ipCmd),
                                 "10.0.1.11 via 10.0.0.1 dev enP5p9s0 src 10.0.0.5 uid 1000\n    cache\n");

        auto result = GetRoutingInterfacesForHosts({ "10.0.0.5", "10.0.1.10", "10.0.1.11" }, &mockExecutor, ipCmd);
        REQUIRE(result == "enP5p9s0");
    }

    SECTION("All loopback returns empty string")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        mockExecutor.SetResponse(fmt::format("{} route get 127.0.0.1", ipCmd),
                                 "local 127.0.0.1 dev lo src 127.0.0.1 uid 1000\n    cache <local>\n");
        mockExecutor.SetResponse(fmt::format("{} route get 127.0.0.2", ipCmd),
                                 "local 127.0.0.2 dev lo src 127.0.0.2 uid 1000\n    cache <local>\n");
        mockExecutor.SetResponse(fmt::format("{} route get 127.0.0.3", ipCmd),
                                 "local 127.0.0.3 dev lo src 127.0.0.3 uid 1000\n    cache <local>\n");

        auto result = GetRoutingInterfacesForHosts({ "127.0.0.1", "127.0.0.2", "127.0.0.3" }, &mockExecutor, ipCmd);
        REQUIRE(result.empty());
    }

    SECTION("Empty host list returns empty string")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        REQUIRE(GetRoutingInterfacesForHosts({}, &mockExecutor, ipCmd).empty());
    }
}

// ---------------------------------------------------------------------------
// MCA interface injection tests via SetRoutingInterfaceResolver
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiRunner MCA interface injection in command")
{
    SECTION("MCA args injected when resolver returns an interface name")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "enP5p9s0"; });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "enP5p9s0");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "enP5p9s0");
    }

    SECTION("MCA args injected with multiple interfaces")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0,eth1"; });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "eth0,eth1");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "eth0,eth1");
    }

    SECTION("No MCA args injected when resolver returns empty string")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return ""; });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include").empty());
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include").empty());
    }

    SECTION("MCA args appear before the mnubergemm binary")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetTestBinaryPath("/path/to/mnubergemm");
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "enP5p9s0"; });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        auto binaryIt = std::find(cmdArgs.begin(), cmdArgs.end(), "/path/to/mnubergemm");
        auto btlIt    = std::find(cmdArgs.begin(), cmdArgs.end(), "btl_tcp_if_include");
        auto oobIt    = std::find(cmdArgs.begin(), cmdArgs.end(), "oob_tcp_if_include");

        REQUIRE(binaryIt != cmdArgs.end());
        REQUIRE(btlIt != cmdArgs.end());
        REQUIRE(oobIt != cmdArgs.end());
        REQUIRE(btlIt < binaryIt);
        REQUIRE(oobIt < binaryIt);
    }

    SECTION("Resolver receives the correct host list")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        std::vector<std::string> capturedHosts;
        mpiRunner.SetRoutingInterfaceResolver([&capturedHosts](std::vector<std::string> const &hosts) {
            capturedHosts = hosts;
            return "enP5p9s0";
        });

        auto diagStruct = CreateTestDiagStruct(); // hosts: host1, host2, host3
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        REQUIRE(capturedHosts == std::vector<std::string> { "host1", "host2", "host3" });
    }
}
// ---------------------------------------------------------------------------
// openmpi.* test parameter and OMPI_MCA_* env var precedence tests
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiRunner openmpi.detect_interfaces=0 disables auto-detection")
{
    SECTION("detect_interfaces=0 prevents MCA injection even when resolver would return interfaces")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        bool resolverCalled = false;
        mpiRunner.SetRoutingInterfaceResolver([&resolverCalled](std::vector<std::string> const &) {
            resolverCalled = true;
            return "enP5p9s0";
        });

        auto diagStruct = CreateTestDiagStruct();
        SafeCopyTo(diagStruct.testParms[3], "openmpi.detect_interfaces=0");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include").empty());
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include").empty());
        REQUIRE_FALSE(resolverCalled);
    }

    SECTION("detect_interfaces=false also disables auto-detection")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        bool resolverCalled = false;
        mpiRunner.SetRoutingInterfaceResolver([&resolverCalled](std::vector<std::string> const &) {
            resolverCalled = true;
            return "enP5p9s0";
        });

        auto diagStruct = CreateTestDiagStruct();
        SafeCopyTo(diagStruct.testParms[3], "openmpi.detect_interfaces=false");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include").empty());
        REQUIRE_FALSE(resolverCalled);
    }

    SECTION("detect_interfaces=1 keeps auto-detection enabled")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return "eth0"; });

        auto diagStruct = CreateTestDiagStruct();
        SafeCopyTo(diagStruct.testParms[3], "openmpi.detect_interfaces=1");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "eth0");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "eth0");
    }
}

TEST_CASE("MnDiagMpiRunner OMPI_MCA env vars are NOT promoted to -mca args")
{
    SECTION("OMPI_MCA_btl_tcp_if_include present — auto-detection still runs, env var not forwarded")
    {
        ScopedEnvVars envVars("OMPI_MCA_btl_tcp_if_include", "ib0");

        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        bool resolverCalled = false;
        mpiRunner.SetRoutingInterfaceResolver([&resolverCalled](std::vector<std::string> const &) {
            resolverCalled = true;
            return "enP5p9s0";
        });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(resolverCalled);
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "enP5p9s0");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "enP5p9s0");
    }

    SECTION("OMPI_MCA_oob_tcp_if_include present — auto-detection still runs, env var not forwarded")
    {
        ScopedEnvVars envVars("OMPI_MCA_oob_tcp_if_include", "ib0");

        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        bool resolverCalled = false;
        mpiRunner.SetRoutingInterfaceResolver([&resolverCalled](std::vector<std::string> const &) {
            resolverCalled = true;
            return "eth0";
        });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(resolverCalled);
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "eth0");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "eth0");
    }

    SECTION("Both OMPI_MCA env vars present — auto-detection still runs, env vars not forwarded")
    {
        ScopedEnvVars envVars({
            { "OMPI_MCA_btl_tcp_if_include", "eth0" },
            { "OMPI_MCA_oob_tcp_if_include", "eth1" },
        });

        MnDiagMpiMnubergemmRunnerTests mpiRunner;

        bool resolverCalled = false;
        mpiRunner.SetRoutingInterfaceResolver([&resolverCalled](std::vector<std::string> const &) {
            resolverCalled = true;
            return "enP5p9s0";
        });

        auto diagStruct = CreateTestDiagStruct();
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE(resolverCalled);
        REQUIRE(FindMcaValue(cmdArgs, "btl_tcp_if_include") == "enP5p9s0");
        REQUIRE(FindMcaValue(cmdArgs, "oob_tcp_if_include") == "enP5p9s0");
    }

    SECTION("LogMpiInterfaceEnvVars does not crash with no env vars set")
    {
        ScopedEnvVars envVars({
            { "OMPI_MCA_btl_tcp_if_include", std::nullopt },
            { "OMPI_MCA_oob_tcp_if_include", std::nullopt },
            { "PRTE_MCA_oob_tcp_if_include", std::nullopt },
        });

        REQUIRE_NOTHROW(MnDiagMpiRunnerInternal::LogMpiInterfaceEnvVars());
    }

    SECTION("LogMpiInterfaceEnvVars does not crash with env vars set")
    {
        ScopedEnvVars envVars({
            { "OMPI_MCA_btl_tcp_if_include", "eth0" },
            { "OMPI_MCA_oob_tcp_if_include", "eth1" },
            { "PRTE_MCA_oob_tcp_if_include", "eth2" },
        });

        REQUIRE_NOTHROW(MnDiagMpiRunnerInternal::LogMpiInterfaceEnvVars());
    }
}

TEST_CASE("MnDiagMpiRunner openmpi.* params do not leak into mnubergemm args")
{
    SECTION("openmpi.detect_interfaces does not appear as --detect_interfaces in mnubergemm args")
    {
        MnDiagMpiMnubergemmRunnerTests mpiRunner;
        mpiRunner.SetRoutingInterfaceResolver([](std::vector<std::string> const &) { return ""; });

        auto diagStruct = CreateTestDiagStruct();
        SafeCopyTo(diagStruct.testParms[3], "openmpi.detect_interfaces=0");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        REQUIRE_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--detect_interfaces"));
    }
}

// ---------------------------------------------------------------------------
// ResolveHostnameToIp tests
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiRunnerInternal::ResolveHostnameToIp")
{
    using MnDiagMpiRunnerInternal::ResolveHostnameToIp;

    SECTION("IPv4 address is returned unchanged")
    {
        REQUIRE(ResolveHostnameToIp("10.0.0.1") == "10.0.0.1");
        REQUIRE(ResolveHostnameToIp("192.168.1.100") == "192.168.1.100");
        REQUIRE(ResolveHostnameToIp("127.0.0.1") == "127.0.0.1");
    }

    SECTION("localhost resolves to 127.0.0.1")
    {
        REQUIRE(ResolveHostnameToIp("localhost") == "127.0.0.1");
    }

    SECTION("Unresolvable hostname falls back to the original string")
    {
        auto result = ResolveHostnameToIp("this-host-definitely-does-not-exist.invalid");
        REQUIRE(result == "this-host-definitely-does-not-exist.invalid");
    }
}

// ---------------------------------------------------------------------------
// GetRoutingInterfacesForHosts with IP addresses (resolution is a no-op)
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiRunnerInternal::GetRoutingInterfacesForHosts with IP addresses")
{
    using MnDiagMpiRunnerInternal::GetRoutingInterfacesForHosts;

    std::string const ipCmd = "/test/ip";

    SECTION("IP addresses are passed directly to ip route get")
    {
        MnubergemmMockCommandExecutor mockExecutor;
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.1", ipCmd),
                                 "10.0.0.1 via 10.0.0.254 dev eth0 src 10.0.0.5 uid 1000\n    cache\n");
        mockExecutor.SetResponse(fmt::format("{} route get 10.0.0.2", ipCmd),
                                 "10.0.0.2 via 10.0.0.254 dev eth0 src 10.0.0.5 uid 1000\n    cache\n");

        auto result = GetRoutingInterfacesForHosts({ "10.0.0.1", "10.0.0.2" }, &mockExecutor, ipCmd);
        REQUIRE(result == "eth0");
    }
}

TEST_CASE("MnDiagMpiMnubergemmRunner ResolveTestBinaryPath [mndiag]")
{
    SECTION("Should use default path when no environment variable is set")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), std::nullopt);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnubergemm") != std::string::npos);
    }

    SECTION("Should use custom path when environment variable points to valid executable")
    {
        std::string customPath = "/bin/true";
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), customPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path == customPath);
    }

    SECTION("Should fallback to default path when environment variable is empty")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), "");

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnubergemm") != std::string::npos);
    }

    SECTION("Should fallback to default path when environment variable points to non-existent file")
    {
        std::string nonExistentPath = "/path/to/nonexistent/binary";
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), nonExistentPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnubergemm") != std::string::npos);
    }

    SECTION("Should fallback to default path when environment variable points to non-executable file")
    {
        auto nonExecutableFile = TempFile::Random("not an executable", "mnubergemm-non-exec-");
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), nonExecutableFile.Path().string());

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnubergemm") != std::string::npos);
    }

    SECTION("Should fallback to default path when environment variable points to directory")
    {
        auto directoryPath = TempDir::Random("mnubergemm-dir-");
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), directoryPath.Path().string());

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("mnubergemm") != std::string::npos);
    }

    SECTION("CUDA version is zero with no environment variable, default path should fail")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), std::nullopt);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(0);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_NO_DATA);
        CHECK(path.empty());

        // Second call returns the same cached error without re-resolving
        result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_NO_DATA);
    }

    SECTION("Should fail when environment variable path is too long")
    {
        std::string longPath(DCGM_MAX_STR_LENGTH + 10, 'a');
        ScopedEnvVars envVars(MnDiagConstants::ENV_MNUBERGEMM_PATH.data(), longPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiMnubergemmRunner runner(*mockCoreProxy, MNUBERGEMM_EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_BADPARAM);
        CHECK(path.empty());

        // Second call returns the same cached error without re-resolving
        result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_BADPARAM);
    }
}
