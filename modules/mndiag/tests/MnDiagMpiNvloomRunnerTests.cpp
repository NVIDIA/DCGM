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
#include <Defer.hpp>
#include <MnDiagMpiNvloomRunner.h>
#include <MnDiagMpiRunner.h>
#include <MnDiagOutputHelpers.h>
#include <ScopedEnvVars.hpp>
#include <UniquePtrUtil.h>
#include <array>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <dcgm_errors.h>
#include <dcgm_structs.h>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>

#include "MockDcgmCoreProxy.h"

static constexpr uid_t EFFECTIVE_UID = 0;

namespace
{

int CreateMemFdFromString(std::string const &data)
{
    int fd = memfd_create("test_data", 0);
    if (fd < 0)
    {
        return -1;
    }
    if (write(fd, data.data(), data.size()) != static_cast<ssize_t>(data.size()))
    {
        close(fd);
        return -1;
    }
    lseek(fd, 0, SEEK_SET);
    return fd;
}

std::unique_ptr<dcgmMnDiagResponse_v2> CreateTestResponseStruct()
{
    auto responseUptr     = MakeUniqueZero<dcgmMnDiagResponse_v2>();
    responseUptr->version = dcgmMnDiagResponse_version2;
    return responseUptr;
}

dcgmRunMnDiag_v2 CreateNvloomTestDiagStruct()
{
    dcgmRunMnDiag_v2 diagStruct = {};

    diagStruct.version = dcgmRunMnDiag_version2;

    SafeCopyTo(diagStruct.hostList[0], "host1");
    SafeCopyTo(diagStruct.hostList[1], "host2");
    SafeCopyTo(diagStruct.hostList[2], "host3");

    SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=300");

    return diagStruct;
}

} // namespace

class MnDiagMpiNvloomRunnerTests
{
public:
    explicit MnDiagMpiNvloomRunnerTests(uid_t uid = EFFECTIVE_UID)
        : m_mockCoreProxy(std::make_unique<MockDcgmCoreProxy>())
        , m_runner(*m_mockCoreProxy, uid)
    {
        m_runner.m_testBinaryPath.emplace("/fake/test/nvloom");
    }

    dcgmReturn_t MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
    {
        return m_runner.MnDiagOutputCallback(fd, responseStruct, nodeInfo);
    }

    void SetTestBinaryPath(std::string const &path)
    {
        m_runner.m_testBinaryPath.emplace(path);
    }

    void ConstructMpiCommand(void const *params)
    {
        m_runner.ConstructMpiCommand(params);
    }

    void ParseDcgmMnDiagToMpiCommand(dcgmRunMnDiag_v2 const &drmnd)
    {
        m_runner.ParseDcgmMnDiagToMpiCommand(drmnd);
    }

    std::string GetLastCommand() const
    {
        return m_runner.GetLastCommand();
    }

    std::vector<std::string> GetLastCommandArgs() const
    {
        return m_runner.m_lastCommand;
    }

    void SetMockCudaVersion(int version)
    {
        m_mockCoreProxy->SetMockCudaVersion(version);
    }

private:
    std::unique_ptr<MockDcgmCoreProxy> m_mockCoreProxy;
    MnDiagMpiNvloomRunner m_runner;
};

SCENARIO("MnDiagMpiNvloomRunner: compile-time interface properties")
{
    STATIC_REQUIRE(std::is_base_of_v<MnDiagMpiRunner, MnDiagMpiNvloomRunner>);
    STATIC_REQUIRE(!std::copyable<MnDiagMpiNvloomRunner>);
    STATIC_REQUIRE(std::is_nothrow_destructible_v<MnDiagMpiNvloomRunner>);
}

SCENARIO("MnDiagMpiNvloomRunner: test prefix and defaults")
{
    GIVEN("a default-constructed runner")
    {
        MockDcgmCoreProxy coreProxy;
        MnDiagMpiNvloomRunner realRunner(coreProxy, EFFECTIVE_UID);

        THEN("the test prefix is nvloom.")
        {
            CHECK(realRunner.GetTestPrefix() == "nvloom.");
        }

        THEN("the log file prefix is mndiag_nvloom")
        {
            CHECK(realRunner.GetLogFilePrefix() == "mndiag_nvloom");
        }

        THEN("the default parameters map contains expected nvloom defaults")
        {
            auto defaults = realRunner.GetDefaultParametersMap();
            REQUIRE(defaults.size() == 5);
            CHECK(defaults.at("time_to_run") == "3600");
            CHECK(defaults.at("suite") == "gpu-to-rack");
            CHECK(defaults.at("buffer_size") == "536870912");
            CHECK(defaults.at("iterations") == "16");
            CHECK(defaults.at("allocation_strategy") == "reuse");
        }
    }
}

// Clean nvloom output: 2 hosts, 4 GPUs each, 2 sub-tests, all pass
const std::string nvloom_success_2node = R"(nvloom_cli 2.0.0
git commit:
NVCC version: 13.0
CUDA version: 13.4
Driver version: 615.27
Allocation strategy: reuse
Buffer size: 536870912 bytes
Iteration count: 16
Process 0 (compute05): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 1234567890003
Process 1 (compute05): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 1234567890003
Process 2 (compute05): device 2: NVIDIA GB200 (00000018:01:00); rackGuid: 1234567890003
Process 3 (compute05): device 3: NVIDIA GB200 (00000019:01:00); rackGuid: 1234567890003
Process 4 (compute04): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 1234567890002
Process 5 (compute04): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 1234567890002
Process 6 (compute04): device 2: NVIDIA GB200 (00000018:01:00); rackGuid: 1234567890002
Process 7 (compute04): device 3: NVIDIA GB200 (00000019:01:00); rackGuid: 1234567890002

Running gpu_to_rack_device_to_device_bidir_read_ce
	1234567890002	1234567890003
0	165.15	167.07
1	165.28	167.06
2	165.21	167.07
3	165.25	167.07
4	167.06	165.25
5	167.07	165.28
6	167.07	165.09
7	167.06	165.25

Min gpu_to_rack_device_to_device_bidir_read_ce 165.09 GB/s
Avg gpu_to_rack_device_to_device_bidir_read_ce 166.14 GB/s
Max gpu_to_rack_device_to_device_bidir_read_ce 167.07 GB/s
Total gpu_to_rack_device_to_device_bidir_read_ce 2658.29 GB/s

ExecutionTime gpu_to_rack_device_to_device_bidir_read_ce 6.61 s
Done gpu_to_rack_device_to_device_bidir_read_ce

Running gpu_to_rack_device_to_device_bidir_read_sm
	1234567890002	1234567890003
0	151.16	151.19
1	151.15	151.19
2	151.17	151.19
3	151.15	151.19
4	151.19	151.16
5	151.19	151.16
6	151.19	151.16
7	151.20	151.16

Min gpu_to_rack_device_to_device_bidir_read_sm 151.15 GB/s
Avg gpu_to_rack_device_to_device_bidir_read_sm 151.18 GB/s
Max gpu_to_rack_device_to_device_bidir_read_sm 151.20 GB/s
Total gpu_to_rack_device_to_device_bidir_read_sm 2418.81 GB/s

ExecutionTime gpu_to_rack_device_to_device_bidir_read_sm 7.19 s
Done gpu_to_rack_device_to_device_bidir_read_sm
)";

// Output with stderr errors from specific processes
// [hostname:N] where N is the local CUDA device index (not the MPI rank)
const std::string nvloom_failure_with_errors = R"(nvloom_cli 2.0.0
Process 0 (node01): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (node01): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111
Process 2 (node02): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 222
Process 3 (node02): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 222

Running multicast_all_to_all_ce
[node01:0]: [CUDA_ERROR_UNKNOWN] unknown error in expression cuMulticastBindMem(multicastHandle, 0, handle, 0, roundedUpAllocationSize, 0) in void MultinodeMemoryAllocationMulticast::allocateMemory()() : /tmp/nvloom-2.0.0/library/allocators.cpp:186
[node02:0]: [CUDA_ERROR_UNKNOWN] unknown error in expression cuMulticastBindMem(multicastHandle, 0, handle, 0, roundedUpAllocationSize, 0) in void MultinodeMemoryAllocationMulticast::allocateMemory()() : /tmp/nvloom-2.0.0/library/allocators.cpp:186
)";

// Mixed output: some sub-tests pass, then errors on multicast sub-test
const std::string nvloom_mixed_results = R"(nvloom_cli 2.0.0
Process 0 (host-a): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (host-a): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111
Process 2 (host-b): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 222
Process 3 (host-b): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 222

Running gpu_to_rack_device_to_device_bidir_read_ce
0	165.15	167.07
1	165.28	167.06
2	167.06	165.25
3	167.07	165.28

Min gpu_to_rack_device_to_device_bidir_read_ce 165.15 GB/s
Avg gpu_to_rack_device_to_device_bidir_read_ce 166.15 GB/s
Max gpu_to_rack_device_to_device_bidir_read_ce 167.07 GB/s
Total gpu_to_rack_device_to_device_bidir_read_ce 1329.15 GB/s

ExecutionTime gpu_to_rack_device_to_device_bidir_read_ce 6.5 s
Done gpu_to_rack_device_to_device_bidir_read_ce

Running multicast_all_to_all_ce
[host-a:0]: [CUDA_ERROR_UNKNOWN] cuMulticastBindMem failed
)";

// Output with MPI timestamp prefixes interleaved mid-line (real-world format)
const std::string nvloom_interleaved_timestamps = R"(Mon Jul 20 09:42:51 2026<stdout>:nvloom_cli 2.0.0
Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostX): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 1 (hostX): device 1: NVIDIA GB200 (00000009:01:00)Mon Jul 20 09:42:51 2026<stdout>:; rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 2 (hostY): device 0: NVIDIA GB200 (00000018:01:00); rackGuid: 222
Mon Jul 20 09:42:51 2026<stdout>:Process 3 (hostY): device Mon Jul 20 09:42:51 2026<stdout>:1: NVIDIA GB200 (00000019:01:00); rackGuid: 222
Mon Jul 20 09:42:52 2026<stdout>:Running gpu_to_rack_test
Mon Jul 20 09:42:53 2026<stdout>:0	Mon Jul 20 09:42:53 2026<stdout>:165.15	Mon Jul 20 09:42:53 2026<stdout>:167.07
Mon Jul 20 09:42:53 2026<stdout>:
Mon Jul 20 09:42:53 2026<stdout>:Min gpu_to_rack_test Mon Jul 20 09:42:53 2026<stdout>:165.15 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Avg gpu_to_rack_test 166.11 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Max gpu_to_rack_testMon Jul 20 09:42:53 2026<stdout>: 167.07 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Total Mon Jul 20 09:42:53 2026<stdout>:gpu_to_rack_test 332.22 GB/s
Mon Jul 20 09:42:53 2026<stdout>:
Mon Jul 20 09:42:53 2026<stdout>:ExecutionTime gpu_to_rack_test 1.5 s
Mon Jul 20 09:42:53 2026<stdout>:Done gpu_to_rack_test
)";

// Output with ANSI escape codes wrapping content
const std::string nvloom_ansi_codes
    = "\x1B[32mnvloom_cli 2.0.0\x1B[0m\n"
      "\x1B[32mProcess 0 (ansi-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111\x1B[0m\n"
      "\x1B[32mProcess 1 (ansi-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111\x1B[0m\n"
      "\x1B[32mRunning test_with_ansi\x1B[0m\n"
      "0\t100.0\n"
      "1\t100.0\n"
      "\x1B[32mMin test_with_ansi 100.0 GB/s\x1B[0m\n"
      "\x1B[32mAvg test_with_ansi 100.0 GB/s\x1B[0m\n"
      "\x1B[32mMax test_with_ansi 100.0 GB/s\x1B[0m\n"
      "\x1B[32mTotal test_with_ansi 200.0 GB/s\x1B[0m\n"
      "\x1B[32mDone test_with_ansi\x1B[0m\n";

// Output with OpenMPI rank/tag prefixes on every line ([rank,local]<stdout>:)
// This reproduces the failure from --merge-stderr-to-stdout where OpenMPI
// wraps each line with [1,0]<stdout>: style prefixes.
const std::string nvloom_mpi_rank_prefixed = R"([1,0]<stdout>:nvloom_cli 2.0.0
[1,0]<stdout>:Process 0 (mpi-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
[1,0]<stdout>:Process 1 (mpi-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111
[1,0]<stdout>:
[1,0]<stdout>:Running gpu_to_rack_device_to_device_bidir_read_ce
[1,0]<stdout>:0	165.15	167.07
[1,0]<stdout>:1	165.28	167.06
[1,0]<stdout>:
[1,0]<stdout>:Min gpu_to_rack_device_to_device_bidir_read_ce 165.15 GB/s
[1,0]<stdout>:Avg gpu_to_rack_device_to_device_bidir_read_ce 166.14 GB/s
[1,0]<stdout>:Max gpu_to_rack_device_to_device_bidir_read_ce 167.07 GB/s
[1,0]<stdout>:Total gpu_to_rack_device_to_device_bidir_read_ce 1329.12 GB/s
[1,0]<stdout>:
[1,0]<stdout>:ExecutionTime gpu_to_rack_device_to_device_bidir_read_ce 6.61 s
[1,0]<stdout>:Done gpu_to_rack_device_to_device_bidir_read_ce
)";

// OpenMPI 5 --timestamp-output format captured from the containerized mndiag test.
const std::string nvloom_openmpi5_timestamp_prefixed = R"([Tue Aug 11 00:04:15 2026]<stdout>: nvloom_cli 2.0.0
[Tue Aug 11 00:04:15 2026]<stdout>: Process 0 (localhost): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 1234567890001
[Tue Aug 11 00:04:15 2026]<stdout>: Process 1 (localhost): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 1234567890001
[Tue Aug 11 00:04:15 2026]<stdout>:
[Tue Aug 11 00:04:15 2026]<stdout>: Running gpu_to_rack_device_to_device_bidir_read_ce
[Tue Aug 11 00:04:15 2026]<stdout>: 0	165.00
[Tue Aug 11 00:04:15 2026]<stdout>: 1	165.10
[Tue Aug 11 00:04:15 2026]<stdout>:
[Tue Aug 11 00:04:15 2026]<stdout>: Min gpu_to_rack_device_to_device_bidir_read_ce 165.00 GB/s
[Tue Aug 11 00:04:15 2026]<stdout>: Avg gpu_to_rack_device_to_device_bidir_read_ce 165.05 GB/s
[Tue Aug 11 00:04:15 2026]<stdout>: Max gpu_to_rack_device_to_device_bidir_read_ce 165.10 GB/s
[Tue Aug 11 00:04:15 2026]<stdout>: Total gpu_to_rack_device_to_device_bidir_read_ce 330.10 GB/s
[Tue Aug 11 00:04:15 2026]<stdout>:
[Tue Aug 11 00:04:15 2026]<stdout>: ExecutionTime gpu_to_rack_device_to_device_bidir_read_ce 5.0 s
[Tue Aug 11 00:04:15 2026]<stdout>: Done gpu_to_rack_device_to_device_bidir_read_ce
)";

// MPI timestamps interleaved at every token boundary within Process lines:
//   - Between rank number and hostname parenthesis (Process 3<ts> (hostA))
//   - Between hostname and close-paren  ((hostA<ts>):)
//   - Between close-paren and colon     ((hostA)<ts>: device)
//   - Between "device" keyword and device index (device <ts>3:)
//   - Between device index and colon    (device 2<ts>:)
// StripMpiTimestamps replaces each with a space; the processLineRe uses \s*
// before colons and trims the hostname capture to tolerate the extra whitespace.
const std::string nvloom_timestamp_multipoint_splits
    = R"(Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostA): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 1 (hostAMon Jul 20 09:42:51 2026<stdout>:): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 2 (hostA)Mon Jul 20 09:42:51 2026<stdout>:: device 2: NVIDIA GB200 (00000018:01:00); rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 3Mon Jul 20 09:42:51 2026<stdout>: (hostA): device 3: NVIDIA GB200 (00000019:01:00); rackGuid: 111
Mon Jul 20 09:42:51 2026<stdout>:Process 4 (hostB): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 222
Mon Jul 20 09:42:51 2026<stdout>:Process 5 (hostB): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 222
Mon Jul 20 09:42:51 2026<stdout>:Process 6Mon Jul 20 09:42:51 2026<stdout>: (hostB): device 2: NVIDIA GB200 (00000018:01:00); rackGuid: 222
Mon Jul 20 09:42:51 2026<stdout>:Process 7 (hostB): device Mon Jul 20 09:42:51 2026<stdout>:3: NVIDIA GB200 (00000019:01:00); rackGuid: 222

Mon Jul 20 09:42:52 2026<stdout>:Running gpu_to_rack_test
Mon Jul 20 09:42:53 2026<stdout>:Min gpu_to_rack_testMon Jul 20 09:42:53 2026<stdout>: 165.15 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Avg gpu_to_rack_test 166.11 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Max gpu_to_rack_test 167.07 GB/s
Mon Jul 20 09:42:53 2026<stdout>:Total gpu_to_rack_test 1329.12 GB/s
Mon Jul 20 09:42:54 2026<stdout>:ExecutionTime gpu_to_rack_test 1.5 s
Mon Jul 20 09:42:54 2026<stdout>:Done gpu_to_rack_test
)";

// Stderr-only output: errors without any stdout process lines.
// [hostname:N] where N is the local CUDA device index.
const std::string nvloom_stderr_only = R"([err-host:0]: [CUDA_ERROR_UNKNOWN] fatal error
[err-host:1]: [CUDA_ERROR_OUT_OF_MEMORY] allocation failed
)";

// Mixed known/unknown devices: device 0 has a Process line, device 99 does not.
// [hostname:N] where N is the local CUDA device index.
const std::string nvloom_mixed_mapped_unmapped = R"(nvloom_cli 2.0.0
Process 0 (mix-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (mix-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running test_mix
[mix-host:0]: GPU memory error on device 0
[mix-host:99]: unknown device crashed
)";

// Incomplete topology: output has only 1 of 2 expected hosts
const std::string nvloom_incomplete_topology = R"(nvloom_cli 2.0.0
Process 0 (host-present): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (host-present): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running gpu_test
Min gpu_test 150.0 GB/s
Avg gpu_test 155.0 GB/s
Max gpu_test 160.0 GB/s
Total gpu_test 310.0 GB/s

ExecutionTime gpu_test 5.0 s
Done gpu_test
)";

// Multi-node output: 4 hosts
const std::string nvloom_4node = R"(nvloom_cli 2.0.0
Process 0 (rack1-node1): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 1001
Process 1 (rack1-node1): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 1001
Process 2 (rack1-node2): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 1002
Process 3 (rack1-node2): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 1002
Process 4 (rack2-node1): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 2001
Process 5 (rack2-node1): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 2001
Process 6 (rack2-node2): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 2002
Process 7 (rack2-node2): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 2002

Running fabric_test
Min fabric_test 150.0 GB/s
Avg fabric_test 155.0 GB/s
Max fabric_test 160.0 GB/s
Total fabric_test 1240.0 GB/s

ExecutionTime fabric_test 10.0 s
Done fabric_test
)";

// Output with benign cuGetExportTable errors that should be skipped
const std::string nvloom_benign_stderr = R"(nvloom_cli 2.0.0
Process 0 (benign-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (benign-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running test_benign
Min test_benign 100.0 GB/s
Avg test_benign 100.0 GB/s
Max test_benign 100.0 GB/s
Total test_benign 200.0 GB/s
Done test_benign
[benign-host:0]: [09:42:51.504][1068900][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuGetExportTable
[benign-host:1]: [09:42:51.872][1068900][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuGetExportTable
)";

const std::string nvloom_all_waived = R"(nvloom_cli 2.0.0
Process 0 (waived-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (waived-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running multicast_waived
testcase WAIVED
Done multicast_waived
)";

const std::string nvloom_all_waived_without_process_records = R"(nvloom_cli 2.0.0
Running multicast_waived
testcase WAIVED
Done multicast_waived
)";

const std::string nvloom_mixed_measured_and_waived = R"(nvloom_cli 2.0.0
Process 0 (mixed-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (mixed-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running multicast_waived
testcase WAIVED
Done multicast_waived
Running multicast_measured
Min multicast_measured 100.0 GB/s
Avg multicast_measured 100.0 GB/s
Max multicast_measured 100.0 GB/s
Total multicast_measured 200.0 GB/s
Done multicast_measured
)";

const std::string nvloom_all_na_without_waiver = R"(nvloom_cli 2.0.0
Process 0 (na-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (na-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running rack_to_rack_test
Min rack_to_rack_test N/A GB/s
Avg rack_to_rack_test N/A GB/s
Max rack_to_rack_test N/A GB/s
Total rack_to_rack_test N/A GB/s
Done rack_to_rack_test
)";

const std::string nvloom_mixed_waived_and_na = R"(nvloom_cli 2.0.0
Process 0 (mixed-na-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (mixed-na-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running multicast_waived
testcase WAIVED
Done multicast_waived
Running rack_to_rack_na
Min rack_to_rack_na N/A GB/s
Avg rack_to_rack_na N/A GB/s
Max rack_to_rack_na N/A GB/s
Total rack_to_rack_na N/A GB/s
Done rack_to_rack_na
)";

const std::string nvloom_waived_with_numeric_summary_named_na = R"(nvloom_cli 2.0.0
Process 0 (named-na-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (named-na-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running multicast_waived
testcase WAIVED
Min multicast N/A measured 100.0 GB/s
Avg multicast N/A measured 100.0 GB/s
Max multicast N/A measured 100.0 GB/s
Total multicast N/A measured 200.0 GB/s
Done multicast_waived
)";

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: all-pass 2-node [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_success_2node);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["compute05"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["compute04"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->version == dcgmMnDiagResponse_version2);
    REQUIRE(responseStruct->numTests == 1);
    REQUIRE(std::string(responseStruct->tests[0].name) == "NVLOOM");
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);

    REQUIRE(responseStruct->numHosts == 2);
    REQUIRE(responseStruct->numResults == 8);
    REQUIRE(responseStruct->numEntities == 8);
    REQUIRE(responseStruct->numErrors == 0);

    // Verify info messages captured summary stats (2 sub-tests x 4 lines each = 8)
    CHECK(responseStruct->numInfos == 8);
    CHECK(responseStruct->tests[0].numInfo == 8);
    for (unsigned char i = 0; i < responseStruct->tests[0].numInfo; i++)
    {
        CHECK(responseStruct->tests[0].infoIndices[i] == i);
    }

    // All entity results should be PASS
    for (unsigned int i = 0; i < responseStruct->numResults; i++)
    {
        CHECK(responseStruct->results[i].result == DCGM_DIAG_RESULT_PASS);
        CHECK(responseStruct->results[i].entity.entityGroupId == DCGM_FE_GPU);
    }

    // Verify host info populated from nodeInfo
    std::set<std::string> foundHostnames;
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        foundHostnames.insert(responseStruct->hosts[i].hostname);
        CHECK(responseStruct->hosts[i].numEntities == 4);
    }
    CHECK(foundHostnames.contains("compute05"));
    CHECK(foundHostnames.contains("compute04"));

    // Verify result indices
    CHECK(responseStruct->tests[0].numResults == 8);
    for (unsigned int i = 0; i < responseStruct->tests[0].numResults; i++)
    {
        CHECK(responseStruct->tests[0].resultIndices[i] == i);
    }
}

SCENARIO("MnDiagMpiNvloomRunner: reports explicit waived-only coverage as skipped")
{
    GIVEN("completed NVLoom output containing only waived executions")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_all_waived);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["waived-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("the test and every entity are skipped without errors")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_SKIP);
                CHECK(response->numErrors == 0);
                REQUIRE(response->numResults == 2);
                for (auto const &entityResult : std::span(response->results, response->numResults))
                {
                    CHECK(entityResult.result == DCGM_DIAG_RESULT_SKIP);
                }
            }
        }
    }

    GIVEN("completed waived output without host process records")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_all_waived_without_process_records);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["expected-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("the missing-host failure is not overwritten by the waiver classification")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_FAIL);
                CHECK(response->numResults == 0);
                REQUIRE(response->numErrors == 1);
                CHECK(std::string_view(response->errors[0].msg).find("No hosts detected") != std::string_view::npos);
            }
        }
    }
}

SCENARIO("MnDiagMpiNvloomRunner: preserves passing measured coverage")
{
    GIVEN("completed NVLoom output containing measured and waived executions")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_mixed_measured_and_waived);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["mixed-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("the measured coverage remains passed")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
                CHECK(response->numErrors == 0);
                REQUIRE(response->numResults == 2);
                for (auto const &entityResult : std::span(response->results, response->numResults))
                {
                    CHECK(entityResult.result == DCGM_DIAG_RESULT_PASS);
                }
            }
        }
    }

    GIVEN("completed NVLoom output containing N/A summaries without an explicit waiver")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_all_na_without_waiver);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["na-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("N/A alone does not change the existing pass policy")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
                CHECK(response->numErrors == 0);
                REQUIRE(response->numResults == 2);
                for (auto const &entityResult : std::span(response->results, response->numResults))
                {
                    CHECK(entityResult.result == DCGM_DIAG_RESULT_PASS);
                }
            }
        }
    }

    GIVEN("completed NVLoom output containing both a waiver and a separate N/A execution")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_mixed_waived_and_na);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["mixed-na-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("a partial waiver does not classify every completed execution as skipped")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
                CHECK(response->numErrors == 0);
                REQUIRE(response->numResults == 2);
                for (auto const &entityResult : std::span(response->results, response->numResults))
                {
                    CHECK(entityResult.result == DCGM_DIAG_RESULT_PASS);
                }
            }
        }
    }

    GIVEN("a numeric summary whose test name contains N/A")
    {
        MnDiagMpiNvloomRunnerTests runner;
        auto response = CreateTestResponseStruct();
        int dataFd    = CreateMemFdFromString(nvloom_waived_with_numeric_summary_named_na);
        REQUIRE(dataFd >= 0);
        DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

        nodeInfoMap_t nodeInfo;
        nodeInfo["named-na-host"] = { .dcgmVersion = "4.7.0", .driverVersion = "580.186" };

        WHEN("the output is parsed")
        {
            auto const result = runner.MnDiagOutputCallback(dataFd, response.get(), nodeInfo);

            THEN("the captured numeric value prevents a false skipped result")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(response->tests[0].result == DCGM_DIAG_RESULT_PASS);
                CHECK(response->numErrors == 0);
                CHECK(response->numInfos == 4);
            }
        }
    }
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: failure with stderr errors [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_failure_with_errors);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["node01"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["node02"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    REQUIRE(responseStruct->numHosts == 2);
    REQUIRE(responseStruct->numResults == 4);
    REQUIRE(responseStruct->numErrors == 2);

    // Verify errors attributed to correct entities
    std::map<std::string, std::string> entityErrors;
    for (unsigned int i = 0; i < responseStruct->numErrors; i++)
    {
        auto const &error = responseStruct->errors[i];
        std::string host  = responseStruct->hosts[error.hostId].hostname;
        std::string key   = host + ":" + std::to_string(error.entity.entityId);
        entityErrors[key] = error.msg;
    }
    CHECK(entityErrors.contains("node01:0"));
    CHECK(entityErrors.contains("node02:0"));
    CHECK(std::string_view(entityErrors["node01:0"]).find("CUDA_ERROR_UNKNOWN") != std::string_view::npos);

    // Verify entities without errors are PASS
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
    CHECK(passCount == 2);
    CHECK(failCount == 2);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: mixed results [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_mixed_results);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["host-a"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["host-b"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    REQUIRE(responseStruct->numHosts == 2);
    REQUIRE(responseStruct->numResults == 4);
    REQUIRE(responseStruct->numErrors == 1);

    // Info messages from the passing sub-test should be captured
    CHECK(responseStruct->numInfos > 0);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: empty output with expected hosts [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString("");
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["expected-host-1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["expected-host-2"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    REQUIRE(responseStruct->numErrors == 1);
    CHECK(std::string_view(responseStruct->errors[0].msg).find("No hosts detected") != std::string_view::npos);
    CHECK(responseStruct->errors[0].code == DCGM_FR_INTERNAL);
    CHECK(responseStruct->numHosts == 0);
    CHECK(responseStruct->numResults == 0);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: empty output no expected hosts [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString("");
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfoMap_t());

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    CHECK(responseStruct->numHosts == 0);
    CHECK(responseStruct->numResults == 0);
    CHECK(responseStruct->numErrors == 0);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: malformed input [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    std::string garbage = "totally garbage input\n"
                          "this is not nvloom output at all\n"
                          "random numbers 12345 67890\n"
                          "Process fake line without proper format\n";

    int dataFd = CreateMemFdFromString(garbage);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["expected-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    CHECK(responseStruct->numErrors == 1);
    CHECK(std::string_view(responseStruct->errors[0].msg).find("No hosts detected") != std::string_view::npos);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: ANSI escape codes stripped [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_ansi_codes);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["ansi-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 1);
    CHECK(std::string(responseStruct->hosts[0].hostname) == "ansi-host");
    REQUIRE(responseStruct->numResults == 2);
    CHECK(responseStruct->numErrors == 0);
    CHECK(responseStruct->numInfos > 0);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: interleaved MPI timestamps [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_interleaved_timestamps);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["hostX"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["hostY"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 2);
    REQUIRE(responseStruct->numResults == 4);
    CHECK(responseStruct->numErrors == 0);

    // Verify timestamps were stripped and hosts were properly detected
    std::set<std::string> foundHostnames;
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        foundHostnames.insert(responseStruct->hosts[i].hostname);
    }
    CHECK(foundHostnames.contains("hostX"));
    CHECK(foundHostnames.contains("hostY"));

    // Verify info messages captured (Min/Avg/Max/Total lines)
    CHECK(responseStruct->numInfos > 0);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: timestamps at multiple split points [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_timestamp_multipoint_splits);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["hostA"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["hostB"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    CHECK(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 2);
    REQUIRE(responseStruct->numResults == 8);
    CHECK(responseStruct->numErrors == 0);

    std::set<std::string> foundHostnames;
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        foundHostnames.insert(responseStruct->hosts[i].hostname);
        CHECK(responseStruct->hosts[i].numEntities == 4);
    }
    CHECK(foundHostnames.contains("hostA"));
    CHECK(foundHostnames.contains("hostB"));
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: OpenMPI rank/tag prefixes [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_mpi_rank_prefixed);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["mpi-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 1);
    REQUIRE(responseStruct->numResults == 2);
    CHECK(responseStruct->numErrors == 0);

    CHECK(std::string(responseStruct->hosts[0].hostname) == "mpi-host");

    for (unsigned int i = 0; i < responseStruct->numResults; i++)
    {
        CHECK(responseStruct->results[i].result == DCGM_DIAG_RESULT_PASS);
    }

    CHECK(responseStruct->numInfos == 4);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: OpenMPI 5 timestamp prefixes [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_openmpi5_timestamp_prefixed);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["localhost"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 1);
    REQUIRE(responseStruct->numResults == 2);
    CHECK(responseStruct->numErrors == 0);
    CHECK(responseStruct->numInfos == 4);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: partial output fails closed [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    // "Running" without "Done" -- fail-closed: no completed subtests detected
    std::string partialOutput = R"(nvloom_cli 2.0.0
Process 0 (partial-host): device 0: NVIDIA GB200 (00000008:01:00); rackGuid: 111
Process 1 (partial-host): device 1: NVIDIA GB200 (00000009:01:00); rackGuid: 111

Running gpu_to_rack_test
0	165.15
1	165.28
)";

    int dataFd = CreateMemFdFromString(partialOutput);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["partial-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    REQUIRE(responseStruct->numHosts == 1);
    REQUIRE(responseStruct->numResults == 2);
    REQUIRE(responseStruct->numErrors >= 1);
    bool foundNoSubtestsError = false;
    for (unsigned int i = 0; i < responseStruct->numErrors; i++)
    {
        if (std::string_view(responseStruct->errors[i].msg).find("No completed subtests") != std::string_view::npos)
        {
            foundNoSubtestsError = true;
        }
    }
    CHECK(foundNoSubtestsError);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: stderr-only output [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_stderr_only);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["err-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    CHECK(responseStruct->numHosts == 1);
    // With direct device attribution, both devices appear as entities
    CHECK(responseStruct->numResults == 2);
    REQUIRE(responseStruct->numErrors == 2);

    std::map<unsigned int, std::string> deviceErrors;
    for (unsigned int i = 0; i < responseStruct->numErrors; i++)
    {
        auto const &error = responseStruct->errors[i];
        CHECK(error.entity.entityGroupId == DCGM_FE_GPU);
        CHECK(error.code == DCGM_FR_UNKNOWN);
        deviceErrors[error.entity.entityId] = error.msg;
    }
    CHECK(std::string_view(deviceErrors[0]).find("CUDA_ERROR_UNKNOWN") != std::string_view::npos);
    CHECK(std::string_view(deviceErrors[1]).find("CUDA_ERROR_OUT_OF_MEMORY") != std::string_view::npos);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: mixed known and unknown devices [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_mixed_mapped_unmapped);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["mix-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    CHECK(responseStruct->numHosts == 1);
    // 3 entities: device 0 and 1 from Process lines, device 99 from stderr
    REQUIRE(responseStruct->numResults == 3);
    REQUIRE(responseStruct->numErrors == 2);

    bool foundDevice0Error  = false;
    bool foundDevice99Error = false;
    for (unsigned int i = 0; i < responseStruct->numErrors; i++)
    {
        auto const &err = responseStruct->errors[i];
        CHECK(err.entity.entityGroupId == DCGM_FE_GPU);
        if (err.entity.entityId == 0)
        {
            foundDevice0Error = true;
            CHECK(std::string_view(err.msg).find("GPU memory error") != std::string_view::npos);
        }
        else if (err.entity.entityId == 99)
        {
            foundDevice99Error = true;
            CHECK(std::string_view(err.msg).find("unknown device crashed") != std::string_view::npos);
        }
    }
    CHECK(foundDevice0Error);
    CHECK(foundDevice99Error);

    // Device 1 had no errors, should be PASS
    for (unsigned int i = 0; i < responseStruct->numResults; i++)
    {
        auto const &res = responseStruct->results[i];
        if (res.entity.entityId == 0 || res.entity.entityId == 99)
        {
            CHECK(res.result == DCGM_DIAG_RESULT_FAIL);
        }
        else if (res.entity.entityId == 1)
        {
            CHECK(res.result == DCGM_DIAG_RESULT_PASS);
        }
    }
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: incomplete topology [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_incomplete_topology);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["host-present"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["host-missing"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL);
    CHECK(responseStruct->numHosts == 1);
    CHECK(responseStruct->numResults == 2);

    bool foundMissingHostError = false;
    for (unsigned int i = 0; i < responseStruct->numErrors; i++)
    {
        auto const &error = responseStruct->errors[i];
        if (error.entity.entityGroupId == DCGM_FE_NONE && error.code == DCGM_FR_INTERNAL)
        {
            if (std::string_view(error.msg).find("host-missing") != std::string_view::npos)
            {
                foundMissingHostError = true;
            }
        }
    }
    CHECK(foundMissingHostError);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: multi-node 4 hosts [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_4node);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["rack1-node1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["rack1-node2"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["rack2-node1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["rack2-node2"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    REQUIRE(responseStruct->numHosts == 4);
    REQUIRE(responseStruct->numResults == 8);
    CHECK(responseStruct->numErrors == 0);

    std::set<std::string> foundHostnames;
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        foundHostnames.insert(responseStruct->hosts[i].hostname);
        CHECK(responseStruct->hosts[i].numEntities == 2);
        CHECK(std::string(responseStruct->hosts[i].dcgmVersion) == "4.0.0");
        CHECK(std::string(responseStruct->hosts[i].driverVersion) == "615.27");
    }
    CHECK(foundHostnames.contains("rack1-node1"));
    CHECK(foundHostnames.contains("rack1-node2"));
    CHECK(foundHostnames.contains("rack2-node1"));
    CHECK(foundHostnames.contains("rack2-node2"));

    for (unsigned int i = 0; i < responseStruct->numResults; i++)
    {
        CHECK(responseStruct->results[i].result == DCGM_DIAG_RESULT_PASS);
    }
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: benign cuGetExportTable errors skipped [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_benign_stderr);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["benign-host"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    CHECK(responseStruct->numErrors == 0);
    CHECK(responseStruct->numResults == 2);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: null response struct returns BADPARAM [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;

    int dataFd = CreateMemFdFromString("test output");
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, nullptr, nodeInfoMap_t());
    CHECK(result == DCGM_ST_BADPARAM);
}

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: entity-host version info populated [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_success_2node);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    nodeInfoMap_t nodeInfo;
    nodeInfo["compute05"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["compute04"] = { .dcgmVersion = "3.9.0", .driverVersion = "600.10" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);
    REQUIRE(result == DCGM_ST_OK);

    // Find each host and verify version info
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        std::string hostname = responseStruct->hosts[i].hostname;
        if (hostname == "compute05")
        {
            CHECK(std::string(responseStruct->hosts[i].dcgmVersion) == "4.0.0");
            CHECK(std::string(responseStruct->hosts[i].driverVersion) == "615.27");
        }
        else if (hostname == "compute04")
        {
            CHECK(std::string(responseStruct->hosts[i].dcgmVersion) == "3.9.0");
            CHECK(std::string(responseStruct->hosts[i].driverVersion) == "600.10");
        }
    }
}

TEST_CASE("MnDiagMpiNvloomRunner ResolveTestBinaryPath [mndiag]")
{
    SECTION("Should use default path when no environment variable is set")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), std::nullopt);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("nvloom") != std::string::npos);
    }

    SECTION("Should use custom path when environment variable points to valid executable")
    {
        std::string customPath = "/bin/true";
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), customPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path == customPath);
    }

    SECTION("Should fallback to default path when environment variable is empty")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), std::string(""));

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("nvloom") != std::string::npos);
    }

    SECTION("Should fallback to default path when environment variable points to non-existent file")
    {
        std::string nonExistentPath = "/path/to/nonexistent/binary";
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), nonExistentPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_OK);
        CHECK(path.find("cuda12") != std::string::npos);
        CHECK(path.find("nvloom") != std::string::npos);
    }

    SECTION("CUDA version is zero with no environment variable, default path should fail")
    {
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), std::nullopt);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(0);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_NO_DATA);
        CHECK(path.empty());

        result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_NO_DATA);
    }

    SECTION("Should fail when environment variable path is too long")
    {
        std::string longPath(DCGM_MAX_STR_LENGTH + 10, 'a');
        ScopedEnvVars envVars(MnDiagConstants::ENV_NVLOOM_PATH.data(), longPath);

        auto mockCoreProxy = std::make_unique<MockDcgmCoreProxy>();
        mockCoreProxy->SetMockCudaVersion(12000);
        MnDiagMpiNvloomRunner runner(*mockCoreProxy, EFFECTIVE_UID);

        std::string path;
        dcgmReturn_t result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_BADPARAM);
        CHECK(path.empty());

        result = runner.GetTestBinaryPath(path);
        CHECK(result == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("MnDiagMpiNvloomRunner MPI command construction [mndiag]")
{
    SECTION("ConstructMpiCommand produces command with nvloom binary, hostfile, and translated params")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ConstructMpiCommand(&diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        REQUIRE_FALSE(cmdArgs.empty());
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--hostfile"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--oversubscribe"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--timestamp-output"));
        REQUIRE(CommandLineArgUtils::ContainsArg(cmdArgs, "--merge-stderr-to-stdout"));

        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "-np") == "9");
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--map-by") == "ppr:3:node");

        auto binaryIt = std::find(cmdArgs.begin(), cmdArgs.end(), "/fake/test/nvloom");
        REQUIRE(binaryIt != cmdArgs.end());

        constexpr unsigned int timeToRun        = 300;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int gpuToRackCount   = 8;
        constexpr unsigned int expectedDuration = (timeToRun - margin) / gpuToRackCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == "gpu-to-rack");
    }

    SECTION("ConstructMpiCommand with null parameters produces empty command")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;

        mpiRunner.ConstructMpiCommand(nullptr);

        auto const &command = mpiRunner.GetLastCommand();
        REQUIRE(command.empty());
    }

    SECTION("ParseDcgmMnDiagToMpiCommand translates time_to_run to --duration")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int timeToRun        = 300;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int gpuToRackCount   = 8;
        constexpr unsigned int expectedDuration = (timeToRun - margin) / gpuToRackCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--time_to_run"));
    }

    SECTION("ParseDcgmMnDiagToMpiCommand ignores mnubergemm-prefixed parameters")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "mnubergemm.workload=GN");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        REQUIRE_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--workload"));
    }

    SECTION("ParseDcgmMnDiagToMpiCommand writes hostfile with correct hosts")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

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

    SECTION("ParseDcgmMnDiagToMpiCommand with empty host list produces no command")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        for (int i = 0; i < DCGM_MNDIAG_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();
        CHECK(cmdArgs.empty());
    }

    SECTION("Default parameters produce correct nvloom_cli flags")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        for (int i = 0; i < DCGM_MAX_TEST_PARMS; i++)
        {
            diagStruct.testParms[i][0] = '\0';
        }

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int defaultTimeToRun = 3600;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int gpuToRackCount   = 8;
        constexpr unsigned int expectedDuration = (defaultTimeToRun - margin) / gpuToRackCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == "gpu-to-rack");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--time_to_run"));
    }

    SECTION("User-supplied suite overrides default")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.suite=pairwise");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == "pairwise");
    }

    SECTION("buffer_size is translated to --bufferSize")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.buffer_size=1G");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--bufferSize") == "1G");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--buffer_size"));
    }

    SECTION("rich_output flag is translated to --richOutput")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.rich_output");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--richOutput"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--rich_output"));
    }

    SECTION("iterations is passed through as --iterations")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.iterations=32");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--iterations") == "32");
    }

    SECTION("testcase is passed through as --testcase")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.testcase=bisect_device_to_device_write_sm");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testcase") == "bisect_device_to_device_write_sm");
    }

    SECTION("Unknown nvloom parameter is removed from command")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[1], "nvloom.unknown_param=42");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--unknown_param"));
    }

    SECTION("NET_link_order is not injected for nvloom even with odd process count")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct           = CreateNvloomTestDiagStruct();
        diagStruct.hostList[2][0] = '\0'; // 2 hosts * 3 GPUs = 6 processes (even), change to odd

        SafeCopyTo(diagStruct.hostList[0], "host1");
        diagStruct.hostList[1][0] = '\0'; // 1 host * 3 GPUs = 3 processes (odd)

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--NET_link_order"));
    }

    SECTION("Multiple nvloom parameters combined")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=600");
        SafeCopyTo(diagStruct.testParms[1], "nvloom.suite=fabric-stress");
        SafeCopyTo(diagStruct.testParms[2], "nvloom.buffer_size=256M");
        SafeCopyTo(diagStruct.testParms[3], "nvloom.iterations=32");
        SafeCopyTo(diagStruct.testParms[4], "nvloom.rich_output");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int timeToRun         = 600;
        constexpr unsigned int margin            = 30;
        constexpr unsigned int fabricStressCount = 4;
        constexpr unsigned int expectedDuration  = (timeToRun - margin) / fabricStressCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == "fabric-stress");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--bufferSize") == "256M");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--iterations") == "32");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--richOutput"));
    }

    SECTION("Duration computation: testcase specified uses N=1")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=300");
        SafeCopyTo(diagStruct.testParms[1], "nvloom.testcase=gpu_to_rack_device_to_device_bidir_read_ce");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int timeToRun        = 300;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int singleTestcase   = 1;
        constexpr unsigned int expectedDuration = (timeToRun - margin) / singleTestcase;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--testcase") == "gpu_to_rack_device_to_device_bidir_read_ce");
    }

    SECTION("Duration computation: unknown suite uses conservative fallback")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=600");
        SafeCopyTo(diagStruct.testParms[1], "nvloom.suite=future-suite");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int timeToRun        = 600;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int fallbackCount    = 8;
        constexpr unsigned int expectedDuration = (timeToRun - margin) / fallbackCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
    }

    SECTION("Duration computation: small time_to_run clamps duration to 1")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=10");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int minDuration = 1;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(minDuration));
    }

    SECTION("Duration computation: pairwise suite with 8 testcases")
    {
        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        SafeCopyTo(diagStruct.testParms[0], "nvloom.time_to_run=500");
        SafeCopyTo(diagStruct.testParms[1], "nvloom.suite=pairwise");

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");

        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        constexpr unsigned int timeToRun        = 500;
        constexpr unsigned int margin           = 30;
        constexpr unsigned int pairwiseCount    = 8;
        constexpr unsigned int expectedDuration = (timeToRun - margin) / pairwiseCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
    }

    SECTION("Duration computation uses NVLoom 1.4.1 and 2.0.0 shared suite testcase counts")
    {
        auto const [suite, testcaseCount] = GENERATE(table<std::string_view, unsigned int>({
            { "gpu-to-rack", 8u },
            { "pairwise", 8u },
            { "pairwise-tma", 4u },
            { "fabric-stress", 4u },
            { "all-to-one", 4u },
            { "multicast", 8u },
            { "multicast-tma", 4u },
            { "egm", 8u },
            { "rack-to-rack", 8u },
            { "latency", 2u },
        }));
        CAPTURE(suite, testcaseCount);

        constexpr unsigned int timeToRun = 510;
        constexpr unsigned int margin    = 30;

        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        auto const timeToRunParameter = fmt::format("nvloom.time_to_run={}", timeToRun);
        auto const suiteParameter     = fmt::format("nvloom.suite={}", suite);
        SafeCopyTo(diagStruct.testParms[0], timeToRunParameter.c_str());
        SafeCopyTo(diagStruct.testParms[1], suiteParameter.c_str());

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs                        = mpiRunner.GetLastCommandArgs();
        unsigned int const expectedDuration = (timeToRun - margin) / testcaseCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == suite);
    }

    SECTION("Duration computation uses NVLoom 2.0.0-only suite testcase counts")
    {
        auto const [suite, testcaseCount] = GENERATE(table<std::string_view, unsigned int>({
            { "rack-aware-all-to-one", 4u },
            { "egm-gpu-to-rack", 8u },
        }));
        CAPTURE(suite, testcaseCount);

        constexpr unsigned int timeToRun = 510;
        constexpr unsigned int margin    = 30;

        MnDiagMpiNvloomRunnerTests mpiRunner;
        auto diagStruct = CreateNvloomTestDiagStruct();

        auto const timeToRunParameter = fmt::format("nvloom.time_to_run={}", timeToRun);
        auto const suiteParameter     = fmt::format("nvloom.suite={}", suite);
        SafeCopyTo(diagStruct.testParms[0], timeToRunParameter.c_str());
        SafeCopyTo(diagStruct.testParms[1], suiteParameter.c_str());

        ScopedEnvVars envVars("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun");
        mpiRunner.ParseDcgmMnDiagToMpiCommand(diagStruct);

        auto cmdArgs                        = mpiRunner.GetLastCommandArgs();
        unsigned int const expectedDuration = (timeToRun - margin) / testcaseCount;
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--duration") == std::to_string(expectedDuration));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--suite") == suite);
    }
}

SCENARIO("MnDiagMpiNvloomRunner: AllowedParamNames completeness")
{
    GIVEN("the set of allowed nvloom parameter names")
    {
        auto const &allowed = MnDiagMpiNvloomRunner::AllowedParamNames();

        THEN("it contains all documented parameters")
        {
            CHECK(allowed.contains("time_to_run"));
            CHECK(allowed.contains("suite"));
            CHECK(allowed.contains("testcase"));
            CHECK(allowed.contains("buffer_size"));
            CHECK(allowed.contains("iterations"));
            CHECK(allowed.contains("rich_output"));
            CHECK(allowed.contains("allocation_strategy"));
        }

        THEN("it does not contain mnubergemm-specific parameters")
        {
            CHECK_FALSE(allowed.contains("workload"));
            CHECK_FALSE(allowed.contains("NET_link_order"));
            CHECK_FALSE(allowed.contains("dynamic_adj"));
        }
    }
}

// ---------------------------------------------------------------------------
// BuildHostAliasMap tests
// ---------------------------------------------------------------------------

TEST_CASE("BuildHostAliasMap: exact match returns empty map [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["compute05"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["compute04"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "compute04", "compute05" };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo);
    CHECK(aliasMap.empty());
}

TEST_CASE("BuildHostAliasMap: alias resolved via hostname-to-IP [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["10.0.0.1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["10.0.0.2"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "host-a.example.com", "host-b.example.com" };

    auto mockResolver = [](std::string const &host) -> std::string {
        if (host == "host-a.example.com")
        {
            return "10.0.0.1";
        }
        if (host == "host-b.example.com")
        {
            return "10.0.0.2";
        }
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, mockResolver);
    REQUIRE(aliasMap.size() == 2);
    CHECK(aliasMap.at("host-a.example.com") == "10.0.0.1");
    CHECK(aliasMap.at("host-b.example.com") == "10.0.0.2");
}

TEST_CASE("BuildHostAliasMap: partial exact match plus alias [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["compute05"]   = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["192.168.1.2"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "compute05", "compute06" };

    auto mockResolver = [](std::string const &host) -> std::string {
        if (host == "compute06")
        {
            return "192.168.1.2";
        }
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, mockResolver);
    REQUIRE(aliasMap.size() == 1);
    CHECK(aliasMap.at("compute06") == "192.168.1.2");
}

TEST_CASE("BuildHostAliasMap: ambiguous mapping produces no alias [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["10.0.0.1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["10.0.0.1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "host-a.example.com", "host-b.example.com" };

    // Both output hosts resolve to the same IP — ambiguous for the second one
    auto mockResolver = [](std::string const &host) -> std::string {
        if (host == "host-a.example.com" || host == "host-b.example.com")
        {
            return "10.0.0.1";
        }
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, mockResolver);
    // First one claims 10.0.0.1; second finds it already claimed — only one alias
    REQUIRE(aliasMap.size() == 1);
}

TEST_CASE("BuildHostAliasMap: unresolvable output host produces no alias [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["10.0.0.1"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "unresolvable.invalid" };

    // Resolver returns original hostname on failure — won't match any expected IP
    auto mockResolver = [](std::string const &host) -> std::string {
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, mockResolver);
    CHECK(aliasMap.empty());
}

TEST_CASE("BuildHostAliasMap: case-insensitive match without DNS [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["Compute05"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "compute05" };

    auto noopResolver = [](std::string const &host) -> std::string {
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, noopResolver);
    REQUIRE(aliasMap.size() == 1);
    CHECK(aliasMap.at("compute05") == "Compute05");
}

TEST_CASE("BuildHostAliasMap: trailing dot normalization [mndiag]")
{
    nodeInfoMap_t nodeInfo;
    nodeInfo["compute05.example.com."] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };

    std::set<std::string> outputHosts = { "compute05.example.com" };

    auto noopResolver = [](std::string const &host) -> std::string {
        return host;
    };

    auto aliasMap = BuildHostAliasMap(outputHosts, nodeInfo, noopResolver);
    REQUIRE(aliasMap.size() == 1);
    CHECK(aliasMap.at("compute05.example.com") == "compute05.example.com.");
}

// ---------------------------------------------------------------------------
// Nvloom parser: case-insensitive host alias handling
// ---------------------------------------------------------------------------

TEST_CASE("MnDiagMpiNvloomRunner ParseTestOutput: case-insensitive host alias passes [mndiag]")
{
    MnDiagMpiNvloomRunnerTests runner;
    auto responseStruct = CreateTestResponseStruct();

    int dataFd = CreateMemFdFromString(nvloom_success_2node);
    REQUIRE(dataFd >= 0);
    DcgmNs::Defer closeFd([dataFd] { close(dataFd); });

    // nodeInfo keyed by uppercase hostnames while nvloom output reports lowercase
    // (compute04, compute05). The alias map resolves these via case-insensitive
    // normalized exact match, without DNS resolution.
    nodeInfoMap_t nodeInfo;
    nodeInfo["COMPUTE05"] = { .dcgmVersion = "4.0.0", .driverVersion = "615.27" };
    nodeInfo["COMPUTE04"] = { .dcgmVersion = "3.9.0", .driverVersion = "600.10" };

    dcgmReturn_t result = runner.MnDiagOutputCallback(dataFd, responseStruct.get(), nodeInfo);

    REQUIRE(result == DCGM_ST_OK);
    CHECK(responseStruct->tests[0].result == DCGM_DIAG_RESULT_PASS);
    CHECK(responseStruct->numHosts == 2);
    CHECK(responseStruct->numErrors == 0);

    // Verify host metadata was populated via alias lookup
    for (unsigned int i = 0; i < responseStruct->numHosts; i++)
    {
        std::string hostname = responseStruct->hosts[i].hostname;
        CHECK_FALSE(hostname.empty());
        CHECK(std::string(responseStruct->hosts[i].dcgmVersion) != "unknown");
        CHECK(std::string(responseStruct->hosts[i].driverVersion) != "See nvidia-smi");
    }
}

TEST_CASE("StripMpiTimestamps preserves content whitespace [mndiag]")
{
    SECTION("Line-leading timestamp is stripped cleanly")
    {
        auto result = StripMpiTimestamps("Mon Jul 20 09:42:51 2026<stdout>:Running gpu_test");
        CHECK(result == "Running gpu_test");
    }

    SECTION("Mid-line timestamp deleted preserves surrounding content whitespace")
    {
        auto result = StripMpiTimestamps(
            "Mon Jul 20 09:42:51 2026<stdout>:Process 3Mon Jul 20 09:42:51 2026<stdout>: (hostA): device 0");
        CHECK(result == "Process 3 (hostA): device 0");
    }

    SECTION("Mid-line timestamp between device keyword and index")
    {
        auto result
            = StripMpiTimestamps("Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostA): device Mon Jul 20 09:42:51 "
                                 "2026<stdout>:3: NVIDIA GB200");
        CHECK(result == "Process 0 (hostA): device 3: NVIDIA GB200");
    }

    SECTION("Mid-line timestamp between colon and device keyword")
    {
        auto result = StripMpiTimestamps(
            "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostA):Mon Jul 20 09:42:51 2026<stdout>: device 0: GB200");
        CHECK(result == "Process 0 (hostA): device 0: GB200");
    }

    SECTION("Mid-line timestamp between hostname and close-paren")
    {
        auto result = StripMpiTimestamps(
            "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostAMon Jul 20 09:42:51 2026<stdout>:): device 0: GB200");
        CHECK(result == "Process 0 (hostA): device 0: GB200");
    }

    SECTION("Mid-line timestamp between close-paren and colon")
    {
        auto result = StripMpiTimestamps(
            "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostA)Mon Jul 20 09:42:51 2026<stdout>:: device 0: GB200");
        CHECK(result == "Process 0 (hostA): device 0: GB200");
    }

    SECTION("Mid-line timestamp between device index and colon")
    {
        auto result = StripMpiTimestamps(
            "Mon Jul 20 09:42:51 2026<stdout>:Process 0 (hostA): device 2Mon Jul 20 09:42:51 2026<stdout>:: GB200");
        CHECK(result == "Process 0 (hostA): device 2: GB200");
    }

    SECTION("No timestamp returns input unchanged")
    {
        auto result = StripMpiTimestamps("Process 0 (hostA): device 0: NVIDIA GB200");
        CHECK(result == "Process 0 (hostA): device 0: NVIDIA GB200");
    }

    SECTION("OpenMPI 5 bracket timestamp stripped from line-leading position")
    {
        auto result = StripMpiTimestamps("[Tue Aug 11 00:04:15 2026]<stdout>: Process 0 (hostA)");
        CHECK(result == "Process 0 (hostA)");
    }

    SECTION("Whitespace-only result after stripping returns empty")
    {
        auto result = StripMpiTimestamps("Mon Jul 20 09:42:51 2026<stdout>:");
        CHECK(result.empty());
    }
}
