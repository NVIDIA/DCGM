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

#include <DcgmStringHelpers.h>
#include <MnDiagMpiRunner.h>
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <dcgm_errors.h>
#include <dcgm_structs.h>
#include <optional>
#include <ranges>

#include "mocks/MockDcgmCoreProxy.h"

namespace
{
/**
 * Helper function to create a populated dcgmRunMnDiag_v1 struct for testing
 */
dcgmRunMnDiag_v1 CreateTestDiagStruct()
{
    dcgmRunMnDiag_v1 diagStruct = {};

    // Set version
    diagStruct.version = dcgmRunMnDiag_version1;

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
 * Helper function to create a populated dcgmMnDiagResponse_v1 struct for testing
 */
std::unique_ptr<dcgmMnDiagResponse_v1> CreateTestResponseStruct()
{
    auto responseUptr     = MakeUniqueZero<dcgmMnDiagResponse_v1>();
    responseUptr->version = dcgmMnDiagResponse_version1;
    return responseUptr;
}

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

class MnDiagMpiRunnerTests
{
public:
    MnDiagMpiRunnerTests()
        : m_mockCoreProxy(std::make_unique<MockDcgmCoreProxy>())
        , m_runner(*m_mockCoreProxy)
    {}

    // Expose methods for testing
    dcgmReturn_t MnDiagOutputCallback(std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &nodeInfo)
    {
        return m_runner.MnDiagOutputCallback(dataStream, responseStruct, nodeInfo);
    }

    void ConstructMpiCommand(void const *params)
    {
        m_runner.ConstructMpiCommand(params);
    }

    void ParseDcgmMnDiagToMpiCommand_v1(dcgmRunMnDiag_v1 const &drmnd)
    {
        m_runner.ParseDcgmMnDiagToMpiCommand_v1(drmnd);
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

    void SetUserInfo(std::pair<std::string, uid_t> userInfo)
    {
        m_runner.SetUserInfo(userInfo);
    }

    void SetMnubergemmPath(std::string const &mnubergemmPath)
    {
        m_runner.SetMnubergemmPath(mnubergemmPath);
    }

private:
    std::unique_ptr<MockDcgmCoreProxy> m_mockCoreProxy;
    MnDiagMpiRunner m_runner;
};

TEST_CASE("MnDiagMpiRunner command construction")
{
    SECTION("ConstructMpiCommand with valid parameters")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set environment variables to fixed values for testing
        setenv("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun", 1);
        setenv("DCGM_MNDIAG_MNUBERGEMM_PATH", "/usr/bin/test_mnubergemm", 1);

        // Construct the command
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Get the generated command as a string
        std::string commandStr = mpiRunner.GetLastCommand();

        // Verify the command is not empty
        REQUIRE_FALSE(commandStr.empty());

        // Verify expected parts are present in the command string
        REQUIRE(commandStr.find("host1,host2,host3") != std::string::npos);
        REQUIRE(commandStr.find("-np 9") != std::string::npos);
        REQUIRE(commandStr.find("--map-by ppr:3:node") != std::string::npos);
        REQUIRE(commandStr.find("--time_to_run 300") != std::string::npos);
        REQUIRE(commandStr.find("--workload GN") != std::string::npos);
        REQUIRE(commandStr.find("--dynamic_adj") != std::string::npos);
        // With 9 processes (odd), NET_link_order should be set to snake
        REQUIRE(commandStr.find("--NET_link_order snake") != std::string::npos);

        // Clean up environment
        unsetenv("DCGM_MNDIAG_MPIRUN_PATH");
        unsetenv("DCGM_MNDIAG_MNUBERGEMM_PATH");
    }

    SECTION("ConstructMpiCommand with null parameters")
    {
        MnDiagMpiRunnerTests mpiRunner;

        // This should not crash
        mpiRunner.ConstructMpiCommand(nullptr);

        // Verify the command is empty
        auto const &command = mpiRunner.GetLastCommand();
        REQUIRE(command.empty());
    }

    SECTION("ConstructMpiCommand with invalid version")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set an invalid version
        diagStruct.version = 9999;

        // This should not crash
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Verify the command is empty
        auto const &command = mpiRunner.GetLastCommand();
        REQUIRE(command.empty());
    }
}

TEST_CASE("MnDiagMpiRunner binary paths")
{
    std::string customMnubergemmPath = "/custom/path/to/mnubergemm";
    std::string customMpirunPath     = "/custom/path/to/mpirun";

    SECTION("GetMpiBinPath with custom environment")
    {
        // Set environment variable
        setenv(MnDiagConstants::ENV_MPIRUN_PATH.data(), customMpirunPath.c_str(), 1);

        MnDiagMpiRunnerTests mpiRunner;
        auto mpiPath = mpiRunner.GetMpiBinPath();

        REQUIRE(mpiPath == customMpirunPath);

        // Clean up environment
        unsetenv(MnDiagConstants::ENV_MPIRUN_PATH.data());
    }

    SECTION("GetMpiBinPath with default environment")
    {
        // Ensure the environment variable is not set
        unsetenv(MnDiagConstants::ENV_MPIRUN_PATH.data());

        MnDiagMpiRunnerTests mpiRunner;
        auto mpiPath = mpiRunner.GetMpiBinPath();

        REQUIRE(mpiPath == MnDiagConstants::DEFAULT_MPIRUN_PATH);
    }

    SECTION("Custom mnubergemm path affects command construction")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set the mnubergemm path
        mpiRunner.SetMnubergemmPath(customMnubergemmPath);

        // Construct the command
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Get the command as a string
        std::string commandStr = mpiRunner.GetLastCommand();

        // The custom mnubergemm path should be in the command
        REQUIRE(commandStr.find(customMnubergemmPath) != std::string::npos);

        // Clean up environment
        unsetenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
    }

    SECTION("Default mnubergemm path is used when environment variable is not set")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Construct the command
        mpiRunner.ConstructMpiCommand(&diagStruct);

        // Get the command as a string
        std::string commandStr = mpiRunner.GetLastCommand();

        // The default mnubergemm path should be in the command
        REQUIRE(commandStr.find(MnDiagConstants::DEFAULT_MNUBERGEMM_PATH) != std::string::npos);
    }
}

TEST_CASE("MnDiagMpiRunner output parsing")
{
    SECTION("MnDiagOutputCallback with null response struct")
    {
        MnDiagMpiRunnerTests mpiRunner;

        // This should not crash
        std::istringstream dataStream("test output");
        mpiRunner.MnDiagOutputCallback(dataStream, nullptr, nodeInfoMap_t());
    }

    SECTION("MnDiagOutputCallback with stdout output")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Call the callback with sample output
        std::istringstream dataStream("Sample mnubergemm output");
        mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfoMap_t());

        // Version should be maintained
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version1);
    }

    SECTION("MnDiagOutputCallback with invalid version")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Set an invalid version
        responseStruct->version = 9999;

        // This should not crash
        std::istringstream dataStream("test output");
        mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfoMap_t());

        // Version should be maintained
        REQUIRE(responseStruct->version == 9999);
    }
}

TEST_CASE("MnDiagMpiRunner ParseDcgmMnDiagToMpiCommand_v1")
{
    SECTION("Parse dcgmRunMnDiag_v1 with standard parameters")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set environment variables to fixed values for testing
        setenv("DCGM_MNDIAG_MPIRUN_PATH", "/usr/bin/test_mpirun", 1);
        setenv("DCGM_MNDIAG_MNUBERGEMM_PATH", "/usr/bin/test_mnubergemm", 1);

        // Call the parse function directly
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify the host list
        unsigned int hostIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "--host") - cmdArgs.begin();
        REQUIRE(hostIdx < cmdArgs.size());
        REQUIRE(cmdArgs[hostIdx + 1] == "host1,host2,host3");

        // Verify process count (should be hosts * gpus per node, which is 3*3=9)
        unsigned int npIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "-np") - cmdArgs.begin();
        REQUIRE(npIdx < cmdArgs.size());
        REQUIRE(cmdArgs[npIdx + 1] == "9"); // 3 hosts * 3 GPUs per node

        // Verify --map-by is present and configured correctly
        unsigned int mapByIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "--map-by") - cmdArgs.begin();
        REQUIRE(mapByIdx < cmdArgs.size());
        REQUIRE(cmdArgs[mapByIdx + 1] == "ppr:3:node"); // 3 GPUs per node

        // Verify workload parameter
        bool foundWorkload = false;
        for (size_t i = 0; i < cmdArgs.size(); i++)
        {
            if (cmdArgs[i] == "--workload" && i + 1 < cmdArgs.size())
            {
                REQUIRE(cmdArgs[i + 1] == "GN");
                foundWorkload = true;
                break;
            }
        }
        REQUIRE(foundWorkload);

        // Verify time_to_run parameter
        bool foundTimeToRun = false;
        for (size_t i = 0; i < cmdArgs.size(); i++)
        {
            if (cmdArgs[i] == "--time_to_run" && i + 1 < cmdArgs.size())
            {
                REQUIRE(cmdArgs[i + 1] == "300");
                foundTimeToRun = true;
                break;
            }
        }
        REQUIRE(foundTimeToRun);

        // Verify dynamic_adj flag is present
        bool foundDynamicAdj = false;
        for (auto const &arg : cmdArgs)
        {
            if (arg == "--dynamic_adj")
            {
                foundDynamicAdj = true;
                break;
            }
        }
        REQUIRE(foundDynamicAdj);

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));

        // Clean up environment
        unsetenv("DCGM_MNDIAG_MPIRUN_PATH");
        unsetenv("DCGM_MNDIAG_MNUBERGEMM_PATH");
    }

    SECTION("Parse dcgmRunMnDiag_v1 with empty host list")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Clear host list
        for (int i = 0; i < DCGM_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // The command should be created but with 0 hosts
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify process count is 0
        unsigned int npIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "-np") - cmdArgs.begin();
        REQUIRE(npIdx < cmdArgs.size());
        REQUIRE(cmdArgs[npIdx + 1] == "0");

        // Verify --map-by is present
        unsigned int mapByIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "--map-by") - cmdArgs.begin();
        REQUIRE(mapByIdx < cmdArgs.size());
        REQUIRE(cmdArgs[mapByIdx + 1] == "ppr:3:node"); // 3 GPUs per node based on MockDcgmCoreProxy

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));
    }

    SECTION("Parse dcgmRunMnDiag_v1 with hosts containing port numbers")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set hosts with port numbers
        SafeCopyTo(diagStruct.hostList[0], "host1:5555");
        SafeCopyTo(diagStruct.hostList[1], "host2:6666");
        SafeCopyTo(diagStruct.hostList[2], "host3:7777");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify the host list (ports should be stripped)
        unsigned int hostIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "--host") - cmdArgs.begin();
        REQUIRE(hostIdx < cmdArgs.size());
        REQUIRE(cmdArgs[hostIdx + 1] == "host1,host2,host3");

        // Verify process count is hosts * gpus per node (3*3=9)
        unsigned int npIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "-np") - cmdArgs.begin();
        REQUIRE(npIdx < cmdArgs.size());
        REQUIRE(cmdArgs[npIdx + 1] == "9");

        // Verify --map-by is present with correct format
        unsigned int mapByIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "--map-by") - cmdArgs.begin();
        REQUIRE(mapByIdx < cmdArgs.size());
        REQUIRE(cmdArgs[mapByIdx + 1] == "ppr:3:node");

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));
    }

    SECTION("Verify default parameters when not specified")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Clear all test parameters
        for (int i = 0; i < DCGM_MAX_TEST_PARMS; i++)
        {
            diagStruct.testParms[i][0] = '\0';
        }

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));

        // Define a helper lambda to check for a parameter and its value
        auto checkParam = [&cmdArgs](const std::string &param, const std::string &expectedValue) -> bool {
            for (size_t i = 0; i < cmdArgs.size(); i++)
            {
                if (cmdArgs[i] == ("--" + param) && i + 1 < cmdArgs.size())
                {
                    REQUIRE(cmdArgs[i + 1] == expectedValue);
                    return true;
                }
            }
            return false;
        };

        // Check all default parameters from GetDefaultMnuberGemmParametersMap_v1
        REQUIRE(checkParam("workload", "N"));
        REQUIRE(checkParam("time_to_run", "3600"));
        REQUIRE(checkParam("max_workload", "65536"));
        REQUIRE(checkParam("MM_max_workload", "65536"));
        REQUIRE(checkParam("NET_sm_count", "152"));
        // With 9 processes (odd), NET_link_order should be set to snake
        REQUIRE(checkParam("NET_link_order", "snake"));
        REQUIRE(checkParam("NET_size", "2048000000"));
        REQUIRE(checkParam("freq", "1"));
        REQUIRE(checkParam("duty", "1.0"));

        // Check flags (parameters with no values)
        bool foundDynamicAdj = false;
        bool foundNoGraphs   = false;
        for (const auto &arg : cmdArgs)
        {
            if (arg == "--dynamic_adj")
            {
                foundDynamicAdj = true;
            }
            else if (arg == "--no_graphs")
            {
                foundNoGraphs = true;
            }
        }
        REQUIRE(foundDynamicAdj);
        REQUIRE(foundNoGraphs);
    }

    SECTION("Verify NET_link_order set to snake with odd number of processes")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // With 3 GPUs per node from MockDcgmCoreProxy, we can create an odd process count easily

        // Clear host list first
        for (int i = 0; i < DCGM_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        // Add a single host to get exactly 3 processes (odd number)
        SafeCopyTo(diagStruct.hostList[0], "single-host");

        // Override default NET_link_order to verify it gets changed
        SafeCopyTo(diagStruct.testParms[3], "mnubergemm.NET_link_order=pair");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify we have 3 processes total (odd number)
        unsigned int npIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "-np") - cmdArgs.begin();
        REQUIRE(npIdx < cmdArgs.size());
        REQUIRE(cmdArgs[npIdx + 1] == "3"); // 1 host * 3 GPUs = 3 processes

        // Verify NET_link_order is set to snake for odd number of processes
        bool foundNetLinkOrderSnake = false;
        for (size_t i = 0; i < cmdArgs.size(); i++)
        {
            if (cmdArgs[i] == "--NET_link_order" && i + 1 < cmdArgs.size())
            {
                REQUIRE(cmdArgs[i + 1] == "snake"); // Should be overridden to snake
                foundNetLinkOrderSnake = true;
                break;
            }
        }
        REQUIRE(foundNetLinkOrderSnake);

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));
    }

    SECTION("Verify NET_link_order stays pair with even number of processes")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // With 3 GPUs per node from MockDcgmCoreProxy, we need 2 hosts to get 6 processes (even)

        // Clear host list first
        for (int i = 0; i < DCGM_MAX_NUM_HOSTS; i++)
        {
            diagStruct.hostList[i][0] = '\0';
        }

        // Add exactly 2 hosts to get 6 processes (even number)
        SafeCopyTo(diagStruct.hostList[0], "host1");
        SafeCopyTo(diagStruct.hostList[1], "host2");

        // Set NET_link_order to verify it doesn't get changed
        SafeCopyTo(diagStruct.testParms[3], "mnubergemm.NET_link_order=pair");

        // Call the parse function
        mpiRunner.ParseDcgmMnDiagToMpiCommand_v1(diagStruct);

        // Get the generated command args
        auto cmdArgs = mpiRunner.GetLastCommandArgs();

        // Verify we have 6 processes total (even number)
        unsigned int npIdx = std::find(cmdArgs.begin(), cmdArgs.end(), "-np") - cmdArgs.begin();
        REQUIRE(npIdx < cmdArgs.size());
        REQUIRE(cmdArgs[npIdx + 1] == "6"); // 2 hosts * 3 GPUs = 6 processes

        // Verify NET_link_order remains pair for even number of processes
        bool foundNetLinkOrderPair = false;
        for (size_t i = 0; i < cmdArgs.size(); i++)
        {
            if (cmdArgs[i] == "--NET_link_order" && i + 1 < cmdArgs.size())
            {
                REQUIRE(cmdArgs[i + 1] == "pair"); // Should not be changed
                foundNetLinkOrderPair = true;
                break;
            }
        }
        REQUIRE(foundNetLinkOrderPair);

        // Verify --oversubscribe is present
        REQUIRE(std::ranges::contains(cmdArgs, "--oversubscribe"));
    }
}

TEST_CASE("MnDiagMpiRunner SetUserInfo")
{
    SECTION("Verify SetUserInfo with non-root user")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // First construct a command without setting user info
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto argsWithoutUser = mpiRunner.GetLastCommandArgs();

        // Now set a regular user (non-root) and construct the command again
        std::pair<std::string, uid_t> userInfo = std::make_pair("testuser", 1000);
        mpiRunner.SetUserInfo(userInfo);
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto argsWithUser = mpiRunner.GetLastCommandArgs();

        // Verify both commands are not empty
        REQUIRE_FALSE(argsWithoutUser.empty());
        REQUIRE_FALSE(argsWithUser.empty());

        // With a non-root user, "--allow-run-as-root" should NOT be present
        auto findAllowRunAsRootWithUser = std::find(argsWithUser.begin(), argsWithUser.end(), "--allow-run-as-root");
        REQUIRE(findAllowRunAsRootWithUser == argsWithUser.end());
    }

    SECTION("Verify SetUserInfo with root user")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto diagStruct = CreateTestDiagStruct();

        // Set root user and construct the command
        std::pair<std::string, uid_t> rootUserInfo = std::make_pair("root", 0);
        mpiRunner.SetUserInfo(rootUserInfo);
        mpiRunner.ConstructMpiCommand(&diagStruct);
        auto argsWithRootUser = mpiRunner.GetLastCommandArgs();

        // Verify commands are not empty
        REQUIRE_FALSE(argsWithRootUser.empty());

        // With a root user, "--allow-run-as-root" should be present only if the environment variable is set
        unsetenv(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data());
        auto findAllowRunAsRootWithRootUser
            = std::find(argsWithRootUser.begin(), argsWithRootUser.end(), "--allow-run-as-root");
        REQUIRE(findAllowRunAsRootWithRootUser == argsWithRootUser.end());

        // Set the environment variable to allow run as root
        setenv(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data(), "1", 1);
        mpiRunner.ConstructMpiCommand(&diagStruct);
        argsWithRootUser = mpiRunner.GetLastCommandArgs();
        // Verify commands are not empty
        REQUIRE_FALSE(argsWithRootUser.empty());
        findAllowRunAsRootWithRootUser
            = std::find(argsWithRootUser.begin(), argsWithRootUser.end(), "--allow-run-as-root");
        REQUIRE(findAllowRunAsRootWithRootUser != argsWithRootUser.end());
        unsetenv(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data());
    }
}

TEST_CASE("MnDiagMpiRunner ParseMnUberGemmOutput_v1")
{
    SECTION("Parse MNUB output with mixed success and error messages")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create input stream from the test_string
        std::istringstream dataStream(test_string);

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
        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version1);

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

            // Verify entity indices match what we found in results
            std::string hostname = host.hostname;
            for (unsigned int j = 0; j < host.numEntities; j++)
            {
                REQUIRE(hostToEntities[hostname].count(host.entityIndices[j]) == 1);
            }
        }
    }

    SECTION("Parse empty MNUB output")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create empty input stream
        std::istringstream dataStream("");

        // Call the parser
        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfoMap_t());

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version1);

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

    SECTION("Parse MNUB output with only INFO messages")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Create input stream with only INFO messages (no errors)
        std::string infoOnlyString = "MNUB [I] G: 0 test-host L: 0  INFO  hosthash=123456789 B0I=632386609\n"
                                     "MNUB [I] G: 1 test-host L: 1  INFO  hosthash=123456789 B0I=632386610\n"
                                     "MNUB [I] G: 0 test-host L: 0  T:G FINAL values L:14057 IDLE:31878952\n"
                                     "MNUB [I] G: 1 test-host L: 1  T:N FINISHED\n";
        std::istringstream dataStream(infoOnlyString);

        // Call the parser
        nodeInfoMap_t nodeInfo;
        std::string hostname             = "test-host";
        std::string dcgmVersion          = "3.1.1";
        std::string driverVersion        = "570.15";
        nodeInfo[hostname].dcgmVersion   = dcgmVersion;
        nodeInfo[hostname].driverVersion = driverVersion;
        dcgmReturn_t result              = mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version1);

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

    SECTION("Parse MNUB output with timestamp prefix edge cases")
    {
        MnDiagMpiRunnerTests mpiRunner;
        auto responseStruct = CreateTestResponseStruct();

        // Test string with various timestamp scenarios
        std::string timestampEdgeCases =
            // Normal timestamp with MNUB message
            "[2025-01-15 10:30:45.123] MNUB [I] G: 0 host1 L: 0  INFO test message\n"
            // Timestamp with no space before MNUB
            "[2025-01-15 10:30:45.456]MNUB [E] G: 1 host1 L: 1  T:N Error: test error\n"
            // Timestamp with multiple spaces/tabs before MNUB
            "[2025-01-15 10:30:45.789]   \t  MNUB [I] G: 2 host2 L: 0  INFO another test\n"
            // Timestamp with only whitespace after (should be skipped as empty line)
            "[2025-01-15 10:30:45.012]   \t   \n"
            // Timestamp that's exactly the timestamp length (should result in empty line)
            "[2025-01-15 10:30:45.345]\n"
            // Normal MNUB message without timestamp
            "MNUB [I] G: 3 host2 L: 1  INFO no timestamp message\n"
            // Malformed timestamp (should not be stripped)
            "[2025-01-15 10:30:45] MNUB [E] G: 4 host3 L: 0  T:N malformed timestamp\n"
            // Another malformed timestamp (wrong format)
            "[25-01-15 10:30:45.123] MNUB [I] G: 5 host3 L: 1  INFO wrong year format\n"
            // Timestamp with ANSI escape codes
            "\x1B[31m[2025-01-15 10:30:45.999] MNUB [E] G: 6 host4 L: 0  T:N colored error\x1B[0m\n";

        std::istringstream dataStream(timestampEdgeCases);

        // Set up node info for all hosts
        nodeInfoMap_t nodeInfo;
        std::vector<std::string> hostnames = { "host1", "host2", "host3", "host4" };
        for (const auto &hostname : hostnames)
        {
            nodeInfo[hostname].dcgmVersion   = "3.1.1";
            nodeInfo[hostname].driverVersion = "570.15";
        }

        dcgmReturn_t result = mpiRunner.MnDiagOutputCallback(dataStream, responseStruct.get(), nodeInfo);

        // Verify the call succeeded
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(responseStruct->version == dcgmMnDiagResponse_version1);

        // Verify test information
        REQUIRE(responseStruct->numTests == 1);
        REQUIRE(std::string(responseStruct->tests[0].name) == "MNUBERGEMM");
        REQUIRE(responseStruct->tests[0].result == DCGM_DIAG_RESULT_FAIL); // Should fail due to errors

        // Should find 4 unique hosts
        REQUIRE(responseStruct->numHosts == 4);

        // Verify we have the expected number of results (7 MNUB messages total)
        // - Lines with valid timestamps: 3 + 1 + 1 = 5 (includes ANSI escape line)
        // - Line without timestamp: 1
        // - Malformed timestamp lines: 2 (should still be parsed since they start with MNUB after the malformed
        // timestamp)
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
        REQUIRE(entityErrors.contains("host1:1")); // G:1 L:1 from timestamped error
        REQUIRE(entityErrors.contains("host3:0")); // G:4 L:0 from malformed timestamp line
        REQUIRE(entityErrors.contains("host4:0")); // G:6 L:0 from ANSI escape line

        // Verify error messages contain expected content
        REQUIRE(entityErrors["host1:1"][0].contains("test error"));
        REQUIRE(entityErrors["host3:0"][0].contains("malformed timestamp"));
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
}