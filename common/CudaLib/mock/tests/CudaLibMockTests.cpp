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

#include <CudaLibMock.h>
#include <fmt/format.h>

TEST_CASE("CudaMockDev::UUID")
{
    DcgmNs::CudaMockDev dev;
    std::string uuid = "GPU-22222222-2222-2222-2222-222222222222";

    dev.SetUuid(uuid);
    REQUIRE(dev.GetUuid() == uuid);
}

TEST_CASE("CudaLibMock::cuDeviceGetCount")
{
    DcgmNs::CudaLibMock cudaLibMock;
    DcgmNs::CudaMockDev dev1;
    DcgmNs::CudaMockDev dev2;

    cudaLibMock.AddMockDev(dev1);
    cudaLibMock.AddMockDev(dev2);

    int count = 0;
    REQUIRE(cudaLibMock.cuDeviceGetCount(&count) == CUDA_SUCCESS);
    REQUIRE(count == 2);
}

TEST_CASE("CudaLibMock::cuDeviceGet")
{
    DcgmNs::CudaLibMock cudaLibMock;
    DcgmNs::CudaMockDev dev1;

    cudaLibMock.AddMockDev(dev1);

    SECTION("Valid Index")
    {
        CUdevice device;
        REQUIRE(cudaLibMock.cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    }

    SECTION("Invalid Index")
    {
        CUdevice device;
        REQUIRE(cudaLibMock.cuDeviceGet(&device, 1) == CUDA_ERROR_INVALID_VALUE);
    }
}

TEST_CASE("CudaLibMock::cuDeviceGetUuid_v2")
{
    DcgmNs::CudaLibMock cudaLibMock;
    DcgmNs::CudaMockDev dev1;
    std::string uuid = "GPU-22222222-2222-2222-2222-222222222222";

    dev1.SetUuid(uuid);
    cudaLibMock.AddMockDev(dev1);

    CUdevice device;
    CUuuid cudaUuid;
    REQUIRE(cudaLibMock.cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    REQUIRE(cudaLibMock.cuDeviceGetUuid_v2(&cudaUuid, device) == CUDA_SUCCESS);

    std::string const retrievedUuid = fmt::format("{:02x}{:02x}{:02x}{:02x}-"
                                                  "{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                                                  "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                                                  (unsigned char)cudaUuid.bytes[0],
                                                  (unsigned char)cudaUuid.bytes[1],
                                                  (unsigned char)cudaUuid.bytes[2],
                                                  (unsigned char)cudaUuid.bytes[3],
                                                  (unsigned char)cudaUuid.bytes[4],
                                                  (unsigned char)cudaUuid.bytes[5],
                                                  (unsigned char)cudaUuid.bytes[6],
                                                  (unsigned char)cudaUuid.bytes[7],
                                                  (unsigned char)cudaUuid.bytes[8],
                                                  (unsigned char)cudaUuid.bytes[9],
                                                  (unsigned char)cudaUuid.bytes[10],
                                                  (unsigned char)cudaUuid.bytes[11],
                                                  (unsigned char)cudaUuid.bytes[12],
                                                  (unsigned char)cudaUuid.bytes[13],
                                                  (unsigned char)cudaUuid.bytes[14],
                                                  (unsigned char)cudaUuid.bytes[15]);
    // remove "GPU-" prefix
    uuid = uuid.substr(4);
    REQUIRE(uuid == retrievedUuid);
}