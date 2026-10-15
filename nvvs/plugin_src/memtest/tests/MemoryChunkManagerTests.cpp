/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <MemoryChunkManager.h>
#include <MockCudaOps.h>

#include <CudaOpsAnyPtr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdlib>

struct MemoryChunkManagerTestFixture
{
    MockCudaOps cudaOps;
    CudaOpsAnyPtr cudaOpsPtr { &cudaOps };

    MemoryChunkManagerTestFixture()
    {
        cudaOps.cudaHostAllocResult            = cudaSuccess;
        cudaOps.cudaHostGetDevicePointerResult = cudaSuccess;
        cudaOps.cudaMallocResult               = cudaSuccess;
        cudaOps.cudaStubAllocateMemory         = true;
    }
};

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: rejects invalid initialization sizes")
{
    GIVEN("a memory chunk manager")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("zero total blocks are requested")
        {
            cudaError_t const result = manager.Initialize(0, 1, false);

            THEN("initialization fails without creating chunks")
            {
                CHECK(result == cudaErrorInvalidValue);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetChunkPtr(0) == nullptr);
                CHECK(manager.GetChunkStart(0) == nullptr);
                CHECK(manager.GetChunkEnd(0) == nullptr);
                CHECK(manager.GetChunkBlockCount(0) == 0);
            }
        }

        WHEN("zero desired chunks are requested")
        {
            cudaError_t const result = manager.Initialize(1, 0, false);

            THEN("initialization fails without creating chunks")
            {
                CHECK(result == cudaErrorInvalidValue);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetChunkPtr(0) == nullptr);
                CHECK(manager.GetChunkStart(0) == nullptr);
                CHECK(manager.GetChunkEnd(0) == nullptr);
                CHECK(manager.GetChunkBlockCount(0) == 0);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: partitions regular CUDA allocations")
{
    GIVEN("a CUDA stub that allocates host-backed memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("five blocks are split across two chunks")
        {
            cudaError_t const result = manager.Initialize(5, 2, false);

            THEN("chunks cover all blocks with ceil-sized first and smaller final chunk")
            {
                REQUIRE(result == cudaSuccess);
                REQUIRE(manager.GetNumChunks() == 2);
                CHECK(manager.GetTotalBlocks() == 5);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 3);
                REQUIRE(manager.GetChunkPtr(0) != nullptr);
                CHECK(manager.GetChunkStart(0) == manager.GetChunkPtr(0));
                CHECK(manager.GetChunkEnd(0) == manager.GetChunkPtr(0) + (3 * BLOCKSIZE));
                CHECK(manager.GetChunkBlockCount(0) == 3);
                CHECK(manager.GetChunkBlockCount(1) == 2);
                CHECK(manager.GetChunkPtr(2) == nullptr);
                CHECK(manager.GetChunkBlockCount(2) == 0);
            }
        }

        WHEN("five blocks are split across three chunks")
        {
            cudaError_t const result = manager.Initialize(5, 3, false);

            THEN("chunks cover all blocks with a smaller final chunk")
            {
                REQUIRE(result == cudaSuccess);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetTotalBlocks() == 5);
                CHECK(manager.GetNumChunks() == 3);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 2);
                REQUIRE(manager.GetChunkStart(0) != nullptr);
                REQUIRE(manager.GetChunkStart(1) != nullptr);
                REQUIRE(manager.GetChunkStart(2) != nullptr);
                CHECK(manager.GetChunkBlockCount(0) == 2);
                CHECK(manager.GetChunkBlockCount(1) == 2);
                CHECK(manager.GetChunkBlockCount(2) == 1);
                CHECK(manager.GetChunkEnd(0) - manager.GetChunkStart(0) == static_cast<std::ptrdiff_t>(2 * BLOCKSIZE));
                CHECK(manager.GetChunkEnd(2) - manager.GetChunkStart(2) == static_cast<std::ptrdiff_t>(BLOCKSIZE));
                CHECK(manager.GetChunkStart(3) == nullptr);
                CHECK(manager.GetChunkEnd(3) == nullptr);
            }
        }

        WHEN("more chunks than blocks are requested")
        {
            cudaError_t const result = manager.Initialize(2, 8, false);

            THEN("the chunk count is capped to the block count")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 2);
                CHECK(manager.GetChunkBlockCount(0) == 1);
                CHECK(manager.GetChunkBlockCount(1) == 1);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: supports mapped CUDA allocations")
{
    GIVEN("a CUDA stub that can return mapped host pointers")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("mapped memory is requested")
        {
            cudaError_t const result = manager.Initialize(3, 2, true);

            THEN("the manager records mapped mode and exposes chunk boundaries")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.IsUsingMappedMemory());
                CHECK(manager.GetNumChunks() == 2);
                CHECK(manager.GetChunkBlockCount(0) == 2);
                CHECK(manager.GetChunkBlockCount(1) == 1);
                REQUIRE(manager.GetChunkPtr(0) != nullptr);
                REQUIRE(manager.GetChunkPtr(1) != nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: clears partial allocations on CUDA failures")
{
    GIVEN("a memory chunk manager with CUDA allocation stubs")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("cudaMalloc fails")
        {
            cudaOps.cudaMallocResult = cudaErrorMemoryAllocation;
            cudaError_t const result = manager.Initialize(1, 1, false);

            THEN("the error is returned and no chunks remain")
            {
                CHECK(result == cudaErrorMemoryAllocation);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
            }
        }

        WHEN("cudaMalloc fails after an earlier chunk was allocated")
        {
            int callCount             = 0;
            cudaOps.cudaMallocHandler = [&](void **buf, size_t size) -> cudaError_t {
                if (++callCount > 1)
                {
                    return cudaErrorMemoryAllocation;
                }
                if (buf == nullptr)
                {
                    return cudaErrorInvalidValue;
                }
                *buf = std::malloc(size);
                return *buf == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
            };

            cudaError_t const result = manager.Initialize(4, 2, false);

            THEN("the error is returned and previously owned chunks are cleared")
            {
                CHECK(result == cudaErrorMemoryAllocation);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetChunkPtr(0) == nullptr);
            }
        }

        WHEN("cudaHostAlloc fails after an earlier mapped chunk was allocated")
        {
            int callCount                = 0;
            cudaOps.cudaHostAllocHandler = [&](void **buf, size_t size, unsigned int) -> cudaError_t {
                if (++callCount > 1)
                {
                    return cudaErrorMemoryAllocation;
                }
                if (buf == nullptr)
                {
                    return cudaErrorInvalidValue;
                }
                *buf = std::malloc(size);
                return *buf == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
            };

            cudaError_t const result = manager.Initialize(4, 2, true);

            THEN("the error is returned and previously owned chunks are cleared")
            {
                CHECK(result == cudaErrorMemoryAllocation);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetChunkPtr(0) == nullptr);
            }
        }

        WHEN("cudaHostGetDevicePointer fails after an earlier mapped chunk was allocated")
        {
            int callCount                           = 0;
            cudaOps.cudaHostGetDevicePointerHandler = [&](void **pDevice, void *pHost, unsigned int) -> cudaError_t {
                if (++callCount > 1)
                {
                    return cudaErrorInvalidDevicePointer;
                }
                if (pDevice == nullptr)
                {
                    return cudaErrorInvalidValue;
                }
                *pDevice = pHost;
                return cudaSuccess;
            };

            cudaError_t const result = manager.Initialize(4, 2, true);

            THEN("the error is returned and previously owned chunks are cleared")
            {
                CHECK(result == cudaErrorInvalidDevicePointer);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetChunkPtr(0) == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: Clear resets reusable state")
{
    GIVEN("an initialized memory chunk manager")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(3, 2, false) == cudaSuccess);

        WHEN("the manager is cleared")
        {
            manager.Clear();

            THEN("all chunk metadata returns to the empty state")
            {
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetChunkBlockCount(0) == 0);
                CHECK(manager.GetChunkStart(0) == nullptr);
                CHECK(manager.GetChunkEnd(0) == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: reinitialization replaces previous chunk layout")
{
    GIVEN("a manager initialized with mapped chunks")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(6, 4, true) == cudaSuccess);
        REQUIRE(manager.IsUsingMappedMemory());
        REQUIRE(manager.GetNumChunks() == 3);

        WHEN("the same manager is initialized again for regular memory")
        {
            cudaError_t const result = manager.Initialize(4, 2, false);

            THEN("only the new allocation metadata remains observable")
            {
                REQUIRE(result == cudaSuccess);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetTotalBlocks() == 4);
                CHECK(manager.GetNumChunks() == 2);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 2);
                CHECK(manager.GetChunkBlockCount(0) == 2);
                CHECK(manager.GetChunkBlockCount(1) == 2);
                CHECK(manager.GetChunkBlockCount(2) == 0);
                CHECK(manager.GetChunkStart(2) == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture,
                "MemoryChunkManager: failed reinitialization clears stale chunks and remains reusable")
{
    GIVEN("a manager with an existing regular CUDA allocation")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(4, 2, false) == cudaSuccess);
        REQUIRE(manager.GetNumChunks() == 2);
        REQUIRE(manager.GetChunkPtr(0) != nullptr);

        WHEN("a later regular allocation fails")
        {
            cudaOps.cudaMallocResult = cudaErrorMemoryAllocation;
            cudaError_t const result = manager.Initialize(8, 4, false);

            THEN("the previous allocation metadata is cleared with the failed attempt")
            {
                CHECK(result == cudaErrorMemoryAllocation);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
                CHECK(manager.GetChunkPtr(0) == nullptr);
            }
        }

        WHEN("a failed reinitialization is followed by a mapped allocation")
        {
            cudaOps.cudaMallocResult = cudaErrorMemoryAllocation;
            REQUIRE(manager.Initialize(8, 4, false) == cudaErrorMemoryAllocation);

            cudaOps.cudaMallocResult               = cudaSuccess;
            cudaOps.cudaHostAllocResult            = cudaSuccess;
            cudaOps.cudaHostGetDevicePointerResult = cudaSuccess;
            cudaError_t const result               = manager.Initialize(5, 2, true);

            THEN("the manager can be reused with a fresh mapped layout")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.IsUsingMappedMemory());
                CHECK(manager.GetTotalBlocks() == 5);
                CHECK(manager.GetNumChunks() == 2);
                CHECK(manager.GetChunkBlockCount(0) == 3);
                CHECK(manager.GetChunkBlockCount(1) == 2);
                REQUIRE(manager.GetChunkPtr(0) != nullptr);
                REQUIRE(manager.GetChunkPtr(1) != nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: handles exact and default chunk boundaries")
{
    GIVEN("a CUDA stub that allocates memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("blocks divide evenly across requested chunks")
        {
            cudaError_t const result = manager.Initialize(8, 4, false);

            THEN("each chunk receives the same number of blocks")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 4);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 2);
                for (size_t chunkIndex = 0; chunkIndex < manager.GetNumChunks(); ++chunkIndex)
                {
                    CAPTURE(chunkIndex);
                    CHECK(manager.GetChunkBlockCount(chunkIndex) == 2);
                    REQUIRE(manager.GetChunkStart(chunkIndex) != nullptr);
                    CHECK(manager.GetChunkEnd(chunkIndex) - manager.GetChunkStart(chunkIndex)
                          == static_cast<std::ptrdiff_t>(2 * BLOCKSIZE));
                }
            }
        }

        WHEN("the default chunk count exceeds the block count")
        {
            cudaError_t const result = manager.Initialize(3);

            THEN("the manager creates one chunk per block")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 3);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 1);
                CHECK(manager.GetChunkBlockCount(0) == 1);
                CHECK(manager.GetChunkBlockCount(1) == 1);
                CHECK(manager.GetChunkBlockCount(2) == 1);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture,
                "MemoryChunkManager: invalid reinitialization clears previous allocations")
{
    GIVEN("a manager with an existing allocation")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(2, 2, false) == cudaSuccess);
        REQUIRE(manager.GetNumChunks() == 2);
        REQUIRE(manager.GetTotalBlocks() == 2);

        WHEN("the next initialization request has zero total blocks")
        {
            cudaError_t const result = manager.Initialize(0, 1, false);

            THEN("the stale allocation metadata is no longer observable")
            {
                CHECK(result == cudaErrorInvalidValue);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK(manager.GetChunkBlockCount(0) == 0);
                CHECK(manager.GetChunkStart(0) == nullptr);
                CHECK(manager.GetChunkEnd(0) == nullptr);
            }
        }

        WHEN("the next initialization request has zero desired chunks")
        {
            cudaError_t const result = manager.Initialize(2, 0, true);

            THEN("the manager returns to the empty state before reporting the validation error")
            {
                CHECK(result == cudaErrorInvalidValue);
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK_FALSE(manager.IsUsingMappedMemory());
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: single chunk boundaries span the full allocation")
{
    GIVEN("a CUDA stub that allocates memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("all blocks are assigned to one chunk")
        {
            cudaError_t const result = manager.Initialize(7, 1, false);

            THEN("the single chunk exposes the full block range")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 1);
                CHECK(manager.GetTotalBlocks() == 7);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 7);
                CHECK(manager.GetChunkBlockCount(0) == 7);
                REQUIRE(manager.GetChunkStart(0) != nullptr);
                CHECK(manager.GetChunkEnd(0) - manager.GetChunkStart(0) == static_cast<std::ptrdiff_t>(7 * BLOCKSIZE));
                CHECK(manager.GetChunkBlockCount(1) == 0);
                CHECK(manager.GetChunkPtr(1) == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: reports stable empty-state boundaries")
{
    GIVEN("a default-constructed memory chunk manager")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("out-of-range chunk metadata is queried before initialization")
        {
            auto const numChunks   = manager.GetNumChunks();
            auto const *chunkPtr   = manager.GetChunkPtr(0);
            auto const *chunkStart = manager.GetChunkStart(0);
            auto const *chunkEnd   = manager.GetChunkEnd(0);
            auto const blockCount  = manager.GetChunkBlockCount(0);
            auto const usingMapped = manager.IsUsingMappedMemory();

            THEN("the manager reports no addressable chunks")
            {
                CHECK(numChunks == 0);
                CHECK(chunkPtr == nullptr);
                CHECK(chunkStart == nullptr);
                CHECK(chunkEnd == nullptr);
                CHECK(blockCount == 0);
                CHECK_FALSE(usingMapped);
            }
        }

        WHEN("the empty manager is cleared repeatedly")
        {
            manager.Clear();
            manager.Clear();

            THEN("the observable empty state is preserved")
            {
                CHECK(manager.GetNumChunks() == 0);
                CHECK(manager.GetTotalBlocks() == 0);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 0);
                CHECK(manager.GetChunkPtr(1) == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture,
                "MemoryChunkManager: preserves chunk metadata around out-of-range queries")
{
    GIVEN("a manager with an uneven regular allocation")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(9, 4, false) == cudaSuccess);

        WHEN("valid and out-of-range chunks are queried together")
        {
            auto const numChunks   = manager.GetNumChunks();
            auto const totalBlocks = manager.GetTotalBlocks();
            auto const nominal     = manager.GetNominalChunkSizeInBlocks();
            auto const *oobPtr     = manager.GetChunkPtr(manager.GetNumChunks());
            auto const *oobStart   = manager.GetChunkStart(manager.GetNumChunks());
            auto const *oobEnd     = manager.GetChunkEnd(manager.GetNumChunks());
            auto const oobCount    = manager.GetChunkBlockCount(manager.GetNumChunks());

            THEN("valid chunk boundaries remain stable and invalid indexes are empty")
            {
                CHECK(numChunks == 3);
                CHECK(totalBlocks == 9);
                CHECK(nominal == 3);
                for (size_t chunkIndex = 0; chunkIndex < manager.GetNumChunks(); ++chunkIndex)
                {
                    CAPTURE(chunkIndex);
                    REQUIRE(manager.GetChunkStart(chunkIndex) != nullptr);
                    CHECK(manager.GetChunkEnd(chunkIndex) - manager.GetChunkStart(chunkIndex)
                          == static_cast<std::ptrdiff_t>(manager.GetChunkBlockCount(chunkIndex) * BLOCKSIZE));
                    CHECK(manager.GetChunkBlockCount(chunkIndex) == 3);
                }
                CHECK(oobPtr == nullptr);
                CHECK(oobStart == nullptr);
                CHECK(oobEnd == nullptr);
                CHECK(oobCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture, "MemoryChunkManager: reinitialization switches allocation modes cleanly")
{
    GIVEN("a manager initialized with regular CUDA memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        REQUIRE(manager.Initialize(6, 2, false) == cudaSuccess);
        REQUIRE_FALSE(manager.IsUsingMappedMemory());
        REQUIRE(manager.GetNumChunks() == 2);

        WHEN("the manager is reinitialized with mapped memory")
        {
            cudaError_t const result = manager.Initialize(5, 4, true);

            THEN("old regular chunks are replaced with the mapped chunk layout")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.IsUsingMappedMemory());
                CHECK(manager.GetTotalBlocks() == 5);
                CHECK(manager.GetNumChunks() == 3);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 2);
                CHECK(manager.GetChunkBlockCount(0) == 2);
                CHECK(manager.GetChunkBlockCount(1) == 2);
                CHECK(manager.GetChunkBlockCount(2) == 1);
                REQUIRE(manager.GetChunkStart(2) != nullptr);
                CHECK(manager.GetChunkEnd(2) - manager.GetChunkStart(2) == static_cast<std::ptrdiff_t>(BLOCKSIZE));
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture,
                "MemoryChunkManager: ceil partitions do not expose zero-sized trailing chunks")
{
    GIVEN("a CUDA stub that allocates regular device memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);

        WHEN("ten blocks are split into four requested chunks")
        {
            cudaError_t const result = manager.Initialize(10, 4, false);

            THEN("three ceil-sized chunks and one one-block chunk cover the allocation")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 4);
                CHECK(manager.GetTotalBlocks() == 10);
                CHECK(manager.GetNominalChunkSizeInBlocks() == 3);
                CHECK(manager.GetChunkBlockCount(0) == 3);
                CHECK(manager.GetChunkBlockCount(1) == 3);
                CHECK(manager.GetChunkBlockCount(2) == 3);
                CHECK(manager.GetChunkBlockCount(3) == 1);
                CHECK(manager.GetChunkBlockCount(4) == 0);
                for (size_t chunkIndex = 0; chunkIndex < manager.GetNumChunks(); ++chunkIndex)
                {
                    CAPTURE(chunkIndex);
                    REQUIRE(manager.GetChunkStart(chunkIndex) != nullptr);
                    CHECK(manager.GetChunkEnd(chunkIndex) - manager.GetChunkStart(chunkIndex)
                          == static_cast<std::ptrdiff_t>(manager.GetChunkBlockCount(chunkIndex) * BLOCKSIZE));
                }
            }
        }
    }
}

SCENARIO_METHOD(MemoryChunkManagerTestFixture,
                "MemoryChunkManager: non-owning stub allocations expose null pointers safely")
{
    GIVEN("a CUDA stub that reports success without allocating backing memory")
    {
        MemoryChunkManager manager(cudaOpsPtr);
        cudaOps.cudaStubAllocateMemory = false;

        WHEN("regular chunks are initialized without stub allocations")
        {
            cudaError_t const result = manager.Initialize(3, 2, false);

            THEN("chunk metadata is still bounded and null chunk pointers are exposed safely")
            {
                REQUIRE(result == cudaSuccess);
                CHECK(manager.GetNumChunks() == 2);
                CHECK(manager.GetChunkPtr(0) == nullptr);
                CHECK(manager.GetChunkStart(0) == nullptr);
                CHECK(manager.GetChunkEnd(0) == nullptr);
                CHECK(manager.GetChunkBlockCount(0) == 2);
                CHECK(manager.GetChunkBlockCount(1) == 1);
            }
        }
    }
}
