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

#pragma once

#include "misc.h"
#include <DcgmUtilities.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

// RAII wrapper for both regular and mapped CUDA memory
struct CudaMemoryDeleter
{
    bool isMapped;
    void *hostPtr; // Only used for mapped memory

    CudaMemoryDeleter(bool mapped = false, void *host = nullptr)
        : isMapped(mapped)
        , hostPtr(host)
    {}

    void operator()(char *ptr) const noexcept
    {
        if (ptr != nullptr)
        {
            if (isMapped && hostPtr)
            {
                if (auto st = cudaFreeHost(hostPtr); st != cudaSuccess)
                {
                    log_error("cudaFreeHost failed: {}", cudaGetErrorString(st));
                }
            }
            else
            {
                if (auto st = cudaFree(ptr); st != cudaSuccess)
                {
                    log_error("cudaFree failed: {}", cudaGetErrorString(st));
                }
            }
        }
    }
};

using CudaMemoryChunk = std::unique_ptr<char, CudaMemoryDeleter>;

// Container for managing multiple memory chunks
class MemoryChunkManager
{
private:
    std::vector<CudaMemoryChunk> m_chunks;
    size_t m_blocksPerChunk { 1 };
    size_t m_totalBlocks { 1 };
    bool m_useMappedMemory { false };

public:
    static constexpr size_t DEFAULT_NUM_CHUNKS = 8; // Configurable

    MemoryChunkManager() = default;

    // Initialize with the total number of blocks we want to allocate
    cudaError_t Initialize(size_t tot_num_blocks,
                           size_t desired_chunks = DEFAULT_NUM_CHUNKS,
                           bool useMappedMemory  = false);

    // Get pointer and size for a specific chunk
    char *GetChunkPtr(size_t chunkIndex) const;
    size_t GetNominalChunkSizeInBlocks() const
    {
        return m_blocksPerChunk;
    }
    size_t GetNumChunks() const
    {
        return m_chunks.size();
    }

    // For compatibility with existing code - get start/end pointers for a chunk
    char *GetChunkStart(size_t chunkIndex) const;
    char *GetChunkEnd(size_t chunkIndex) const;

    // Get the actual number of blocks in a specific chunk
    size_t GetChunkBlockCount(size_t chunkIndex) const;

    // Get total blocks across all chunks
    size_t GetTotalBlocks() const
    {
        return m_totalBlocks;
    }

    bool IsUsingMappedMemory() const
    {
        return m_useMappedMemory;
    }

    void Clear() noexcept;
};
