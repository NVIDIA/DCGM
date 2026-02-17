
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
#include "MemoryChunkManager.h"
#include <algorithm>

cudaError_t MemoryChunkManager::Initialize(size_t tot_num_blocks, size_t desired_chunks, bool useMappedMemory)
{
    cudaError_t st = cudaSuccess;
    Clear();

    if (tot_num_blocks == 0 || desired_chunks == 0)
    {
        return cudaErrorInvalidValue;
    }

    m_totalBlocks     = tot_num_blocks;
    size_t numChunks  = std::min(desired_chunks, tot_num_blocks);     // Don't create more chunks than blocks
    m_blocksPerChunk  = (tot_num_blocks + numChunks - 1) / numChunks; // Round up
    m_useMappedMemory = useMappedMemory;

    m_chunks.reserve(numChunks);

    // Allocate each chunk
    for (size_t i = 0; i < numChunks; i++)
    {
        // Calculate start block for this chunk
        size_t startBlock = i * m_blocksPerChunk;

        // Prevent underflow - if we've run out of blocks, reduce chunk count
        if (startBlock >= tot_num_blocks)
        {
            break;
        }

        // Calculate size for this chunk (last chunk might be smaller)
        size_t remainingBlocks = tot_num_blocks - startBlock;
        size_t thisChunkBlocks = std::min(m_blocksPerChunk, remainingBlocks);

        // Skip zero-sized chunks
        if (thisChunkBlocks == 0)
        {
            break;
        }

        size_t chunkSize = thisChunkBlocks * BLOCKSIZE;

        char *devicePtr = nullptr;
        void *hostPtr   = nullptr;

        if (m_useMappedMemory)
        {
            // Allocate mapped memory
            if (st = cudaHostAlloc(&hostPtr, chunkSize, cudaHostAllocMapped); st != cudaSuccess)
            {
                log_error("cudaHostAlloc failed: {}", cudaGetErrorString(st));
                Clear();
                return st;
            }

            // Get device pointer
            if (st = cudaHostGetDevicePointer((void **)&devicePtr, hostPtr, 0); st != cudaSuccess)
            {
                log_error("cudaHostGetDevicePointer failed: {}", cudaGetErrorString(st));
                cudaFreeHost(hostPtr);
                Clear();
                return st;
            }
        }
        else
        {
            if (st = cudaMalloc((void **)&devicePtr, chunkSize); st != cudaSuccess)
            {
                // Allocation failed - clean up and return error
                log_error("cudaMalloc({} bytes) failed: {}", chunkSize, cudaGetErrorString(st));
                Clear();
                return st;
            }
        }

        // Create unique_ptr with custom deleter
        CudaMemoryDeleter deleter(m_useMappedMemory, hostPtr);
        m_chunks.emplace_back(devicePtr, deleter);
    }

    return st;
}

char *MemoryChunkManager::GetChunkPtr(size_t chunkIndex) const
{
    if (chunkIndex >= m_chunks.size())
    {
        return nullptr;
    }
    return m_chunks[chunkIndex].get();
}

char *MemoryChunkManager::GetChunkStart(size_t chunkIndex) const
{
    return GetChunkPtr(chunkIndex);
}

char *MemoryChunkManager::GetChunkEnd(size_t chunkIndex) const
{
    char *start = GetChunkPtr(chunkIndex);
    if (!start)
    {
        return nullptr;
    }

    size_t thisChunkBlocks = GetChunkBlockCount(chunkIndex);

    return start + (thisChunkBlocks * BLOCKSIZE);
}

void MemoryChunkManager::Clear() noexcept
{
    m_chunks.clear(); // RAII will automatically free all memory
    m_blocksPerChunk  = 0;
    m_totalBlocks     = 0;
    m_useMappedMemory = false;
}

// Helper function to get actual number of blocks in a specific chunk
size_t MemoryChunkManager::GetChunkBlockCount(size_t chunkIndex) const
{
    if (chunkIndex >= m_chunks.size())
    {
        return 0;
    }

    // Calculate size for this chunk (last chunk might be smaller)
    size_t remainingBlocks = m_totalBlocks - (chunkIndex * m_blocksPerChunk);
    return std::min(m_blocksPerChunk, remainingBlocks);
}
