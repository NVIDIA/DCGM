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

#include "DcgmUtilities.h"
#include "FileSystemOperator.h"
#include <dcgm_structs.h>

#include <MurmurHash3.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <random>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <utility>

/* Task Identifier, used as Fingerprint Store key */
struct PidTidPair
{
    pid_t pid;
    std::optional<int> tid;
};

/* Container for the task fingerprint */
using TaskFingerprint = std::array<uint64_t, 2>;

/**
 * Specialize fmt::formatter for TaskFingerprint
 */
template <>
struct fmt::formatter<TaskFingerprint>
{
    // Parse format specifications
    constexpr auto parse(format_parse_context &ctx)
    {
        return ctx.begin();
    }

    // Format the TaskFingerprint
    template <typename FormatContext>
    auto format(const TaskFingerprint &fp, FormatContext &ctx)
    {
        return fmt::format_to(ctx.out(), "{:016x}{:016x}", fp[0], fp[1]);
    }
};

/**
 * Equality operator for PidTidPair
 *
 * @param a PidTidPair to compare
 * @param b PidTidPair to compare
 */
inline bool operator==(PidTidPair const &a, PidTidPair const &b)
{
    return (a.pid == b.pid)
           && ((a.tid.has_value() && b.tid.has_value() && *(a.tid) == *(b.tid))
               || (!a.tid.has_value() && !b.tid.has_value()));
}

/**
 * Hash function for PidTidPair
 */
struct PidTidPairHash
{
    size_t operator()(PidTidPair const &x) const noexcept
    {
        size_t const prime = 31;
        size_t hash        = std::hash<uint64_t>()(x.pid);
        hash               = hash * prime ^ (x.tid.has_value() ? std::hash<uint64_t>()(*(x.tid)) : 1);
        return hash;
    }
};

/* Return type for FingerprintStore operations that may fail to return a fingerprint */
using FingerprintStoreRet = std::pair<dcgmReturn_t, std::optional<TaskFingerprint>>;

/**
 * Task Fingerprint Store. Computes and stores task fingerprints for task state monitoring.
 */
class FingerprintStore
{
public:
    /**
     * Constructor for FingerprintStore
     *
     * Only one thread will instantiate the FingerprintStore.
     *
     * @param fileOp Optional pointer to a FileSystemOperator instance (typically a mock for testing)
     */
    FingerprintStore(std::unique_ptr<FileSystemOperator> fileOp = std::make_unique<FileSystemOperator>())
        : m_mmhSeed(GenerateUniqueSeed())
        , m_store {}
        , m_fileOp(std::move(fileOp))
    {}

    /**
     * Compute a fingerprint from task state data
     *
     * @param data Task state data from procfs
     * @return Pair of status and optional fingerprint
     * @returns DCGM_ST_OK and the fingerprint if found
     * @returns DCGM_ST_NO_DATA if the fingerprint is not found
     * @returns DCGM_ST_GENERIC_ERROR if there is an error retrieving the fingerprint
     */
    FingerprintStoreRet Compute(std::string_view data);

    /**
     * Compute a fingerprint for a task by reading its procfs entry
     *
     * @param pid Process ID
     * @param tid Optional thread ID
     * @return Pair of status and optional fingerprint
     * @returns DCGM_ST_OK and the fingerprint if found
     * @returns DCGM_ST_NO_DATA if the fingerprint is not found
     * @returns DCGM_ST_GENERIC_ERROR if there is an error retrieving the fingerprint
     */
    FingerprintStoreRet ComputeForTask(pid_t pid, std::optional<int> tid = std::nullopt);

    /**
     * Store a fingerprint for a task
     *
     * @param key Task identifier
     * @param fp Fingerprint to store
     */
    void Update(PidTidPair const &key, TaskFingerprint const &fp);

    /**
     * Retrieve stored fingerprint for a task
     *
     * @param key Task identifier
     * @return Pair of status and optional fingerprint
     * @returns DCGM_ST_OK and the fingerprint if found
     * @returns DCGM_ST_NO_DATA if the fingerprint is not found
     * @returns DCGM_ST_GENERIC_ERROR if there is an error retrieving the fingerprint
     */
    FingerprintStoreRet Retrieve(PidTidPair const &key);

    /**
     * Delete stored fingerprint for a task
     *
     * @param key Task identifier
     * @return Status code
     * @returns DCGM_ST_OK if the fingerprint is Deleted
     * @returns DCGM_ST_NO_DATA if the fingerprint is not found
     */
    dcgmReturn_t Delete(PidTidPair const &key);

private:
    /**
     * Get the path to the proc file for the specified task.
     *
     * @param pid Process ID
     * @param tid Optional thread ID
     * @return Path to the proc file
     */
    static std::string GetProcStatPath(pid_t pid, std::optional<int> tid = std::nullopt);

    /**
     * Generate a unique seed for the murmur hash function.
     * This is used to ensure that the hash function is unique for each instance of the FingerprintStore.
     *
     * Only one thread will instantiate the FingerprintStore, so there is no need to lock the seed.
     *
     * @returns uint32_t
     */
    static uint32_t GenerateUniqueSeed()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<uint32_t> dist;

        return dist(gen);
    }

    std::mutex m_mutex;                                                      // Mutex to protect m_store
    uint32_t const m_mmhSeed;                                                // murmurhash seed
    std::unordered_map<PidTidPair, TaskFingerprint, PidTidPairHash> m_store; // Store of task fingerprints
    std::unique_ptr<FileSystemOperator> m_fileOp;                            // File system operator

    friend class FingerprintStoreTest;
};
