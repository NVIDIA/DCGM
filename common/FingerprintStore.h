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
#include <ranges>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/* Task Identifier, used as Fingerprint Store key */
struct PidTidPair
{
    pid_t pid;
    std::optional<pid_t> tid;
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
 * Field indices in /proc/[pid]/stat
 * See proc(5) for details
 */
enum class ProcStatField
{
    Pid   = 1, //!< Process ID
    Comm  = 2, //!< Command name
    State = 3, //!< Process state (R=running, S=sleeping, D=disk sleep, Z=zombie, T=stopped, t=tracing stop, W=paging,
               //!< X=dead, x=dead, K=wakekill, P=parked)
    Ppid                = 4,  //!< Parent process ID
    Pgrp                = 5,  //!< Process group ID
    Session             = 6,  //!< Session ID
    TtyNr               = 7,  //!< Controlling terminal
    Tpgid               = 8,  //!< Terminal process group ID
    Flags               = 9,  //!< Kernel flags
    Minflt              = 10, //!< Minor faults (no page load)
    Cminflt             = 11, //!< Minor faults of children
    Majflt              = 12, //!< Major faults (page load)
    Cmajflt             = 13, //!< Major faults of children
    Utime               = 14, //!< User CPU time
    Stime               = 15, //!< System CPU time
    Cutime              = 16, //!< Children's user CPU time
    Cstime              = 17, //!< Children's system CPU time
    Priority            = 18, //!< Process priority
    Nice                = 19, //!< Nice level
    NumThreads          = 20, //!< Number of threads
    Itrealvalue         = 21, //!< Time in jiffies before next SIGALRM
    Starttime           = 22, //!< Time in jiffies since system boot
    Vsz                 = 23, //!< Virtual memory size in bytes
    Rss                 = 24, //!< Resident set size in pages
    Rsslim              = 25, //!< Current soft limit of RSS in bytes
    Startcode           = 26, //!< Address above which program text can run
    Endcode             = 27, //!< Address below which program text can run
    Startstack          = 28, //!< Address of the start of the stack
    Kstkesp             = 29, //!< Current value of ESP/RSP
    Kstkeip             = 30, //!< Current EIP/RIP
    Signal              = 31, //!< Bitmap of pending signals
    Blocked             = 32, //!< Bitmap of blocked signals
    Sigignore           = 33, //!< Bitmap of ignored signals
    Sigcatch            = 34, //!< Bitmap of caught signals
    Wchan               = 35, //!< Channel in which process is waiting
    Nswap               = 36, //!< Number of pages swapped (not maintained)
    Cnswap              = 37, //!< Cumulative nswap for child processes (not maintained)
    ExitSignal          = 38, //!< Signal to be sent to parent when we die
    Processor           = 39, //!< CPU number last executed on
    RtPriority          = 40, //!< Real-time scheduling priority
    Policy              = 41, //!< Scheduling policy
    DelayacctBlkioTicks = 42, //!< Aggregated block I/O delays in clock ticks
    GuestTime           = 43, //!< Guest time of the process (time spent running a virtual CPU for a guest OS)
    CguestTime          = 44, //!< Guest time of the process's children
    StartData           = 45, //!< Address above which program data+bss is placed
    EndData             = 46, //!< Address below which program data+bss is placed
    StartBrk            = 47, //!< Address above which program heap can be expanded with brk()
    ArgStart            = 48, //!< Address above which program command line is placed
    ArgEnd              = 49, //!< Address below which program command line is placed
    EnvStart            = 50, //!< Address above which program environment is placed
    EnvEnd              = 51, //!< Address below which program environment is placed
    ExitCode            = 52  //!< The thread's exit status in form reported by waitpid
};

/**
 * Configuration for proc stat field filtering
 *
 * Fields may need to be excluded from fingerprinting for several reasons:
 *   - If values change frequently and may not be a good indicator of hang state
 *   - If values are process-wide and not meaningful for individual threads (e.g., Vsz, Rss)
 *   - If values are unreliable or platform-dependent
 *
 * The default configurations provide sensible exclusions, but alternatives
 * may be needed to avoid false positives or other issues.
 */
struct StatFieldConfig
{
    std::unordered_set<ProcStatField> excludedFields {};

    /**
     * Create a default configuration for process monitoring
     * Start with no exclusions - fields will be excluded as issues are discovered
     */
    static StatFieldConfig ForProcess()
    {
        return StatFieldConfig {};
    }

    /**
     * Create a default configuration for thread monitoring
     * Excludes fields known to cause issues with thread fingerprinting
     */
    static StatFieldConfig ForThread()
    {
        return StatFieldConfig { .excludedFields = {
                                     ProcStatField::Vsz, // Virtual memory size - varies between threads of same process
                                     ProcStatField::Rss, // Resident set size - varies between threads of same process
                                     ProcStatField::NumThreads // Thread count - not meaningful for individual threads
                                 } };
    }

    /**
     * Check if a field should be included
     */
    [[nodiscard]] bool ShouldIncludeField(size_t fieldNum) const noexcept
    {
        auto field = static_cast<ProcStatField>(fieldNum);
        return !excludedFields.contains(field);
    }
};

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
     * @param config Optional configuration for filtering proc stat fields
     * @return Pair of status and optional fingerprint
     * @returns DCGM_ST_OK and the fingerprint if found
     * @returns DCGM_ST_NO_DATA if the fingerprint is not found
     * @returns DCGM_ST_GENERIC_ERROR if there is an error retrieving the fingerprint
     */
    FingerprintStoreRet Compute(std::string_view data, std::optional<StatFieldConfig> config = std::nullopt);

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
    FingerprintStoreRet ComputeForTask(pid_t pid, std::optional<pid_t> tid = std::nullopt);

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
     * Filter proc stat fields based on configuration
     *
     * @param data Raw contents of proc stat file
     * @param config Configuration specifying which fields to include
     * @return Filtered proc stat data containing only desired fields, empty string if invalid
     */
    static std::string FilterProcStatFields(std::string_view data, StatFieldConfig const &config);

    /**
     * Get the path to the proc file for the specified task.
     *
     * @param pid Process ID
     * @param tid Optional thread ID
     * @return Path to the proc file
     */
    static std::string GetProcStatPath(pid_t pid, std::optional<pid_t> tid = std::nullopt);

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
