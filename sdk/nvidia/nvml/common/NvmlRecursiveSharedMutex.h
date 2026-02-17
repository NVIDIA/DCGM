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
#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>

// A custom class that implements a recursive shared mutex.
// It allows a single thread to acquire the exclusive lock multiple times.
// It also allows a single thread to acquire shared locks multiple times.
// The implementation uses a standard shared_mutex and a mutex to manage state.
class NvmlRecursiveSharedMutex
{
public:
    /**
     * @brief Acquires an exclusive (writer) lock, recursively.
     *
     * Blocks until the lock is acquired. If the calling thread already holds
     * the lock, the internal lock count is incremented. Throws an exception
     * if the thread already holds a shared lock (lock escalation).
     */
    void lock();

    /**
     * @brief Releases an exclusive (writer) lock.
     *
     * Decrements the recursive lock count. If the count reaches zero, the
     * lock is fully released and waiting threads are notified.
     */
    void unlock();

    /**
     * @brief Acquires a shared (reader) lock, recursively.
     *
     * Blocks until the lock is acquired. If the calling thread already holds
     * a shared lock, the internal lock count for that thread is incremented.
     */
    void lock_shared();

    /**
     * @brief Releases a shared (reader) lock.
     *
     * Decrements the recursive lock count for the calling thread. If the
     * count reaches zero, the thread is no longer considered a reader. If it
     * is the very last reader, it notifies a waiting writer.
     */
    void unlock_shared();

private:
    std::mutex m_mtx;
    std::condition_variable m_cvReader;
    std::condition_variable m_cvWriter;

    // State for readers
    int m_activeReaders = 0;                                // Total count of shared locks (including recursive)
    std::unordered_map<std::thread::id, int> m_readerLocks; // Per-thread lock counts

    // State for writers
    int m_waitingWriters             = 0;
    bool m_writerActive              = false;
    std::thread::id m_writerThreadId = std::thread::id {};
    int m_writerLockCount            = 0; // Recursive lock count for the writer
};

class NvmlSharedLockGuard
{
public:
    NvmlSharedLockGuard(NvmlRecursiveSharedMutex &mutex, bool deferLock = false)
        : m_mutex(mutex)
    {
        if (!deferLock)
        {
            lock();
        }
    }

    void lock()
    {
        m_mutex.lock_shared();
        m_locked = true;
    }

    void unlock()
    {
        if (m_locked)
        {
            m_locked = false;
            m_mutex.unlock_shared();
        }
    }

    ~NvmlSharedLockGuard()
    {
        if (m_locked)
        {
            m_mutex.unlock_shared();
        }
    }

private:
    NvmlRecursiveSharedMutex &m_mutex;
    bool m_locked = false;

    NvmlSharedLockGuard(NvmlSharedLockGuard const &)            = delete;
    NvmlSharedLockGuard &operator=(NvmlSharedLockGuard const &) = delete;
    NvmlSharedLockGuard(NvmlSharedLockGuard &&other) noexcept   = delete;
    NvmlSharedLockGuard &operator=(NvmlSharedLockGuard &&)      = delete;
};

class NvmlExclusiveLockGuard
{
public:
    NvmlExclusiveLockGuard(NvmlRecursiveSharedMutex &mutex, bool deferLock = false)
        : m_mutex(mutex)
    {
        if (!deferLock)
        {
            lock();
        }
    }

    void lock()
    {
        m_mutex.lock();
        m_locked = true;
    }

    void unlock()
    {
        if (m_locked)
        {
            m_locked = false;
            m_mutex.unlock();
        }
    }

    ~NvmlExclusiveLockGuard()
    {
        if (m_locked)
        {
            m_mutex.unlock();
        }
    }

private:
    NvmlRecursiveSharedMutex &m_mutex;
    bool m_locked = false;

    NvmlExclusiveLockGuard(NvmlExclusiveLockGuard const &)            = delete;
    NvmlExclusiveLockGuard &operator=(NvmlExclusiveLockGuard const &) = delete;
    NvmlExclusiveLockGuard(NvmlExclusiveLockGuard &&other) noexcept   = delete;
    NvmlExclusiveLockGuard &operator=(NvmlExclusiveLockGuard &&)      = delete;
};
