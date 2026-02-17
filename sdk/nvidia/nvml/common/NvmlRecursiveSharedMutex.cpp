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

#include "NvmlRecursiveSharedMutex.h"

void NvmlRecursiveSharedMutex::lock()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    const auto tid = std::this_thread::get_id();

    // Prevent lock escalation from shared to exclusive
    if (m_readerLocks.contains(tid))
    {
        throw std::logic_error("Cannot acquire exclusive lock while holding a shared lock "
                               "(lock escalation).");
    }

    // Handle recursive writer lock
    if (m_writerThreadId == tid)
    {
        m_writerLockCount += 1;
        return;
    }

    m_waitingWriters += 1;
    // Wait until there are no active readers and no active writer.
    m_cvWriter.wait(lock, [this] { return m_activeReaders == 0 && !m_writerActive; });
    m_waitingWriters -= 1;
    m_writerActive    = true;
    m_writerThreadId  = tid;
    m_writerLockCount = 1;
}

void NvmlRecursiveSharedMutex::unlock()
{
    auto const tid = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock(m_mtx);
    if (m_writerThreadId != tid)
    {
        throw std::logic_error("Cannot unlock: not the owner of the exclusive lock.");
    }

    if (m_readerLocks.contains(tid))
    {
        throw std::logic_error("Cannot unlock: holding a shared lock.");
    }

    m_writerLockCount -= 1;
    if (m_writerLockCount == 0)
    {
        m_writerActive   = false;
        m_writerThreadId = std::thread::id();
        // Prioritize notifying a waiting writer.
        if (m_waitingWriters > 0)
        {
            lock.unlock();
            m_cvWriter.notify_one();
        }
        else
        {
            // If no writers are waiting, notify all readers.
            lock.unlock();
            m_cvReader.notify_all();
        }
    }
}

void NvmlRecursiveSharedMutex::lock_shared()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    auto const tid = std::this_thread::get_id();

    // Handle recursive shared lock
    if (m_writerThreadId == tid || m_readerLocks.contains(tid))
    {
        m_readerLocks[tid] += 1;
        m_activeReaders += 1;
        return;
    }

    // Wait until there are no active or waiting writers.
    m_cvReader.wait(lock, [this] { return m_waitingWriters == 0 && !m_writerActive; });
    m_readerLocks[tid] = 1;
    m_activeReaders += 1;
}

void NvmlRecursiveSharedMutex::unlock_shared()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    const auto tid = std::this_thread::get_id();
    auto it        = m_readerLocks.find(tid);

    if (it == m_readerLocks.end())
    {
        throw std::runtime_error("Cannot unlock: not holding a shared lock.");
    }

    it->second -= 1;
    if (it->second == 0)
    {
        m_readerLocks.erase(it);
    }

    m_activeReaders -= 1;
    // If there are no more readers and a writer is waiting,
    // notify the writer.
    if (m_activeReaders == 0 && m_waitingWriters > 0)
    {
        lock.unlock();
        m_cvWriter.notify_one();
    }
}
