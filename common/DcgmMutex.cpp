/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmMutex.h"
#include "DcgmLogging.h"
#include "timelib.h"

#include <chrono>
#include <ratio>

/*****************************************************************************/
DcgmMutex::DcgmMutex(int timeoutMs)
    // Cast to long long to avoid overflowing before widening to a long long
    : m_timeoutUsec(static_cast<long long>(timeoutMs) * 1000)
    , m_handleInit(1)
    , m_debugLogging(false)
    , m_lockCount(0)
    , m_mutex()
    , m_locker()
{
    if (m_debugLogging)
        PRINT_DEBUG("%p", "Mutex %p allocated", (void *)this);
}

/*****************************************************************************/
DcgmMutex::~DcgmMutex()
{
    auto mutexStatus = Poll();
    switch (mutexStatus)
    {
        case DCGM_MUTEX_ST_LOCKEDBYOTHER:
            DCGM_LOG_ERROR << "Trying to destroy a locked by-other thread mutex";
            std::terminate();
        case DCGM_MUTEX_ST_LOCKEDBYME:
            DCGM_LOG_WARNING << "Destroying a locked Mutex";
            Unlock(__FILE__, __LINE__);
            break;
        default:;
    }

    m_handleInit = 0;

    if (m_debugLogging)
    {
        DCGM_LOG_DEBUG << "Mutex " << std::hex << (void *)this << " destroyed";
    }
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Poll(void)
{
    std::thread::id myTid = std::this_thread::get_id();

    /* Locked mutex? */
    if (m_locker.ownerTid == myTid)
        return DCGM_MUTEX_ST_LOCKEDBYME;
    else if (m_locker.ownerTid != std::thread::id())
        return DCGM_MUTEX_ST_LOCKEDBYOTHER;

    return DCGM_MUTEX_ST_NOTLOCKED;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Unlock(const char *file, int line)
{
    std::thread::id myTid = std::this_thread::get_id();

    if (m_locker.ownerTid == std::thread::id())
    {
        PRINT_ERROR("%s %d", "%s[%d] passed in an unlocked mutex to Unlock\n", file, line);
        return DCGM_MUTEX_ST_NOTLOCKED;
    }
    else if (m_locker.ownerTid != myTid)
    {
        PRINT_ERROR("%s %d %zu %s %d",
                    "%s[%d] passed in locked by tid %zu %s[%d]\n",
                    file,
                    line,
                    std::hash<std::thread::id> {}(m_locker.ownerTid),
                    m_locker.file,
                    m_locker.line);
        return DCGM_MUTEX_ST_LOCKEDBYOTHER;
    }

    /* Clear locker info */
    m_locker = dcgm_mutex_locker_t {};

    m_mutex.unlock();

    if (m_debugLogging)
    {
        PRINT_DEBUG("%p %zu %s %d",
                    "Mutex %p unlocked by tid %zu from %s[%d]",
                    (void *)this,
                    std::hash<std::thread::id> {}(myTid),
                    file,
                    line);
    }
    return DCGM_MUTEX_ST_OK;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::Lock(int complainMe, const char *file, int line)
{
    // When acquiring the lock with a timeout, sleep for 100 microseconds between attempts to lock
    std::thread::id myTid   = std::this_thread::get_id();
    dcgmMutexReturn_t retSt = DCGM_MUTEX_ST_OK;
    timelib64_t diff;

    if (m_locker.ownerTid == myTid)
    {
        if (complainMe)
        {
            PRINT_ERROR("%s %d %s %d",
                        "%s[%d] mutex already locked by me from %s[%d]\n",
                        file,
                        line,
                        m_locker.file,
                        m_locker.line);
        }
        return DCGM_MUTEX_ST_LOCKEDBYME;
    }

    /* Try and get the lock */
    if (!m_timeoutUsec)
    {
        m_mutex.lock();

        /* Got lock */
        retSt = DCGM_MUTEX_ST_OK;
    }
    else
    {
        std::chrono::microseconds timeout(m_timeoutUsec);
        auto end    = std::chrono::steady_clock::now() + timeout;
        bool locked = false;

        while (!locked && std::chrono::steady_clock::now() <= end)
        {
            locked = m_mutex.try_lock();
            if (!locked)
            {
                std::this_thread::yield();
            }
        }

        if (!locked)
        {
            // Couldn't acquire the lock during the timeout
            retSt = DCGM_MUTEX_ST_TIMEOUT;
        }
        else
        {
            // Lock acquired
            retSt = DCGM_MUTEX_ST_OK;
        }
    }

    /* Handle the mutex statuses */
    switch (retSt)
    {
        case DCGM_MUTEX_ST_OK:
            break; /* Keep going */

        case DCGM_MUTEX_ST_TIMEOUT:
        {
            timelib64_t now = timelib_usecSince1970();
            diff            = now - m_locker.whenLockedUsec;
            PRINT_ERROR("%zu %s %d %zu %s %d %lld",
                        "Mutex timeout by tid %zu %s[%d] owned by tid "
                        "%zu %s[%d] for %lld usec\n",
                        std::hash<std::thread::id> {}(myTid),
                        file,
                        line,
                        std::hash<std::thread::id> {}(m_locker.ownerTid),
                        m_locker.file,
                        m_locker.line,
                        (long long)diff);
            return retSt;
        }

        // coverity[dead_error_begin] unreachable code to avoid compiler warnings
        default:
            PRINT_ERROR("%d", "Unexpected retSt %d\n", (int)retSt);
            return retSt;
    }

    /* Finally got lock Populate info */
    m_locker.ownerTid = myTid;
    m_locker.line     = line;
    m_locker.file     = file;
    m_lockCount++;
    if (m_timeoutUsec)
    {
        m_locker.whenLockedUsec = timelib_usecSince1970();
    }

    if (m_debugLogging)
    {
        PRINT_DEBUG("%p %zu %s %d %lld",
                    "Mutex %p locked by tid %zu %s[%d] lockCount %lld\n",
                    (void *)this,
                    std::hash<decltype(m_locker.ownerTid)> {}(m_locker.ownerTid),
                    m_locker.file,
                    m_locker.line,
                    m_lockCount);
    }

    return DCGM_MUTEX_ST_OK;
}

/*****************************************************************************/
void DcgmMutex::EnableDebugLogging(bool enabled)
{
    m_debugLogging = enabled;
}

/*****************************************************************************/
dcgmMutexReturn_t DcgmMutex::CondWait(std::condition_variable &cv,
                                      unsigned int timeoutMs,
                                      std::function<bool()> const &pred)
{
    dcgmMutexReturn_t lockSt;
    dcgmMutexReturn_t retSt;
    dcgm_mutex_locker_t backupLocker;

    /* Make sure the mutex is actually locked by us. This also checks the mutex
       status */
    lockSt = Lock(0, __FILE__, __LINE__);
    if (lockSt != DCGM_MUTEX_ST_OK && lockSt != DCGM_MUTEX_ST_LOCKEDBYME)
    {
        PRINT_ERROR("%p %d", "CondWait of mutex %p call to Lock() returned unexpected %d", (void *)this, lockSt);
        return DCGM_MUTEX_ST_ERROR;
    }

    // adopt_lock tells unique_lock we don't want to take ownership of the lock locally
    std::unique_lock<std::mutex> lock(m_mutex, std::adopt_lock);

    /* Back up the owner and clear it since wait_for will unlock the mutex */
    backupLocker      = m_locker;
    m_locker          = dcgm_mutex_locker_t {};
    m_locker.ownerTid = std::thread::id();

    if (timeoutMs == 0)
    {
        timeoutMs = 60000U;
    }

    /* never sleep for longer than 1 minute. That should protect us against missed notifications */
    timeoutMs = std::min(60000U, timeoutMs);

    try
    {
        using std::chrono::milliseconds;
        retSt = cv.wait_for(lock, milliseconds(timeoutMs), pred) ? DCGM_MUTEX_ST_OK : DCGM_MUTEX_ST_TIMEOUT;
    }
    catch (std::runtime_error const &ex)
    {
        DCGM_LOG_ERROR << "An exception occured while waiting for a CondVar. Ex: " << ex.what();
        retSt = DCGM_MUTEX_ST_ERROR;
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "A generic exception occured while waiting for a CondVar.";
        retSt = DCGM_MUTEX_ST_ERROR;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "An unknown exception occured while waiting for a CondVar";
        retSt = DCGM_MUTEX_ST_ERROR;
    }

    lock.release(); // Instruct unique_lock to not release the mutex at the end of the function

    /* We now have the lock again. Restore the locker info */
    m_locker = backupLocker;

    if (m_debugLogging)
    {
        DCGM_LOG_DEBUG << "CondWait finished on mutex" << (void *)this << ". retSt " << retSt;
    }

    return retSt;
}

/*****************************************************************************/
long long DcgmMutex::GetLockCount(void)
{
    return m_lockCount;
}

/*****************************************************************************/
