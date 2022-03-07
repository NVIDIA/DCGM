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
#ifndef DCGMMUTEX_H
#define DCGMMUTEX_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

/* API Status codes */
typedef enum dcgmMutexSt
{
    DCGM_MUTEX_ST_OK            = 0,  /* OK */
    DCGM_MUTEX_ST_LOCKEDBYOTHER = 1,  /* Another thread has the lock */
    DCGM_MUTEX_ST_LOCKEDBYME    = -2, /* My thread already owns the lock */
    DCGM_MUTEX_ST_TIMEOUT       = -3, /* Tried to wait for the lock but didn't get it before our
                                          timeout expired */
    DCGM_MUTEX_ST_NOTLOCKED = -4,     /* Mutex is currently not locked */
    DCGM_MUTEX_ST_ERROR     = -5      /* Generic, unspecified error */
} dcgmMutexReturn_t;

struct dcgm_mutex_locker_t
{
    dcgm_mutex_locker_t()
        : file(nullptr)
        , line(0)
        , unused(0)
        , whenLockedUsec(0)
        , ownerTid(std::thread::id())
    {}
    const char *file;         /* Pointer to the filename that locked this. */
    int line;                 /* Line of code in leaf this was locked from */
    int unused;               /* padding to 8-byte alignment */
    long long whenLockedUsec; /* usec since 1970 of when this sem was locked */
    std::thread::id ownerTid; /* Thread id of the current locker. std::thread::id() if no one */
};
using dcgm_mutex_locker_p = dcgm_mutex_locker_t *;


/* DcgmMutex class. Instantiate this class to get a mutex */
class DcgmMutex
{
public:
    /*************************************************************************/
    /* Constructor
     *
     * timeoutMs IN: How long we should wait when locking this mutex before
     *               giving up and returning DCGM_MUTEX_ST_TIMEOUT.
     *               0 = never timeout. This is slightly faster because no timing
     *               information is recorded in this case
     *
     */
    DcgmMutex(int timeoutMs);

    /*************************************************************************/
    /**
     * Destructor
     */
    ~DcgmMutex();

    /*************************************************************************/
    /**
     * Lock this mutex
     *
     * complainMe  IN: Whether or not to complain if the mutex is already locked
     *                 by my thread.
     * file        IN: Should be __FILE__ or some other heap-allocated pointer to
     *                 a source code line.
     * line        IN: Should be __LINE__
     *
     * RETURNS: DCGM_MUTEX_ST_OK if OK
     *          DCGM_MUTEX_ST_LOCKEDBYME if mutex was locked by me already
     *          DCGM_MUTEX_ST_? enum on error
     */
    dcgmMutexReturn_t Lock(int complainMe, const char *file, int line);

/* Convenience macros for using Lock() on a DcgmMutex pointer */
#define dcgm_mutex_lock(m)    (m)->Lock(1, __FILE__, __LINE__)
#define dcgm_mutex_lock_me(m) (m)->Lock(0, __FILE__, __LINE__)

    /*************************************************************************/
    /**
     * Unlock this mutex
     *
     * file IN: Should be __FILE__ or some other heap-allocated pointer to
     *          a source code line.
     * line IN: Should be __LINE__
     *
     *   RETURNS: 0 if OK
     *            DCGM_MUTEX_ST_? enum on error
     */
    dcgmMutexReturn_t Unlock(const char *file, int line);

#define dcgm_mutex_unlock(m) (m)->Unlock(__FILE__, __LINE__)

    /*************************************************************************/
    /*
     * Query the current state of this mutex
     *
     * RETURNS: DCGM_MUTEX_ST_? enum state of the mutex
     */
    dcgmMutexReturn_t Poll(void);

    /*************************************************************************/
    /* Enable or disable debug logging spew from this mutex
     *
     * Default is disabled
     *
     */
    void EnableDebugLogging(bool enabled);

    /*************************************************************************/
    /* Wait on a condition variable using this mutex as the underlying mutex
     *
     * The mutex will still be locked by the calling thread after this call.
     *
     * cv        IN: condition_variable to wait on
     * timeoutMs IN: How long to wait on this condition in ms. If the timeoutMs is 0, then CondVar is waited for
     *               a maximum allowed time.
     * pred      IN: Predicate which returns false if the waiting should be continued.
     *               The signature of the predicate function should be equivalent to the following: bool pred();
     *
     * RETURNS: DCGM_MUTEX_ST_OK if the condition was signalled.
     *          DCGM_MUTEX_ST_TIMEOUT if timeoutMs elapsed before the condition was signalled
     *          DCGM_MUTEX_ST_ERROR if an error happened (e.g. pred throws an exception)
     */
    dcgmMutexReturn_t CondWait(std::condition_variable &cv, unsigned int timeoutMs, std::function<bool()> const &pred);

    /*************************************************************************/
    /*
     * Get the number of times this mutex has ever been locked. This is useful for
     * detecting places that lock a mutex too many times. Note that this value
     * only counts successful locks. Recursive locks where the mutex was already locked
     * don't increment this counter.
     *
     * RETURNS: Number of times this mutex was locked.
     */
    long long GetLockCount(void);

private:
    /*************************************************************************/

    /* OS Handle to the mutex */
    long long m_timeoutUsec; /* How long to wait in usec before timing out. 0=never timeout */
    int m_handleInit;        /* Is handle/critSec is initialized? */
    bool m_debugLogging;     /* Should we log verbose debug logs? true=yes */
    long long m_lockCount;   /* Number of times this mutex has been locked. This doesn't count recursive locks */
    std::mutex m_mutex;

    dcgm_mutex_locker_t m_locker; /* Information about the locker of this mutex */

    /*************************************************************************/
};

/**
 * RAII style locking mechanism.  Meant to be similar to C++11 lock_guard.
 * Non-copyable.
 */
class DcgmLockGuard
{
public:
    explicit DcgmLockGuard(DcgmMutex *mutex) noexcept
        : m_mutex(mutex)
    {
        /* Use recursive version of lock. The destructor will handle this properly */
        m_mutexReturn = dcgm_mutex_lock_me(m_mutex);
    }

    ~DcgmLockGuard() noexcept
    {
        if (m_mutex != nullptr && m_mutexReturn == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    DcgmLockGuard &operator=(DcgmLockGuard const &) = delete;
    DcgmLockGuard(DcgmLockGuard const &)            = delete;

    /**
     * @brief Move assignment operator.
     * @note If current guard has its mutex locked, the mutex will be unlocked before assigning.
     * @param r Another lock guard (locked or unlocked)
     * @return *this
     */
    DcgmLockGuard &operator=(DcgmLockGuard &&r) noexcept
    {
        if (this != &r)
        {
            if (m_mutex != nullptr && m_mutexReturn == DCGM_MUTEX_ST_OK)
            {
                dcgm_mutex_unlock(m_mutex);
            }
            m_mutex         = r.m_mutex;
            m_mutexReturn   = r.m_mutexReturn;
            r.m_mutex       = nullptr;
            r.m_mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
        }
        return *this;
    }

    DcgmLockGuard(DcgmLockGuard &&r) noexcept
    {
        m_mutex         = r.m_mutex;
        m_mutexReturn   = r.m_mutexReturn;
        r.m_mutex       = nullptr;
        r.m_mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
    }

private:
    DcgmMutex *m_mutex;
    dcgmMutexReturn_t m_mutexReturn;
};

#endif // DCGMMUTEX_H
