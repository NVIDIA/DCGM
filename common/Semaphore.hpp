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
#pragma once

#include "DcgmLogging.h"

#include <semaphore.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>


namespace DcgmNs
{
/**
 * Simple Semaphore synchronization object
 * This semaphore is unnamed and cannot be used for inter-process synchronization.
 *
 * @code{.cpp}
 * ```
 *  auto sm = std::make_shared<Semaphore>();
 *  //thread 1:
 *  sm->Release(2);
 *  //thread 2 and 3:
 *  sm->Wait(); //This function will return only after the sm is Released or destroyed
 *  //thread 4:
 *  sm->Destroy(); //This will notify all waiters that the semaphore is going to be destroyed
 * ```
 * @endcode
 *
 * @note    Destructor of this semaphore will mark the object as Destroyed and will wait until all Waiters are exited.
 *          Any caller of the Wait* functions should not assume the semaphore is alive and can be used once the Wait*
 *          functions return *::Destroyed status. Even if a caller holds a shared_ptr for the semaphore object.
 * @note    An instance of the Semaphore class is neither copyable to moveable.
 */
class Semaphore : public std::enable_shared_from_this<Semaphore>
{
private:
    /**
     * This is a helper object to count all active calls to the Wait* functions.
     * Usually an instance of this object should be created at the very beginnign of every Wait* function.
     */
    struct WaitersGuard
    {
        /**
         * Constructor.
         * @param[in,out] val   A reference to the counter that will be used to track Wait* calls.
         */
        explicit WaitersGuard(std::atomic_uint &val) noexcept
            : m_val(val)
        {
            ++m_val;
        }

        ~WaitersGuard() noexcept
        {
            --m_val;
        }

        WaitersGuard(WaitersGuard const &) = delete;
        WaitersGuard(WaitersGuard &&)      = delete;

        WaitersGuard &operator=(WaitersGuard &&) = delete;
        WaitersGuard &operator=(WaitersGuard const &) = delete;

    private:
        std::atomic_uint &m_val;
    };

public:
    enum class [[nodiscard]] ReleaseResult {
        Ok,        //!< Semaphore was successfully released
        Destroyed, //!< Semaphore was marked to be destroyed. The semaphore may or may not be released.
        Overflow,  //!< Too many releases were made to the semaphore.
    };
    enum class [[nodiscard]] WaitResult {
        Ok,        //!< The semaphore was successfully acquired and its counter was decreased by 1
        Destroyed, //!< The semaphore was marked to be destroyed. The semaphore may or may not be acquired.
    };
    enum class [[nodiscard]] TimedWaitResult {
        Ok,        //!< The semaphore was successfully acquired and its counter was decreased by 1
        Destroyed, //!< The semaphore was marked to be destroyed. The semaphore may or may not be acquired.
        TimedOut,  //!< The semaphore was not acquired as due to the timeout
    };
    enum class [[nodiscard]] AsyncWaitResult {
        Ok,         //!< The semaphore was acquired and its counter was decreased by 1
        Destroyed,  //!< The semaphore was marked to be destroyed. The semaphore may or may not be acquired.
        LockNeeded, //!< The semaphore cannot be acquired without full lock: its counter is 0.
    };

    /**
     * Creates an instance of the Semaphore class.
     * @throw std::system_error if the semaphore cannot be created due to some underlying OS issues.
     */
    Semaphore()
        : m_systemSemaphore {}
    {
        if (0 != sem_init(&m_systemSemaphore, 0, 0))
        {
            auto err = errno;
            DCGM_LOG_ERROR << "Unable to initialize semaphore. Errno: " << err;
            throw std::system_error(std::error_code(err, std::generic_category()));
        }
    }

    /**
     * Destroys the semaphore.
     * Marks the semaphore for destruction and waits till all Wait* function have finished.
     */
    ~Semaphore() noexcept
    {
        try
        {
            Destroy();
        }
        catch (std::exception const &ex)
        {
            DCGM_LOG_ERROR << "An exception caught during a Semaphore destruction: " << ex.what();
        }

        if (m_debugLog.load(std::memory_order_relaxed))
        {
            DCGM_LOG_DEBUG << "Destroying a semaphore with " << m_numOfWaiters.load(std::memory_order_relaxed)
                           << " waiters";
        }

        while (m_numOfWaiters != 0)
        {
            std::this_thread::yield();
        }

        sem_destroy(&m_systemSemaphore);
    }

    Semaphore(Semaphore const &) = delete;
    Semaphore &operator=(Semaphore const &) = delete;

    Semaphore(Semaphore &&) = delete;
    Semaphore &operator=(Semaphore &&) = delete;

    /**
     * Marks the semaphore to be destroyed.
     * @throw std::system_error If underlying semaphore returns an error code (via Release method call)
     */
    void Destroy() noexcept(false)
    {
        m_markForDestroy = true;

        //
        // Timed waiters may wait for a significant amount of time and we want to wake them up forcely to handle
        // the destruction in a timely manner.
        //
        [[maybe_unused]] auto discard = Release(m_numOfWaiters * 2, true);
    }

    /**
     * Releases the Semaphore.
     * Increments internal semaphore counter is case of success.
     * @param[in] number            How many releases to perform.
     * @param[in] isCalledInDestroy Forcely release semaphore even if it's already marked as destroyed
     * @return `ReleaseResult` status
     * @throw std::system_error If underlying semaphore returns an error code
     * @sa `enum class ReleaseResult` for details on the possible result values.
     * @note ReleaseResult type is marked as [[nodiscard]]
     */
    ReleaseResult Release(std::uint32_t number = 1, bool isCalledInDestroy = false) noexcept(false)
    {
        for (std::uint32_t i = 0; i < number; ++i)
        {
            if (m_markForDestroy.load(std::memory_order_relaxed) && (!isCalledInDestroy))
            {
                if (m_debugLog.load(std::memory_order_relaxed))
                {
                    DCGM_LOG_DEBUG << "A " << __func__ << " was called on a destroyed semaphore";
                }

                return ReleaseResult::Destroyed;
            }
            int res = sem_post(&m_systemSemaphore);
            if (res != 0)
            {
                auto err = errno;
                if (err == EOVERFLOW)
                {
                    return ReleaseResult::Overflow;
                }
                DCGM_LOG_ERROR << "Unable to release a semaphore. Errno: " << err;
                throw std::system_error(std::error_code(err, std::generic_category()));
            }
        }
        return ReleaseResult::Ok;
    }

    /**
     * Tries to acquire a semaphore without locking.
     * @return AsyncWaitResult Status
     * @throw std::system_error If underlying semaphore returns an error code
     * @sa `enum class AsyncWaitResult` for details on the possible result values.
     * @note AsyncWaitResult type is marked as [[nodiscard]]
     */
    AsyncWaitResult TryWait() noexcept(false)
    {
        WaitersGuard grd(m_numOfWaiters);
        if (m_markForDestroy)
        {
            if (m_debugLog.load(std::memory_order_relaxed))
            {
                DCGM_LOG_DEBUG << "A " << __func__ << " was called on a destroyed semaphore";
            }

            return AsyncWaitResult::Destroyed;
        }

        for (;;)
        {
            auto res = sem_trywait(&m_systemSemaphore);
            if (res == 0)
            {
                return m_markForDestroy ? AsyncWaitResult::Destroyed : AsyncWaitResult::Ok;
            }

            auto err = errno;
            if (err == EINTR)
            {
                DCGM_LOG_DEBUG << "Semaphore was interrupted by a signal";
                continue;
            }
            if (err != EAGAIN)
            {
                DCGM_LOG_ERROR << "Unable to trywait a semaphore. Errno: " << err;
                throw std::system_error(std::error_code(err, std::generic_category()));
            }

            break;
        }

        return AsyncWaitResult::LockNeeded;
    }


    /**
     * Tries to acquire the semaphore until the specified time is reached.
     * @param[in] waitUntil     A moment in the future until when semaphore should be acquired.
     * @return TimedWaitResult  Status
     * @throw std::system_error If underlying semaphore returns an error code
     * @sa `enum class TimedWaitResult` for details on the possible result values.
     * @note TimedWaitResult type is marked as [[nodiscard]]
     * @note If the specified time is already expired and the semaphore cannot be acquired immediately,
     *       this function returns TimedOut status.
     */
    TimedWaitResult WaitUntil(std::chrono::system_clock::time_point waitUntil) noexcept(false)
    {
        // TODO(nkonyuchenko): system_clock should be replaced to utc_clock once we are C++20 ready
        using std::chrono::nanoseconds;
        using std::chrono::seconds;
        using std::chrono::time_point_cast;

        if (m_markForDestroy.load(std::memory_order_relaxed))
        {
            if (m_debugLog.load(std::memory_order_relaxed))
            {
                DCGM_LOG_DEBUG << "A " << __func__ << " was called on a destroyed semaphore";
            }

            return TimedWaitResult::Destroyed;
        }

        WaitersGuard grd(m_numOfWaiters);

        auto secs  = time_point_cast<seconds>(waitUntil);
        auto nsecs = time_point_cast<nanoseconds>(waitUntil) - time_point_cast<nanoseconds>(secs);
        timespec ts { secs.time_since_epoch().count(), nsecs.count() };

        for (;;)
        {
            int res = sem_timedwait(&m_systemSemaphore, &ts);
            if (res == 0)
            {
                return m_markForDestroy ? TimedWaitResult::Destroyed : TimedWaitResult::Ok;
            }

            auto err = errno;
            if (err == EINTR)
            {
                DCGM_LOG_DEBUG << "Semaphore was interrupted by a signal";
                continue;
            }
            if (err != ETIMEDOUT)
            {
                DCGM_LOG_ERROR << "Unable to timedwait for a semaphore. Errno: " << err;
                throw std::system_error(std::error_code(err, std::generic_category()));
            }

            break;
        }

        if (m_markForDestroy)
        {
            return TimedWaitResult::Destroyed;
        }

        return TimedWaitResult::TimedOut;
    }

    /**
     * Tries to acquire the semaphore for a given amount of time.
     * @param[in] timeout   How long should it try to acquire the semaphore
     * @return TimedWaitResult Status
     * @throw std::system_error If underlying semaphore returns an error code
     * @sa `enum class TimedWaitResult` for details on the possible result values.
     * @note TimedWaitResult type is marked as [[nodiscard]]
     */
    TimedWaitResult TimedWait(std::chrono::milliseconds timeout) noexcept(false)
    {
        return WaitUntil(std::chrono::system_clock::now() + timeout);
    }

    /**
     * Infinitely wait for a semaphore to be acquired.
     * It uses `TimedWait()` under the hood.
     * @return WaitResult Status
     * @throw std::system_error If underlying semaphore return an error code
     * @sa `enum class WaitResult` for details on the possible result values.
     * @note This method will periodically check if the semaphore is marked to be destroyed or not.
     */
    WaitResult Wait() noexcept(false)
    {
        WaitersGuard grd(m_numOfWaiters);
        while (true)
        {
            switch (TimedWait(std::chrono::seconds(1)))
            {
                case TimedWaitResult::Ok:
                    return WaitResult::Ok;
                case TimedWaitResult::Destroyed:
                    return WaitResult::Destroyed;
                case TimedWaitResult::TimedOut:
                    std::this_thread::yield();
                    break;
                default:
                    break;
            };
        }
    }

    /**
     * Enable or disable debug logging for the Semaphore
     * @param enabled Debug loggig will be enabled if value is True.
     */
    void SetDebugLogging(bool enabled)
    {
        m_debugLog.store(enabled, std::memory_order_relaxed);
        if (enabled)
        {
            PRINT_INFO("", "Debug logging is enabled for the Sepamore at %p", (void *)this);
        }
    }

private:
    sem_t m_systemSemaphore;
    std::atomic_bool m_markForDestroy = false;
    std::atomic_bool m_debugLog       = false;
    std::atomic_uint m_numOfWaiters   = 0;
};

} // namespace DcgmNs
