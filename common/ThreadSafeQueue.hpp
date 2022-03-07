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

#include <cassert>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <thread>


namespace DcgmNs
{
template <class T>
class ThreadSafeQueue;

/**
 * A class to provide thread safe read-write access to the ThreadSafeQueue.
 * All operation on the queue must be proxied via this class.
 * An instance of this class is acquired via calling `ThreadSafeQueue::LockRW()` function.
 * Once the instance is destroyed, the queue will be unlocked automatically.
 * An instance of this class cannot be transferred to another thread.
 *
 * @code{.cpp}
 * ```
 *      auto&& handle = queue.LockRW();
 *      handle.Enqueue(val);
 * ```
 * @endcode
 *
 * @note An instance of this object is not copyable but movable
 */
template <class T>
class ThreadSafeQueueHandle
{
public:
    /**
     * Creates an instance of the `ThreadSafeQueueHandle` class.
     * This handle implies exclusive read-write lock for the queue.
     * @param[in] owner Owning queue where all operations will be proxied.
     * @param[in] lock  A locked `std::unique_lock`. The ownership of the lock will be transferred to the newly created
     *                  instance of the handle class.
     */
    ThreadSafeQueueHandle(ThreadSafeQueue<T> &owner, std::unique_lock<std::shared_mutex> &&lock)
        : m_owner(std::ref(owner))
        , m_lock(std::move(lock))
        , m_owningThread(std::this_thread::get_id())
    {}

    ThreadSafeQueueHandle(ThreadSafeQueueHandle &&) noexcept = default;
    ThreadSafeQueueHandle &operator=(ThreadSafeQueueHandle &&) noexcept = default;

    ThreadSafeQueueHandle(ThreadSafeQueueHandle const &) = delete;
    ThreadSafeQueueHandle &operator=(ThreadSafeQueueHandle const &) = delete;

    ~ThreadSafeQueueHandle() noexcept
    {
        if (m_owningThread != std::this_thread::get_id())
        {
            abort();
        }
    }

    /**
     * Adds a new object to the queue.
     * @param[in] val   An object to add to the end of the queue.
     *
     * @note    This function gets the object by value which implies that he ownership of the value is transferred to
     *          the queue. It's still possible to store pointers or std::ref in the queue.
     */
    void Enqueue(T val) const
    {
        assert(m_owningThread == std::this_thread::get_id());
        m_owner.get().m_queue.push(std::move(val));
    }

    /**
     * Removes an element from the head of the queue.
     * @return  An object from the queue. The object ownership is transferred from the queue to the caller of this
     *          function.
     */
    [[nodiscard]] T Dequeue() const
    {
        assert(m_owningThread == std::this_thread::get_id());
        auto &queue = m_owner.get().m_queue;
        auto result = std::move(queue.front());
        queue.pop();
        return result;
    }

    /**
     * Returns the size of the underlying queue.
     * As this method is called for a locked queue, the result value is the exact size of the queue as no
     * insertions/deletions are allowed at the moment.
     *
     * @return Size of the locked queue.
     */
    [[nodiscard]] std::size_t GetSize() const
    {
        assert(m_owningThread == std::this_thread::get_id());
        return m_owner.get().m_queue.size();
    }

    /**
     * Returns if the queue is empty or not.
     *
     * @return Emptiness flag for the locked queue.
     */
    [[nodiscard]] bool IsEmpty() const
    {
        assert(m_owningThread == std::this_thread::get_id());
        return m_owner.get().m_queue.empty();
    }

private:
    std::reference_wrapper<ThreadSafeQueue<T>> m_owner;
    std::unique_lock<std::shared_mutex> m_lock;
    std::thread::id m_owningThread;
};

/**
 * @brief Provides a thread-safe read only access to the ThreadSafeQueue.
 * Only non-mutating operations are provided.
 * An instance of this class should be acquired via `ThreadSafeQueue::LockRO()` method call.
 * An instance of this class cannot be transferred to another thread.
 * @tparam T
 */
template <class T>
class ThreadSafeQueueReadHandle
{
public:
    ThreadSafeQueueReadHandle(ThreadSafeQueue<T> &owner, std::shared_lock<std::shared_mutex> &&lock)
        : m_owner(std::ref(owner))
        , m_readOnlyLock(std::move(lock))
        , m_owningThread(std::this_thread::get_id())
    {
        assert(m_readOnlyLock.owns_lock() == true);
    }

    ThreadSafeQueueReadHandle(ThreadSafeQueueReadHandle &&) noexcept = default;
    ThreadSafeQueueReadHandle &operator=(ThreadSafeQueueReadHandle &&) noexcept = default;

    ThreadSafeQueueReadHandle(ThreadSafeQueueReadHandle const &) = delete;
    ThreadSafeQueueReadHandle &operator=(ThreadSafeQueueReadHandle const &) = delete;

    ~ThreadSafeQueueReadHandle() noexcept
    {
        if (m_owningThread != std::this_thread::get_id())
        {
            abort();
        }
    }

    /**
     * Returns if the queue is empty or not.
     *
     * @return Emptiness flag for the locked queue.
     */
    [[nodiscard]] bool IsEmpty() const
    {
        assert(m_owningThread == std::this_thread::get_id());
        return m_owner.get().m_queue.empty();
    }

    /**
     * Returns the size of the underlying queue.
     * As this method is called for a locked queue, the result value is the exact size of the queue as no
     * insertions/deletions are allowed at the moment.
     *
     * @return Size of the locked queue.
     */
    [[nodiscard]] std::size_t GetSize() const
    {
        assert(m_owningThread == std::this_thread::get_id());
        return m_owner.get().m_queue.size();
    }

private:
    std::reference_wrapper<ThreadSafeQueue<T>> m_owner;
    std::shared_lock<std::shared_mutex> m_readOnlyLock;
    std::thread::id m_owningThread;
};

/**
 * Thread-safe wrapper around std::queue.
 * @tparam T    A type of the stored objects in the queue
 *
 * @note The only operation that is permitted to be done with an instance of this object directly is the `Lock()` call.
 *       All the rest operations must be done via the returned proxy object.
 * @sa `class ThreadSafeQueueHandle`
 */
template <class T>
class ThreadSafeQueue
{
public:
    /**
     * Lock the queue and return a proxy object.
     * @return Proxy object that must be used to perform all operations with the queue.
     */
    [[nodiscard]] ThreadSafeQueueHandle<T> LockRW()
    {
        std::unique_lock<std::shared_mutex> lck(m_mutex);
        return ThreadSafeQueueHandle<T> { *this, std::move(lck) };
    }

    [[nodiscard]] ThreadSafeQueueReadHandle<T> LockRO()
    {
        std::shared_lock<std::shared_mutex> lck(m_mutex);
        return ThreadSafeQueueReadHandle<T> { *this, std::move(lck) };
    }

private:
    friend class ThreadSafeQueueHandle<T>;
    friend class ThreadSafeQueueReadHandle<T>;

    std::shared_mutex m_mutex;
    std::queue<T> m_queue;
};
} // namespace DcgmNs
