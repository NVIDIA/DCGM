/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>
#include <mutex>
#include <queue>

namespace DcgmNs
{
template <class T>
class ThreadSafeQueue;

/**
 * A class to provide thread safe access to the ThreadSafeQueue.
 * All operation on the queue must be proxied via this class.
 * An instance of this class is acquired via calling `ThreadSafeQueue::Lock()` function.
 * Once the instance is destroyed, the queue will be unlocked automatically.
 *
 * @code{.cpp}
 * ```
 *      auto&& handle = queue.Lock();
 *      handle.Enqueue(val);
 * ```
 * @endcode
 *
 * @note An instance of this object is not copyable but moveable
 */
template <class T>
class ThreadSafeQueueHandle
{
public:
    /**
     * Creates an instance of the `ThreadSafeQueueHandle` class.
     * @param[in] owner Owning queue where all operations will be proxied.
     * @param[in] lock  A locked `std::unique_lock`. The ownership of the lock will be transferred to the newely created
     *                  instance of the handle class.
     */
    ThreadSafeQueueHandle(ThreadSafeQueue<T> &owner, std::unique_lock<std::mutex> &&lock)
        : m_owner(std::ref(owner))
        , m_lock(std::move(lock))
    {}

    ThreadSafeQueueHandle(ThreadSafeQueueHandle &&) noexcept = default;
    ThreadSafeQueueHandle &operator=(ThreadSafeQueueHandle &&) noexcept = default;

    /**
     * Add a new object to the queue.
     * @param[in] val   An object to add to the end of the queue.
     *
     * @note    This function gets the object by value which implies that he ownership of the value is transferred to
     *          the queue. It's still possible to store pointers or std::ref in the queue.
     */
    void Enqueue(T val) const
    {
        m_owner.get().m_queue.push(std::move(val));
    }

    /**
     * Remove and element from the head of the queue.
     * @return  An object from the queue. The object ownership is transferred from the queue to the caller of this
     *          function.
     */
    [[nodiscard]] T Dequeue() const
    {
        auto result = std::move(m_owner.get().m_queue.front());
        m_owner.get().m_queue.pop();
        return result;
    }

    /**
     * Return the size of the underlying queue.
     * As this method is called for a locked queue, the result value is the exact size of the queue as no
     * insertions/deletions are allowed at the moment.
     *
     * @return Size of the locked queue.
     */
    [[nodiscard]] std::size_t GetSize() const
    {
        return m_owner.get().m_queue.size();
    }

    /**
     * Returns if the queue is empty or not.
     *
     * @return Emptines flag for the locked queue.
     */
    [[nodiscard]] bool IsEmpty() const
    {
        return m_owner.get().m_queue.empty();
    }

private:
    std::reference_wrapper<ThreadSafeQueue<T>> m_owner;
    std::unique_lock<std::mutex> m_lock;
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
    ThreadSafeQueueHandle<T> Lock()
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        return ThreadSafeQueueHandle<T> { *this, std::move(lck) };
    }

private:
    friend class ThreadSafeQueueHandle<T>;

    std::mutex m_mutex;
    std::queue<T> m_queue;
};
} // namespace DcgmNs
