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

#include "Semaphore.hpp"
#include "Task.hpp"
#include "ThreadSafeQueue.hpp"

#include "DcgmLogging.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>


namespace DcgmNs
{
template <class T>
struct UnwrapFutureNestedType
{
    using type = std::decay_t<T>;
};

template <class T>
struct UnwrapFutureNestedType<std::future<T>>
{
    using type = typename UnwrapFutureNestedType<T>::type;
};


/**
 * This is a helper trait that allows to extract type T from a chain like
 * future<future<future<T>>>
 */
template <class T>
using UnwrapFutureNestedType_t = typename UnwrapFutureNestedType<T>::type;

/**
 * A class that represents Multiple-Producers-Single-Consumer working queue.
 * Only one thread is running the scheduled tasks and multiple threads can enqueue tasks for execution.
 *
 * @note    It's technically possible to make multiple threads running the `Run()` function. But there are no quarantees
 *          that workload will be evenly distributed between such threads.
 */
class TaskRunner
{
public:
    TaskRunner()
        : m_runInterval(std::chrono::minutes(1))
        , m_runnerSemaphore(std::make_shared<Semaphore>())
        , m_stop(false)
        , m_debugLogging(false)
    {
        auto *envVar = getenv("__DCGM_TASK_RUNNER_QUEUE_SIZE");
        if (envVar != nullptr)
        {
            try
            {
                m_queueCapacity = std::min(std::stoul(envVar), 100000UL);
            }
            catch (...)
            {
                DCGM_LOG_ERROR
                    << "Bad value of the __DCGM_TASK_RUNNER_QUEUE_SIZE environment variable. A positive integer is expected";
            }
        }

        envVar = getenv("__DCGM_TASK_RUNNER_DEBUG");
        if (envVar != nullptr)
        {
            try
            {
                m_debugLogging = std::stoul(envVar) == 1UL;
            }
            catch (...)
            {
                DCGM_LOG_ERROR
                    << "Bad value of the __DCGM_TASK_RUNNER_DEBUG environment variable. A 0 or 1 numbers are expected";
            }
        }
    }

    virtual ~TaskRunner() = default;

    /**
     * Enable debug logging for the operations
     * @param[in] enabled   Whether the debug logging should be enabled or not.
     * @note    The underlying flag is thread safe and its change will be eventually visible to the working thread(s).
     *          Thus, you are not required to set this value before running the `Run()` working thread.
     */
    void SetDebugLogging(bool enabled)
    {
        m_debugLogging = enabled;
        if (enabled)
        {
            log_info("Debug logging is enabled for the TaskRunner at {}", (void *)this);
        }
        m_runnerSemaphore->SetDebugLogging(enabled);
    }

    void SetQueueCapacity(size_t newCapacity)
    {
        m_queueCapacity.store(newCapacity, std::memory_order_relaxed);
    }

    /**
     * Schedule a new task for execution.
     * This method is for simple (not deferred tasks)
     * @tparam T    A type of the final task result. Can be void. @sa `class DcgmNs::Task<void>` for details.
     *
     * @param[in] task  A task to add to the queue. Task ownership is transferred to the TaskRunner
     * @param[in] isContinuation    The \a task is a continuation for another task that is already in the queue so the
     *                              queue capacity should not be checked.
     *
     * @return An optional instance of std::shared_future<T>. A caller may wait on this future till the task is
     *         finished. The value will be \c std::nullopt if the \a task cannot be added due to the queue capacity
     *         exhaustion.
     */
    template <class T, class P>
    [[nodiscard]] auto Enqueue(std::unique_ptr<NamedBasicTask<T, P>> task,
                               bool isContinuation = false) -> std::optional<std::shared_future<P>>
    {
        if (m_debugLogging.load(std::memory_order_relaxed))
        {
            DCGM_LOG_DEBUG << fmt::format(
                "Enqueueing simple task '{}' for the {:#x} TaskRunner", task->GetName(), (size_t)this);
        }

        if (!isContinuation)
        {
            auto queueHandleRo = m_queue.LockRO();
            if (queueHandleRo.GetSize() > m_queueCapacity.load(std::memory_order_relaxed))
            {
                DCGM_LOG_ERROR << fmt::format(
                    "Unable to add task {} to the TaskRunner {:#x} queue as the queue is full",
                    task->GetName(),
                    (size_t)this);
                [[maybe_unused]] auto releaseResult = m_runnerSemaphore->Release();
                if (releaseResult != Semaphore::ReleaseResult::Ok)
                {
                    auto const msg = fmt::format(
                        "Unable to trigger consumers for the TaskRunner {:#x} while the event queue is full",
                        (size_t)this);
                    DCGM_LOG_FATAL << msg;
                    throw std::runtime_error(msg);
                }
                return std::nullopt;
            }
        }

        std::promise<P> prom;
        auto result = prom.get_future().share();

        task->SetPromise(std::move(prom)); /* after this prom is invalid */

        auto queueHandle = m_queue.LockRW();
        if (!isContinuation && queueHandle.GetSize() > m_queueCapacity.load(std::memory_order::relaxed))
        {
            DCGM_LOG_ERROR << fmt::format("Unable to add task {} to the TaskRunner {:#x} queue as the queue is full",
                                          task->GetName(),
                                          (size_t)this);
            [[maybe_unused]] auto releaseResult = m_runnerSemaphore->Release();
            if (releaseResult != Semaphore::ReleaseResult::Ok)
            {
                auto const msg = fmt::format(
                    "Unable to trigger consumers for the TaskRunner {:#x} while the event queue is full", (size_t)this);
                DCGM_LOG_FATAL << msg;
                throw std::runtime_error(msg);
            }
            return std::nullopt;
        }
        queueHandle.Enqueue(std::move(task));
        [[maybe_unused]] auto _ = m_runnerSemaphore->Release();

        return result;
    }

    /**
     * Schedule a deferred task for execution.
     * Deferred task returns another future instead of the final result.
     * @tparam T    This may or may not be a type of the final task result. See notes.
     *
     * @param[in] task  A deferred task to execute
     *
     * @return An instance of std::optional<std::shared_future<Y>>. A caller may wait on this future till the task is
     *         finished.
     * @note    The optional result type of this function is std::shared_future<Y> where Y is the unwrapped type of the
     *          final task result. That means if a Task<shared_future<shared_future<T>> was passed as the `task` param,
     *          the result of this method will have std::shared_future<T> type. That is needed to allow callers to call
     *          `.get()` only once on the returned future instead of making a chain of `.get().get().get()` calls.
     *
     * This is mostly a helper function which allows to make chains of asynchronous tasks.
     * @code{.cpp}
     * ```
     *      TaskRunner tr;
     *      auto fut = tr.Enqueue(make_task([&tr]() mutable {
     *          auto f1 = tr.Enqueue(make_task([]{ return LongFunctionResult(); }));
     *          auto f2 = tr.Enqueue(make_task([&tr]() mutable { return AnotherLongFuncResult(tr); }));
     *          return tr.Enqueue(make_task([f1=std::move(f1), f2=std::move(f2)]{
     *              if (f1.has_value()) {
     *                  if (f2.has_value()) {
     *                      return (*f1).get() + (*f2).get();
     *                  } else {
     *                      return (*f1).get();
     *                  }
     *              } else if (f2.has_value()) {
     *                  return (*f2).get();
     *              }
     *              throw std::runtime_error("Neither task was scheduled");
     *          }));
     *      }));
     *      printf("Result: %d", fut.has_value() ? (*fut).get() : "Not executed");
     *
     *      auto AnotherLongFuncResult(TaskRunner &tr){
     *          return tr.Enqueue(make_task([]{ return SomeElseResult(); }));
     *      }
     * ```
     * @endcode
     */
    template <class T, class P>
    auto Enqueue(std::unique_ptr<NamedBasicTask<std::shared_future<T>, P>> task)
        -> std::optional<std::shared_future<UnwrapFutureNestedType_t<T>>>
    {
        if (m_debugLogging.load(std::memory_order_relaxed))
        {
            DCGM_LOG_DEBUG << fmt::format(
                "Enqueueing deferred task '{}' for the {:#x} TaskRunner", task->GetName(), (size_t)this);
        }
        {
            auto queueHandleRo = m_queue.LockRO();
            if (queueHandleRo.GetSize() > m_queueCapacity.load(std::memory_order::relaxed))
            {
                DCGM_LOG_ERROR << fmt::format(
                    "Unable to add task {} to the TaskRunner {:#x} queue as the queue is full",
                    task->GetName(),
                    (size_t)this);
                [[maybe_unused]] auto releaseResult = m_runnerSemaphore->Release();
                if (releaseResult != Semaphore::ReleaseResult::Ok)
                {
                    auto const msg = fmt::format(
                        "Unable to trigger consumers for the TaskRunner {:#x} while the event queue is full",
                        (size_t)this);
                    DCGM_LOG_FATAL << msg;
                    throw std::runtime_error(msg);
                }
                return std::nullopt;
            }
        }

        std::promise<std::shared_future<T>> contProm;
        auto contFuture = contProm.get_future().share();

        task->SetPromise(std::move(contProm)); /* after this prom is invalid */
        auto taskNameCopy = task->GetName();

        {
            auto queueHandle = m_queue.LockRW();
            if (queueHandle.GetSize() > m_queueCapacity.load(std::memory_order_relaxed))
            {
                DCGM_LOG_ERROR << "Unable to add task " << task->GetName() << " to the TaskRunner " << (size_t)this
                               << " queue as the queue is full";
                [[maybe_unused]] auto releaseResult = m_runnerSemaphore->Release();
                if (releaseResult != Semaphore::ReleaseResult::Ok)
                {
                    auto const msg = fmt::format(
                        "Unable to trigger consumers for the TaskRunner {:#x} while the event queue is full",
                        (size_t)this);
                    DCGM_LOG_FATAL << msg;
                    throw std::runtime_error(msg);
                }
                return std::nullopt;
            }
            queueHandle.Enqueue(std::move(task));
            [[maybe_unused]] auto _ = m_runnerSemaphore->Release();
        }

        auto continuation = [contFuture = std::move(contFuture)]() mutable -> std::optional<T> {
            if (contFuture.get().wait_for(std::chrono::microseconds(0)) != std::future_status::ready)
            {
                return std::nullopt;
            }

            return contFuture.get().get();
        };

        using namespace std::literals::string_literals;

        return Enqueue(make_task("Auxiliary for a deferred task '"s + taskNameCopy + "'"s, std::move(continuation)),
                       true);
    }

    /**
     * This enum describes possible results of the Run() method
     */
    enum class [[nodiscard]] RunResult : std::uint8_t
    {
        Ok,            //!< The Run method can be called again
        NoLongerValid, //!< The Run method should not be called again
    };

    /**
     * Executes scheduled tasks.
     * This method has the following semantics:
     *  It waits for a given amount of time ( @sa `SetRunInterval()` ) for new tasks.
     *  If there is no signal about new tasks, the runner will check its queue anyway in case there are
     *      some postponed tasks, and return.
     *  If new tasks were signalled (via semaphore), worker will process all tasks in the queue.
     *      Postponed (deferred) tasks will be re-added to the queue but will not be signalled.
     *
     * Here is a example how this method could be used:
     * @code{.cpp}
     * ```
     *  TaskRunner tr;
     *  ...
     *  while(!ShouldStop()) {
     *      if (TaskRunner::RunResult::NoLongerValid == tr.Run()){
     *          break;
     *      };
     *      DoSomeMoreWorkHere();
     *  }
     * ```
     * @endcode
     *
     * @param[in] oneIteration      Do not spin TaskRunner until the given run interval is exhausted.
     *                              If true, TaskRunner will wake up at semaphore event, process the queue and return.
     *                              If false, TaskRunner will wake up at semaphore event, process the queue and wait on
     *                                  the semaphore if the run interval is not exhausted yet.
     * @return
     *        - RunResult::Ok               If the Run finished successfully and can be called again
     *        - RunResult::NoLongerValid    If the TaskRunner is already stopped or Run cannot be called again
     *
     * @note It's recommended to call this method from a dedicated thread.
     */
    RunResult Run(bool oneIteration = false) noexcept(false)
    {
        if (m_stop.load(std::memory_order_relaxed))
        {
            return RunResult::NoLongerValid;
        }

        using namespace std::literals::chrono_literals;

        auto waitResult = Semaphore::TimedWaitResult::Ok;
        auto wakeUpTime = std::chrono::system_clock::now() + m_runInterval.load(std::memory_order_relaxed);

        std::vector<std::unique_ptr<ITask>> tasks;
        std::vector<std::unique_ptr<ITask>> deferredTasks;

        while (waitResult == Semaphore::TimedWaitResult::Ok && !m_stop.load(std::memory_order_relaxed))
        {
            tasks.clear();
            deferredTasks.clear();
            waitResult = m_runnerSemaphore->WaitUntil(wakeUpTime);
            if (waitResult == Semaphore::TimedWaitResult::Destroyed)
            {
                if (m_debugLogging.load(std::memory_order_relaxed))
                {
                    DCGM_LOG_DEBUG << "Underlying semaphore was destroyed during the TaskRunner run loop";
                }
                return RunResult::NoLongerValid;
            }

            if (m_stop.load(std::memory_order_relaxed))
            {
                DCGM_LOG_DEBUG << "TaskRunner is interrupted by a stop signal and will not process remaining tasks";
                return RunResult::NoLongerValid;
            }

            /*
             * We need to check the queue even if the semaphore has timed out.
             */
            std::queue<decltype(m_queue)::QueueItemType> tmpTasks;
            {
                {
                    auto queueHandle = m_queue.LockRO();
                    if (queueHandle.IsEmpty())
                    {
                        continue;
                    }
                }

                auto queueHandle = m_queue.LockRW();
                if (m_debugLogging.load(std::memory_order_relaxed))
                {
                    DCGM_LOG_DEBUG << "TaskRunner is consuming " << std::to_string(queueHandle.GetSize())
                                   << " tasks from the queue";
                }

                tasks.reserve(queueHandle.GetSize());
                deferredTasks.reserve(queueHandle.GetSize());

                queueHandle.Swap(tmpTasks);
            }

            while (!tmpTasks.empty())
            {
                tasks.emplace_back(std::move(tmpTasks.front()));
                tmpTasks.pop();
            }

            for (auto &task : tasks)
            {
                if (m_stop.load(std::memory_order_relaxed))
                {
                    if (m_debugLogging.load(std::memory_order_relaxed))
                    {
                        DCGM_LOG_DEBUG
                            << "TaskRunner is interrupted by a stop signal and will not process remaining tasks";
                    }
                    return RunResult::NoLongerValid;
                }

                if (m_debugLogging.load(std::memory_order_relaxed))
                {
                    DCGM_LOG_DEBUG << "TaskRunner is going to run the '" << task->GetName() << "' task";
                }

                switch (task->Run())
                {
                    case ITask::RunResult::Deferred:
                        if (m_debugLogging.load(std::memory_order_relaxed))
                        {
                            DCGM_LOG_DEBUG << "The task '" << task->GetName() << "' is not ready and will be deferred";
                        }
                        deferredTasks.push_back(std::move(task));
                        break;
                    case ITask::RunResult::Ok:
                        if (m_debugLogging.load(std::memory_order_relaxed))
                        {
                            DCGM_LOG_DEBUG << "The task '" << task->GetName() << "' is done";
                        }
                        break;
                }
            }

            if (!deferredTasks.empty())
            {
                auto queueHandle = m_queue.LockRW();
                for (auto &task : deferredTasks)
                {
                    queueHandle.Enqueue(std::move(task));
                }
                [[maybe_unused]] auto _ = m_runnerSemaphore->Release();
            }

            if (oneIteration)
            {
                break;
            }
        }

        return RunResult::Ok;
    }

    /**
     * Notify the TaskRunner that it should stop its processing.
     * @sa See `TaskRunner::Run()` documentation for details.
     */
    void Stop() noexcept(noexcept(std::declval<Semaphore>().Destroy()))
    {
        if (m_debugLogging.load(std::memory_order_relaxed))
        {
            DCGM_LOG_DEBUG << fmt::format("The TaskRunner {:#x} is going to stop", (std::size_t)this);
        }
        m_stop.store(true, std::memory_order_relaxed);
        m_runnerSemaphore->Destroy(); // Semaphore.Destroy will wake up all waiters (runner threads)
    }

protected:
    /**
     * Change the semaphore waiting timeout in the Run() method.
     * If this value is too big, the task runner becomes irresponsible if tasks are rare.
     */
    void SetRunInterval(std::chrono::milliseconds const interval)
    {
        m_runInterval.store(interval, std::memory_order_relaxed);
    }

private:
    ThreadSafeQueue<std::unique_ptr<ITask>> m_queue;      //!< A queue of scheduled tasks. Thread Safe.
    std::atomic<std::chrono::milliseconds> m_runInterval; //!< How long a worker will wait for a new task signal.
    std::shared_ptr<Semaphore> m_runnerSemaphore;         //!< Signals if there is new task in the queue.
    std::atomic_bool m_stop;                              //!< Signals that the task runner should stop its work.

    std::atomic_bool m_debugLogging; //!< If the task runner methods need to write debug logs.
                                     //!< Can be overridden by env variable __DCGM_TASK_RUNNER_QUEUE_SIZE

    std::atomic_size_t m_queueCapacity = 100; //!< How many events can be stored in the queue.
                                              //!< Attempt to add more events will fail.
                                              //!< Can be overridden by env variable __DCGM_TASK_RUNNER_QUEUE_SIZE
};

} // namespace DcgmNs
