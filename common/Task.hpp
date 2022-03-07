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

#include <functional>
#include <future>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

namespace DcgmNs
{
template <class T>
struct OptionalNestedType
{
    using type = T;
};

template <class T>
struct OptionalNestedType<std::optional<T>>
{
    using type = typename OptionalNestedType<T>::type;
};

/**
 * This is a helper trait that allows to get type T from the optional<T> type
 */
template <class T>
using OptionalNestedType_t = typename OptionalNestedType<T>::type;

/**
 * This is a base class for all scheduled tasks.
 * TaskRunner keeps only this interface in its queue.
 * All concrete tasks should inherit this interface publicly.
 */
struct ITask
{
    enum class RunResult
    {
        Ok,       //!< The task finished its work
        Deferred, //!< The task is still working and should be put back in a queue
    };
    ITask()          = default;
    virtual ~ITask() = default;
    ITask(ITask &&)  = default;
    ITask &operator=(ITask const &) = default;

    /**
     * Perform real work inside this function
     *
     * @return  RunResult::Ok           When the task finished its work and can be reported to the original requester.
     *          RunResult::Deferred     When the task cannot be finished right now and should be rescheduled to run
     *                                  later. This function will be called again sometime in the future.
     */
    virtual RunResult Run() = 0;

    /**
     * Return a name for the given task.
     * It's up to the implementation to decide what those names should be.
     *
     * The name will be used for logging purposes and in other places where
     * human readable explanation and destinction of the task is preferred.
     */
    [[nodiscard]] virtual char const *GetName() const noexcept = 0;
};

namespace
{
    /**
     * A helper function to represent size_t (usually an address) in a hexadecimal form.
     * @param val   A numeric value to represent in the hex form
     *
     * @return Hexadecimal string in the following form `0xabcdef0123`
     */
    inline std::string to_hex_string(size_t val)
    {
        std::stringstream ss;
        ss << "0x" << std::hex << val;
        return ss.str();
    }
} // namespace

/**
 * Base class for all templated tasks and deferred.
 * @tparam T            The type of the task result
 * @tparam PromiseType  The type of the returning promise/future.
 *                      This type is equal T by default.
 *                      This type cannot be `void`
 *
 * Objects of this class are not copyable but moveable.
 *
 * @sa class Task<void>
 * @sa class Task<T>
 */
template <class T, class PromiseType = T>
class NamedBasicTask : public ITask
{
public:
    /**
     * Creates NamedBasicTask with given task name and a function
     *
     * @param[in] taskName  A mnemonic human readable name of the task.
     *                      The name cannot be changed during the livetime of the object.
     * @param[in] func      A function that fill be executed inside `NamedBasicTask::Run()`
     */
    NamedBasicTask(std::string taskName, std::function<std::optional<T>()> func)
        : m_func(std::move(func))
        , m_taskName(std::move(taskName))
    {}

    /**
     * Creates NamedBasicTask with a function that will be executed inside the `NamedBasicTask::Run()`
     * The name of the task will be generated automatically based on the address of the object.
     * The name cannot be changed during the livetime of the object.
     *
     * @note If the object is created on a stack and moved to a heap later (e.g. moved to a unique_ptr)
     *       the name will still have an address in the stack. That may have implication if name is used for debugging
     *       purposes.
     * @param[in] func  A function that will be executed inside `NamedBasicTask::Run()`
     */
    NamedBasicTask(std::function<std::optional<T>()> func)
        : NamedBasicTask(std::string("Unknown at 0x") + to_hex_string(std::size_t(this)), std::move(func))
    {}

    NamedBasicTask(NamedBasicTask const &) = delete;
    NamedBasicTask &operator=(NamedBasicTask const &) = delete;

    NamedBasicTask(NamedBasicTask &&) noexcept = default;
    NamedBasicTask &operator=(NamedBasicTask &&) noexcept = default;

    /**
     * Specified a promise which value will be set once the `NamedBasicTask::Run()` is done
     *
     * @param[in] promise   This promise will be used to set value and notify that the task is ready.
     *
     * @note It is expected that a caller has already got a future from this promise.
     */
    void SetPromise(std::promise<PromiseType> promise)
    {
        m_promise = std::make_unique<std::promise<PromiseType>>(std::move(promise));
    }

    /**
     * Perform the main work of the task.
     *
     * @return  RunResult::Ok           This task is completely done and the promise value has been set.
     *          RunResult::Deferred     This task is not ready yet and this function should be called later again.
     */
    [[nodiscard]] RunResult Run() override
    {
        auto &&result = std::invoke(m_func);
        if (!result.has_value())
        {
            return RunResult::Deferred;
        }

        if (m_promise)
        {
            if constexpr (std::is_same_v<void, PromiseType>)
            {
                m_promise->set_value();
            }
            else
            {
                m_promise->set_value(std::move(result.value()));
            }
        }

        return RunResult::Ok;
    }

    /**
     * Returns the name associated with the task
     */
    [[nodiscard]] char const *GetName() const noexcept override
    {
        return m_taskName.c_str();
    }

private:
    std::unique_ptr<std::promise<PromiseType>> m_promise;
    std::function<std::optional<T>()> m_func;
    std::string m_taskName;
};

template <class T>
class Task;

/**
 * Task specialization for the case when the passed to the constructor function does not return anything (aka void).
 */
template <>
class Task<void> : public NamedBasicTask<int, void>
{
public:
    /**
     * Creates a Task<void> instance with given task name and function.
     * @sa `NamedBasicTask::NamedBasicTask(std::string, std::function)`
     */
    Task(std::string taskName, std::function<void()> func)
        : NamedBasicTask<int, void>(std::move(taskName), [func = std::move(func)] {
            std::invoke(func);
            return 0;
        })
    {}

    /**
     * Creates a Task<void> instance with given function
     * @sa `NamedBasicTask::NamedBasicTask(std::function)`
     */
    Task(std::function<void()> func)
        : NamedBasicTask<int, void>([func = std::move(func)] {
            std::invoke(func);
            return 0;
        })
    {}
};

/**
 * Task specialization for cases when the passed to the constructor function returns some value.
 * @tparam T    Type of the final task result
 */
template <class T>
class Task : public NamedBasicTask<T>
{
public:
    /**
     * Creates a Task<T> instance with given task name and function.
     * @sa `NamedBasicTask::NamedBasicTask(std::string, std::function)`
     */
    Task(std::string taskName, std::function<std::optional<T>()> func)
        : NamedBasicTask<T>(std::move(taskName), std::move(func))
    {}

    /**
     * Creates a Task<T> instance with given function
     * @sa `NamedBasicTask::NamedBasicTask(std::function)`
     */
    Task(std::function<std::optional<T>()> func)
        : NamedBasicTask<T>(std::move(func))
    {}
};

/**
 * Helper function to create proper Task<T> from a given function
 * @param[in] func  A function that will perform main task work. Can be a lambda with `void` return type
 */
template <class Fn>
auto make_task(Fn &&func) -> Task<OptionalNestedType_t<std::invoke_result_t<Fn>>>
{
    return Task<OptionalNestedType_t<std::invoke_result_t<Fn>>> { std::forward<Fn>(func) };
}

/**
 * Helper function to create proper Task<T> from a given function
 * @param[in] taskName  Name of the crated task. @sa See `NamedBasicTask::NamedBasicTask()` for the name definition.
 * @param[in] func      A function that will perform main task work. Can be a lambda with `void` return type
 */
template <class Fn>
auto make_task(std::string taskName, Fn &&func) -> Task<OptionalNestedType_t<std::invoke_result_t<Fn>>>
{
    return Task<OptionalNestedType_t<std::invoke_result_t<Fn>>> { std::move(taskName), std::forward<Fn>(func) };
}

} // namespace DcgmNs
