/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifdef __cplusplus
#include <algorithm>
#include <forward_list>
#include <gettid.h>
#include <mutex>
#include <sys/types.h>

/**
 * This class is used to track the state of tasks monitored for hang detection.
 * We plan to replace this initial tracking implementation with a lock-free list.
 */
class TaskContextManager
{
public:
    /**
     * Context will be tracked for the specified task.
     * @param taskId The task ID to check.
     */
    void addTask(pid_t const tid)
    {
        std::forward_list<pid_t> newTasks = { tid };
        std::lock_guard<std::mutex> lock(m_mutex);
        m_tasks.splice_after(m_tasks.before_begin(), newTasks);
    }

    /**
     * Add tracking context for the current task.
     */
    void addTask()
    {
        addTask(gettid());
    }

    /**
     * End tracking context for the specified task.
     * @param taskId The task ID to check.
     */
    void removeTask(pid_t const tid) noexcept
    {
        std::forward_list<pid_t> toRemove;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto cur = m_tasks.begin(), prev = m_tasks.before_begin(); cur != m_tasks.end();
             prev = cur, cur = std::next(cur))
        {
            if (*cur == tid)
            {
                toRemove.splice_after(toRemove.before_begin(), m_tasks, prev);
                return;
            }
        }
    }

    /**
     * End tracking context for the current task.
     */
    void removeTask() noexcept
    {
        removeTask(gettid());
    }

    /**
     * Check if the specified task matches criteria for selection.
     * @param taskId The task ID to check.
     */
    bool isTaskIncluded(pid_t const tid) const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::find(m_tasks.begin(), m_tasks.end(), tid) != m_tasks.end();
    }

    /**
     * Check if the current task matches criteria for selection.
     */
    bool isTaskIncluded() const noexcept
    {
        return isTaskIncluded(gettid());
    }

private:
    std::forward_list<pid_t> m_tasks;
    mutable std::mutex m_mutex;
};

extern "C" {
#endif // __cplusplus

/**
 * Get the TaskContextManager instance.
 */
void *GetTaskContextManager();

/**
 * Add the current task to the TaskContextManager.
 */
void TaskContextManagerAddCurrentTask();

/**
 * Remove the current task from the TaskContextManager.
 */
void TaskContextManagerRemoveCurrentTask();

#ifdef __cplusplus
}
#endif // __cplusplus
