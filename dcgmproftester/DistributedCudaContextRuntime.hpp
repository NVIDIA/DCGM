/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <sys/types.h>

namespace DcgmNs::ProfTester
{
class DistributedCudaContextRuntime
{
public:
    /**
     * Create a pipe.
     *
     * @param[out] pipeFds Receives the read and write file descriptors.
     * @return 0 on success, or -1 on failure with errno set.
     */
    int Pipe(int pipeFds[2]);

    /**
     * Create a child process.
     *
     * @return 0 in the child, the child PID in the parent, or -1 on failure with errno set.
     */
    pid_t Fork();

    /**
     * Wait for a child process state change.
     *
     * @param[in] pid Child selection passed to waitpid.
     * @param[out] status Receives the child status when non-null.
     * @param[in] options Options passed to waitpid.
     * @return The selected child PID, 0 for a nonblocking wait with no state change, or -1 on failure.
     */
    pid_t WaitPid(pid_t pid, int *status, int options);

    /**
     * Send a signal to a process.
     *
     * @param[in] pid Process to signal.
     * @param[in] signal Signal number to send.
     * @return 0 on success, or -1 on failure with errno set.
     */
    int Kill(pid_t pid, int signal);
};

} // namespace DcgmNs::ProfTester
