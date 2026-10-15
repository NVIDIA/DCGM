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

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

/**
 * Helpers for tests that need to isolate exit-oriented code in child processes.
 */
class ChildProcessUtils
{
public:
    /**
     * Runs a callable in a forked child process and verifies the child exits with the expected status.
     *
     * @tparam ChildBody Callable type executed in the child process.
     * @param[in] childBody Callable executed in the child process.
     * @param[in] expectedExitStatus Exit code required from the child process.
     */
    template <typename ChildBody>
    static void RunChildAndRequireExitStatus(ChildBody childBody, int expectedExitStatus)
    {
        pid_t const pid = fork();
        REQUIRE(pid >= 0);

        if (pid == 0)
        {
            childBody();
            _exit(EXIT_SUCCESS);
        }

        int status = 0;
        REQUIRE(waitpid(pid, &status, 0) == pid);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == expectedExitStatus);
    }
};
