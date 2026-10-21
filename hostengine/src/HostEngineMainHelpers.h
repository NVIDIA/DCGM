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

#include "dcgm_agent.h"

#include <csignal>
#include <sys/types.h>

/** Global stop flag set by hostengine signal handlers and read by the main loop. */
extern volatile std::sig_atomic_t g_stopLoop;

/**
 * Writes a process ID to a pid file.
 *
 * @param[in] pidfile Path to the pid file to overwrite.
 * @param[in] pid Process ID to write.
 * @return 0 on success; -1 if the file cannot be opened.
 */
int write_to_pidfile(char const *pidfile, pid_t pid);

/**
 * Reads a process ID from a pid file.
 *
 * @param[in] pidfile Path to the pid file to read.
 * @param[out] pid Destination for the parsed process ID on success.
 * @return 0 on success; -1 if the file cannot be opened or parsed.
 */
int read_from_pidfile(char const *pidfile, pid_t *pid);

/**
 * Checks whether a process appears to be alive.
 *
 * @param[in] pid Process ID to query.
 * @return 1 when the process exists or cannot be signaled due to permissions; 0 otherwise.
 */
int isDaemonProcessAlive(pid_t pid);

/**
 * Checks whether a pid file points to a live hostengine process.
 *
 * @param[in] pidfile Path to the pid file to inspect.
 * @return true when the pid file contains a live process ID; false otherwise.
 */
bool isDaemonRunning(char const *pidfile);

/**
 * Attempts to terminate the daemon described by a pid file.
 *
 * @param[in] pidfile Path to the pid file to inspect.
 * @return true when a running daemon was signaled and exited; false otherwise.
 */
bool termDaemonIfRunning(char const *pidfile);

/**
 * Removes a stale pid file or exits if it points at a live daemon.
 *
 * @param[in] pidfile Path to the pid file to validate and possibly remove.
 */
void delete_pidfile_if_not_running_or_exit(char const *pidfile);

/**
 * Creates the daemon pid file for the current process.
 *
 * @param[in] pidfile Path to the pid file to create.
 * @param[in] parentPid Parent process to notify if pid file creation fails.
 */
void create_daemon_pid_file(char const *pidfile, pid_t parentPid);

/**
 * Terminates the daemon described by a pid file and exits with the operation result.
 *
 * @param[in] pidFilePath Path to the daemon pid file.
 * @return This function exits the process before returning.
 */
int terminateHostEngineDaemon(char const *pidFilePath);

/**
 * Redirects daemon console output to /dev/null and notifies the parent process.
 *
 * @param[in] parentPid Parent process to signal with SIGUSR1.
 */
void daemonCloseConsoleOutput(pid_t parentPid);

/**
 * Hostengine signal handler that requests main-loop shutdown.
 *
 * @param[in] signum Signal number delivered by the OS.
 */
void sig_handler(int signum);

/**
 * Installs hostengine control signal handlers.
 *
 * @return 0 on success; -1 if any signal disposition cannot be installed.
 */
int InstallCtrlHandler();

/**
 * Stops embedded DCGM state and returns the supplied hostengine status.
 *
 * @param[in] dcgmHandle Embedded DCGM handle to stop, or nullptr when not started.
 * @param[in] status Status code to return after cleanup.
 * @param[in] parentPid Parent process to notify when initialization failed.
 * @return The supplied status value.
 */
int cleanup(dcgmHandle_t dcgmHandle, int status, int parentPid);
