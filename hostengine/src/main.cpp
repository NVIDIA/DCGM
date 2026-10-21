/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include "HostEngineCommandLine.h"
#include "HostEngineMainHelpers.h"

#include <DcgmBuildInfo.hpp>
#include <UserCredentials.hpp>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <syslog.h>
#include <thread>

#include <fcntl.h>
#include <unistd.h>


// Exit when SIG_USR1 is received
// Block the signal so that it is not printed to the console
static void awaitDeathBySigUsr1()
{
    sigset_t sigset;
    siginfo_t sig;

    memset(&sigset, 0, sizeof(sigset));
    memset(&sig, 0, sizeof(sig));

    sigaddset(&sigset, SIGUSR1);
    sigprocmask(SIG_BLOCK, &sigset, NULL); // block signal

    struct timespec timeout;
    // Set the timeout value for the nv-hostengine parent process signal wait time
    // to be long enough for the child process to finish all the initializations.
    timeout.tv_sec  = 120;
    timeout.tv_nsec = 0;

    // await signal to die.  In the rare case that the signal never arrives, die
    // after a timeout that is longer than how long initialization should reasonably take
    int result = sigtimedwait(&sigset, &sig, &timeout);
    if (result < 0)
    {
        printf("Got error %d while waiting for SIGUSR1 from child process.\n", errno);
        exit(EXIT_FAILURE);
    }
    else
    {
        // printf("Caught signal %d from our child process\n", result);
        exit(EXIT_SUCCESS);
    }
}

/*****************************************************************************
 * Method to daemonize the host engine
 *****************************************************************************/
static void heDaemonize()
{
    pid_t pid;
    int fd;

    /* Fork off the parent process */
    pid = fork();

    /* An error occurred */
    if (pid < 0)
        exit(EXIT_FAILURE);

    // Success,
    // first parent should stay alive until after DCGM initialization so that error messages
    // can still be directed to the terminal.  After init, it will be killed by the daemon sending it a signal
    if (pid > 0)
    {
        awaitDeathBySigUsr1();
    }

    /* On success: The child process becomes session leader */
    if (setsid() < 0)
        exit(EXIT_FAILURE);

    /* Fork off for the second time*/
    pid = fork();

    /* An error occurred */
    if (pid < 0)
        exit(EXIT_FAILURE);

    /* Success: Let the parent terminate */
    if (pid > 0)
        exit(EXIT_SUCCESS);

    /* Set new file permissions */
    umask(0);

    /* Close all open file descriptors except stdout/stderr.
     * These are closed later after initialization so that error messages still be sent to the terminal */
    for (fd = sysconf(_SC_OPEN_MAX); fd > 0; fd--)
    {
        if (fd == STDOUT_FILENO || fd == STDERR_FILENO)
            continue;
        close(fd);
    }

    fd = open("/dev/null", 02, 0);
    if (fd != -1)
    {
        dup2(fd, STDIN_FILENO);
    }

    /* Open the log file */
    openlog("nvhostengine_daemon", LOG_PID, LOG_DAEMON);
    // coverity[leaked_handle] leave fd open
}


static void TryCreateDcgmHomeDir(std::string const &serviceAccount)
{
    auto const *dcgmHomeDir = getenv(DCGM_HOME_DIR_VAR_NAME);
    if (dcgmHomeDir == nullptr)
    {
        return;
    }

    std::error_code ec;
    if (std::filesystem::exists(dcgmHomeDir, ec) || ec)
    {
        return;
    }

    if (!std::filesystem::create_directory(dcgmHomeDir, ec))
    {
        syslog(LOG_ERR, "failed to create folder: [%s] with error: [%s].", dcgmHomeDir, ec.message().c_str());
        return;
    }

    // If no --service-account, leave ownership as is
    if (serviceAccount.empty())
    {
        return;
    }

    try
    {
        auto const creds = GetUserCredentials(serviceAccount.c_str());
        if (!creds.has_value())
        {
            syslog(LOG_ERR, "service account [%s] not found for [%s]", serviceAccount.c_str(), dcgmHomeDir);
        }
        else if (chown(dcgmHomeDir, creds->uid, creds->gid) != 0)
        {
            syslog(LOG_ERR, "failed to chown [%s] to [%s]: %s", dcgmHomeDir, serviceAccount.c_str(), strerror(errno));
        }
    }
    catch (std::exception const &ex)
    {
        syslog(LOG_ERR, "failed to resolve service account [%s]: %s", serviceAccount.c_str(), ex.what());
    }
}

int main(int argc, char **argv)
{
    dcgmReturn_t ret;
    dcgmHandle_t dcgmHandle = (dcgmHandle_t) nullptr;
    pid_t parentPid         = getpid();

    HostEngineCommandLine cmdLine;
    try
    {
        cmdLine = ParseCommandLine(argc, argv);
    }
    catch (std::exception const &ex)
    {
        syslog(LOG_ERR, "nv-hostengine failed to parse command line arguments: %s", ex.what());
        fprintf(stderr, "nv-hostengine failed to parse command line arguments: %s\n", ex.what());
        return -1;
    }

    /* Check if the user is not root. Return if not */
    if (geteuid() != 0)
    {
        syslog(LOG_NOTICE, "nv-hostengine running as non-root. Some functionality will be limited.");
        fprintf(stderr, "nv-hostengine running as non-root. Some functionality will be limited.\n");
    }

    // coverity[var_deref_model] If we get this far, cmdLine is initialized
    if (cmdLine.ShouldTerminate())
    {
        terminateHostEngineDaemon(cmdLine.GetPidFilePath().c_str());
        return 0;
    }

    // Should we daemonize?
    if (cmdLine.ShouldDaemonize())
    {
        // Create daemon process
        if (isDaemonRunning(cmdLine.GetPidFilePath().c_str()))
        {
            exit(EXIT_FAILURE);
        }

        delete_pidfile_if_not_running_or_exit(cmdLine.GetPidFilePath().c_str());

        // Create Daemon
        heDaemonize();

        {
            auto version = DcgmNs::DcgmBuildInfo().GetVersion();
            syslog(LOG_NOTICE, "nv-hostengine version %.*s daemon started", (int)version.size(), version.data());
        }
    }

    /* Initialize DCGM Host Engine */
    ret = dcgmInit();
    if (DCGM_ST_OK != ret)
    {
        // assume that error message has already been printed
        syslog(LOG_NOTICE, "Error: DCGM engine failed to initialize");
        return cleanup(dcgmHandle, -1, parentPid);
    }

    dcgmStartEmbeddedV2Params_v3 params {};
    params.version  = dcgmStartEmbeddedV2Params_version3;
    params.opMode   = DCGM_OPERATION_MODE_AUTO;
    params.logFile  = cmdLine.GetLogFileName().c_str();
    params.severity = LoggingSeverityFromString(cmdLine.GetLogLevel().c_str(), DcgmLoggingSeverityUnspecified);

    params.denyListCount = 0;
    memset(params.denyList, 0, sizeof(params.denyList));

    /* Modules to place on the denylist before anyone connects via socket */
    for (auto moduleId : cmdLine.GetDenylistedModules())
    {
        params.denyList[params.denyListCount] = moduleId;
        params.denyListCount++;
    }

    auto const &serviceAccount = cmdLine.GetServiceAccount();
    params.serviceAccount      = serviceAccount.c_str();

    auto const &diagHomeDir = cmdLine.GetHomeDir();
    if (diagHomeDir.empty() == false)
    {
        setenv(DCGM_HOME_DIR_VAR_NAME, diagHomeDir.c_str(), 1);
    }

    TryCreateDcgmHomeDir(serviceAccount);
    ret = dcgmStartEmbedded_v2((dcgmStartEmbeddedV2Params_v1 *)&params);

    dcgmHandle = params.dcgmHandle;
    if (DCGM_ST_OK != ret)
    {
        // assume that error message has already been printed
        syslog(LOG_NOTICE, "Error: DCGM failed to start embedded engine");
        return cleanup(dcgmHandle, -1, parentPid);
    }


    syslog(LOG_NOTICE, "DCGM initialized");

    /* Hostengine depends on these handlers for graceful shutdown and ignoring SIGPIPE. */
    if (InstallCtrlHandler() != 0)
    {
        fprintf(stderr, "Error: Failed to install hostengine signal handlers\n");
        return cleanup(dcgmHandle, -1, parentPid);
    }

    switch (cmdLine.GetConnectionType())
    {
        case DcgmConnectionTypeDomainSocket:
            ret = dcgmEngineRun(cmdLine.GetPort(), cmdLine.GetUnixSocketPath().c_str(), DcgmConnectionTypeDomainSocket);
            break;
        case DcgmConnectionTypeTcp:
            ret = dcgmEngineRun(cmdLine.GetPort(), cmdLine.GetBindInterface().c_str(), DcgmConnectionTypeTcp);
            break;
        case DcgmConnectionTypeVsock:
            ret = dcgmEngineRun(cmdLine.GetPort(), cmdLine.GetVsockCid().c_str(), DcgmConnectionTypeVsock);
            break;
        default:
            syslog(
                LOG_NOTICE, "Error: Invalid connection type used to start hostengine: %d", cmdLine.GetConnectionType());
            ret = DCGM_ST_BADPARAM;
            break;
    }

    if (DCGM_ST_OK != ret)
    {
        printf("Err: Failed to start DCGM Server: %d\n", ret);
        syslog(LOG_NOTICE, "Err: Failed to start DCGM Server");
        return cleanup(dcgmHandle, -1, parentPid);
    }

    auto version = DcgmNs::DcgmBuildInfo().GetVersion();
    switch (cmdLine.GetConnectionType())
    {
        case DcgmConnectionTypeDomainSocket:
            fmt::println("Started host engine version {} using socket path: {}", version, cmdLine.GetUnixSocketPath());
            break;
        case DcgmConnectionTypeTcp:
            fmt::println("Started host engine version {} using port number: {}", version, cmdLine.GetPort());
            break;
        case DcgmConnectionTypeVsock:
            fmt::println("Started host engine version {} using vsock CID:PORT: {}:{}",
                         version,
                         cmdLine.GetVsockCid().empty() ? "any" : cmdLine.GetVsockCid(),
                         cmdLine.GetPort());
            break;
    }
    fflush(stdout);

    if (cmdLine.ShouldDaemonize())
    {
        create_daemon_pid_file(cmdLine.GetPidFilePath().c_str(), parentPid);
        daemonCloseConsoleOutput(parentPid);
    }

    while (true)
    {
        auto const shouldStop = g_stopLoop;
        if (shouldStop != 0)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return cleanup(dcgmHandle, 0, parentPid);
}
