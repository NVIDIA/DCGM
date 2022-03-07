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
#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include "HostEngineCommandLine.h"

#define DCGM_INIT_UUID
#include "dcgm_agent.h"
#include "dcgm_test_apis.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <DcgmBuildInfo.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


int g_stopLoop = 0;

// Forward declarations
void daemonCloseConsoleOutput(pid_t parentPid);

/*****************************************************************************/
/* Writes pid to file */
int write_to_pidfile(char const *pidfile, pid_t pid)
{
    FILE *fp;

    fp = fopen(pidfile, "w");
    if (!fp)
    {
        printf("Error writing pidfile '%s':\n%s.\n", pidfile, strerror(errno));
        printf("If the current user cannot to write to the PID file, update the permissions or use "
               "the --pid option to specify an alternate location for the PID file.\n");
        return -1;
    }

    fprintf(fp, "%ld\n", (long)pid);
    fclose(fp);
    return 0;
}


/**
 * This method reads a PID from the specified pidfile
 * Returns:
 *  0 on Success
 * -1 on Error
 */
int read_from_pidfile(char const *pidfile, pid_t *pid)
{
    FILE *fp;
    long p;

    /* Read PID out of pidfile*/
    fp = fopen(pidfile, "r");
    if (fp)
    {
        if (fscanf(fp, "%ld", &p) > 0)
        {
            *pid = p;
            fclose(fp);
            return 0;
        }

        /* Implies that fscanf failed to read pid from file */
        fclose(fp);
        return -1;
    }

    /* If the control reaches here then it implies that fopen failed to open
       the pidfile */
    return -1;
}

/**
 *  Method to check if the PID is alive on the system
 * Returns:
 * 1 : If the Process is alive
 * 0 : If the process is not alive
 */
int isDaemonProcessAlive(pid_t pid)
{
    char procfs_path[PATH_MAX];
    struct stat sb;
    int ret;

    procfs_path[0] = 0;

    /* For VMware, check if the PID is listed in by ps*/
#if defined(NV_VMWARE)
    sprintf(procfs_path, "/bin/ps | grep -q %ld", (long)pid);
    ret = system(procfs_path);
#else
    /* For others, check if the PID exists as part of proc fs */
    sprintf(procfs_path, "/proc/%ld", (long)pid);
    ret = stat(procfs_path, &sb);
#endif
    if (0 == ret)
    { /* File exists and Implies that the process is running */
        return 1;
    }
    else
    {
        return 0;
    }
}

/* Check if the daemon is already running */
bool isDaemonRunning(char const *pidfile)
{
    pid_t pid;
    bool isRunning = false;

    if (0 == read_from_pidfile(pidfile, &pid))
    {
        if (isDaemonProcessAlive(pid))
        {
            printf("Host engine already running with pid %ld\n", (long)pid);
            isRunning = true;
        }
    }

    return isRunning;
}

/**
 * Terminate Daemon if it's running
 *
 * Returns: true if we successfully terminated the host engine
 *          false if we could not find the host engine's pid file or could not terminate the host engine
 */
bool termDaemonIfRunning(char const *pidfile)
{
    pid_t pid;

    if (read_from_pidfile(pidfile, &pid))
    {
        syslog(LOG_NOTICE, "nv-hostengine pidfile %s could not be read.", pidfile);
        return false;
    }

    if (!isDaemonProcessAlive(pid))
        return false; /* Wasn't running anymore. Parent handles this */

    syslog(LOG_NOTICE, "Killing nv-hostengine");
    (void)kill(pid, SIGTERM);

    /* Wait 30 seconds for the daemon to exit. */
    int totalWaitMsec = 30000;
    int incrementMsec = 100;
    for (int waitMsec = 0; waitMsec < totalWaitMsec; waitMsec += incrementMsec)
    {
        if (!isDaemonProcessAlive(pid))
            return true;
        usleep(incrementMsec * 1000);
    }

    syslog(LOG_NOTICE,
           "Sent a SIGTERM to nv-hostengine pid %u but it did not exit after %d seconds.",
           pid,
           totalWaitMsec / 1000);
    return false;
}

/* Update PID file /var/run/ accordingly */
void update_daemon_pid_file(char const *pidfile, pid_t parentPid)
{
    pid_t pid;
    mode_t default_umask;

    if (0 == read_from_pidfile(pidfile, &pid))
    {
        if (isDaemonProcessAlive(pid))
        {
            printf("Host engine already running with pid %ld\n", (long)pid);
            exit(EXIT_FAILURE);
        }
    }

    /* write the pid of this process to the pidfile */
    default_umask = umask(0112);
    unlink(pidfile);
    if (0 != write_to_pidfile(pidfile, getpid()))
    {
        printf("Host engine failed to write to pid file %s\n", pidfile);
        // Signal the parent process to exit before exiting to prevent parent process from hanging
        daemonCloseConsoleOutput(parentPid);
        umask(default_umask);
        exit(EXIT_FAILURE);
    }

    umask(default_umask);
}

int terminateHostEngineDaemon(char const *pidFilePath)
{
    if (termDaemonIfRunning(pidFilePath))
    {
        printf("Host engine successfully terminated.\n");
        exit(EXIT_SUCCESS);
    }

    if (!isDaemonRunning(pidFilePath))
    {
        printf("Unable to terminate host engine, it may not be running.\n");
        exit(EXIT_FAILURE);
    }

    printf("Unable to terminate host engine.\n");

    exit(EXIT_FAILURE);
}

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

// Intended to be run by the daemon after initialization
// this closes stout/stderr from being hooked to the console and then kills the original parent
void daemonCloseConsoleOutput(pid_t parentPid)
{
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    int fd = open("/dev/null", 02, 0);
    if (fd != -1)
    {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
    }

    kill(parentPid, SIGUSR1);
    // coverity[leaked_handle] leave fd open
}

/*****************************************************************************/
void sig_handler(int signum)
{
    g_stopLoop = 1;
}


/*****************************************************************************
 * This method provides mechanism to register Sighandler callbacks for
 * SIGHUP, SIGINT, SIGQUIT, and SIGTERM
 *****************************************************************************/
int InstallCtrlHandler()
{
    if (signal(SIGHUP, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        return -1;
    if (signal(SIGTERM, sig_handler) == SIG_ERR)
        return -1;

    /* Ignore SIGPIPE so that socket hang-ups don't cause us to crash */
    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR)
        return -1;

    return 0;
}

int cleanup(dcgmHandle_t dcgmHandle, int status, int parentPid)
{
    // error in initialization, still need to close console and kill parent
    if (status != 0)
    {
        daemonCloseConsoleOutput(parentPid);
    }

    if (dcgmHandle != (dcgmHandle_t) nullptr)
    {
        (void)dcgmStopEmbedded(dcgmHandle);
    }
    (void)dcgmShutdown();

    return status;
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

        // Create Daemon
        heDaemonize();

        {
            auto version = DcgmNs::DcgmBuildInfo().GetVersion();
            syslog(LOG_NOTICE, "nv-hostengine version %.*s daemon started", (int)version.size(), version.data());
        }

        update_daemon_pid_file(cmdLine.GetPidFilePath().c_str(), parentPid);
    }

    /* Initialize DCGM Host Engine */
    ret = dcgmInit();
    if (DCGM_ST_OK != ret)
    {
        // assume that error message has already been printed
        syslog(LOG_NOTICE, "Error: DCGM engine failed to initialize");
        return cleanup(dcgmHandle, -1, parentPid);
    }

    dcgmStartEmbeddedV2Params_v1 params {};
    params.version  = dcgmStartEmbeddedV2Params_version1;
    params.opMode   = DCGM_OPERATION_MODE_AUTO;
    params.logFile  = cmdLine.GetLogFileName().c_str();
    params.severity = DcgmLogging::severityFromString(cmdLine.GetLogLevel().c_str(), DcgmLoggingSeverityUnspecified);

    params.blackListCount = 0;
    memset(params.blackList, 0, sizeof(params.blackList));

    /* Blacklist any modules before anyone connects via socket */
    for (auto moduleId : cmdLine.GetBlacklistedModules())
    {
        params.blackList[params.blackListCount] = moduleId;
        params.blackListCount++;
    }

    ret = dcgmStartEmbedded_v2(&params);

    dcgmHandle = params.dcgmHandle;
    if (DCGM_ST_OK != ret)
    {
        // assume that error message has already been printed
        syslog(LOG_NOTICE, "Error: DCGM failed to start embedded engine");
        return cleanup(dcgmHandle, -1, parentPid);
    }


    syslog(LOG_NOTICE, "DCGM initialized");

    InstallCtrlHandler();

    /* Should we start in TCP mode? */
    if (cmdLine.IsConnectionTcp())
    {
        ret = dcgmEngineRun(cmdLine.GetPort(), cmdLine.GetBindInterface().c_str(), cmdLine.IsConnectionTcp() ? 1 : 0);
    }
    else
    {
        /* Start in unix domain socket mode */
        ret = dcgmEngineRun(cmdLine.GetPort(), cmdLine.GetUnixSocketPath().c_str(), cmdLine.IsConnectionTcp() ? 1 : 0);
    }

    if (DCGM_ST_OK != ret)
    {
        printf("Err: Failed to start DCGM Server: %d\n", ret);
        syslog(LOG_NOTICE, "Err: Failed to start DCGM Server");
        return cleanup(dcgmHandle, -1, parentPid);
    }

    {
        auto version = DcgmNs::DcgmBuildInfo().GetVersion();
        if (cmdLine.IsConnectionTcp())
        {
            printf("Started host engine version %.*s using port number: %u \n",
                   (int)version.size(),
                   version.data(),
                   cmdLine.GetPort());

            fflush(stdout);
        }
        else
        {
            printf("Started host engine version %.*s using socket path: %s \n",
                   (int)version.size(),
                   version.data(),
                   cmdLine.GetUnixSocketPath().c_str());
        }
    }

    if (cmdLine.ShouldDaemonize())
    {
        daemonCloseConsoleOutput(parentPid);
    }

    while (g_stopLoop == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return cleanup(dcgmHandle, 0, parentPid);
}
