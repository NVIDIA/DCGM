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
#include "HostEngineMainHelpers.h"

#include <DcgmLogging.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <syslog.h>
#include <unistd.h>

#include <fcntl.h>

volatile std::sig_atomic_t g_stopLoop = 0;

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
    if (kill(pid, 0) == 0)
    {
        return 1;
    }

    return errno == EPERM ? 1 : 0;
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

/* Delete PID file /var/run/ accordingly */
void delete_pidfile_if_not_running_or_exit(char const *pidfile)
{
    pid_t pid;

    if (0 == read_from_pidfile(pidfile, &pid))
    {
        if (isDaemonProcessAlive(pid))
        {
            printf("Host engine already running with pid %ld\n", (long)pid);
            exit(EXIT_FAILURE);
        }
    }

    unlink(pidfile);
}

/* Create PID file /var/run/ accordingly */
void create_daemon_pid_file(char const *pidfile, pid_t parentPid)
{
    mode_t default_umask;

    /* write the pid of this process to the pidfile */
    default_umask = umask(0112);
    unlink(pidfile); // Just in case

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
void sig_handler(int /* signum */)
{
    g_stopLoop = 1;
}


/*****************************************************************************
 * This method provides mechanism to register Sighandler callbacks for
 * SIGHUP, SIGINT, SIGQUIT, and SIGTERM
 *****************************************************************************/
int InstallCtrlHandler()
{
    struct SignalRegistration
    {
        int signum;
        char const *name;
        void (*handler)(int);
    };

    /* SIGPIPE is ignored so socket hang-ups don't cause us to crash. */
    SignalRegistration const registrations[] = {
        { SIGHUP, "SIGHUP", sig_handler },   { SIGINT, "SIGINT", sig_handler }, { SIGQUIT, "SIGQUIT", sig_handler },
        { SIGTERM, "SIGTERM", sig_handler }, { SIGPIPE, "SIGPIPE", SIG_IGN },
    };

    for (auto const &registration : registrations)
    {
        if (signal(registration.signum, registration.handler) == SIG_ERR)
        {
            syslog(
                LOG_ERR, "Failed to install hostengine signal handler for %s: %s", registration.name, strerror(errno));
            return -1;
        }
    }

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
