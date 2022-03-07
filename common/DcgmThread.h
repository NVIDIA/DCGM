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
#ifndef DCGMTHREAD_H
#define DCGMTHREAD_H

#ifndef __linux__
#include <Windows.h>
#else
#include "errno.h"
#include "pthread.h"
#include <signal.h>
#endif
#include <string>

#include <atomic>
#include <condition_variable>
#include <mutex>

#define DCGM_THREAD_SIGNUM SIGUSR2 /* Which signal should DcgmThread use to wake wake up threads? */

class DcgmThread
{
private:
    std::atomic_bool m_hasRun;     /* Has this thread run yet? */
    std::atomic_bool m_shouldStop; /* Has our thread been signalled to quit or not? */
    std::atomic_bool m_hasExited;  /* Has our thread finished running yet? */
    std::atomic_bool m_hasStarted; /* Has this thread been started yet? (call to pthread_create
                                      was made but pthread might not be running yet) */

#ifndef __linux__
    /* Windows specific stuff */
    HANDLE m_handle;
    DWORD m_threadId;
#else
    pthread_t m_pthread;
    bool m_alreadyJoined;    /* Already called pthread_join; calling a second time is undefined behavior */
    bool m_sendSignalOnStop; /* Should we send a signal to this thread when someone calls Stop() or
                                     StopAndWait() on it? Only set this to true if you have a signal handler
                                     installed or it will cause a seg fault in this thread due to an unhandled
                                     exception. */
#endif

    std::mutex m_mutex;                 /* Mutex used for m_stopCond */
    std::condition_variable m_stopCond; /* Condition used to signal that it's time to stop */
    std::string m_threadName;           /* Textual name of this thread that will appear in gdb */

public:
    /*************************************************************************/
    /*
     * Constructor
     *
     * sendSignalOnStop IN: Should we send a signal to this thread when someone calls Stop() or
                            StopAndWait() on it? Only set this to true if you have a signal
                            handler installed for signal DCGM_THREAD_SIGNUM or it will cause
                            a seg fault in this thread due to an unhandled. exception.
                            Alternatively, you can call DcgmThread::InstallSignalHandler() to
                            mitigate this.
        threadName      IN: Text name for this thread that will appear in GDB and other debugging tools.
     */

    DcgmThread(bool sendSignalOnStop = false, std::string threadName = "");

    /*************************************************************************/
    /*
     * Destructor (virtual to satisfy ancient compiler)
     */
    virtual ~DcgmThread();


    /*************************************************************************/
    /*
     * Static method for getting the current thread's thread ID on linux/windows
     *
     */
    static unsigned long GetTid();


    /*************************************************************************/
    /*
     * Signal this thread that it should stop at its earliest convenience
     *
     * This will invoke the OnStop() method of your derived class if you've
     * implemented it.
     */
    void Stop();


    /*************************************************************************/
    /*
     * Terminate this thread using OS calls to stop it in its tracks
     */
    void Kill();

    /*************************************************************************/
    /*
     * Spawn the separate thread and call its run() method
     *
     * RETURNS: 0 if OK
     *         !0 on error
     */
    int Start();

    /*************************************************************************/
    /*
     * Wait for this thread to exit. Stop() will be called on this
     * thread before waiting to ask the thread to stop
     *
     * timeoutMs IN: Milliseconds to wait for the thread to stop before returning
     *               0=forever
     *
     *  RETURNS: 0 if the thread is no longer running
     *            1 if the thread is still running after timeoutMs
     */
    int StopAndWait(int timeoutMs);

    /*************************************************************************/
    int Wait(int timeoutMs);
    /*
     * Wait for this thread to exit. Call StopAndWait() if you actually want to
     * signal the thread to quit
     *
     * timeoutMs IN: Milliseconds to wait for the thread to stop before returning
     *               0=forever
     *
     * RETURNS: 0 if the thread is no longer running
     *          1 if the thread is still running after timeoutMs
     */

    /*************************************************************************/
    /*
     * Call this method from within your run callback to see if you have
     * been signaled to stop or not
     *
     *  0 = No. Keep running
     * !0 = Yes. Stop running
     */
    int ShouldStop();

    /*************************************************************************/
    /*
     * Implement this virtual method within your class to be run
     * from the separate thread that is created.
     *
     * RETURNS: Nothing.
     */
    virtual void run(void) = 0;

    /*************************************************************************/
    /*
     * Internal method that only need to be public to satisfy C++
     */
    void RunInternal();

    /*************************************************************************/
    /*
     * Sleep for a specified time
     *
     * How long to sleep in microseconds (at least). This will return
     * immediately if ShouldStop() is true
     */
    void Sleep(long long howLongUsec);


    /*************************************************************************/
    /*
     * Method to check if the this thread has run yet
     *
     * RETURNS: 1 if the thread has started processing
     *          0 If the thread has not run yet
     */
    int HasRun();

    /*************************************************************************/
    /*
     * Method to send a signal to this thread. This can be used to wake
     * up a thread that's sleeping inside a poll, select..etc.
     *
     * WARNING: if you do not have a signal handler installed in your process,
     *          this will cause a segfault in the receiving thread.
     *
     * signum  IN Signal to send (SIGHUP, SIGUSR1..etc). See signal.h
     *
     * RETURNS: Nothing
     */
    void SendSignal(int signum);

    /*************************************************************************/
    /*
     * Method to check if the this thread has exited
     *
     * RETURNS: 1 if the thread has exited
     *          0 If the thread has not exited yet
     */
    int HasExited();

    /*************************************************************************/
    /*
     * Method to install a signal handler for DCGM_THREAD_SIGNUM if one isn't
     * already installed.
     *
     * Note: This routine is not thread safe.
     *
     * RETURNS: Nothing.
     */
    static void InstallSignalHandler();

    /*************************************************************************/
    /*
     * Implement this virtual method within your derived class if you want a
     * notification when Stop() gets called on your thread. This allows you to
     * do any custom signalling to wake your thread up.
     *
     * RETURNS: Nothing.
     */
    virtual void OnStop(void)
    {}

private:
    void resetStatusFlags();
};


#endif /* DCGMTHREAD_H_ */
