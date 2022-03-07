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
#ifndef NVVSTHREAD_H
#define NVVSTHREAD_H

#include "DcgmMutex.h"
#include "errno.h"
#include "pthread.h"

class NvvsThread
{
private:
    int m_shouldStop;  /* Has our thread been signalled to quit or not? */
    int m_hasExited;   /* Has our thread finished running yet? */
    DcgmMutex m_mutex; /* Mutex for controlling concurrent access to internal variables */
    pthread_t m_pthread;
    int m_hasRun; /* Has this thread run yet? Is needed for slow-starting posix threads */


protected:
    static DcgmMutex m_sync_mutex; /* Synchronization mutex for use by subclasses to control access to global data */

public:
    /*************************************************************************/
    /*
     * Constructor
     */

    NvvsThread(void);

    /*************************************************************************/
    /*
     * Destructor (virtual to satisfy ancient compiler)
     */
    virtual ~NvvsThread();


    /*************************************************************************/
    /*
     * Static method for getting the current thread's thread ID on linux/windows
     *
     */
    static unsigned long GetTid();


    /*************************************************************************/
    /*
     * Signal this thread that it should stop at its earliest convenience
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
};


#endif /* NVVSTHREAD_H_ */
