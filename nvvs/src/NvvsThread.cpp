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


#include "NvvsThread.h"
#include "NvvsCommon.h"
#include <stdio.h>
#include <sys/syscall.h> //syscall()
#include <unistd.h>      //usleep()

/*****************************************************************************/
/*
    Helper static function to pass to CreateThread
*/
void *nvvsthread_starter(void *parm)
{
    NvvsThread *nvvsThread = (NvvsThread *)parm;
    nvvsThread->RunInternal();
    return 0;
}

DcgmMutex NvvsThread::m_sync_mutex(0);

/*****************************************************************************/
NvvsThread::NvvsThread()
    : m_mutex(0)
{
    m_shouldStop = 0;
    m_hasExited  = 0;
    m_pthread    = 0;
    m_hasRun     = 0;
}

/*****************************************************************************/
NvvsThread::~NvvsThread()
{}

/*****************************************************************************/
int NvvsThread::Start()
{
    if (m_pthread != 0)
    {
        PRINT_ERROR("%u", "Can't start thread. Already running as handle %u\n", (unsigned int)m_pthread);
        return -100;
    }

    int st = pthread_create(&m_pthread, 0, nvvsthread_starter, this);
    if (st)
    {
        m_pthread = 0;
        PRINT_ERROR("%d", "Unable to pthread_create. errno=%d\n", st);
        return -200;
    }

    PRINT_INFO("%u", "Created thread ID %u\n", (unsigned int)m_pthread);
    return 0;
}

/*****************************************************************************/
void NvvsThread::Stop()
{
    DcgmLockGuard lock(&m_mutex);
    m_shouldStop = 1;
}

/*****************************************************************************/
void NvvsThread::Kill()
{
    if (m_pthread == 0 || m_hasExited)
    {
        /* Nothing to do */
        return;
    }

    int st = pthread_cancel(m_pthread);
    if (st == 0 || st == ESRCH)
    {
        return; /* Thread terminated */
    }

    PRINT_WARNING("%u %d", "pthread_cancel(%u) returned %d\n", (unsigned int)m_pthread, st);
}

/*****************************************************************************/
int NvvsThread::Wait(int timeoutMs)
{
    {
        DcgmLockGuard lock(&m_mutex);
        if (m_pthread == 0 || m_hasExited)
        {
            /* Already terminated */
            return 0;
        }
    }

    void *retVal;
    dcgm_mutex_lock_me((&m_mutex));
    int hasRun = m_hasRun;
    dcgm_mutex_unlock((&m_mutex));

    /* Infinite timeout? */
    if (timeoutMs == 0)
    {
        /* Does this thread exist yet? */
        while (!hasRun)
        {
            usleep(10000);
            {
                DcgmLockGuard lock(&m_mutex);
                hasRun = m_hasRun;
            }
        }

        int st = pthread_join(m_pthread, &retVal);
        if (st)
        {
            return 0; /* Thread is gone */
        }

        PRINT_ERROR("%p %d", "pthread_join(%p) returned st %d\n", (void *)m_pthread, st);
        return 1;
    }
    else
    {
        /* Does this thread exist yet? */
        while (!hasRun && timeoutMs > 0)
        {
            /* Sleep for small intervals until we exhaust our timeout */
            usleep(10000);
            timeoutMs -= 10;
            {
                DcgmLockGuard lock(&m_mutex);
                hasRun = m_hasRun;
            }
        }

        if (timeoutMs < 0)
        {
            return 1; /* Hasn't started yet. I guess it's running */
        }

        dcgm_mutex_lock_me((&m_mutex));
        int hasExited = m_hasExited;
        dcgm_mutex_unlock((&m_mutex));
#if 1
        while ((!hasExited) && m_pthread && timeoutMs > 0)
        {
            usleep(10000);
            timeoutMs -= 10;
            {
                DcgmLockGuard lock(&m_mutex);
                hasExited = m_hasExited;
            }
        }

        if (!hasExited)
        {
            return 1; /* Running still */
        }
        else
        {
            int st = pthread_join(m_pthread, &retVal);
            if (st)
            {
                PRINT_ERROR("%p %d", "pthread_join(%p) returned st %d\n", (void *)m_pthread, st);
            }
            return 0;
        }

#else /* This code won't work until we upgrade to glibc 2.2.3 */
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeoutMs / 1000;
        ts.tv_nsec += (timeoutMs % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000)
        {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }

        int st = pthread_timedjoin_np(m_pthread, &retVal, &ts);
        if (st == ETIMEDOUT)
            return 1;
        else if (!st)
            return 0; /* Thread was gone */

        PRINT_ERROR("%u %d", "pthread_timedjoin_np(%u) returned st %d\n", m_pthread, st);
        return 1;
#endif
    }
}

/*****************************************************************************/
int NvvsThread::StopAndWait(int timeoutMs)
{
    int st;
    if (m_pthread == 0)
    {
        /* Already terminated */
        return 0;
    }

    /* Tell the thread it should stop */
    Stop();

    st = Wait(timeoutMs);
    return st;
}

/*****************************************************************************/
void NvvsThread::RunInternal(void)
{
    dcgm_mutex_lock_me((&m_mutex));
    m_hasRun = 1;
    dcgm_mutex_unlock((&m_mutex));

    PRINT_DEBUG("%u", "Thread handle %u running\n", (unsigned int)m_pthread);
    run();
    PRINT_DEBUG("%u", "Thread id %u stopped\n", (unsigned int)m_pthread);

    dcgm_mutex_lock_me((&m_mutex));
    m_hasExited = 1;
    dcgm_mutex_unlock((&m_mutex));
}

/*****************************************************************************/
int NvvsThread::ShouldStop(void)
{
    DcgmLockGuard lock(&m_mutex);
    return m_shouldStop || main_should_stop;
}

/*****************************************************************************/
unsigned long NvvsThread::GetTid()
{
    pid_t tid = syscall(SYS_gettid);
    return (unsigned long)tid;
}

/*****************************************************************************/
void NvvsThread::Sleep(long long howLongUsec)
{
    if (ShouldStop())
    {
        return; /* Return immediately if we're supposed to shut down */
    }
    if (howLongUsec < 0)
    {
        return; /* Bad value */
    }

    usleep(howLongUsec);
}

/*****************************************************************************/
