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
/* See timelib.h for license details */

#ifndef NV_UNIX
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include "timelib.h"
#include <limits.h>
#include <memory.h>
#include <stdio.h>
#include <time.h>

/*****************************************************************************/
timelib64_t cyclesPerSec        = 2000000000; /* Initialize to 2 GHZ */
double cyclesPerUsec            = 2000.0;
timelib64_t resyncCycle         = 0;
timelib64_t resyncUsecSince1970 = 0;

/* The following is the magic number for the windows file time of 1/1/1970 */
timelib64_t jan1st1970HunNano = 116444736000000000LL;

/*****************************************************************************/
#ifndef NV_UNIX //Windows
/*****************************************************************************/
void timelib_syncClock(void)
{
    FILETIME oldFiletime, filetime;
    timelib64_t hunNanoSince1970;
    ULARGE_INTEGER filetimeUlarge;

    /* Synchronize with the system clock the first time this is called. This
   should be done while your app is still single threaded for
   consistency */
    QueryPerformanceFrequency((PLARGE_INTEGER)&cyclesPerSec);

    cyclesPerUsec = (DOUBLE)cyclesPerSec / 1000000.0;

    GetSystemTimeAsFileTime(&oldFiletime);
    filetime = oldFiletime;

    /* Spin until we tick over. This should take about 8 ms */
    while (!memcmp(&filetime, &oldFiletime, sizeof(filetime)))
    {
        GetSystemTimeAsFileTime(&filetime);
        QueryPerformanceCounter((PLARGE_INTEGER)&resyncCycle);
    }

    /* Now we've got a file time and cycle count that are about 500 ns out of
   sync with each other, which is just fine. Find out what usec since 1970 this
   cycle counter represents and what second it was at 0 */
    memmove(&filetimeUlarge, &filetime, sizeof(filetimeUlarge));
    hunNanoSince1970    = filetimeUlarge.QuadPart - jan1st1970HunNano;
    resyncUsecSince1970 = hunNanoSince1970 / 10ui64;
    return;
}

/*****************************************************************************/
void timelib_addToClock(timelib64_t howMuch)
{
    /* Add an offset to the clock so it's always off by a certain amount */

    if (!resyncUsecSince1970)
        timelib_syncClock();

    resyncUsecSince1970 += howMuch;
}

/*****************************************************************************/
timelib64_t timelib_usecSince1970(void)
{
    static timelib64_t lastNowCycles = 0;
    static int haveSynced            = 0;
    timelib64_t retTime              = 0;
    timelib64_t nowCycles;
    timelib64_t usecSinceResync;

    if (!haveSynced)
    {
        timelib_syncClock();
        haveSynced = 1;
    }

    QueryPerformanceCounter((PLARGE_INTEGER)&nowCycles);
    if (nowCycles < lastNowCycles)
        nowCycles = lastNowCycles; /* Don't go backwards without a flux capacitor */
    else
        lastNowCycles = nowCycles;

    usecSinceResync = (timelib64_t)((DOUBLE)(nowCycles - resyncCycle) / cyclesPerUsec);

    retTime = usecSinceResync + resyncUsecSince1970;
    return retTime;
}

unsigned long timelib_getMillis(void)
{
    return GetTickCount();
}

/*****************************************************************************/
#else //Linux
/*****************************************************************************/
timelib64_t timelib_usecSince1970(void)
{
    struct timeval timeVal;
    timelib64_t retTime;

    gettimeofday(&timeVal, NULL);

    retTime = timeVal.tv_sec * (timelib64_t)1000000;
    retTime += timeVal.tv_usec;
    return retTime;
}

#if 0
/*****************************************************************************/
unsigned long timelib_getMillis(void)
{
unsigned long retVal;
struct timespec now;
if (clock_gettime(CLOCK_MONOTONIC, &now))
	return 0;
retVal = (now.tv_sec * 1000) + (now.tv_nsec / 1000000);
return retVal;
}
#endif

#endif

/*****************************************************************************/
timelib32_t timelib_secSince1970(void)
{
    timelib32_t time32;

    time32 = (timelib32_t)(timelib_usecSince1970() / (timelib64_t)1000000);
    return time32;
}

/*****************************************************************************/
double timelib_dsecSince1970(void)
{
    double retVal;
    timelib64_t val;


    val    = timelib_usecSince1970();
    retVal = ((double)val) / 1000000.0;
    return retVal;
}


/*****************************************************************************/
#if 1
#endif


/*****************************************************************************/
#ifndef NV_UNIX //Windows
timelib32_t timelib_gmt_to_local(timelib32_t timestamp)
{
    struct tm *tptr;
    time_t secs, local_secs, gmt_secs;
    timelib32_t diff_secs;

    time(&secs); // Current time in GMT

    tptr       = localtime(&secs);
    local_secs = mktime(tptr);
    tptr       = gmtime(&secs);
    gmt_secs   = mktime(tptr);
    diff_secs  = (timelib32_t)(local_secs - gmt_secs);

    return (timestamp + diff_secs);
}
#else

static timelib32_t getGMTOffset(void)
{
#ifdef __sun
    time_t t     = time(NULL);
    struct tm lt = { 0 };
    struct tm gt = { 0 };
    time_t local, gmt;
    timelib32_t dst_offset = 0;

    localtime_r(&t, &lt);
    gmtime_r(&t, &gt);

    local = mktime(&lt);
    gmt   = mktime(&gt);

    if (lt.tm_isdst)
    {
        dst_offset = -3600; // 1 less hour
    }

    return (timelib32_t)difftime(gmt, local) + dst_offset;
#else
    time_t t     = time(NULL);
    struct tm lt = { 0 };

    /* Get Local time to know the offset from gmt */
    localtime_r(&t, &lt);

#if defined(__USE_BSD) || defined(NV_BSD) || defined(__USE_MISC)
    return (timelib32_t)lt.tm_gmtoff;
#else
    return (timelib32_t)lt.__tm_gmtoff;
#endif
#endif
}

timelib32_t timelib_gmt_to_local(timelib32_t timestamp)
{
    return timestamp + getGMTOffset();
}


#endif


/*****************************************************************************/
#ifndef NV_UNIX //Windows
timelib32_t timelib_local_to_gmt(timelib32_t timestamp)
{
    struct tm *tptr;
    time_t secs, local_secs, gmt_secs;
    timelib32_t diff_secs;

    time(&secs); // Current time in GMT

    tptr       = localtime(&secs);
    local_secs = mktime(tptr);
    tptr       = gmtime(&secs);
    gmt_secs   = mktime(tptr);
    diff_secs  = (timelib32_t)(local_secs - gmt_secs);

    return (timestamp - diff_secs);
}
#else
timelib32_t timelib_local_to_gmt(timelib32_t timestamp)
{
    return timestamp - getGMTOffset();
}
#endif
