/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef TIMELIB_H
#define TIMELIB_H

/*
  Portable Time library
*/

#ifdef __cplusplus
extern "C"
{
#endif

/*****************************************************************************/
#ifdef NV_UNIX
#include <inttypes.h>
#include <time.h>
    typedef int64_t __int64;
#endif


/* 2000-03-01 (mod 400 year, immediately after feb29 */
#define LEAPOCH (946684800LL + 86400 * (31 + 29))
#define DAYS_PER_400Y (365 * 400 + 97)
#define DAYS_PER_100Y (365 * 100 + 24)
#define DAYS_PER_4Y (365 * 4 + 1)

    typedef __int64 timelib64_t;
    typedef unsigned int timelib32_t;

    /*****************************************************************************/
    timelib64_t timelib_usecSince1970(void);
    /*
	Returns microseconds since 1970 as an int64
*/

    /*****************************************************************************/
    timelib32_t timelib_secSince1970(void);
    /*
	Returns seconds since 1970
*/

    /*****************************************************************************/
    double timelib_dsecSince1970(void);
    /*
	Returns microseconds since 1970 as a double
*/

    /*****************************************************************************/
    void timelib_addToClock(timelib64_t howMuch);
    /*
	Permanently adjust the clock by a certain amount
*/

#if 0
/*****************************************************************************/
unsigned long timelib_getMillis(void);
/*
	Get milliseconds since some unspecified starting time. This will wrap
	every 49.7 days but is fast and can be used for getting the duration
	between two events
*/
#endif

    /*****************************************************************************/

    /*****************************************************************************/
    int timelib_secs_to_tm(long long t, struct tm *tm);
    /*
 convert timestamp in seconds to tm struct
*/

    /*****************************************************************************/
    int timelib_ts_to_date(const struct tm *pTimeTm, char *dateString, unsigned int length);
    /*
 Get date in string format from timestamp in seconds
 */

    /*****************************************************************************/
    int timelib_ts_to_readable_time(const struct tm *pTimeTm, char *timeString, unsigned int length);
    /*
 Get human readable time in string format from timestamp in seconds
*/


    timelib32_t timelib_gmt_to_local(timelib32_t timestamp);
    /*
 convert timestamp from gmt to local time zone
 */

    timelib32_t timelib_local_to_gmt(timelib32_t timestamp);
    /*
 convert timestamp from local time zone to gmt
 */

    int timelib_tm_to_ts(struct tm tm, timelib32_t *timestamp);
    /*
 convert tm to timestamp
*/

#ifdef __cplusplus
}
#endif

#endif //TIMELIB_H
