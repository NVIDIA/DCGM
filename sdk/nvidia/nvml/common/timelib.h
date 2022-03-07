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

    timelib32_t timelib_gmt_to_local(timelib32_t timestamp);
    /*
 convert timestamp from gmt to local time zone
 */

    timelib32_t timelib_local_to_gmt(timelib32_t timestamp);

#ifdef __cplusplus
}
#endif

#endif //TIMELIB_H
