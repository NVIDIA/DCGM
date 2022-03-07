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
#include "TestDcgmMutex.h"
#include <iostream>
#include <stdexcept>
#include <string>

/*****************************************************************************/
TestDcgmMutex::TestDcgmMutex()
{}

/*****************************************************************************/
TestDcgmMutex::~TestDcgmMutex()
{}

/*************************************************************************/
std::string TestDcgmMutex::GetTag()
{
    return std::string("mutex");
}

/*****************************************************************************/
int TestDcgmMutex::Cleanup(void)
{
    return 0;
}

/*****************************************************************************/
int TestDcgmMutex::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    m_gpus = gpus;
    return 0;
}

/*****************************************************************************/
void TestDcgmMutex::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestDcgmMutex::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    }
    else
    {
        std::cout << "TestDcgmMutex::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestDcgmMutex::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestPerf", TestPerf(), Nfailed);
        CompleteTest("TestDoubleLock", TestDoubleLock(), Nfailed);
        CompleteTest("TestDoubleUnlock", TestDoubleUnlock(), Nfailed);
    }
    // fatal test return ocurred
    catch (const std::runtime_error &e)
    {
        return -1;
    }

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    printf("All tests passed\n");

    return 0;
}

/*****************************************************************************/
int TestDcgmMutex::TestDoubleLock(void)
{
    int Nfailed = 0;
    dcgmMutexReturn_t mutexReturn;
    DcgmMutex *mutex = new DcgmMutex(0);

    mutexReturn = dcgm_mutex_lock(mutex);
    if (mutexReturn != DCGM_MUTEX_ST_OK)
    {
        std::cerr << "TestDoubleLock first lock failed. Got " << mutexReturn << " expected DCGM_MUTEX_ST_OK"
                  << std::endl;
        Nfailed++;
    }

    // coverity[double_lock] - false positive. DcgmMutex handles this case, and that's what this test is testing
    mutexReturn = dcgm_mutex_lock(mutex);
    if (mutexReturn != DCGM_MUTEX_ST_LOCKEDBYME)
    {
        std::cerr << "TestDoubleLock second lock failed. Got " << mutexReturn << " expected DCGM_MUTEX_ST_LOCKEDBYME"
                  << std::endl;
        Nfailed++;
    }

    /* Unlock the mutex before we free it */
    dcgm_mutex_unlock(mutex);

    delete (mutex);
    return Nfailed;
}

/*****************************************************************************/
int TestDcgmMutex::TestPerf(void)
{
    int i;
    int Ntimes                  = 10000000;
    DcgmMutex *mutex            = new DcgmMutex(0);
    DcgmMutex *mutexWithTimeout = new DcgmMutex(1000);
    double t1, perTime, timeDiff;

    std::cout << "Mutex Timing:" << std::endl;

    t1 = timelib_dsecSince1970();

    for (i = 0; i < Ntimes; i++)
    {
        dcgm_mutex_lock(mutex);
        dcgm_mutex_unlock(mutex);
    }

    timeDiff = timelib_dsecSince1970() - t1;

    perTime = (1000000.0 * timeDiff) / (double)Ntimes;

    std::cout << "Mutex Lock+Unlock Pair: " << perTime << " usec" << std::endl;

    t1 = timelib_dsecSince1970();

    for (i = 0; i < Ntimes; i++)
    {
        dcgm_mutex_lock(mutexWithTimeout);
        dcgm_mutex_unlock(mutexWithTimeout);
    }

    timeDiff = timelib_dsecSince1970() - t1;

    perTime = (1000000.0 * timeDiff) / (double)Ntimes;

    std::cout << "Mutex w/Timeout Lock+Unlock Pair: " << perTime << " usec" << std::endl;

    delete (mutex);
    delete (mutexWithTimeout);
    return 0;
}

/*****************************************************************************/
int TestDcgmMutex::TestDoubleUnlock(void)
{
    int Nfailed = 0;
    dcgmMutexReturn_t mutexReturn;
    DcgmMutex *mutex = new DcgmMutex(0);

    mutexReturn = dcgm_mutex_lock(mutex);
    if (mutexReturn != DCGM_MUTEX_ST_OK)
    {
        std::cerr << "TestDoubleUnlock first lock failed. Got " << mutexReturn << " expected DCGM_MUTEX_ST_OK"
                  << std::endl;
        Nfailed++;
    }

    mutexReturn = dcgm_mutex_unlock(mutex);
    if (mutexReturn != DCGM_MUTEX_ST_OK)
    {
        std::cerr << "TestDoubleLock unlock failed. Got " << mutexReturn << " expected DCGM_MUTEX_ST_OK" << std::endl;
        Nfailed++;
    }

    // coverity[double_unlock] - We are intentionally double unlocking as part of the test.
    mutexReturn = dcgm_mutex_unlock(mutex);
    if (mutexReturn != DCGM_MUTEX_ST_NOTLOCKED)
    {
        std::cerr << "TestDoubleLock unlock failed. Got " << mutexReturn << " expected DCGM_MUTEX_NOT_LOCKED"
                  << std::endl;
        Nfailed++;
    }

    delete (mutex);
    return Nfailed;
}


/*****************************************************************************/
