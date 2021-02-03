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
#include "TestDcgmConnections.h"
#include "DcgmThread.h"
#include "dcgm_agent.h"
#include "dcgm_test_apis.h"
#include "timelib.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

/*****************************************************************************/
TestDcgmConnections::TestDcgmConnections()
{}

/*****************************************************************************/
TestDcgmConnections::~TestDcgmConnections()
{}

/*************************************************************************/
std::string TestDcgmConnections::GetTag()
{
    return std::string("connections");
}

/*****************************************************************************/
int TestDcgmConnections::Cleanup(void)
{
    return 0;
}

/*****************************************************************************/
int TestDcgmConnections::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    m_gpus = gpus;
    return 0;
}

/*****************************************************************************/
void TestDcgmConnections::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestDcgmConnections::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    }
    else
    {
        std::cout << "TestDcgmConnections::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestDcgmConnections::TestDeadlockSingle(void)
{
    dcgmReturn_t dcgmReturn;
    int i;
    dcgmHandle_t dcgmHandle;
    char *badHostname    = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */
    int startTime        = timelib_secSince1970();
    int testDurationSecs = 30;

    printf("TestDeadlockSingle running for %d seconds.\n", testDurationSecs);

    for (i = 0; (int)timelib_secSince1970() - startTime < testDurationSecs; i++)
    {
        dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
        if (dcgmReturn == DCGM_ST_OK)
        {
            dcgmDisconnect(dcgmHandle);
            std::cerr << "TestDeadlockSingle skipping due to actually being able to connect to " << badHostname
                      << ::std::endl;
            return 0;
        }
        else if (dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
        {
            std::cerr << "Unexpected return " << dcgmReturn << " from dcgmConnect() iteration" << i << std::endl;
            return 100;
        }
    }

    printf("TestDeadlockSingle finished without deadlocking.\n");
    return 0;
}

/*****************************************************************************/
class TestDeadlockMultiThread : public DcgmThread
{
public:
    int m_threadIndex;

    TestDeadlockMultiThread(int threadIndex)
    {
        m_threadIndex = threadIndex;
    }

    ~TestDeadlockMultiThread()
    {}

    void run(void)
    {
        dcgmReturn_t dcgmReturn;
        const int Ntimes = 100;
        int i;
        dcgmHandle_t dcgmHandle;
        char *badHostname = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */

        for (i = 0; !ShouldStop(); i++)
        {
            dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
            if (dcgmReturn == DCGM_ST_OK)
            {
                dcgmDisconnect(dcgmHandle);
                std::cerr << "TestDeadlockMulti thread " << m_threadIndex
                          << " aborting due to actually being able to connect to " << badHostname << ::std::endl;
                break;
            }
            else if (dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
            {
                std::cerr << "TestDeadlockMulti thread " << m_threadIndex << " got unexpected return " << dcgmReturn
                          << " from dcgmConnect() iteration" << i << std::endl;
                break;
            }

            if (i % (Ntimes / 10) == 0) // Log every 10%
            {
                std::cout << "TestDeadlockMulti at iteration " << (i + 1) << "/" << Ntimes << std::endl;
            }
        }

        std::cout << "Thread " << m_threadIndex << " exiting." << std::endl;
    }
};

/*****************************************************************************/
#define TDC_NUM_WORKERS 4

int TestDcgmConnections::TestDeadlockMulti(void)
{
    dcgmReturn_t dcgmReturn;
    int i;
    TestDeadlockMultiThread *workers[TDC_NUM_WORKERS] = { 0 };
    dcgmHandle_t dcgmHandle;
    char *badHostname = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */
    int workersLeft   = TDC_NUM_WORKERS;
    int startTime     = timelib_secSince1970();

    /* Make a single connection to make sure our threads will work */
    dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
    if (dcgmReturn == DCGM_ST_OK)
    {
        std::cerr << "TestDeadlockMulti skipping due to actually being able to connect to " << badHostname
                  << ::std::endl;
        dcgmDisconnect(dcgmHandle);
        return 0;
    }
    else if (dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
    {
        std::cerr << "Unexpected return " << dcgmReturn << " from dcgmConnect() " << std::endl;
        return 100;
    }

    std::cout << "Starting " << (int)TDC_NUM_WORKERS << " workers." << std::endl;

    for (i = 0; i < TDC_NUM_WORKERS; i++)
    {
        workers[i] = new TestDeadlockMultiThread(i);
        workers[i]->Start();
    }

    std::cout << "Waiting for " << (int)TDC_NUM_WORKERS << " workers." << std::endl;

    while (workersLeft > 0)
    {
        for (i = 0; i < TDC_NUM_WORKERS; i++)
        {
            if (!workers[i])
                continue;

            if (timelib_secSince1970() - startTime > 30)
            {
                std::cout << "Requesting stop of worker " << i << " after 30 seconds." << std::endl;
                workers[i]->Stop();
            }

            /* Wait() returns 0 if the thread is gone */
            if (!workers[i]->Wait(1000))
            {
                delete workers[i];
                workers[i] = 0;
                workersLeft--;
            }
        }
    }

    std::cout << "All workers exited." << std::endl;
    return 0;
}

/*****************************************************************************/
class TestThrashThread : public DcgmThread
{
public:
    int m_threadIndex;

    TestThrashThread(int threadIndex)
    {
        m_threadIndex = threadIndex;
    }

    ~TestThrashThread()
    {}

    void run(void)
    {
        dcgmReturn_t dcgmReturn;
        int i;
        dcgmHandle_t dcgmHandle;
        bool connected      = false;
        char *connectString = (char *)"127.0.0.1:5555";

        for (i = 0; !ShouldStop(); i++)
        {
            if (!connected)
            {
                dcgmReturn = dcgmConnect(connectString, &dcgmHandle);
                if (dcgmReturn != DCGM_ST_OK)
                {
                    std::cerr << "Unable to connect to host engine from worker " << m_threadIndex << " return "
                              << errorString(dcgmReturn);
                    break;
                }
                connected = true;
            }

            int op = rand() % 100;

            if (op >= 99)
            {
                dcgmReturn = dcgmDisconnect(dcgmHandle);
                assert(dcgmReturn == DCGM_ST_OK);
                connected = false;
            }
            else if (op >= 50)
            {
                unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES] = {};
                int count                                    = 0;
                dcgmReturn = dcgmGetAllSupportedDevices(dcgmHandle, gpuIdList, &count);
                assert(dcgmReturn == DCGM_ST_OK);
            }
            else
            {
                dcgmHostengineHealth_t health = { dcgmHostengineHealth_version, 0 };
                dcgmReturn                    = dcgmHostengineIsHealthy(dcgmHandle, &health);
                assert(dcgmReturn == DCGM_ST_OK);
            }

            if (i % 10000 == 0) // Log every 0.01%
            {
                std::cout << "TestThrash worker " << m_threadIndex << " at iteration " << (i + 1) << std::endl;
            }
        }

        std::cout << "Thread " << m_threadIndex << " exiting." << std::endl;

        /* Intentionally (possibly) leaving the connection open here */
    }
};

/*****************************************************************************/
#define THRASH_NUM_WORKERS 10

int TestDcgmConnections::TestThrash(void)
{
    dcgmReturn_t dcgmReturn;
    int i;
    std::vector<std::unique_ptr<TestThrashThread>> workers;

    dcgmHandle_t dcgmHandle;
    char *hostname = (char *)"127.0.0.1";
    int startTime  = timelib_secSince1970();
    int duration   = 10;

    dcgmReturn = dcgmEngineRun(5555, hostname, 1);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "dcgmEngineRun returned %d\n", dcgmReturn);
        return -1;
    }

    std::cout << "Starting " << THRASH_NUM_WORKERS << " workers." << std::endl;

    for (i = 0; i < THRASH_NUM_WORKERS; i++)
    {
        workers.emplace_back(std::make_unique<TestThrashThread>(i));
        workers[i]->Start();
    }

    std::cout << "Waiting for " << (int)workers.size() << " workers." << std::endl;

    while (timelib_secSince1970() - startTime < duration)
    {
        /* Sleep 1 second at a time */
        usleep(1000000);
    }

    /* First trigger all workers stopping. Then wait for them */
    for (auto &w : workers)
    {
        w->Stop();
    }

    for (auto &w : workers)
    {
        if (w->StopAndWait(10000))
        {
            std::cerr << "A worker returned an error while stopping.";
            break;
        }
    }

    std::cout << "All workers exited. Stopping host engine." << std::endl;

    /* Trigger clean-up and prepare for further tests */
    dcgmShutdown();
    dcgmInit();
    dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle);
    return 0;
}

/*****************************************************************************/
int TestDcgmConnections::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestThrash", TestThrash(), Nfailed);
        CompleteTest("TestDeadlockSingle", TestDeadlockSingle(), Nfailed);
        CompleteTest("TestDeadlockMultiThread", TestDeadlockMulti(), Nfailed);
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
