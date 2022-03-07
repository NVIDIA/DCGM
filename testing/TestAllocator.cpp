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
/*
 * Testallocator que tests performance of Malloc.
 * It tests the malloc by allocating blocks in a range and
 * less than 16GB.
 *
 */
#include "TestAllocator.h"

const int MAX_RUNS         = 100000000;
const int MAX_POINTERS     = 1000000;
const int FIXED_SIZE_BLOCK = 1024;

TestAllocator::TestAllocator()
{}

TestAllocator::~TestAllocator()
{}

int TestAllocator::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

std::string TestAllocator::GetTag()
{
    return std::string("allocator");
}

int TestAllocator::Cleanup()
{
    return 0;
}

int TestAllocator::Run()
{
    unsigned int Nfailed = 0;

    std::cout << "\n\nRunning Time profiling tests......\n\n";
    std::cout << "Time Taken to run(in seconds) \n_____________________________\n\n";

    RUN_MODULE_SUBTEST(TestAllocator, testAllocateVarSize_FreeInorder, Nfailed);
    RUN_MODULE_SUBTEST(TestAllocator, testAllocateVarSize_FreeReverse, Nfailed);
    RUN_MODULE_SUBTEST(TestAllocator, testAllocateFixedSize_FreeInorder, Nfailed);
    RUN_MODULE_SUBTEST(TestAllocator, testAllocateFixedSize_FreeReverse, Nfailed);
    RUN_MODULE_SUBTEST(TestAllocator, testAllocateVarSize_FreeSerial, Nfailed);
    RUN_MODULE_SUBTEST(TestAllocator, testAllocateFixedSize_FreeSerial, Nfailed);

    std::cout << "\n\nRunning Memory profiling tests......\n\n";
    std::cout << "Memory stats (in bytes) \n_____________________________\n\n";
    RUN_MODULE_SUBTEST(TestAllocator, testInternalFragFixedSize, Nfailed);

    if (Nfailed > 0)
    {
        fprintf(stderr, "TestAllocator: %d tests FAILED\n", Nfailed);
        return 1;
    }

    fprintf(stderr, "TestAllocator: all tests PASSED\n");
    return 0;
}


int allocateFixedSize_FreeSerial()
{
    int count = 0;
    int *i    = NULL;
    int ret   = 0;


    for (count = 0; count < MAX_RUNS; count++)
    {
        i = (int *)malloc(FIXED_SIZE_BLOCK);
        if (i == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            break;
        }
        free(i);
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : " << FIXED_SIZE_BLOCK << " bytes"
              << std::endl;

    return ret;
}

int allocateVarSize_FreeSerial()
{
    int k     = 0;
    int count = 0;
    int *i    = NULL;
    int ret   = 0;


    for (count = 0; count < MAX_RUNS / 2; count++)
    {
        k = rand() % (FIXED_SIZE_BLOCK - 63) + 64; // Generate random values between 64 Kb and
                                                   // 1 Mb.
        i = (int *)malloc(k * FIXED_SIZE_BLOCK);
        if (i == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            break;
        }
        free(i);
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : variable between 64kb and 1mb"
              << std::endl;

    return ret;
}

int allocateFixedSize_FreeReverse()
{
    int count = 0;
    int **i   = (int **)malloc(MAX_POINTERS * sizeof(int *));
    int ret   = 0;

    if (i == NULL)
    {
        std::cout << "Out of memory at count " << count << std::endl;
        ret = 1;
        return ret;
    }

    // Check also if allocated blocks exceed 16Gb
    for (count = 0; count < MAX_POINTERS && (((FIXED_SIZE_BLOCK * count) / 1048576) < 16384); count++)
    {
        i[count] = (int *)malloc(FIXED_SIZE_BLOCK);
        if (i[count] == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            count--;
            break;
        }
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : " << FIXED_SIZE_BLOCK << " bytes"
              << std::endl;

    int allocatedPtr = count;
    // Free in reverse order
    for (count = allocatedPtr; count >= 0; count--)
    {
        free(i[count]);
    }

    free(i);

    return ret;
}

int allocateVarSize_FreeReverse()
{
    int count      = 0;
    int k          = 0;
    int **i        = (int **)malloc(MAX_POINTERS * sizeof(int *));
    int ret        = 0;
    long allocated = 0;

    if (i == NULL)
    {
        std::cout << "Out of memory at count " << count << std::endl;
        ret = 1;
        return ret;
    }

    // Check also if allocated blocks exceed 16Gb
    for (count = 0; count < MAX_POINTERS && ((allocated / 1048576) < 16384); count++)
    {
        k        = rand() % (FIXED_SIZE_BLOCK - 63) + 64;
        i[count] = (int *)malloc(k * FIXED_SIZE_BLOCK);
        allocated += (k * FIXED_SIZE_BLOCK);
        if (i[count] == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            count--;
            break;
        }
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : variable between 64kb and 1mb"
              << std::endl;

    int allocatedPtr = count;
    // Free in reverse order
    for (count = allocatedPtr; count >= 0; count--)
    {
        free(i[count]);
    }
    free(i);

    return ret;
}


int allocateFixedSize_FreeInorder()
{
    int count = 0;
    int ret   = 0;

    int **i = (int **)malloc(MAX_POINTERS * sizeof(int *));

    if (i == NULL)
    {
        std::cout << "Failed to allocate block";
        ret = 1;
        return ret;
    }

    // Check also if allocated blocks exceed 16Gb
    for (count = 0; count < MAX_POINTERS && (((FIXED_SIZE_BLOCK * count) / 1048576) < 16384); count++)
    {
        i[count] = (int *)malloc(FIXED_SIZE_BLOCK);
        if (i[count] == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            count--;
            break;
        }
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : " << FIXED_SIZE_BLOCK << " bytes"
              << std::endl;

    int allocatedPtr = count;

    for (count = 0; count <= allocatedPtr; count++)
    {
        free(i[count]);
    }

    free(i);

    return ret;
}

int allocateVarSize_FreeInorder()
{
    int count      = 0;
    int ret        = 0;
    int k          = 0;
    int **i        = (int **)malloc(MAX_POINTERS * sizeof(int *));
    long allocated = 0;

    if (i == NULL)
    {
        ret = 1;
        std::cout << "Failed to allocate block";
        return ret;
    }

    // Check also if allocated blocks exceed 16Gb
    for (count = 0; count < MAX_POINTERS && ((allocated / 1048576) < 16384); count++)
    {
        k        = rand() % (FIXED_SIZE_BLOCK - 63) + 64;
        i[count] = (int *)malloc(k * FIXED_SIZE_BLOCK);
        allocated += (k * FIXED_SIZE_BLOCK);

        if (i[count] == NULL)
        {
            std::cout << "Out of memory at count " << count << std::endl;
            ret = 1;
            count--;
            break;
        }
    }

    std::cout << "Number of blocks allocated :" << count << "\nSize of each block : variable between 64kb and 1mb"
              << std::endl;

    int allocatedPtr = count;

    for (count = 0; count <= allocatedPtr; count++)
    {
        free(i[count]);
    }

    free(i);

    return ret;
}

/********************** Tests Begin ************************/
/**
 * ------------Time performance Test----------------
 * This test allocates fixed size blocks of 1024 bytes and immediately
 * frees them after allocation.
 */
int TestAllocator::testAllocateFixedSize_FreeSerial()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret                    = 0;

    m_begin             = clock();
    ret                 = allocateFixedSize_FreeSerial();
    m_end               = clock();
    m_time_spent_malloc = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateFixedSize_FreeSerial - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / MAX_RUNS) * 1000000 << " usec" << std::endl;
    return ret;
}

/**
 * ------------Time performance Test----------------
 * The test allocates blocks of variable size from 64 kb to
 * 1 mb by using a random number and frees them immediately after allocation.
 */
int TestAllocator::testAllocateVarSize_FreeSerial()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret                    = 0;

    // test
    m_begin             = clock();
    ret                 = allocateVarSize_FreeSerial();
    m_end               = clock();
    m_time_spent_malloc = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateVarSize_FreeSerial - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / (MAX_RUNS / 2)) * 1000000 << " usec"
              << std::endl;
    return ret;
}

/**
 * ------------Time performance Test----------------
 * The below test creates pointer of pointers to free the allocated blocks
 * in reverse order of allocation.
 */
int TestAllocator::testAllocateFixedSize_FreeReverse()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret                    = 0;
    m_begin                    = clock();
    ret                        = allocateFixedSize_FreeReverse();
    m_end                      = clock();
    m_time_spent_malloc        = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateFixedSize_FreeReverse - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / MAX_POINTERS) * 1000000 << " usec"
              << std::endl;
    return ret;
}

/**
 * ------------Time performance Test----------------
 * The test allocates the blocks of same size that is 1kb and
 * it frees them in the second loop in the same order as they were allocated.
 */
int TestAllocator::testAllocateFixedSize_FreeInorder()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret                    = 0;

    m_begin             = clock();
    ret                 = allocateFixedSize_FreeInorder();
    m_end               = clock();
    m_time_spent_malloc = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateFixedSize_FreeInorder - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / MAX_POINTERS) * 1000000 << " usec"
              << std::endl;
    return ret;
}

/**
 * ------------Time performance Test----------------
 * The below test creates variable sized blocks of size between 64kb to
 * 1 mb.The blocks are freed in the reverse order of allocation in the
 * second loop.
 */
int TestAllocator::testAllocateVarSize_FreeReverse()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret                    = 0;

    m_begin             = clock();
    ret                 = allocateVarSize_FreeReverse();
    m_end               = clock();
    m_time_spent_malloc = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateVarSize_FreeReverse - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / MAX_POINTERS) * 1000000 << " usec"
              << std::endl;
    return ret;
}

/**
 * ------------Time performance Test----------------
 * The test allocates variables sized blocks from 64kb to 1 Mb
 * and frees them in order they were allocated.
 */
int TestAllocator::testAllocateVarSize_FreeInorder()
{
    clock_t m_begin;
    clock_t m_end;
    double m_time_spent_malloc = 0;
    int ret;

    m_begin             = clock();
    ret                 = allocateVarSize_FreeInorder();
    m_end               = clock();
    m_time_spent_malloc = (double)(m_end - m_begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken to run testAllocateVarSize_FreeInorder - " << m_time_spent_malloc << " seconds"
              << std::endl;
    std::cout << "Time taken to run per block - " << (m_time_spent_malloc / MAX_POINTERS) * 1000000 << " usec"
              << std::endl;
    return ret;
}

/************************** Frag Tests ****************************/

/* ---------------Memory Overhead test-------------------
 * The below test tries to find memory overhead by measuring internal fragmentation
 * in allocation. The blocks are not freed so that internal fragmentation
 * can be measured.
 */
int TestAllocator::testInternalFragFixedSize()
{
    unsigned int blockSize = 64 * FIXED_SIZE_BLOCK; // 64 Kb
    int *current           = (int *)malloc(blockSize);
    int ret                = 0;

    std::vector<int *> allocMem;

    if (current == NULL)
    {
        ret = 1;
        return ret;
    }

    int *next  = NULL;
    long total = 0;
    long inUse = 0;
    long sum   = 0;
    char *preloadPath;
    preloadPath = getenv("LD_PRELOAD");

    allocMem.push_back(current);

    // Check also if allocated blocks exceed 16Gb
    // Reducing number of max runs as we are not freeing the memory to calculate fragmentation.
    for (int i = 0; (i < MAX_RUNS / 1000) && ((inUse / 1048576) < 16384); i++)
    {
        next = (int *)malloc(blockSize);
        if (next == NULL)
        {
            ret = 1;
            std::cout << "Failed to allocate block number - " << i;
            break;
        }
        sum     = (long)next - (long)current;
        total   = total + sum;
        inUse   = inUse + blockSize;
        current = next;
        allocMem.push_back(current);
    }

    for (const auto &i : allocMem)
    {
        free(i);
    }

    if (ret == 1)
    {
        return ret;
    }

    std::cout << "\ntestInternalFragFixedSize\n\n";
    std::cout << "Number of blocks allocated : " << MAX_RUNS / 1000 << "\nSize of each block : " << FIXED_SIZE_BLOCK
              << " bytes" << std::endl;
    std::cout << "Requested : " << inUse << " bytes" << std::endl;
    if (preloadPath == NULL)
    {
        std::cout << "Allocated : " << total << " bytes" << std::endl;
        std::cout << "Overhead  : " << total - inUse << "    bytes" << std::endl;
        std::cout << "Overhead per block : " << (total - inUse) / (MAX_RUNS / 1000) << " bytes" << std::endl;
    }
    else
    {
        std::cout << "\n\n";
    }

    return ret;
}
