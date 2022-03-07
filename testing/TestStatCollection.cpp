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

#include "TestStatCollection.h"
#include "DcgmLogging.h"
#include "DcgmStatCollection.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>

/*****************************************************************************/
/* Operation categories */
#define SRT_CAT_GLOBAL 0
#define SRT_CAT_GROUP  1
#define SRT_CAT_GPU    2
#define SRT_CAT_ENTITY 3
#define SRT_CAT_MAX    4 /* One greater than max value above */

#define SRT_OP_MAX (SRT_CAT_MAX * MC_TYPE_COUNT) /* Operation indexes for fake work */

#define SRT_MAKE_OP(cat, mct) ((MC_TYPE_COUNT * (cat)) + (mct))
#define SRT_OP_CAT(op)        ((op) % MC_TYPE_COUNT)
#define SRT_OP_MCT(op)        ((op) / SRT_CAT_MAX)

static int g_groupIndex = 0;
static int g_keyIndex   = 0;

/*****************************************************************************/
TestStatCollection::TestStatCollection()
{}

/*****************************************************************************/
TestStatCollection::~TestStatCollection()
{}

/*****************************************************************************/
int TestStatCollection::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

/*****************************************************************************/
int TestStatCollection::Cleanup()
{
    return 0;
}

/*****************************************************************************/
std::string TestStatCollection::GetTag()
{
    return std::string("statcollection");
}

/*****************************************************************************/
int do_stat_collection_op(DcgmStatCollection *statCollection, int opIndex)
{
    char groupStr[32] = { 0 };
    char keyStr[32]   = { 0 };
    std::string key;
    std::string group;
    unsigned int gpuIndex = rand() % 32;
    /* Pass a timestamp in or the binary-logged timestamp won't match the
     * time series's value
     */
    timelib64_t timestamp = timelib_usecSince1970();

    if (opIndex < 0)
    {
        /* do a random op */
        opIndex = rand() % SRT_OP_MAX;
    }

    sprintf(groupStr, "group%d", (++g_groupIndex) % 1000000);
    group = groupStr;
    sprintf(keyStr, "key%d", (++g_keyIndex) % 1000000);
    key = keyStr;

    switch (SRT_OP_CAT(opIndex))
    {
        default:
        case MC_TYPE_TIMESTAMP:
        case MC_TYPE_INT64:
        {
            long long val = (long long)rand();

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->SetGlobalStat(key, val);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->SetGroupedStat(group, key, val);
                    break;
                case SRT_CAT_GPU:
                    statCollection->SetGpuStat(gpuIndex, key, val);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->SetEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val);
                    break;
            }
            break;
        }

        case MC_TYPE_DOUBLE:
        {
            double val = 0.001 * (double)rand();

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->SetGlobalStat(key, val);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->SetGroupedStat(group, key, val);
                    break;
                case SRT_CAT_GPU:
                    statCollection->SetGpuStat(gpuIndex, key, val);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->SetEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val);
                    break;
            }
            break;
        }

        case MC_TYPE_STRING:
        {
            std::string val = key;

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->SetGlobalStat(key, val);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->SetGroupedStat(group, key, val);
                    break;
                case SRT_CAT_GPU:
                    statCollection->SetGpuStat(gpuIndex, key, val);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->SetEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val);
                    break;
            }
            break;
        }

        case MC_TYPE_TIMESERIES_DOUBLE:
        {
            double val = 0.001 * (double)rand();

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->AppendGlobalStat(key, val, timestamp);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->AppendGroupedStat(group, key, val, timestamp);
                    break;
                case SRT_CAT_GPU:
                    statCollection->AppendGpuStat(gpuIndex, key, val, 0, timestamp);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->AppendEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val, 0.0, timestamp);
                    break;
            }
            break;
        }

        case MC_TYPE_TIMESERIES_INT64:
        {
            long long val = (long long)rand();

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->AppendGlobalStat(key, val, timestamp);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->AppendGroupedStat(group, key, val, timestamp);
                    break;
                case SRT_CAT_GPU:
                    statCollection->AppendGpuStat(gpuIndex, key, val, 0, timestamp);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->AppendEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val, 0, timestamp);
                    break;
            }
            break;
        }

        case MC_TYPE_TIMESERIES_STRING:
        {
            std::string val = "oh hai";

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->AppendGlobalStat(key, val, timestamp);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->AppendGroupedStat(group, key, val, timestamp);
                    break;
                case SRT_CAT_GPU:
                    statCollection->AppendGpuStat(gpuIndex, key, val, timestamp);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->AppendEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, val, timestamp);
                    break;
            }

            break;
        }

        case MC_TYPE_TIMESERIES_BLOB:
        {
            long long val = (long long)rand();

            switch (SRT_OP_MCT(opIndex))
            {
                default:
                case SRT_CAT_GLOBAL:
                    statCollection->AppendGlobalStat(key, &val, sizeof(val), timestamp);
                    break;
                case SRT_CAT_GROUP:
                    statCollection->AppendGroupedStat(group, key, &val, sizeof(val), timestamp);
                    break;
                case SRT_CAT_GPU:
                    statCollection->AppendGpuStat(gpuIndex, key, &val, sizeof(val), timestamp);
                    break;
                case SRT_CAT_ENTITY:
                    statCollection->AppendEntityStat(SC_ENTITY_GROUP_GPU, gpuIndex, key, &val, sizeof(val), timestamp);
                    break;
            }
            break;
        }
    }

    return 0;
}

/*****************************************************************************/
int TestStatCollection::TestCollectionMerge(void)
{
    int i, st;
    DcgmStatCollection *first = 0, *second = 0;

    /* Do one of each possible insert, merge into other collection, and test for equivalence */
    for (i = 0; i < SRT_OP_MAX; i++)
    {
        if (first)
        {
            delete (first);
            first = 0;
        }
        if (second)
        {
            delete (second);
            second = 0;
        }

        first = new DcgmStatCollection();
        // coverity[leaked_storage] if this throws an exception
        second = new DcgmStatCollection();

        do_stat_collection_op(first, i);

        second->MergeFrom(first);

        st = first->EqualTo(second, 1);
        if (!st)
        {
            fprintf(stderr, "Got first != second for op %d\n", i);
            delete (first);
            delete (second);
            return -1;
        }
    }

    if (first)
    {
        delete (first);
        first = 0;
    }
    if (second)
    {
        delete (second);
        second = 0;
    }

    return 0;
}

/*****************************************************************************/
int TestStatCollection::TestPerformance(void)
{
    DcgmStatCollection *statCollection = 0;
    int numGpus                        = 8;
    int numFieldsPerGpu                = 150; /* The statKey code below works up to 255 before the keys repeat */
    int numRecordsPerStat              = 10;
    int i, j, k;
    long long nowUsec;
    double t1, t2;
    char statKey[16] = { 0 };
    long long value1 = 123456, value2 = 789012;

    statCollection = new DcgmStatCollection();

    nowUsec = timelib_usecSince1970();
    t1      = timelib_dsecSince1970();

    /* measure how long it takes to add the first record for each gpu x key combination */
    for (i = 0; i < numGpus; i++)
    {
        for (j = 0; j < numFieldsPerGpu; j++)
        {
            statKey[0] = '0' + (j / 16);
            statKey[1] = '0' + (j % 16);

            /* Note we are adding k to t1 here so that timeseries doesn't have to keep adding
             * a microsecond when adding records (linear search)
             */
            statCollection->AppendGpuStat((unsigned int)i, (std::string)statKey, value1, value2, nowUsec);
        }
    }

    t2 = timelib_dsecSince1970();

    double numInserted   = numGpus * numFieldsPerGpu;
    double usecPerInsert = 1000000.0 * ((t2 - t1) / numInserted);

    printf("TestPerformance FirstInserted %d records at %.2f usec/record (IsDebug %d)\n",
           (int)numInserted,
           usecPerInsert,
           IsDebugBuild());

    nowUsec = timelib_usecSince1970();
    t1      = timelib_dsecSince1970();

    /* Now that the structure has at least one record for each key, measure how long it takes to add records */
    for (i = 0; i < numGpus; i++)
    {
        for (j = 0; j < numFieldsPerGpu; j++)
        {
            statKey[0] = '0' + (j / 16);
            statKey[1] = '0' + (j % 16);

            for (k = 0; k < numRecordsPerStat; k++)
            {
                /* Note we are adding k to t1 here so that timeseries doesn't have to keep adding
                 * a microsecond when adding records (linear search)
                 */
                statCollection->AppendGpuStat((unsigned int)i, (std::string)statKey, value1, value2, nowUsec + k);
            }
        }
    }

    t2 = timelib_dsecSince1970();

    numInserted   = numGpus * numFieldsPerGpu * numRecordsPerStat;
    usecPerInsert = 1000000.0 * ((t2 - t1) / numInserted);

    printf("TestPerformance Appended %d records at %.2f usec/record (IsDebug %d)\n",
           (int)numInserted,
           usecPerInsert,
           IsDebugBuild());

    mcollect_value_p gpuStat;

    t1 = timelib_dsecSince1970();

    for (i = 0; i < numGpus; i++)
    {
        for (j = 0; j < numFieldsPerGpu; j++)
        {
            statKey[0] = '0' + (j / 16);
            statKey[1] = '0' + (j % 16);

            /* Still search for this numRecordsPerStat times since it gives us a better average */
            for (k = 0; k < numRecordsPerStat; k++)
            {
                /* Note we are adding k to t1 here so that timeseries doesn't have to keep adding
                 * a microsecond when adding records (linear search)
                 */
                gpuStat = statCollection->GetGpuStat((unsigned int)i, (std::string)statKey);
                if (!gpuStat)
                {
                    fprintf(stderr, "Unexpected not found: nvmlIndex %d, statKey %s\n", i, statKey);
                    delete statCollection;
                    statCollection = nullptr;
                    return 200;
                }
            }
        }
    }

    t2 = timelib_dsecSince1970();

    double numSearched   = numGpus * numFieldsPerGpu * numRecordsPerStat;
    double usecPerSearch = 1000000.0 * ((t2 - t1) / numSearched);

    printf("TestPerformance Searched for %d records at %.2f usec/record (IsDebug %d)\n",
           (int)numSearched,
           usecPerSearch,
           IsDebugBuild());


    delete (statCollection);
    statCollection = nullptr;

    return 0;
}

/*****************************************************************************/
int TestStatCollection::Run(void)
{
    int st, Nfailed = 0;

    st = TestPerformance();
    if (st)
    {
        fprintf(stderr, "TestPerformance FAILED with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestPerformance PASSED\n");

    st = TestCollectionMerge();
    if (st)
    {
        fprintf(stderr, "TestCollectionMerge FAILED with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestCollectionMerge PASSED\n");

    if (Nfailed > 0)
        return 1;

    return 0;
}

/*****************************************************************************/
