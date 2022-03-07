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

#include "TestKeyedVector.h"
#include "keyedvector.h"
#include "timelib.h"
#include <string.h>

/*****************************************************************************/
typedef struct kv_test_t
{
    int key;
    int value;
    int value2;
} kv_test_t, *kv_test_p;

/*****************************************************************************/
/* Test data passed to keyedvector_alloc */
typedef struct kv_test_user_t
{
    int NfreeCBCalls;         /* Count of number of free callback calls */
    int NfreeCBFailures;      /* Count of errors detected in calling freeCB */
    kv_test_t elemBeingFreed; /* Used for comparing element being freed.
                                 This will rarely match for batch delete calls
                                 so -1 in key is considered "don't compare" */

} kv_test_user_t, *kv_test_user_p;

/*****************************************************************************/
static int kv_test_cmpCB(void *Left, void *Right)
{
    kv_test_p left  = (kv_test_p)Left;
    kv_test_p right = (kv_test_p)Right;

    if (left->key < right->key)
        return -1;
    else if (left->key > right->key)
        return 1;
    else
        return 0;
}

/*****************************************************************************/
static int kv_test_mergeCB(void *current, void *inserting, void *user)
{
    /* Do nothing */
    return 0;
}

/*****************************************************************************/
static void kv_test_freeCB(void *Elem, void *User)
{
    kv_test_user_p user = (kv_test_user_p)User;
    kv_test_p elem      = (kv_test_p)Elem;

    user->NfreeCBCalls++;

    /* If we're being asked to validate the element being freed, make sure
     * it matches our expectations
     */
    if (user->elemBeingFreed.key != -1)
    {
        if (elem->key != user->elemBeingFreed.key)
        {
            fprintf(stderr, "freeCB expected key %d. Got key %d\n", user->elemBeingFreed.key, elem->key);
            user->NfreeCBFailures++;
        }
    }
}

/*****************************************************************************/
int TestKeyedVector::HelperVerifyOrder(keyedvector_p kv)
{
    kv_test_p prevElem;
    kv_test_p elem, elem2;
    int i = 0, st;
    kv_cursor_t cursor, cursor2;

    st = HelperVerifySize(kv);
    if (st)
        return 50;

    prevElem = (kv_test_p)keyedvector_first(kv, &cursor);
    for (elem = (kv_test_p)keyedvector_next(kv, &cursor); elem; elem = (kv_test_p)keyedvector_next(kv, &cursor))
    {
        if (prevElem->key >= elem->key)
        {
            fprintf(stderr,
                    "Out of order at %d (cursor %d,%d). %d > %d\n",
                    i,
                    cursor.blockIndex,
                    cursor.subIndex,
                    prevElem->key,
                    elem->key);
            return 100;
        }

        /* Refind the element */
        elem2 = (kv_test_p)keyedvector_find_by_key(kv, elem, KV_LGE_EQUAL, &cursor2);
        if (!elem2)
        {
            fprintf(stderr, "Unable to refind element key %d\n", elem->key);
            return 150;
        }

        if (elem2->key != elem->key)
        {
            fprintf(stderr, "Error. Set breakpoint\n");
            return 200;
        }

        prevElem = elem;
        i++;
    }

    prevElem = (kv_test_p)keyedvector_last(kv, &cursor);
    for (elem = (kv_test_p)keyedvector_prev(kv, &cursor); elem; elem = (kv_test_p)keyedvector_prev(kv, &cursor))
    {
        if (prevElem->key <= elem->key)
        {
            fprintf(stderr, "Out of order at %d. %d > %d\n", i, prevElem->key, elem->key);
            return 300;
        }

        /* Refind the element */
        elem2 = (kv_test_p)keyedvector_find_by_key(kv, elem, KV_LGE_EQUAL, &cursor2);
        if (!elem2)
        {
            fprintf(stderr, "Unable to refind element key %d\n", elem->key);
            return 350;
        }

        if (elem2->key != elem->key)
        {
            fprintf(stderr, "Error. Set breakpoint\n");
            return 400;
        }

        prevElem = elem;
        i++;
    }

    return 0;
}

/*****************************************************************************/
int TestKeyedVector::HelperVerifySize(keyedvector_p kv)
{
    int kvSize     = keyedvector_size(kv);
    int kvSizeSlow = keyedvector_size_slow(kv);

    if (kvSize != kvSizeSlow)
    {
        fprintf(stderr, "KeyedVector size mismatch. size %d != %d slow size\n", kvSize, kvSizeSlow);
        return 100;
    }

    return 0;
}

/*****************************************************************************/
TestKeyedVector::TestKeyedVector()
{}

/*****************************************************************************/
TestKeyedVector::~TestKeyedVector()
{}

/*************************************************************************/
std::string TestKeyedVector::GetTag()
{
    return std::string("keyedvector");
}

/*****************************************************************************/
int TestKeyedVector::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

/*****************************************************************************/
int TestKeyedVector::Cleanup()
{
    return 0;
}

/*****************************************************************************/
int TestKeyedVector::TestLinearInsertForward(void)
{
    int retSt = 0;
    int i, st;
    kv_test_t testElem;
    kv_cursor_t cursor;
    keyedvector_p kv;
    kv_test_user_t userData = {};
    int NtestElem           = 10000;

    userData.elemBeingFreed.key = -1;

    kv = keyedvector_alloc(sizeof(kv_test_t), 0, kv_test_cmpCB, kv_test_mergeCB, kv_test_freeCB, &userData, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }


    for (i = 0; i < NtestElem; i++)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 200;
            break;
        }

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "HelperVerifyOrder failed after %d\n", testElem.key);
            retSt = 300;
            break;
        }
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }

    /* Verify frees after destroy is called since destroy calls freeCB */
    if (userData.NfreeCBCalls != NtestElem)
    {
        fprintf(stderr, "Expected %d freeCB calls. Got %d\n", NtestElem, userData.NfreeCBCalls);
        retSt = 400;
    }
    if (userData.NfreeCBFailures > 0)
    {
        fprintf(stderr, "Got %d freeCB mismatch errors\n", userData.NfreeCBFailures);
        retSt = 500;
    }

    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestLinearInsertBackward(void)
{
    int retSt = 0;
    int i, st;
    kv_test_t testElem;
    kv_cursor_t cursor;
    keyedvector_p kv;
    kv_test_user_t userData = {};
    int NtestElem           = 10000;

    userData.elemBeingFreed.key = -1;

    kv = keyedvector_alloc(sizeof(kv_test_t), 0, kv_test_cmpCB, kv_test_mergeCB, kv_test_freeCB, &userData, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }


    for (i = NtestElem; i >= 0; i--)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 200;
            break;
        }

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "HelperVerifyOrder failed after %d\n", testElem.key);
            retSt = 300;
            break;
        }
    }

    /* Update expected elements to be deleted based on actual count in case the loop
     * above changes
     */
    NtestElem = keyedvector_size(kv);

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }

    /* Verify frees after destroy is called since destroy calls freeCB */
    if (userData.NfreeCBCalls != NtestElem)
    {
        fprintf(stderr, "Expected %d freeCB calls. Got %d\n", NtestElem, userData.NfreeCBCalls);
        retSt = 400;
    }
    if (userData.NfreeCBFailures > 0)
    {
        fprintf(stderr, "Got %d freeCB mismatch errors\n", userData.NfreeCBFailures);
        retSt = 500;
    }

    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestLinearRemoveByRangeForward(void)
{
    int retSt = 0;
    int i, st;
    int testNelems = 10000;
    int count;
    kv_test_t testElem, *findElem;
    kv_cursor_t cursor, cursor2;
    keyedvector_p kv;
    kv_test_user_t userData = {};

    userData.elemBeingFreed.key = -1;

    kv = keyedvector_alloc(sizeof(kv_test_t), 0, kv_test_cmpCB, kv_test_mergeCB, kv_test_freeCB, &userData, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 150;
            break;
        }
    }


    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        userData.elemBeingFreed.key = testElem.key;

        findElem = (kv_test_p)keyedvector_find_by_key(kv, &testElem, KV_LGE_EQUAL, &cursor);
        if (!findElem)
        {
            fprintf(stderr, "Didn't find elem %d before remove\n", testElem.key);
            retSt = 175;
            break;
        }

        /* We are deleting one element so use the same cursor for start and end */
        cursor2 = cursor;

        st = keyedvector_remove_range_by_cursor(kv, &cursor, &cursor2);
        if (st)
        {
            fprintf(stderr, "Error %d from keyedvector_remove_range_by_cursor of %d\n", st, testElem.key);
            retSt = 200;
            break;
        }

        userData.elemBeingFreed.key = -1;

        findElem = (kv_test_p)keyedvector_find_by_key(kv, &testElem, KV_LGE_EQUAL, &cursor);
        if (findElem)
        {
            fprintf(stderr, "Didn't expect to find elem %d (%d) after remove\n", testElem.key, findElem->key);
            retSt = 300;
            break;
        }

        count = keyedvector_size(kv);
        if (count != testNelems - i - 1)
        {
            fprintf(stderr, "Expected count %d. Got %d at i=%d\n", testNelems - i - 1, count, i);
            retSt = 400;
            break;
        }

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "HelperVerifyOrder failed after %d\n", testElem.key);
            retSt = 500;
            break;
        }

        if (i < testNelems - 1)
        {
            findElem = (kv_test_p)keyedvector_first(kv, &cursor);
            if (!findElem)
            {
                fprintf(stderr, "Unable to find first element at i=%d\n", i);
                retSt = 600;
                break;
            }
        }
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }

    /* Verify frees after destroy is called since destroy calls freeCB */
    if (userData.NfreeCBCalls != testNelems)
    {
        fprintf(stderr, "Expected %d freeCB calls. Got %d\n", testNelems, userData.NfreeCBCalls);
        retSt = 600;
    }
    if (userData.NfreeCBFailures > 0)
    {
        fprintf(stderr, "Got %d freeCB mismatch errors\n", userData.NfreeCBFailures);
        retSt = 700;
    }

    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestLinearRemoveForward(void)
{
    int retSt = 0;
    int i, st;
    int testNelems = 10000;
    int count;
    kv_test_t testElem, *findElem;
    kv_cursor_t cursor;
    keyedvector_p kv;
    kv_test_user_t userData = {};

    userData.elemBeingFreed.key = -1;

    kv = keyedvector_alloc(sizeof(kv_test_t), 0, kv_test_cmpCB, kv_test_mergeCB, kv_test_freeCB, &userData, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 150;
            break;
        }
    }


    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = i;
        testElem.value  = 0;
        testElem.value2 = 0;

        userData.elemBeingFreed.key = testElem.key;

        st = keyedvector_remove(kv, &testElem);
        if (st)
        {
            fprintf(stderr, "Error %d from Remove of %d\n", st, testElem.key);
            retSt = 200;
            break;
        }

        userData.elemBeingFreed.key = -1;

        findElem = (kv_test_p)keyedvector_find_by_key(kv, &testElem, KV_LGE_EQUAL, &cursor);
        if (findElem)
        {
            fprintf(stderr, "Didn't expect to find elem %d (%d) after remove\n", testElem.key, findElem->key);
            retSt = 300;
            break;
        }

        count = keyedvector_size(kv);
        if (count != testNelems - i - 1)
        {
            fprintf(stderr, "Expected count %d. Got %d at i=%d\n", testNelems - i - 1, count, i);
            retSt = 400;
            break;
        }

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "HelperVerifyOrder failed after %d\n", testElem.key);
            retSt = 500;
            break;
        }

        if (i < testNelems - 1)
        {
            findElem = (kv_test_p)keyedvector_first(kv, &cursor);
            if (!findElem)
            {
                fprintf(stderr, "Unable to find first element at i=%d\n", i);
                retSt = 600;
                break;
            }
        }
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }

    /* Verify frees after destroy is called since destroy calls freeCB */
    if (userData.NfreeCBCalls != testNelems)
    {
        fprintf(stderr, "Expected %d freeCB calls. Got %d\n", testNelems, userData.NfreeCBCalls);
        retSt = 600;
    }
    if (userData.NfreeCBFailures > 0)
    {
        fprintf(stderr, "Got %d freeCB mismatch errors\n", userData.NfreeCBFailures);
        retSt = 700;
    }

    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestTimingLinearInsert()
{
    int retSt = 0;
    int i, j, st, testNelems;
    kv_test_t testElem;
    kv_cursor_t cursor;
    keyedvector_p kv;
    timelib64_t t1, diff;
    int numBins = 10; /* Measure N bins of performance so we can see insert performance scale with the size of the KV */
    std::vector<double> perCall(numBins, 0.0);

    kv = keyedvector_alloc(sizeof(kv_test_t), 2048, kv_test_cmpCB, kv_test_mergeCB, 0, 0, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    testNelems = 100000; /* Actual size will be testNelems * numBins */

    for (i = 0; i < numBins; i++)
    {
        t1 = timelib_usecSince1970();

        for (j = 0; j < testNelems; j++)
        {
            testElem.key    = (i * testNelems) + j;
            testElem.value  = 0;
            testElem.value2 = 0;

            st = keyedvector_insert(kv, &testElem, &cursor);
            if (st)
            {
                fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
                retSt = 200;
                goto CLEANUP;
            }
        }

        diff       = timelib_usecSince1970() - t1;
        perCall[i] = (double)diff / (double)testNelems;
    }

    for (i = 0; i < numBins; i++)
    {
        printf("Bin %d: %f usec/call\n", i, perCall[i]);
    }


CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }
    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestTimingRandomInsert()
{
    int retSt = 0;
    int i, st, testNelems;
    kv_test_t testElem;
    kv_cursor_t cursor;
    keyedvector_p kv;
    timelib64_t t1, diff;
    double perCall;


    kv = keyedvector_alloc(sizeof(kv_test_t), 2048, kv_test_cmpCB, kv_test_mergeCB, 0, 0, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    testNelems = 1000000;

    t1 = timelib_usecSince1970();

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = rand() % (testNelems * 10);
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 200;
            goto CLEANUP;
        }
    }

    diff    = timelib_usecSince1970() - t1;
    perCall = (double)diff / (double)testNelems;
    printf("%d random inserts done in %lld usec (%f usec/call)\n", testNelems, (long long)diff, perCall);

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }
    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestRemoveRangeByCursor()
{
    int retSt = 0;
    int startDeleteIdx, endDeleteIdx, beforeStartDeleteIdx, afterEndDeleteIdx;
    int beforeStartKey, afterEndKey;
    int i, st, testNelems, endTestNelem, Nelem, NelemExpected;
    kv_test_t testElem;
    kv_test_p startDeleteElem, endDeleteElem, elem;
    kv_cursor_t cursor, startDeleteCursor, endDeleteCursor, tempCursor;
    keyedvector_p kv;
    kv_test_user_t userData = {};
    int expectedDeletes     = 0;

    userData.elemBeingFreed.key = -1;

    kv = keyedvector_alloc(sizeof(kv_test_t), 2048, kv_test_cmpCB, kv_test_mergeCB, kv_test_freeCB, &userData, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedvector_alloc failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    testNelems   = 10000;
    endTestNelem = testNelems / 10; /* What count to end testing below? */

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = rand() % (testNelems * 10);
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 125;
            goto CLEANUP;
        }

        // printf("Inserted %d at cursor %d,%d\n", testElem.key,
        //       cursor.blockIndex, cursor.subIndex);

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "Verify order failed before test started\n");
            retSt = 150;
            goto CLEANUP;
        }
    }

    /* Verify order before we start */
    st = HelperVerifyOrder(kv);
    if (st)
    {
        fprintf(stderr, "Verify order failed before test started\n");
        retSt = 200;
        goto CLEANUP;
    }


    /* Delete just the first element */
    startDeleteElem = (kv_test_p)keyedvector_first(kv, &startDeleteCursor);
    if (!startDeleteElem)
    {
        fprintf(stderr, "Unexpected null first element\n");
        retSt = 220;
        goto CLEANUP;
    }
    endDeleteCursor = startDeleteCursor;
    /* Do the delete */
    st = keyedvector_remove_range_by_cursor(kv, &startDeleteCursor, &endDeleteCursor);
    if (st)
    {
        fprintf(stderr, "keyedvector_remove_range_by_cursor() failed with %d\n", st);
        retSt = 221;
        goto CLEANUP;
    }

    st = HelperVerifyOrder(kv);
    if (st)
    {
        fprintf(stderr, "Verify order failed before test started\n");
        retSt = 222;
        goto CLEANUP;
    }

    /* Since we used random keys above, we have to dynamically determine how many
     * elements were inserted
     */
    userData.NfreeCBCalls = 0;
    expectedDeletes       = keyedvector_size(kv);

    for (Nelem = keyedvector_size(kv); Nelem >= endTestNelem; Nelem = keyedvector_size(kv))
    {
        startDeleteIdx = rand() % ((Nelem * 9) / 10); /* Start within first 90% of Nelem */
        endDeleteIdx   = startDeleteIdx
                       + rand()
                             % std::min(((Nelem - 1) - startDeleteIdx),
                                        endTestNelem); /* Don't delete too many. Set at max 10% */
        beforeStartDeleteIdx = startDeleteIdx - 1;
        afterEndDeleteIdx    = endDeleteIdx + 1;

        startDeleteElem = (kv_test_p)keyedvector_find_by_index(kv, startDeleteIdx, &startDeleteCursor);
        if (!startDeleteElem)
        {
            fprintf(stderr, "Unexpected null elem at idx %d/%d\n", startDeleteIdx, Nelem);
            retSt = 250;
            goto CLEANUP;
        }

        // printf("Start delete idx %d, cursor %d,%d\n", startDeleteIdx,
        //       startDeleteCursor.blockIndex, startDeleteCursor.subIndex);

        endDeleteElem = (kv_test_p)keyedvector_find_by_index(kv, endDeleteIdx, &endDeleteCursor);
        if (!endDeleteElem)
        {
            fprintf(stderr, "Unexpected null elem at idx %d/%d\n", endDeleteIdx, Nelem);
            retSt = 300;
            goto CLEANUP;
        }

        // printf("End delete idx %d, cursor %d,%d\n", endDeleteIdx,
        //       endDeleteCursor.blockIndex, endDeleteCursor.subIndex);

        /* Get element before beginning element */
        elem = (kv_test_p)keyedvector_find_by_index(kv, beforeStartDeleteIdx, &tempCursor);
        if (!elem)
            beforeStartKey = -1;
        else
            beforeStartKey = elem->key;
        /* Get element after last element */
        elem = (kv_test_p)keyedvector_find_by_index(kv, afterEndDeleteIdx, &tempCursor);
        if (!elem)
            afterEndKey = -1;
        else
            afterEndKey = elem->key;

        /* Do the delete */
        st = keyedvector_remove_range_by_cursor(kv, &startDeleteCursor, &endDeleteCursor);
        if (st)
        {
            fprintf(stderr, "keyedvector_remove_range_by_cursor() failed with %d\n", st);
            retSt = 500;
            goto CLEANUP;
        }

        NelemExpected = Nelem - ((endDeleteIdx - startDeleteIdx) + 1);
        Nelem         = keyedvector_size(kv);
        if (Nelem != NelemExpected)
        {
            fprintf(stderr,
                    "Got unexpected Nelem %d != %d after delete from %d - %d\n",
                    Nelem,
                    NelemExpected,
                    startDeleteIdx,
                    endDeleteIdx);
            retSt = 600;
            goto CLEANUP;
        }

        st = HelperVerifyOrder(kv);
        if (st)
        {
            fprintf(stderr, "Out of order! helper_verify_order failed\n");
            retSt = 650;
            goto CLEANUP;
        }

        /* Now refind the border elements and see if they are adjacent after we deleted all
         * of the elements in between */
        if (beforeStartKey == -1 || afterEndKey == -1)
            continue; /* Can't compare null values */

        testElem.key    = beforeStartKey;
        startDeleteElem = (kv_test_p)keyedvector_find_by_key(kv, &testElem, KV_LGE_EQUAL, &tempCursor);
        if (!startDeleteElem)
        {
            fprintf(stderr, "Unable to refind before start key %d after delete\n", beforeStartKey);
            retSt = 700;
            goto CLEANUP;
        }

        testElem.key  = afterEndKey;
        endDeleteElem = (kv_test_p)keyedvector_find_by_key(kv, &testElem, KV_LGE_EQUAL, &tempCursor);
        if (!endDeleteElem)
        {
            fprintf(stderr, "Unable to refind after end key %d after delete\n", afterEndKey);
            retSt = 800;
            goto CLEANUP;
        }

        /* Now walk backward one element from tempCursor. Hopefully, it's the same key (and value) as startDeleteElem */
        elem = (kv_test_p)keyedvector_prev(kv, &tempCursor);
        if (!elem || elem->key != startDeleteElem->key)
        {
            fprintf(stderr,
                    "Refind mismatch. Expected key %d, got key %d walking back from key %d\n",
                    startDeleteElem->key,
                    elem ? elem->key : -1,
                    endDeleteElem->key);
            retSt = 900;
            goto CLEANUP;
        }
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }

    /* Verify frees after destroy is called since destroy calls freeCB */
    if (userData.NfreeCBCalls != expectedDeletes)
    {
        fprintf(stderr, "Expected %d freeCB calls. Got %d\n", expectedDeletes, userData.NfreeCBCalls);
        retSt = 1000;
    }
    if (userData.NfreeCBFailures > 0)
    {
        fprintf(stderr, "Got %d freeCB mismatch errors\n", userData.NfreeCBFailures);
        retSt = 1100;
    }

    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestFindByIndex()
{
    int retSt = 0;
    int i, st, testNelems;
    kv_test_t testElem;
    kv_test_p iterElem, indexElem;
    kv_cursor_t cursor, iterCursor, indexCursor;
    keyedvector_p kv;


    kv = keyedvector_alloc(sizeof(kv_test_t), 2048, kv_test_cmpCB, kv_test_mergeCB, 0, 0, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    testNelems = 100000;

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = rand() % (testNelems * 10);
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 150;
            goto CLEANUP;
        }
    }

    i = 0;
    for (iterElem = (kv_test_p)keyedvector_first(kv, &iterCursor); iterElem;
         iterElem = (kv_test_p)keyedvector_next(kv, &iterCursor))
    {
        indexElem = (kv_test_p)keyedvector_find_by_index(kv, i, &indexCursor);
        if (!indexElem)
        {
            fprintf(stderr, "keyedvector_find_by_index returned null @ %d\n", i);
            retSt = 200;
            goto CLEANUP;
        }

        if (indexElem->key != iterElem->key)
        {
            fprintf(stderr, "keyedvector_find_by_index mismatch at %d\n", i);
            retSt = 300;
            goto CLEANUP;
        }

        i++;
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }
    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestFindByKeyAtSize(int testNelems)
{
    int retSt = 0;
    int i, st;
    kv_test_t testElem;
    kv_test_t findKey;
    kv_test_p eqElem, ltElem, lteElem, gtElem, gteElem, prevElem, nextElem;
    kv_cursor_t eqCursor, ltCursor, lteCursor, gtCursor, gteCursor, prevCursor, nextCursor;
    kv_cursor_t cursor;
    keyedvector_p kv;

    kv = keyedvector_alloc(sizeof(kv_test_t), 2048, kv_test_cmpCB, kv_test_mergeCB, 0, 0, &st);
    if (!kv)
    {
        fprintf(stderr, "keyedVector->Init failed with %d\n", st);
        retSt = 100;
        goto CLEANUP;
    }

    for (i = 0; i < testNelems; i++)
    {
        testElem.key    = i * 2;
        testElem.value  = 0;
        testElem.value2 = 0;

        st = keyedvector_insert(kv, &testElem, &cursor);
        if (st)
        {
            fprintf(stderr, "Error %d from Insert of %d\n", st, testElem.key);
            retSt = 150;
            goto CLEANUP;
        }
    }


    /* Walk forward. Every 2nd step will hit an element */
    for (i = 1; i < (testNelems)-1; i++)
    {
        findKey.key = i * 2;

        eqElem = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_EQUAL, &eqCursor);
        if (!eqElem)
        {
            fprintf(stderr, "Expected to find element with key %d\n", findKey.key);
            retSt = 200;
            goto CLEANUP;
        }

        ltElem  = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_LESS, &ltCursor);
        lteElem = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_LESSEQUAL, &lteCursor);
        gtElem  = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_GREATER, &gtCursor);
        gteElem = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_GREATEQUAL, &gteCursor);
        memcpy(&prevCursor, &eqCursor, sizeof(kv_cursor_t));
        prevElem = (kv_test_p)keyedvector_prev(kv, &prevCursor);
        memcpy(&nextCursor, &eqCursor, sizeof(kv_cursor_t));
        nextElem = (kv_test_p)keyedvector_next(kv, &nextCursor);

        if (!ltElem || !prevElem || ltElem != prevElem || memcmp(&ltCursor, &prevCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "LT != PREV\n");
            retSt = 300;
            goto CLEANUP;
        }

        if (!gtElem || !nextElem || gtElem != nextElem || memcmp(&gtCursor, &nextCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "GT != NEXT\n");
            retSt = 400;
            goto CLEANUP;
        }

        if (!gteElem || gteElem != eqElem || memcmp(&gteCursor, &eqCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "GTE != EQ\n");
            retSt = 500;
            goto CLEANUP;
        }

        if (!lteElem || lteElem != eqElem || memcmp(&lteCursor, &eqCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "LTE != EQ\n");
            retSt = 600;
            goto CLEANUP;
        }


        /* Now search for the element in between */
        findKey.key = (i * 2) - 1;
        eqElem      = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_EQUAL, &eqCursor);
        if (eqElem)
        {
            fprintf(stderr, "Unexpected element with key %d\n", findKey.key);
            retSt = 700;
            goto CLEANUP;
        }

        ltElem  = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_LESS, &ltCursor);
        lteElem = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_LESSEQUAL, &lteCursor);
        gtElem  = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_GREATER, &gtCursor);
        gteElem = (kv_test_p)keyedvector_find_by_key(kv, &findKey, KV_LGE_GREATEQUAL, &gteCursor);

        if (!ltElem || !lteElem || ltElem != lteElem || memcmp(&ltCursor, &lteCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "LT != LTE\n");
            retSt = 800;
            goto CLEANUP;
        }

        if (ltElem->key != ((i * 2) - 2) || lteElem->key != ((i * 2) - 2))
        {
            fprintf(
                stderr, "Wrong LT/LTE key. Expected %d. Got LT %d, LTE %d\n", ((i * 2) - 2), ltElem->key, lteElem->key);
            retSt = 900;
            goto CLEANUP;
        }

        if (!gtElem || !gteElem || gtElem != gteElem || memcmp(&gtCursor, &gteCursor, sizeof(kv_cursor_t)))
        {
            fprintf(stderr, "GT != GTE\n");
            retSt = 1000;
            goto CLEANUP;
        }

        if (gtElem->key != (i * 2) || gteElem->key != (i * 2))
        {
            fprintf(stderr, "Wrong GT/GTE key. Expected %d. Got GT %d, GTE %d\n", (i * 2), ltElem->key, lteElem->key);
            retSt = 1100;
            goto CLEANUP;
        }
    }

CLEANUP:
    if (kv)
    {
        keyedvector_destroy(kv);
        kv = 0;
    }
    return retSt;
}

/*****************************************************************************/
int TestKeyedVector::TestFindByKey()
{
    /* Run our sub-test at various sizes */
    int i, st;
    int numTests      = 20;
    int testNelem[20] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 50, 100, 1000, 10000 };

    for (i = 0; i < numTests; i++)
    {
        st = TestFindByKeyAtSize(testNelem[i]);
        if (st)
        {
            fprintf(stderr, "TestFindByKeyAtSize(%d) FAILED with %d\n", testNelem[i], st);
            return st;
        }
    }

    return 0;
}

/*****************************************************************************/
int TestKeyedVector::Run()
{
    int st;
    int Nfailed = 0;

    st = TestLinearInsertForward();
    if (st)
    {
        fprintf(stderr, "TestLinearInsertForward failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestLinearInsertForward PASSED\n");

    st = TestLinearRemoveByRangeForward();
    if (st)
    {
        fprintf(stderr, "TestLinearRemoveByRangeForward failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestLinearRemoveByRangeForward PASSED\n");

    st = TestFindByIndex();
    if (st)
    {
        fprintf(stderr, "TestFindByIndex failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestFindByIndex PASSED\n");

    st = TestFindByKey();
    if (st)
    {
        fprintf(stderr, "TestFindByKey failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestFindByKey PASSED\n");

    st = TestRemoveRangeByCursor();
    if (st)
    {
        fprintf(stderr, "TestRemoveRangeByCursor failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestRemoveRangeByCursor PASSED\n");

    st = TestLinearInsertBackward();
    if (st)
    {
        fprintf(stderr, "TestLinearInsertBackward failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestLinearInsertBackward PASSED\n");

    st = TestLinearRemoveForward();
    if (st)
    {
        fprintf(stderr, "TestLinearRemoveForward failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestLinearRemoveForward PASSED\n");


    st = TestTimingRandomInsert();
    if (st)
    {
        fprintf(stderr, "TestTimingRandomInsert failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestTimingRandomInsert PASSED\n");

    st = TestTimingLinearInsert();
    if (st)
    {
        fprintf(stderr, "TestTimingLinearInsert failed with %d\n", st);
        Nfailed++;
    }
    else
        printf("TestTimingLinearInsert PASSED\n");

    return 0;
}


/*****************************************************************************/
