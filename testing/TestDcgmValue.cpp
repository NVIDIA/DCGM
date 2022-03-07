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

#include "TestDcgmValue.h"
#include "nvcmvalue.h"

/*****************************************************************************/
TestDcgmValue::TestDcgmValue()
{}

/*****************************************************************************/
TestDcgmValue::~TestDcgmValue()
{}

/*************************************************************************/
std::string TestDcgmValue::GetTag()
{
    return std::string("nvcmvalue");
}

/*****************************************************************************/
int TestDcgmValue::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

/*****************************************************************************/
int TestDcgmValue::Cleanup()
{
    return 0;
}

/*****************************************************************************/
int TestDcgmValue::TestConversions()
{
    int i;
    int Nerrors = 0;
    int valueInt32;
    long long valueInt64;
    double valueDouble;

    /* All of the values in this list should match at each array index */
    int Nconversions = 6;
    int int32Values[6]
        = { 0, 25, DCGM_INT32_BLANK, DCGM_INT32_NOT_FOUND, DCGM_INT32_NOT_SUPPORTED, DCGM_INT32_NOT_PERMISSIONED };
    long long int64Values[6]
        = { 0, 25, DCGM_INT64_BLANK, DCGM_INT64_NOT_FOUND, DCGM_INT64_NOT_SUPPORTED, DCGM_INT64_NOT_PERMISSIONED };
    double fp64Values[6]
        = { 0.0, 25.0, DCGM_FP64_BLANK, DCGM_FP64_NOT_FOUND, DCGM_FP64_NOT_SUPPORTED, DCGM_FP64_NOT_PERMISSIONED };

    for (i = 0; i < Nconversions; i++)
    {
        valueInt32 = nvcmvalue_int64_to_int32(int64Values[i]);
        if (valueInt32 != int32Values[i])
        {
            fprintf(stderr, "i64 -> i32 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt64 = nvcmvalue_int32_to_int64(int32Values[i]);
        if (valueInt64 != int64Values[i])
        {
            fprintf(stderr, "i32 -> i64 failed for index %d\n", i);
            Nerrors++;
        }

        valueDouble = nvcmvalue_int64_to_double(int64Values[i]);
        if (valueDouble != fp64Values[i])
        {
            fprintf(stderr, "i64 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt64 = nvcmvalue_double_to_int64(fp64Values[i]);
        if (valueInt64 != int64Values[i])
        {
            fprintf(stderr, "fp64 -> int64 failed for index %d\n", i);
            Nerrors++;
        }

        valueDouble = nvcmvalue_int32_to_double(int32Values[i]);
        if (valueDouble != fp64Values[i])
        {
            fprintf(stderr, "i32 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt32 = nvcmvalue_double_to_int32(fp64Values[i]);
        if (valueInt32 != int32Values[i])
        {
            fprintf(stderr, "i32 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }
    }

    if (Nerrors > 0)
    {
        fprintf(stderr, "TestDcgmValue::TestConversions %d conversions failed.\n", Nerrors);
        return 1;
    }

    return 0;
}

/*****************************************************************************/
int TestDcgmValue::Run()
{
    unsigned int numOfFailures = 0;

    RUN_MODULE_SUBTEST(TestDcgmValue, TestConversions, numOfFailures);

    if (numOfFailures == 0)
    {
        printf("All TestDcgmValue tests PASSED\n");
    }

    return 0;
}

/*****************************************************************************/
