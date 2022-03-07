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
#include "TestVersioning.h"
#include <stddef.h>


TestVersioning::TestVersioning()
{}
TestVersioning::~TestVersioning()
{}

int TestVersioning::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

int TestVersioning::Run()
{
    int st;
    int Nfailed = 0;

    st = TestBasicAPIVersions();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestVersioning::TestBasicAPIVersions FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }

    printf("TestVersioning::TestBasicAPIVersions PASSED\n");

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    return 0;
}

int TestVersioning::Cleanup()
{
    return 0;
}

std::string TestVersioning::GetTag()
{
    return std::string("versioning");
}

int recognizeAPIVersion(dcgmVersionTest_t *dummyStruct)
{
    dummyStruct->a = 5;
    if (dummyStruct->version >= dcgmVersionTest_version2)
        dummyStruct->b = 10;
    if (dummyStruct->version > dcgmVersionTest_version2)
        return -1;

    return 0;
}

int TestVersioning::TestBasicAPIVersions()
{
    dcgmVersionTest_v1 api_v1;
    dcgmVersionTest_v2 api_v2;

    api_v1.version = dcgmVersionTest_version1;
    api_v2.version = dcgmVersionTest_version2;

    /* Test the positive */
    if (recognizeAPIVersion((dcgmVersionTest_t *)&api_v1) != 0)
        return -1;
    if (recognizeAPIVersion((dcgmVersionTest_t *)&api_v2) != 0)
        return -1;

    /* Test the negative */
    api_v2.version = dcgmVersionTest_version3;
    if (recognizeAPIVersion((dcgmVersionTest_t *)&api_v2) == 0)
        return -1;
    return 0;
}
