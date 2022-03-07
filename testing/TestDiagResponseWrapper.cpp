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
#include <iostream>
#include <string.h>

#include "DcgmDiagResponseWrapper.h"
#include "NvvsJsonStrings.h"
#include "TestDiagResponseWrapper.h"

TestDiagResponseWrapper::TestDiagResponseWrapper()
{}

TestDiagResponseWrapper::~TestDiagResponseWrapper()
{}

int TestDiagResponseWrapper::Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus)
{
    return 0;
}

int TestDiagResponseWrapper::Cleanup()
{
    return 0;
}

std::string TestDiagResponseWrapper::GetTag()
{
    return std::string("diagresponsewrapper");
}

int TestDiagResponseWrapper::Run()
{
    int nFailed = 0;

    int st = TestInitializeDiagResponse();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestInitializeDiagResponse FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestInitializeDiagResponse PASSED\n");

    st = TestSetPerGpuResponseState();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestSetPerGpuResponseState FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestSetPerGpuResponseState PASSED\n");

    st = TestAddPerGpuMessage();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestAddPerGpuMessage FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestAddPerGpuMessage PASSED\n");

    st = TestSetGpuIndex();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestSetGpuIndex FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestSetGpuIndex PASSED\n");

    st = TestGetBasicTestResultIndex();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestGetBasicTestResultIndex FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestGetBasicTestResultIndex PASSED\n");

    st = TestRecordSystemError();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestRecordSystemError FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestRecordSystemError PASSED\n");

    st = TestAddErrorDetail();
    if (st < 0)
    {
        nFailed++;
        fprintf(stderr, "TestDiagManager::TestAddErrorDetail FAILED with %d\n", st);
    }
    else
    {
        printf("TestDiagManager::TestAddErrorDetail PASSED\n");
    }

    return st;
}

int TestDiagResponseWrapper::TestInitializeDiagResponse()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6 = {};

    // These should be no ops, but make sure there's no crash
    r3.InitializeResponseStruct(6);

    // Set versions and make sure the state is valid
    r3.SetVersion6(&rv6);

    r3.InitializeResponseStruct(8);

    if (rv6.gpuCount != 8)
    {
        fprintf(stderr, "Gpu count was set to %d, but it should've been 8", rv6.gpuCount);
        return -1;
    }

    if (rv6.version != dcgmDiagResponse_version6)
    {
        fprintf(stderr, "Diag Response version wasn't set correctly");
        return -1;
    }

    for (unsigned int i = 0; i < 8; i++)
    {
        for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
        {
            if (rv6.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr, "Initial test status wasn't set correctly");
                return -1;
            }
        }
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetPerGpuResponseState()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6 = {};

    r3.SetVersion6(&rv6);

    r3.InitializeResponseStruct(8);

    r3.SetPerGpuResponseState(0, DCGM_DIAG_RESULT_PASS, 0);

    if (rv6.perGpuResponses[0].results[0].status != DCGM_DIAG_RESULT_PASS)
    {
        fprintf(stderr, "GPU 0 test 0 should be PASS, but is %d\n", rv6.perGpuResponses[0].results[0].status);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddPerGpuMessage()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6 = {};

    r3.SetVersion6(&rv6);

    r3.InitializeResponseStruct(8);

    static const std::string warn("That 3rd ideal can be tricky");
    static const std::string info("There are 5 ideals of radiance");

    r3.AddPerGpuMessage(0, warn, 0, true);

    if (warn != rv6.perGpuResponses[0].results[0].error.msg)
    {
        fprintf(stderr,
                "GPU 0 test 0 warning should be '%s', but found '%s'.\n",
                warn.c_str(),
                rv6.perGpuResponses[0].results[0].error.msg);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetGpuIndex()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6 = {};

    r3.SetVersion6(&rv6);

    r3.SetGpuIndex(2);

    if (rv6.perGpuResponses[2].gpuId != 2)
    {
        fprintf(stderr, "Slot 2 should have gpu id 2 but is %u\n", rv6.perGpuResponses[2].gpuId);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestGetBasicTestResultIndex()
{
    DcgmDiagResponseWrapper drw;

    if (drw.GetBasicTestResultIndex(blacklistName) != DCGM_SWTEST_BLACKLIST)
    {
        fprintf(stderr, "%s didn't match its index.\n", blacklistName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(nvmlLibName) != DCGM_SWTEST_NVML_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", nvmlLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(cudaMainLibName) != DCGM_SWTEST_CUDA_MAIN_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", cudaMainLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(cudaTkLibName) != DCGM_SWTEST_CUDA_RUNTIME_LIBRARY)
    {
        fprintf(stderr, "%s didn't match its index.\n", cudaTkLibName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(permissionsName) != DCGM_SWTEST_PERMISSIONS)
    {
        fprintf(stderr, "%s didn't match its index.\n", permissionsName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(persistenceName) != DCGM_SWTEST_PERSISTENCE_MODE)
    {
        fprintf(stderr, "%s didn't match its index.\n", persistenceName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(envName) != DCGM_SWTEST_ENVIRONMENT)
    {
        fprintf(stderr, "%s didn't match its index.\n", envName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(pageRetirementName) != DCGM_SWTEST_PAGE_RETIREMENT)
    {
        fprintf(stderr, "%s didn't match its index.\n", pageRetirementName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(graphicsName) != DCGM_SWTEST_GRAPHICS_PROCESSES)
    {
        fprintf(stderr, "%s didn't match its index.\n", graphicsName.c_str());
        return -1;
    }

    if (drw.GetBasicTestResultIndex(inforomName) != DCGM_SWTEST_INFOROM)
    {
        fprintf(stderr, "%s didn't match its index.\n", inforomName.c_str());
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestRecordSystemError()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6 = {};

    r3.SetVersion6(&rv6);

    r3.InitializeResponseStruct(8);

    static const std::string horrible("You've Moash'ed things horribly");

    r3.RecordSystemError(horrible);

    if (horrible != rv6.systemError.msg)
    {
        fprintf(stderr, "V4 should've had system error '%s', but found '%s'.\n", horrible.c_str(), rv6.systemError.msg);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddErrorDetail()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v6 rv6;

    r3.SetVersion6(&rv6);

    dcgmDiagErrorDetail_t ed;
    snprintf(ed.msg, sizeof(ed.msg), "Egads! Kaladin failed to say his fourth ideal.");
    ed.code = 20;

    r3.AddErrorDetail(0, 0, "Diagnostic", ed, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv6.perGpuResponses[0].results[0].error.msg, ed.msg))
    {
        fprintf(stderr,
                "Expected to find warning '%s', but found '%s'\n",
                ed.msg,
                rv6.perGpuResponses[0].results[0].error.msg);
        return -1;
    }

    if (rv6.perGpuResponses[0].results[0].error.code != ed.code)
    {
        fprintf(
            stderr, "Expected to find code %u, but found %u\n", ed.code, rv6.perGpuResponses[0].results[0].error.code);
        return -1;
    }

    r3.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT, "Inforom", ed, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv6.levelOneResults[DCGM_SWTEST_INFOROM].error.msg, ed.msg))
    {
        fprintf(stderr,
                "Expected to find error message '%s', but found '%s'\n",
                ed.msg,
                rv6.levelOneResults[DCGM_SWTEST_INFOROM].error.msg);
        return -1;
    }

    if (rv6.levelOneResults[DCGM_SWTEST_INFOROM].error.code != ed.code)
    {
        fprintf(stderr,
                "Expected to find error code %u, but found %u\n",
                ed.code,
                rv6.levelOneResults[DCGM_SWTEST_INFOROM].error.code);
        return -1;
    }

    return DCGM_ST_OK;
}
