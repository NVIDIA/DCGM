/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <fmt/core.h>
#include <iostream>
#include <string.h>

#include "DcgmDiagResponseWrapper.h"
#include "DcgmStringHelpers.h"
#include "NvvsJsonStrings.h"
#include "TestDiagResponseWrapper.h"
#include "dcgm_structs.h"

TestDiagResponseWrapper::TestDiagResponseWrapper() = default;

TestDiagResponseWrapper::~TestDiagResponseWrapper() = default;

int TestDiagResponseWrapper::Init(const TestDcgmModuleInitParams &initParams)
{
    return 0;
}

int TestDiagResponseWrapper::Cleanup()
{
    return 0;
}

std::string TestDiagResponseWrapper::GetTag()
{
    using namespace std::string_literals;
    return "diagresponsewrapper"s;
}

int TestDiagResponseWrapper::Run()
{
    int nFailed = 0;

    int st = TestInitializeDiagResponse();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestInitializeDiagResponse FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestInitializeDiagResponse PASSED\n");
    }

    st = TestSetPerGpuResponseState();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestSetPerGpuResponseState FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestSetPerGpuResponseState PASSED\n");
    }

    st = TestAddPerGpuMessage();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestAddPerGpuMessage FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestAddPerGpuMessage PASSED\n");
    }

    st = TestSetGpuIndex();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestSetGpuIndex FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestSetGpuIndex PASSED\n");
    }

    st = TestGetBasicTestResultIndex();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestGetBasicTestResultIndex FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestGetBasicTestResultIndex PASSED\n");
    }

    st = TestRecordSystemError();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestRecordSystemError FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestRecordSystemError PASSED\n");
    }

    st = TestAddErrorDetail();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestAddErrorDetail FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestAddErrorDetail PASSED\n");
    }

    st = TestAddAuxData();
    if (st < 0)
    {
        nFailed++;
        fmt::print(stderr, "TestDiagResponseWrapper::TestAddAuxData FAILED with {}\n", st);
        fflush(stderr);
    }
    else
    {
        fmt::print("TestDiagResponseWrapper::TestAddAuxData PASSED\n");
    }

    return st;
}

int TestDiagResponseWrapper::TestInitializeDiagResponse()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10 = {};

    // These should be no ops, but make sure there's no crash
    r3.InitializeResponseStruct(6);

    // Set versions and make sure the state is valid
    r3.SetVersion10(&rv10);

    r3.InitializeResponseStruct(8);

    if (rv10.gpuCount != 8)
    {
        fmt::print(stderr, "Gpu count was set to {}, but it should've been 8", rv10.gpuCount);
        fflush(stderr);
        return -1;
    }

    if (rv10.version != dcgmDiagResponse_version10)
    {
        fmt::print(stderr, "Diag Response version wasn't set correctly");
        fflush(stderr);
        return -1;
    }

    for (unsigned int i = 0; i < 8; i++)
    {
        for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
        {
            if (rv10.perGpuResponses[i].results[j].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fmt::print(stderr, "Initial test status wasn't set correctly");
                fflush(stderr);
                return -1;
            }
        }
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetPerGpuResponseState()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10 = {};

    r3.SetVersion10(&rv10);

    r3.InitializeResponseStruct(8);

    r3.SetPerGpuResponseState(0, DCGM_DIAG_RESULT_PASS, 0);

    if (rv10.perGpuResponses[0].results[0].status != DCGM_DIAG_RESULT_PASS)
    {
        fmt::print(stderr, "GPU 0 test 0 should be PASS, but is {}\n", rv10.perGpuResponses[0].results[0].status);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddPerGpuMessage()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10 = {};

    r3.SetVersion10(&rv10);

    r3.InitializeResponseStruct(8);

    static const std::string warn("That 3rd ideal can be tricky");
    static const std::string info("There are 5 ideals of radiance");

    r3.AddPerGpuMessage(0, warn, 0, true);

    if (warn != rv10.perGpuResponses[0].results[0].error[0].msg)
    {
        fmt::print(stderr,
                   "GPU 0 test 0 warning should be '{}', but found '{}'.\n",
                   warn.c_str(),
                   rv10.perGpuResponses[0].results[0].error[0].msg);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestSetGpuIndex()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10 = {};

    r3.SetVersion10(&rv10);

    r3.SetGpuIndex(2);

    if (rv10.perGpuResponses[2].gpuId != 2)
    {
        fmt::print(stderr, "Slot 2 should have gpu id 2 but is {}\n", rv10.perGpuResponses[2].gpuId);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestGetBasicTestResultIndex()
{
    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(denylistName) != DCGM_SWTEST_DENYLIST)
    {
        fmt::print(stderr, "{} didn't match its index.\n", denylistName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(nvmlLibName) != DCGM_SWTEST_NVML_LIBRARY)
    {
        fmt::print(stderr, "{} didn't match its index.\n", nvmlLibName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(cudaMainLibName) != DCGM_SWTEST_CUDA_MAIN_LIBRARY)
    {
        fmt::print(stderr, "{} didn't match its index.\n", cudaMainLibName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(cudaTkLibName) != DCGM_SWTEST_CUDA_RUNTIME_LIBRARY)
    {
        fmt::print(stderr, "{} didn't match its index.\n", cudaTkLibName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(permissionsName) != DCGM_SWTEST_PERMISSIONS)
    {
        fmt::print(stderr, "{} didn't match its index.\n", permissionsName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(persistenceName) != DCGM_SWTEST_PERSISTENCE_MODE)
    {
        fmt::print(stderr, "{} didn't match its index.\n", persistenceName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(envName) != DCGM_SWTEST_ENVIRONMENT)
    {
        fmt::print(stderr, "{} didn't match its index.\n", envName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(pageRetirementName) != DCGM_SWTEST_PAGE_RETIREMENT)
    {
        fmt::print(stderr, "{} didn't match its index.\n", pageRetirementName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(graphicsName) != DCGM_SWTEST_GRAPHICS_PROCESSES)
    {
        fmt::print(stderr, "{} didn't match its index.\n", graphicsName);
        fflush(stderr);
        return -1;
    }

    if (DcgmDiagResponseWrapper::GetBasicTestResultIndex(inforomName) != DCGM_SWTEST_INFOROM)
    {
        fmt::print(stderr, "{} didn't match its index.\n", inforomName);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestRecordSystemError()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10 = {};

    r3.SetVersion10(&rv10);

    r3.InitializeResponseStruct(8);

    static const std::string horrible("You've Moash'ed things horribly");

    r3.RecordSystemError(horrible);

    if (horrible != rv10.systemError.msg)
    {
        fmt::print(
            stderr, "V4 should've had system error '{}', but found '{}'.\n", horrible.c_str(), rv10.systemError.msg);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddErrorDetail()
{
    DcgmDiagResponseWrapper r3;

    dcgmDiagResponse_v10 rv10;

    r3.SetVersion10(&rv10);

    dcgmDiagErrorDetail_v2 ed;
    SafeCopyTo(ed.msg, (const char *)"Egads! Kaladin failed to say his fourth ideal.");
    ed.code = 20;

    r3.AddErrorDetail(0, 0, "Diagnostic", ed, 0, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv10.perGpuResponses[0].results[0].error[0].msg, ed.msg))
    {
        fmt::print(stderr,
                   "Expected to find warning '{}', but found '{}'\n",
                   ed.msg,
                   rv10.perGpuResponses[0].results[0].error[0].msg);
        fflush(stderr);
        return -1;
    }

    if (rv10.perGpuResponses[0].results[0].error[0].code != ed.code)
    {
        fmt::print(stderr,
                   "Expected to find code {}, but found {}\n",
                   ed.code,
                   rv10.perGpuResponses[0].results[0].error[0].code);
        fflush(stderr);
        return -1;
    }

    r3.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT_V8, "Inforom", ed, 0, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[0].msg, ed.msg))
    {
        fmt::print(stderr,
                   "Expected to find error message '{}', but found '{}'\n",
                   ed.msg,
                   rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[0].msg);
        fflush(stderr);
        return -1;
    }

    if (rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[0].code != ed.code)
    {
        fmt::print(stderr,
                   "Expected to find error code {}, but found {}\n",
                   ed.code,
                   rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[0].code);
        fflush(stderr);
        return -1;
    }

    r3.AddErrorDetail(0, DCGM_PER_GPU_TEST_COUNT_V8, "Inforom", ed, 1, DCGM_DIAG_RESULT_FAIL);

    if (strcmp(rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[1].msg, ed.msg))
    {
        fmt::print(stderr,
                   "Expected to find error message '{}', but found '{}'\n",
                   ed.msg,
                   rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[1].msg);
        fflush(stderr);
        return -1;
    }

    if (rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[1].code != ed.code)
    {
        fmt::print(stderr,
                   "Expected to find error code {}, but found {}\n",
                   ed.code,
                   rv10.levelOneResults[DCGM_SWTEST_INFOROM].error[1].code);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}

int TestDiagResponseWrapper::TestAddAuxData()
{
    DcgmDiagResponseWrapper r3;
    dcgmDiagResponse_v10 rv10;
    std::string auxData = "auxdata_from_eud";

    r3.SetVersion10(&rv10);

    // it should not crash if out of range
    r3.AddAuxData(DCGM_PER_GPU_TEST_COUNT_V8 + 1, "unused");

    r3.AddAuxData(DCGM_EUD_TEST_INDEX, auxData);

    if (strcmp(rv10.auxDataPerTest[DCGM_EUD_TEST_INDEX].data, auxData.data()))
    {
        fmt::print(stderr,
                   "Expected to have aux data {}, but found {}\n",
                   auxData,
                   rv10.auxDataPerTest[DCGM_EUD_TEST_INDEX].data);
        fflush(stderr);
        return -1;
    }

    return DCGM_ST_OK;
}