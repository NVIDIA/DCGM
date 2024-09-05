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
#include <fstream>
#include <iostream>
#include <stddef.h>
#include <string>

#include "DcgmDiagCommon.h"
#include "DcgmDiagManager.h"
#include "DcgmDiagResponseWrapper.h"
#include "DcgmError.h"
#include "TestDiagManager.h"
#include "TestDiagManagerStrings.h"
#include <DcgmCoreCommunication.h>

dcgmCoreCallbacks_t g_coreCallbacks;

TestDiagManager::TestDiagManager()
{}
TestDiagManager::~TestDiagManager()
{}

int TestDiagManager::Init(const TestDcgmModuleInitParams &initParams)
{
    g_coreCallbacks.postfunc = PostRequestToCore;
    g_coreCallbacks.version  = dcgmCoreCallbacks_version;
    return 0;
}

int TestDiagManager::Run()
{
    int st;
    int Nfailed = 0;

    st = TestPositiveDummyExecutable();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPositiveDummyExecutable FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestDiagManager::TestPositiveDummyExecutable PASSED\n");

    st = TestNegativeDummyExecutable();
    if (st)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestNegativeDummyExecutable FAILED with %d\n", st);
        if (st < 0)
            return -1;
    }
    else
        printf("TestDiagManager::TestNegativeDummyExecutable PASSED\n");

    st = TestCreateNvvsCommand();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestCreateNvvsCommand FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestCreateNvvsCommand PASSED\n");

    st = TestPopulateRunDiag();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPopulateRunDiag FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestPopulateRunDiag PASSED\n");

    st = TestInvalidVersion();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestInvalidVersion FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestInvalidVersion PASSED\n");

    st = TestFillResponseStructure();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestFillResponseStructure FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestFillRespsonseStructure PASSED\n");

    st = TestPerformExternalCommand();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPerformExternalCommand FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestPerformExternalCommand PASSED\n");

    st = TestErrorsFromLevelOne();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestErrorsFromLevelOne FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestErrorsFromLevelOne PASSED\n");

    st = TestParseExpectedNumEntitiesForGpus();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestParseExpectedNumEntitiesForGpus FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestParseExpectedNumEntitiesForGpus PASSED\n");

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return -1;
    }
    return 0;
}


int TestDiagManager::Cleanup()
{
    return 0;
}

std::string TestDiagManager::GetTag()
{
    return std::string("diagmanager");
}

void TestDiagManager::CreateDummyScript()
{
    std::ofstream dummyScript;
    dummyScript.open("dummy_script");
    dummyScript << "#!/bin/bash" << std::endl;
    dummyScript << "echo \"Dummy script successfully executed with args $1 $2 $3\"" << std::endl;
    dummyScript.close();
    system("chmod 755 dummy_script");
}

void TestDiagManager::RemoveDummyScript()
{
    system("rm -f dummy_script");
}

int TestDiagManager::TestPositiveDummyExecutable()
{
    std::string stdoutStr;
    std::string stderrStr;
    dcgmReturn_t result;

    /*  this is, in practice, really bad... ok if the only public calls made are
     *  the Perform*Execute calls directly
     */
    DcgmDiagManager *am = new DcgmDiagManager(g_coreCallbacks);

    CreateDummyScript();
    result = am->PerformDummyTestExecute(&stdoutStr, &stderrStr);

    delete (am);
    am = 0;

    if (result != DCGM_ST_OK)
        return -1;

    DCGM_LOG_DEBUG << stdoutStr;
    return 0;
}

int TestDiagManager::TestNegativeDummyExecutable()
{
    std::string stdoutStr;
    std::string stderrStr;
    dcgmReturn_t result;

    DcgmDiagManager *am = new DcgmDiagManager(g_coreCallbacks);

    RemoveDummyScript();
    result = am->PerformDummyTestExecute(&stdoutStr, &stderrStr);

    delete (am);
    am = 0;

    if (result == DCGM_ST_OK)
        return -1;

    return 0;
}


// Helper method for TestCreateNvvsCommand
std::string ConvertVectorToCommandString(std::vector<std::string> &v)
{
    std::string out;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        // The CreateNvvsCommand test expects args with spaces to be quoted. We insert the quotes here because
        // DcgmDiagManager no longer needs to add quotes - the args are directly passed to nvvs via argv
        if (v[i].find(' ') == std::string::npos)
        {
            out += v[i];
        }
        else
        {
            out += "\"" + v[i] + "\"";
        }
        if (i < v.size() - 1)
        {
            out += " ";
        }
    }
    return out;
}

int TestDiagManager::TestCreateNvvsCommand()
{
    dcgmReturn_t result;
    std::string command;
    std::vector<std::string> cmdArgs;
    dcgmRunDiag_v8 drd = {};
    DcgmDiagManager am(g_coreCallbacks);

    std::string nvvsBinPath;
    std::vector<std::string> expected;

    const char *nvvsPathEnv = std::getenv("NVVS_BIN_PATH");
    if (nvvsPathEnv)
        nvvsBinPath = std::string(nvvsPathEnv) + "/nvvs";
    else
        nvvsBinPath = "/usr/share/nvidia-validation-suite/nvvs";

    expected.push_back(nvvsBinPath + " -j -z --specifiedtest long --configless -d NONE");
    expected.push_back(nvvsBinPath
                       + " -j -z --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --configless -d WARN");
    expected.push_back(
        nvvsBinPath
        + " -j -z --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --parameters \"memory bandwidth.minimum_bandwidth=5000;sm perf.target_stress=8500;targeted stress.test_duration=600\" --configless -d DEBUG");

    // When no test names are specified, none isn't valid
    drd.validate = DCGM_POLICY_VALID_NONE;
    result       = am.CreateNvvsCommand(cmdArgs, &drd);
    if (result == 0)
    {
        // Should've failed
        return -1;
    }

    // Check a valid scenario
    cmdArgs.clear();
    drd.validate = DCGM_POLICY_VALID_SV_LONG;
    result       = am.CreateNvvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK)
    {
        command = ConvertVectorToCommandString(cmdArgs);
        if (command != expected[0])
        {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[0].c_str(), command.c_str());
            return -1;
        }
    }

    // Specify the test names -- validate should get ignored
    cmdArgs.clear();
    snprintf(drd.testNames[0], sizeof(drd.testNames[0]), "memory bandwidth");
    snprintf(drd.testNames[1], sizeof(drd.testNames[1]), "sm stress");
    snprintf(drd.testNames[2], sizeof(drd.testNames[2]), "targeted stress");
    drd.debugLevel = DcgmLoggingSeverityWarning;
    result         = am.CreateNvvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK)
    {
        command = ConvertVectorToCommandString(cmdArgs);
        if (command != expected[1])
        {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[1].c_str(), command.c_str());
            return -1;
        }
    }

    // Add test parameters
    cmdArgs.clear();
    snprintf(drd.testParms[0], sizeof(drd.testParms[0]), "memory bandwidth.minimum_bandwidth=5000");
    snprintf(drd.testParms[1], sizeof(drd.testParms[1]), "sm perf.target_stress=8500");
    snprintf(drd.testParms[2], sizeof(drd.testParms[2]), "targeted stress.test_duration=600");
    // Invalid severity to test default value
    drd.debugLevel = 20; // TODO (nik, aalsuldani): this is UB. Let's find a better way
    result         = am.CreateNvvsCommand(cmdArgs, &drd);
    if (result == DCGM_ST_OK)
    {
        command = ConvertVectorToCommandString(cmdArgs);
        if (command != expected[2])
        {
            fprintf(stderr, "Expected '%s' but got '%s'\n", expected[2].c_str(), command.c_str());
            return -1;
        }
    }

    cmdArgs.clear();

    return result;
}

int TestDiagManager::TestParseExpectedNumEntitiesForGpus()
{
    dcgmReturn_t result    = DCGM_ST_OK;
    unsigned int numErrors = 0;
    std::string expectedNumEntities;

    auto verifyCountAndError
        = [&numErrors](std::string const &expectedNumEntities, unsigned int const &expectedGpuCount, bool error) {
              unsigned int gpuCount;
              auto err = ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
              if (gpuCount != expectedGpuCount)
              {
                  fprintf(stderr, "Expected gpuCount %u, got %u.\n", expectedGpuCount, gpuCount);
                  numErrors++;
              }
              else if (error && err.empty())
              {
                  fprintf(stderr,
                          "Expected an error, but got no error for expectedNumEntities string '%s'\n",
                          expectedNumEntities.c_str());
                  numErrors++;
              }
              else if (!error && !err.empty())
              {
                  fprintf(stderr,
                          "Expected no error, but got an error for expectedNumEntities string '%s': '%s'\n",
                          expectedNumEntities.c_str(),
                          err.c_str());
                  numErrors++;
              }
          };

    expectedNumEntities = "";
    verifyCountAndError(expectedNumEntities, 0, false);

    expectedNumEntities = "gpu:0";
    verifyCountAndError(expectedNumEntities, 0, false);

    expectedNumEntities = "Gpu:2";
    verifyCountAndError(expectedNumEntities, 2, false);

    expectedNumEntities = "GPU:4";
    verifyCountAndError(expectedNumEntities, 4, false);

    expectedNumEntities = "g:2";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "gpu:";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "cpu:0";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "gpu:2,cpu:3";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "gibberish2";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "gpu0";
    verifyCountAndError(expectedNumEntities, 0, true);

    expectedNumEntities = "gpu";
    verifyCountAndError(expectedNumEntities, 0, true);

    if (numErrors > 0)
    {
        return -1;
    }

    return result;
}

int TestDiagManager::TestPopulateRunDiag()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmRunDiag_v8 drd  = {};

    drd.version = dcgmRunDiag_version8;

    std::string error;

    // Basic test, nothing should get populated
    dcgm_diag_common_populate_run_diag(
        drd, "1", "", "", "", "", false, false, "", "", 0, "", 1, true, 3, 60, "", error);
    if (strlen(drd.testNames[0]) != 0)
    {
        fprintf(stderr, "Expected testNames to be empty but found '%s'\n", drd.testNames[0]);
        return -1;
    }
    else if (strlen(drd.testParms[0]) != 0)
    {
        fprintf(stderr, "Expected testParameters to be empty but found '%s'\n", drd.testParms[0]);
        return -1;
    }
    else if (drd.flags & DCGM_RUN_FLAGS_STATSONFAIL)
    {
        fprintf(stderr, "Expected statsOnlyOnFail to be unset, but it is set.\n");
        return -1;
    }
    else if (drd.flags & DCGM_RUN_FLAGS_VERBOSE)
    {
        fprintf(stderr, "Expected verbose to be unset, but it is set.\n");
        return -1;
    }
    else if (strlen(drd.debugLogFile) > 0)
    {
        fprintf(stderr, "The debug log file shouldn't be set, but found '%s'.\n", drd.debugLogFile);
        return -1;
    }
    else if (strlen(drd.statsPath) > 0)
    {
        fprintf(stderr, "The stats path shouldn't be set, but found '%s'.\n", drd.statsPath);
        return -1;
    }
    else if (drd.debugLevel != 0)
    {
        fprintf(stderr, "Debug level should be 0 but was %u.\n", drd.debugLevel);
        return -1;
    }

    if (drd.pluginPath[0] != 0)
    {
        fprintf(stderr, "Plugin path should be empty, but found '%s'.\n", drd.pluginPath);
        return -1;
    }

    if (strcmp(drd.configFileContents, ""))
    {
        fprintf(stderr, "Config file should be empty (no config file), but found %s.\n", drd.configFileContents);
        return -1;
    }

    if (strcmp(drd.throttleMask, ""))
    {
        fprintf(stderr, "The throttle mask should be empty, but found '%s'.\n", drd.throttleMask);
        return -1;
    }

    if (drd.groupId != (dcgmGpuGrp_t)1)
    {
        fprintf(stderr, "Expected groupid to be 1, but found %llu.\n", (unsigned long long)drd.groupId);
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY))
    {
        fprintf(stderr, "Expected fail early flag to be set, but it is unset.\n");
        return -1;
    }

    if (drd.failCheckInterval != 3)
    {
        fprintf(stderr, "Expected failCheckInterval to be 3, but found %u.\n", drd.failCheckInterval);
        return -1;
    }

    const char *tn1[] = { "pcie", "targeted power" };
    const char *tp1[] = { "pcie.test_pinned=false", "pcie.min_bandwidth=25000", "targeted power.temperature_max=82" };
    std::string debugFileName("kaladin");
    std::string statsPath("/home/aimian/");
    std::string throttleMask("HW_SLOWDOWN");
    drd         = dcgmRunDiag_v8(); // Need to reset the struct
    drd.version = dcgmRunDiag_version8;
    dcgm_diag_common_populate_run_diag(
        drd,
        "pcie,targeted power",
        "pcie.test_pinned=false;pcie.min_bandwidth=25000;targeted power.temperature_max=82",
        "",
        "",
        "0,1,2",
        true,
        true,
        debugFileName,
        statsPath,
        3,
        throttleMask,
        DCGM_GROUP_ALL_GPUS,
        false,
        3,
        60,
        "gpu:8",
        error);
    for (int i = 0; i < 2; i++)
    {
        if (strcmp(drd.testNames[i], tn1[i]))
        {
            fprintf(stderr, "Test name %d. Expected '%s', found '%s'\n", i, tn1[i], drd.testNames[i]);
            return -1;
        }
    }
    for (int i = 0; i < 3; i++)
    {
        if (strcmp(drd.testParms[i], tp1[i]))
        {
            fprintf(stderr, "Test parameter %d. Expected '%s', found '%s'\n", i, tp1[i], drd.testParms[i]);
            return -1;
        }
    }

    if (strcmp(drd.gpuList, "0,1,2"))
    {
        fprintf(stderr, "Gpu list should be '0,1,2' but found '%s'\n", drd.gpuList);
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_STATSONFAIL))
    {
        fprintf(stderr, "Expected stats only on failure to be set, but it wasn't.\n");
        return -1;
    }

    if (!(drd.flags & DCGM_RUN_FLAGS_VERBOSE))
    {
        fprintf(stderr, "Expected verbose to be set, but it wasn't.\n");
        return -1;
    }

    if (debugFileName != drd.debugLogFile)
    {
        fprintf(stderr,
                "Expected the debug file name to be '%s', but found '%s'\n",
                debugFileName.c_str(),
                drd.debugLogFile);
        return -1;
    }

    if (statsPath != drd.statsPath)
    {
        fprintf(stderr, "Expected the stats path to be '%s', but found '%s'.\n", statsPath.c_str(), drd.statsPath);
        return -1;
    }

    if (drd.debugLevel != 3)
    {
        fprintf(stderr, "Debug level should be 3, but found %u.\n", drd.debugLevel);
        return -1;
    }

    // Config file is set, but should be blank since empty contents were given
    if (strcmp(drd.configFileContents, ""))
    {
        fprintf(stderr, "Config file should be empty, but found %s.\n", drd.configFileContents);
        return -1;
    }

    if (throttleMask != drd.throttleMask)
    {
        fprintf(stderr, "Expected throttle mask to be '%s', but found '%s'.\n", throttleMask.c_str(), drd.throttleMask);
        return -1;
    }

    if (drd.pluginPath[0] != '\0')
    {
        fprintf(stderr, "Plugin path should be empty, but found '%s'.\n", drd.pluginPath);
        return -1;
    }

    if (drd.groupId != DCGM_GROUP_ALL_GPUS)
    {
        fprintf(stderr,
                "Expected groupid to be %llu, but found %llu.\n",
                (unsigned long long)DCGM_GROUP_ALL_GPUS,
                (unsigned long long)drd.groupId);
        return -1;
    }

    if (std::string_view(drd.expectedNumEntities) != "gpu:8")
    {
        fprintf(stderr, "Expected expectedNumEntities to be 'gpu:8', but found %s.\n", drd.expectedNumEntities);
        return -1;
    }

    if (drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY)
    {
        fprintf(stderr, "Expected fail early flag to be unset, but it is set.\n");
        return -1;
    }

    if (drd.failCheckInterval != 0)
    {
        fprintf(stderr, "Expected failCheckInterval to be 0, but found %u.\n", drd.failCheckInterval);
        return -1;
    }

    // Config file is set to correct value
    dcgm_diag_common_populate_run_diag(drd,
                                       "pcie,targeted power",
                                       "pcie.test_pinned=false",
                                       "configfile",
                                       "",
                                       "0",
                                       true,
                                       true,
                                       debugFileName,
                                       statsPath,
                                       3,
                                       "",
                                       0,
                                       false,
                                       0,
                                       60,
                                       "",
                                       error);
    if (strcmp(drd.configFileContents, "configfile"))
    {
        fprintf(stderr, "Config file should be 'configfile', but found %s.\n", drd.configFileContents);
        return -1;
    }

    drd              = dcgmRunDiag_v8(); // Need to reset the struct
    dcgmReturn_t ret = dcgm_diag_common_populate_run_diag(
        drd,
        "pcie,targeted power",
        "pcie.test_pinned=false;pcie.min_bandwidth=25000;targeted power.temperature_max=82",
        "",
        "",
        "0,1,2",
        true,
        true,
        debugFileName,
        statsPath,
        3,
        throttleMask,
        0,
        false,
        3,
        60,
        "gpu:8",
        error);
    if (ret != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", ret, error.c_str());
        return -1;
    }

    return result;
}

int TestDiagManager::TestErrorsFromLevelOne()
{
    DcgmCacheManager dcm;
    dcm.Init(1, 3600.0, true);

    std::string errorReported(
        "Persistence mode for GPU 1 is currently disabled. NVVS requires persistence mode to be enabled. Enable persistence mode by running (as root): nvidia-smi -i 1 -pm 1");

    std::string rawJsonOutput = fmt::format(R"(
    {{
        "DCGM GPU Diagnostic" :
            {{
                "test_categories" : [
                    {{
                        "category" : "Deployment",
                        "tests" : [
                            {{
                                "name" : "Denylist",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "PASS" }} ]
                            }},
                            {{
                                "name" : "NVML Library",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "PASS" }} ]
                            }},
                            {{
                                "name" : "CUDA Main Library",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "PASS" }} ]
                            }},
                            {{
                                "name" : "Permissions and OS-related Blocks",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "PASS" }} ]
                            }},
                            {{
                                "name" : "Persistence Mode",
                                "results" : [
                                    {{
                                        "gpu_ids" : "0,1,2,3",
                                        "status" : "FAIL",
                                        "warnings" : ["{}"]
                                    }}
                                ]
                            }},
                            {{
                                "name" : "Environmental Variables",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "SKIP" }} ]
                            }},
                            {{
                                "name" : "Page Retirement/Row Remap",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "SKIP" }} ]
                            }},
                            {{
                                "name" : "Graphics Processes",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "SKIP" }} ]
                            }},
                            {{
                                "name" : "Inforom",
                                "results" : [ {{ "gpu_ids" : "0,1,2,3", "status" : "SKIP" }} ]
                            }}
                        ]
                    }}
                ],
                "version" : "1.7"
            }}
    }})",
                                            errorReported);

    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();

    DcgmCoreCommunication dcc;
    DcgmGroupManager dgm { &dcm };
    dcc.Init(&dcm, &dgm);
    g_coreCallbacks.poster = &dcc;
    DcgmDiagManager am(g_coreCallbacks);
    dcgmDiagResponse_v10 response;
    response.version = dcgmDiagResponse_version10;
    DcgmDiagResponseWrapper drw;
    drw.SetVersion10(&response);

    auto nvvsResults
        = DcgmNs::JsonSerialize::Deserialize<DcgmNs::Nvvs::Json::DiagnosticResults>(std::string_view { rawJsonOutput });
    dcgmReturn_t ret = am.FillResponseStructure(nvvsResults, drw, 0, DCGM_ST_OK);

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Failed to parse JSON, fix the JSON input!");
        return -1;
    }

    // Make sure that we are getting the error message recorded correctly
    for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (i == DCGM_SWTEST_CUDA_RUNTIME_LIBRARY)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(
                    stderr, "Test %d should not have run, but result was %d\n", i, response.levelOneResults[i].status);
                return -1;
            }
        }
        else if (i < DCGM_SWTEST_PERSISTENCE_MODE)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(
                    stderr, "Test %d should have passed, but result was %d\n", i, response.levelOneResults[i].status);
                return -1;
            }
        }
        else if (i == DCGM_SWTEST_PERSISTENCE_MODE)
        {
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_FAIL)
            {
                fprintf(
                    stderr, "Test %d should have failed, but result was %d\n", i, response.levelOneResults[i].status);
                return -1;
            }

            if (errorReported != response.levelOneResults[i].error[0].msg)
            {
                fprintf(stderr,
                        "Expected error '%s' but found '%s'\n",
                        errorReported.c_str(),
                        response.levelOneResults[i].error[0].msg);
                return -1;
            }
        }
        else
        {
            // After the failed tests, all these others should be skipped
            if (response.levelOneResults[i].status != DCGM_DIAG_RESULT_SKIP)
            {
                fprintf(stderr,
                        "Test %d should have been skipped, but result was %d\n",
                        i,
                        response.levelOneResults[i].status);
                return -1;
            }
        }
    }

    return DCGM_ST_OK;
}

int TestDiagManager::TestInvalidVersion()
{
    DcgmCacheManager dcm;
    dcm.Init(1, 3600.0, true);
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();

    DcgmGroupManager dgm { &dcm };

    DcgmCoreCommunication dcc;
    dcc.Init(&dcm, &dgm);

    g_coreCallbacks.poster = &dcc;

    dcgmCoreGetGpuCount_t ggc = {};

    ggc.header.version    = dcgmCoreGetGpuCount_version;
    ggc.header.moduleId   = DcgmModuleIdCore;
    ggc.header.subCommand = DcgmCoreReqIdCMGetGpuCount;
    ggc.header.length     = sizeof(ggc);

    // coverity[overrun-buffer-val]
    dcgmReturn_t ret = g_coreCallbacks.postfunc(&ggc.header, g_coreCallbacks.poster);

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected success, got %d\n", ret);
        return -1;
    }

    ggc.header.version = dcgmCoreGetGpuList_version;

    // coverity[overrun-buffer-val]
    ret = g_coreCallbacks.postfunc(&ggc.header, g_coreCallbacks.poster);

    if (ret != DCGM_ST_VER_MISMATCH)
    {
        fprintf(stderr, "Expected version mismatch, got %d\n", ret);
        return -1;
    }
    return DCGM_ST_OK;
}

int TestDiagManager::TestFillResponseStructure()
{
    dcgmReturn_t result = DCGM_ST_OK;
    DcgmCacheManager dcm;
    dcm.Init(1, 3600.0, true);
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();
    dcm.AddFakeGpu();

    DcgmGroupManager dgm { &dcm };

    DcgmCoreCommunication dcc;
    dcc.Init(&dcm, &dgm);

    g_coreCallbacks.poster = &dcc;
    DcgmDiagManager am(g_coreCallbacks);

    const char *testNames[]
        = { "Denylist",         "NVML Library", "CUDA Main Library",         "CUDA SDK Library",   "Permissions",
            "Persistence Mode", "Environment",  "Page Retirement/Row Remap", "Graphics Processes", "Inforom" };

    /* Version 1.7 (v5), 2.0 (v6) / Per GPU JSON */
    dcgmDiagResponse_v10 perGpuResponse;
    perGpuResponse.version = dcgmDiagResponse_version10;
    DcgmDiagResponseWrapper perGpuDrw;
    perGpuDrw.SetVersion10(&perGpuResponse);

    fprintf(stdout, "Checking Per GPU (v1.7/2.0) JSON parsing");

    auto nvvsResult
        = DcgmNs::JsonSerialize::Deserialize<DcgmNs::Nvvs::Json::DiagnosticResults>(std::string_view { PER_GPU_JSON });
    auto ret = am.FillResponseStructure(nvvsResult, perGpuDrw, 0, DCGM_ST_OK);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected successful parsing, but got %d instead!\n", ret);
        return -1;
    }

    if (perGpuResponse.gpuCount != 4)
    {
        fprintf(stderr, "Expected 4 gpus, got %d\n", perGpuResponse.gpuCount);
        return -1;
    }

    for (int i = 0; i < 4; i++)
    {
        if (i != 1)
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].status != DCGM_DIAG_RESULT_FAIL)
            {
                fprintf(stderr,
                        "Expected the Sm Stress test to have failed with gpu %d, but got %d\n",
                        i,
                        perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].status);
                return -1;
            }
        }
        else
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr,
                        "Expected the Sm Stress test to have passed for gpu %d, but got %d\n",
                        i,
                        perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].status);
                return -1;
            }
        }

        // Make sure all of the GPUs except GPU 1 have warnings
        if ((perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].error[0].msg[0] == '\0') && (i != 1))
        {
            fprintf(stderr, "Gpu %d should have a warning for Sm Stress, but doesn't\n", i);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].info[0] == '\0')
        {
            fprintf(stderr, "Gpu %d has information reported, but we captured none\n", i);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].error[0].code != DCGM_FR_STRESS_LEVEL
            && i != 1)
        {
            fprintf(stderr,
                    "Expected the SM Stress test to have error code %d, but found %d\n",
                    DCGM_FR_STRESS_LEVEL,
                    perGpuResponse.perGpuResponses[i].results[DCGM_SM_STRESS_INDEX].error[0].code);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr,
                    "Expected the Diagnostic test to have been skipped with gpu %d, but got %d\n",
                    i,
                    perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].status);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].error[0].code != 1)
        {
            // Code 1 is used above because the error reported is from old, saved json and doesn't correspond
            // to an error it's actually possible to get today. I manually added it.
            fprintf(stderr,
                    "Expected the diagnostic test to have error code 1, but found %d\n",
                    perGpuResponse.perGpuResponses[i].results[DCGM_DIAGNOSTIC_INDEX].error[0].code);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_STRESS_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr,
                    "Expected the Targeted Stress test to have been skipped with gpu %d, but got %d\n",
                    i,
                    perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_STRESS_INDEX].status);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr,
                    "Expected the Memory Bandwidth test to have been skipped with gpu %d, but got %d\n",
                    i,
                    perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].status);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].error[0].code
            != DCGM_FR_TEST_DISABLED)
        {
            fprintf(stderr,
                    "Expected the Memory Bandwidth test to have error code %d, but found %d.\n",
                    DCGM_FR_TEST_DISABLED,
                    perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_BANDWIDTH_INDEX].error[0].code);
            return -1;
        }

        if (perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status != DCGM_DIAG_RESULT_SKIP)
        {
            fprintf(stderr,
                    "Expected the Targeted Power test to have been skipped with gpu %d, but got %d\n",
                    i,
                    perGpuResponse.perGpuResponses[i].results[DCGM_TARGETED_POWER_INDEX].status);
            return -1;
        }

        if (i == 0)
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_SKIP)
            {
                fprintf(stderr,
                        "Expected GPU %d's Memory test to have been skipped, but found result %d\n",
                        i,
                        perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
        else
        {
            if (perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status != DCGM_DIAG_RESULT_PASS)
            {
                fprintf(stderr,
                        "Expected the Memory test to have been passed with gpu %d, but got %d\n",
                        i,
                        perGpuResponse.perGpuResponses[i].results[DCGM_MEMORY_INDEX].status);
                return -1;
            }
        }
    }

    if (perGpuResponse.perGpuResponses[0].results[DCGM_MEMORY_INDEX].error[0].code != DCGM_FR_ECC_DISABLED)
    {
        fprintf(stderr,
                "Expected to find error code %d but found %d\n",
                DCGM_FR_ECC_DISABLED,
                perGpuResponse.perGpuResponses[0].results[DCGM_MEMORY_INDEX].error[0].code);
        return -1;
    }

    for (int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (i == DCGM_SWTEST_CUDA_RUNTIME_LIBRARY)
        {
            if (perGpuResponse.levelOneResults[i].status != DCGM_DIAG_RESULT_NOT_RUN)
            {
                fprintf(stderr,
                        "Test %d should not have run, but result was %d\n",
                        i,
                        perGpuResponse.levelOneResults[i].status);
                return -1;
            }
        }
        else if (perGpuResponse.levelOneResults[i].status != DCGM_DIAG_RESULT_PASS)
        {
            fprintf(stderr,
                    "%s test should've passed but got %d\n",
                    testNames[i],
                    perGpuResponse.levelOneResults[i].status);

            return -1;
        }
    }

    return result;
}

void TestDiagManager::CreateDummyFailScript()
{
    std::ofstream dummyScript;
    dummyScript.open("dummy_script");
    dummyScript << "#!/bin/bash" << std::endl;
    dummyScript << "kaladin_dalinar_roshar_adolin_renarin_shallan_jasnah" << std::endl;
    dummyScript.close();
    system("chmod 755 dummy_script");
}

int TestDiagManager::TestPerformExternalCommand()
{
    std::string stdoutStr;
    std::string stderrStr;
    std::vector<std::string> dummyCmds;
    dummyCmds.push_back("dummy"); // added to DcgmDiagManager so we can generate stderr
    dcgmReturn_t result = DCGM_ST_OK;

    DcgmDiagManager am(g_coreCallbacks);

    // Make the script that will fail
    CreateDummyFailScript();
    am.PerformExternalCommand(dummyCmds, &stdoutStr, &stderrStr);
    if (stdoutStr.empty() && stderrStr.empty())
    {
        fprintf(stderr, "TestPerformExternalCommand should've captured stderr, but failed to do so.\n");
        return -1;
    }

    // Clean up
    RemoveDummyScript();

    return result;
}