/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "TestDiagManager.h"

#include <DcgmCoreCommunication.h>
#include <DcgmDiagCommon.h>
#include <DcgmDiagManager.h>
#include <DcgmDiagResponseWrapper.h>
#include <DcgmError.h>
#include <UniquePtrUtil.h>
#include <dcgm_structs.h>
#include <fstream>
#include <iostream>
#include <stddef.h>
#include <string>

dcgmCoreCallbacks_t g_coreCallbacks;

TestDiagManager::TestDiagManager()
{}
TestDiagManager::~TestDiagManager()
{}

int TestDiagManager::Init(const TestDcgmModuleInitParams & /* initParams */)
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

    st = TestPerformExternalCommand();
    if (st < 0)
    {
        Nfailed++;
        fprintf(stderr, "TestDiagManager::TestPerformExternalCommand FAILED with %d\n", st);
    }
    else
        printf("TestDiagManager::TestPerformExternalCommand PASSED\n");

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
    DcgmCacheManager dcm;
    dcgmReturn_t ret = dcm.Init(1, 3600.0, true);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected Init success, got %d\n", ret);
        return -1;
    }

    DcgmGroupManager dgm { &dcm };
    DcgmCoreCommunication dcc;
    dcc.Init(&dcm, &dgm);
    g_coreCallbacks.poster = &dcc;

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
    dcgmRunDiag_v10 drd = {};
    DcgmDiagManager am(g_coreCallbacks);

    std::string nvvsBinPath;
    std::vector<std::string> expected;

    const char *nvvsPathEnv = std::getenv("NVVS_BIN_PATH");
    if (nvvsPathEnv)
        nvvsBinPath = std::string(nvvsPathEnv) + "/nvvs";
    else
        nvvsBinPath = "/usr/libexec/datacenter-gpu-manager-4/nvvs";

    std::string diagResponseVersionArg = fmt::format("--response-version {}", dcgmDiagResponse_version12);
    expected.push_back(nvvsBinPath + " --channel-fd 3 " + diagResponseVersionArg
                       + " --specifiedtest long --configless -d NONE");
    expected.push_back(nvvsBinPath + " --channel-fd 3 " + diagResponseVersionArg
                       + " --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --configless -d WARN");
    expected.push_back(
        nvvsBinPath + " --channel-fd 3 " + diagResponseVersionArg
        + " --specifiedtest \"memory bandwidth,sm stress,targeted stress\" --parameters \"memory bandwidth.minimum_bandwidth=5000;sm perf.target_stress=8500;targeted stress.test_duration=600\" --configless -d DEBUG");

    // When no test names are specified, none isn't valid
    drd.validate = DCGM_POLICY_VALID_NONE;
    result       = am.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12);
    if (result == 0)
    {
        // Should've failed
        return -1;
    }

    // Check a valid scenario
    cmdArgs.clear();
    drd.validate = DCGM_POLICY_VALID_SV_LONG;
    result       = am.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12);
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
    result         = am.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12);
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
    result         = am.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12);
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

int TestDiagManager::TestPopulateRunDiag()
{
    dcgmReturn_t result = DCGM_ST_OK;
    dcgmRunDiag_v10 drd = {};

    drd.version = dcgmRunDiag_version10;

    std::string error;
    unsigned int const defaultFrequency = 5000000;

    // Basic test
    result = dcgm_diag_common_populate_run_diag(drd,
                                                "1",
                                                "",
                                                "",
                                                "",
                                                "",
                                                false,
                                                false,
                                                "",
                                                "",
                                                0,
                                                "",
                                                DCGM_GROUP_ALL_GPUS,
                                                true,
                                                3,
                                                60,
                                                "*,cpu:*",
                                                "gpu:1",
                                                defaultFrequency,
                                                "*:*",
                                                error);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected DCGM_ST_OK but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }
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

    if (strcmp(drd.clocksEventMask, ""))
    {
        fprintf(stderr, "The clocks event mask should be empty, but found '%s'.\n", drd.clocksEventMask);
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

    if (std::string_view(drd.entityIds) != "*,cpu:*")
    {
        fprintf(stderr, "Expected entityIds to be '*,cpu:*', but found %s.\n", drd.entityIds);
        return -1;
    }

    if (std::string_view(drd.expectedNumEntities) != "gpu:1")
    {
        fprintf(stderr, "Expected expectedNumEntities to be 'gpu:1', but found %s.\n", drd.expectedNumEntities);
        return -1;
    }

    if (drd.watchFrequency != defaultFrequency)
    {
        fprintf(stderr, "Expected watchFrequency to be %u, but found %u.\n", defaultFrequency, drd.watchFrequency);
        return -1;
    }

    if (std::string_view(drd.ignoreErrorCodes) != "*:*")
    {
        fprintf(stderr, "Expected ignoreErrorCodes to be '*:*', but found %s.\n", drd.entityIds);
        return -1;
    }

    const char *tn1[] = { "pcie", "targeted power" };
    const char *tp1[] = { "pcie.test_pinned=false", "pcie.min_bandwidth=25000", "targeted power.temperature_max=82" };
    std::string debugFileName("kaladin");
    std::string statsPath("/home/aimian/");
    std::string clocksEventMask("HW_SLOWDOWN");
    drd         = dcgmRunDiag_v10(); // Need to reset the struct
    drd.version = dcgmRunDiag_version10;
    result      = dcgm_diag_common_populate_run_diag(
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
        clocksEventMask,
        0,
        false,
        3,
        60,
        "*,cpu:*",
        "",
        defaultFrequency,
        "",
        error);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected DCGM_ST_OK but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }
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

    if (strcmp(drd.entityIds, "0,1,2"))
    {
        fprintf(stderr, "Gpu list should be '0,1,2' but found '%s'\n", drd.entityIds);
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

    if (clocksEventMask != drd.clocksEventMask)
    {
        fprintf(stderr,
                "Expected clocks event mask to be '%s', but found '%s'.\n",
                clocksEventMask.c_str(),
                drd.clocksEventMask);
        return -1;
    }

    if (drd.pluginPath[0] != '\0')
    {
        fprintf(stderr, "Plugin path should be empty, but found '%s'.\n", drd.pluginPath);
        return -1;
    }

    if (drd.groupId != (dcgmGpuGrp_t)0)
    {
        fprintf(stderr, "Expected groupid to be 0, but found %llu.\n", (unsigned long long)drd.groupId);
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
    result = dcgm_diag_common_populate_run_diag(drd,
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
                                                "",
                                                defaultFrequency,
                                                "",
                                                error);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected DCGM_ST_OK but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }
    if (strcmp(drd.configFileContents, "configfile"))
    {
        fprintf(stderr, "Config file should be 'configfile', but found %s.\n", drd.configFileContents);
        return -1;
    }

    // Empty entity ids
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(
        drd,
        "pcie,targeted power",
        "pcie.test_pinned=false;pcie.min_bandwidth=25000;targeted power.temperature_max=82",
        "",
        "",
        "",
        true,
        true,
        debugFileName,
        statsPath,
        3,
        clocksEventMask,
        0,
        false,
        3,
        60,
        "",
        "",
        defaultFrequency,
        "",
        error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    // Too large entity ids
    std::string entityIds(DCGM_ENTITY_ID_LIST_LEN + 1, '6');
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(
        drd,
        "pcie,targeted power",
        "pcie.test_pinned=false;pcie.min_bandwidth=25000;targeted power.temperature_max=82",
        "",
        "",
        "",
        true,
        true,
        debugFileName,
        statsPath,
        3,
        clocksEventMask,
        0,
        false,
        3,
        60,
        entityIds,
        "",
        defaultFrequency,
        "",
        error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    // Too large expectedNumEntities
    std::string invalidExpectedNumEntities(DCGM_EXPECTED_ENTITIES_LEN + 1, '6');
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(
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
        clocksEventMask,
        0,
        false,
        3,
        60,
        "*,cpu:*",
        invalidExpectedNumEntities,
        defaultFrequency,
        "",
        error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    // expectedNumEntities with ALL_ENTITIES groupId
    std::string expectedNumEntities(DCGM_EXPECTED_ENTITIES_LEN, '6');
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(
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
        clocksEventMask,
        DCGM_GROUP_ALL_ENTITIES,
        false,
        3,
        60,
        "*,cpu:*",
        expectedNumEntities,
        defaultFrequency,
        "",
        error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    unsigned int validWatchFrequency = 5000000;
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(drd,
                                                "1",
                                                "",
                                                "",
                                                "",
                                                "",
                                                false,
                                                false,
                                                "",
                                                "",
                                                0,
                                                "",
                                                DCGM_GROUP_ALL_GPUS,
                                                true,
                                                3,
                                                60,
                                                "*,cpu:*",
                                                "gpu:1",
                                                validWatchFrequency,
                                                "",
                                                error);
    if (result != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected DCGM_ST_OK but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    if (drd.watchFrequency != validWatchFrequency)
    {
        fprintf(stderr, "Expected watchFrequency to be %u, but found %u.\n", validWatchFrequency, drd.watchFrequency);
        return -1;
    }

    unsigned int invalidWatchFrequency = 99999;
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(drd,
                                                "1",
                                                "",
                                                "",
                                                "",
                                                "",
                                                false,
                                                false,
                                                "",
                                                "",
                                                0,
                                                "",
                                                DCGM_GROUP_ALL_GPUS,
                                                true,
                                                3,
                                                60,
                                                "*,cpu:*",
                                                "gpu:1",
                                                invalidWatchFrequency,
                                                "",
                                                error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    invalidWatchFrequency = 60000001;
    std::memset(&drd, 0, sizeof(drd));
    result = dcgm_diag_common_populate_run_diag(drd,
                                                "1",
                                                "",
                                                "",
                                                "",
                                                "",
                                                false,
                                                false,
                                                "",
                                                "",
                                                0,
                                                "",
                                                DCGM_GROUP_ALL_GPUS,
                                                true,
                                                3,
                                                60,
                                                "*,cpu:*",
                                                "gpu:1",
                                                invalidWatchFrequency,
                                                "",
                                                error);
    if (result != DCGM_ST_BADPARAM)
    {
        fprintf(stderr, "Expected DCGM_ST_BADPARAM but found '%d', error '%s'\n", result, error.c_str());
        return -1;
    }

    return 0;
}

int TestDiagManager::TestInvalidVersion()
{
    DcgmCacheManager dcm;
    dcgmReturn_t ret = dcm.Init(1, 3600.0, true);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected Init success, got %d\n", ret);
        return -1;
    }

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
    ret = g_coreCallbacks.postfunc(&ggc.header, g_coreCallbacks.poster);

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
    auto diagResponse   = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper wrapper;

    DcgmCacheManager dcm;
    dcgmReturn_t ret = dcm.Init(1, 3600.0, true);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Expected Init success, got %d\n", ret);
        return -1;
    }

    DcgmGroupManager dgm { &dcm };
    DcgmCoreCommunication dcc;
    dcc.Init(&dcm, &dgm);
    g_coreCallbacks.poster = &dcc;

    DcgmDiagManager am(g_coreCallbacks);

    wrapper.SetVersion(diagResponse.get());
    // Make the script that will fail
    CreateDummyFailScript();
    am.PerformExternalCommand(dummyCmds, wrapper, &stdoutStr, &stderrStr);
    if (stdoutStr.empty() && stderrStr.empty())
    {
        fprintf(stderr, "TestPerformExternalCommand should've captured stderr, but failed to do so.\n");
        return -1;
    }

    // Clean up
    RemoveDummyScript();

    return result;
}
