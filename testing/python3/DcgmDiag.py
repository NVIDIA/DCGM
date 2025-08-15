# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dcgm_structs
import dcgm_fields
import dcgm_agent
import logger

g_latestDiagResponseVer = dcgm_structs.dcgmDiagResponse_version12
g_latestRunDiagVer = dcgm_structs.dcgmRunDiag_version10

class DcgmDiag:

    # Maps version codes to simple version values for range comparisons
    _versionMap = {
        dcgm_structs.dcgmRunDiag_version: 10,
        dcgm_structs.dcgmRunDiag_version7: 7,
        dcgm_structs.dcgmRunDiag_version8: 8,
        dcgm_structs.dcgmRunDiag_version9: 9,
        dcgm_structs.dcgmRunDiag_version10: 10,
    }

    def __init__(self, gpuIds=None, cpuIds=None, testNamesStr='', paramsStr='',
                 ignoreErrorCodesStr='', verbose=True, version=dcgm_structs.dcgmRunDiag_version10, 
                 timeout=0):
        # Make sure version is valid
        if version not in DcgmDiag._versionMap:
            raise ValueError("'%s' is not a valid version for dcgmRunDiag." % version)
        self.version = version

        if self.version == dcgm_structs.dcgmRunDiag_version10:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_v10()
        elif self.version == dcgm_structs.dcgmRunDiag_version9:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_v9()
        elif self.version == dcgm_structs.dcgmRunDiag_version8:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_v8()
        elif self.version == dcgm_structs.dcgmRunDiag_version7:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
        else:
            logger.info("Unexpected runDiag version " + self.version + " using RunDiag_t")
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_t()

        self.numTests = 0
        self.numParams = 0
        self.gpuList = ""
        self.cpuList = ""
        self.expectedNumGpus = ""
        self.ignoreErrorCodes = ignoreErrorCodesStr
        self.SetVerbose(verbose)
        if testNamesStr == '':
            # default to a level 1 test
            self.runDiagInfo.validate = 1
        elif testNamesStr == '1':
            self.runDiagInfo.validate = 1
        elif testNamesStr == '2':
            self.runDiagInfo.validate = 2
        elif testNamesStr == '3':
            self.runDiagInfo.validate = 3
        elif testNamesStr == '4':
            self.runDiagInfo.validate = 4
        else:
            # Make sure no number other that 1-4 were submitted
            if testNamesStr.isdigit():
                raise ValueError("'%s' is not a valid test name." % testNamesStr)

            # Copy to the testNames portion of the object
            names = testNamesStr.split(',')
            if len(names) > dcgm_structs.DCGM_MAX_TEST_NAMES:
                err = 'DcgmDiag cannot initialize: %d test names were specified exceeding the limit of %d.' %\
                      (len(names), dcgm_structs.DCGM_MAX_TEST_NAMES)
                raise ValueError(err)

            for testName in names:
                self.AddTest(testName)

        if paramsStr != '':
            params = paramsStr.split(';')
            if len(params) >= dcgm_structs.DCGM_MAX_TEST_PARMS:
                err = 'DcgmDiag cannot initialize: %d parameters were specified, exceeding the limit of %d.' %\
                      (len(params), dcgm_structs.DCGM_MAX_TEST_PARMS)
                raise ValueError(err)

            for param in params:
                self.AddParameter(param)

        self.runDiagInfo.timeoutSeconds = timeout

        self.runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_NULL

        if cpuIds:
            self.cpuList = ",".join(["cpu:" + str(cpuId) for cpuId in cpuIds])

        if gpuIds:
            self.gpuList = ",".join([str(gpuId) for gpuId in gpuIds])

        if hasattr(self.runDiagInfo, 'ignoreErrorCodes'):
            self.runDiagInfo.ignoreErrorCodes = self.ignoreErrorCodes

        if hasattr(self.runDiagInfo, 'entityIds'):
            if len(self.gpuList) > 0 and len(self.cpuList) > 0:
                self.runDiagInfo.entityIds = f"{self.gpuList},{self.cpuList}"
            elif len(self.gpuList) > 0:
                self.runDiagInfo.entityIds = f"{self.gpuList}"
            elif len(self.cpuList) > 0:
                self.runDiagInfo.entityIds = f"{self.cpuList}"
            else:
                self.runDiagInfo.entityIds = "*,cpu:*"
        elif hasattr(self.runDiagInfo, 'gpuList'):
            if len(self.gpuList) > 0:
                self.runDiagInfo.gpuList = self.gpuList
            else:
                self.runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_ALL_GPUS

        if hasattr(self.runDiagInfo, 'expectedNumEntities'):
            self.runDiagInfo.expectedNumEntities = self.expectedNumGpus

        if logger.nvvs_trace_log_filename is not None:
            self.SetDebugLogFile(logger.nvvs_trace_log_filename)
            self.SetDebugLevel(5) # Collect logs at highest level for nvvs.

    def SetVerbose(self, val):
        if val == True:
            self.runDiagInfo.flags |= dcgm_structs.DCGM_RUN_FLAGS_VERBOSE
        else:
            self.runDiagInfo.flags &= ~dcgm_structs.DCGM_RUN_FLAGS_VERBOSE

    def UseFakeGpus(self):
        self.runDiagInfo.fakeGpuList = self.gpuList

    def GetStruct(self):
        return self.runDiagInfo

    def AddParameter(self, parameterStr):
        maxTestParamsLen = dcgm_structs.DCGM_MAX_TEST_PARMS_LEN
        if self.version == dcgm_structs.dcgmRunDiag_version8 or self.version == dcgm_structs.dcgmRunDiag_version9:
            maxTestParamsLen = dcgm_structs.DCGM_MAX_TEST_PARMS_LEN_V2
        elif self.version == dcgm_structs.dcgmRunDiag_version7:
            maxTestParamsLen = dcgm_structs.DCGM_MAX_TEST_PARMS_LEN
        if len(parameterStr) >= maxTestParamsLen:
            err = 'DcgmDiag cannot add parameter \'%s\' because it exceeds max length %d.' % \
                  (parameterStr, maxTestParamsLen)
            raise ValueError(err)

        index = 0
        for c in parameterStr:
            self.runDiagInfo.testParms[self.numParams][index] = ord(c)
            index += 1

        self.numParams += 1

    def AddTest(self, testNameStr):
        if len(testNameStr) >= dcgm_structs.DCGM_MAX_TEST_NAMES_LEN:
            err = 'DcgmDiag cannot add test name \'%s\' because it exceeds max length %d.' % \
                  (testNameStr, dcgm_structs.DCGM_MAX_TEST_NAMES_LEN)
            raise ValueError(err)

        index = 0
        for c in testNameStr:
            self.runDiagInfo.testNames[self.numTests][index] = ord(c)
            index += 1

        self.numTests += 1

    def SetStatsOnFail(self, val):
        if val == True:
            self.runDiagInfo.flags |= dcgm_structs.DCGM_RUN_FLAGS_STATSONFAIL

    def SetClocksEventMask(self, value):
        if DcgmDiag._versionMap[self.version] < 3:
            raise ValueError("Clocks event mask requires minimum version 3 for dcgmRunDiag.")
        if isinstance(value, str) and len(value) >= dcgm_structs.DCGM_CLOCKS_EVENT_MASK_LEN:
            raise ValueError("Clocks event mask value '%s' exceeds max length %d." 
                             % (value, dcgm_structs.DCGM_CLOCKS_EVENT_MASK_LEN - 1))

        self.runDiagInfo.clocksEventMask = str(value)

    # Deprecated: Use SetClocksEventMask instead
    def SetThrottleMask(self, value):
            self.SetClocksEventMask(value)

    def SetFailEarly(self, enable=True, checkInterval=5):
        if DcgmDiag._versionMap[self.version] < 5:
            raise ValueError("Fail early requires minimum version 5 for dcgmRunDiag.")
        if not isinstance(checkInterval, int):
            raise ValueError("Invalid checkInterval value: %s" % checkInterval)

        if enable:
            self.runDiagInfo.flags |= dcgm_structs.DCGM_RUN_FLAGS_FAIL_EARLY
            self.runDiagInfo.failCheckInterval = checkInterval
        else:
            self.runDiagInfo.flags &= ~dcgm_structs.DCGM_RUN_FLAGS_FAIL_EARLY

    def Execute(self, handle):
        return dcgm_agent.dcgmActionValidate_v2(handle, self.runDiagInfo, self.version)

    def SetStatsPath(self, statsPath):
        if len(statsPath) >= dcgm_structs.DCGM_PATH_LEN:
            err = "DcgmDiag cannot set statsPath '%s' because it exceeds max length %d." % \
                   (statsPath, dcgm_structs.DCGM_PATH_LEN)
            raise ValueError(err)

        self.runDiagInfo.statsPath = statsPath

    def SetConfigFileContents(self, configFileContents):
        if len(configFileContents) >= dcgm_structs.DCGM_MAX_CONFIG_FILE_LEN:
            err = "Dcgm Diag cannot set config file contents to '%s' because it exceeds max length %d." \
                  % (configFileContents, dcgm_structs.DCGM_MAX_CONFIG_FILE_LEN)
            raise ValueError(err)

        self.runDiagInfo.configFileContents = configFileContents

    def SetDebugLogFile(self, logFileName):
        if len(logFileName) >= dcgm_structs.DCGM_PATH_LEN:
            raise ValueError("Cannot set debug file to '%s' because it exceeds max length %d."\
                % (logFileName, dcgm_structs.DCGM_PATH_LEN))

        self.runDiagInfo.debugLogFile = logFileName

    def SetDebugLevel(self, debugLevel):
        if debugLevel < 0 or debugLevel > 5:
            raise ValueError("Cannot set debug level to %d. Debug Level must be a value from 0-5 inclusive.")

        self.runDiagInfo.debugLevel = debugLevel
    
    def SetWatchFrequency(self, val):
        if val < 100000 or val > 60000000:
            err = "Cannot set debug level to {}. Watch frequency must be a value from 100000-60000000 inclusive." \
                  % (val)
            raise ValueError(err)
        
        self.runDiagInfo.watchFrequency = val

################# General helpers #################

def find_test_in_response(response, testName):
    assert hasattr(response, 'tests') and hasattr(response, 'results'), "Response does not have tests or results"
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            return test

    return None

def retrieve_diag_failure_message(response, entityPair, testName):
    test = find_test_in_response(response, testName)
    if not test:
        return None
    for index in test.errorIndices[:min(test.numErrors, dcgm_structs.DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)]:
        if response.errors[index].entity == entityPair:
            return response.errors[index].msg

    return None

def check_diag_result_fail(response, entityPair, testName):
    # Returns `True` when there is a FAIL result associated with the specified `entityPair` and `testName`, `False` otherwise.
    test = find_test_in_response(response, testName)
    assert test, "Expected fail result for test %s but none was found" % testName
    if next(filter(lambda cur: cur.result == dcgm_structs.DCGM_DIAG_RESULT_FAIL and cur.entity == entityPair,
                   map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])), None):
        msg = retrieve_diag_failure_message(response, entityPair, testName)
        if not msg:
            msg = "No error was found to accompany the test failure"
        logger.info("Test %s failed, msg: '%s'" % (testName, msg))
        return True
    return False

def check_diag_result_pass(response, entityPair, testName):
    # Returns `True` when there is a PASS result associated with the specified `entityPair` and `testName`, `False` otherwise.
    test = find_test_in_response(response, testName)
    assert test, "Expected pass result for test %s but none was found" % testName
    if next(filter(lambda cur: cur.result == dcgm_structs.DCGM_DIAG_RESULT_PASS and cur.entity == entityPair,
               map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])), None):
        return True

    msg = retrieve_diag_failure_message(response, entityPair, testName)
    if not msg:
        msg = "No error was found to accompany the test failure"
    logger.info("Test %s unexpectedly failed, msg: %s" % (testName, msg))

    return False

def check_diag_result_non_passing(response, entityPair, testName):
    # Returns `False` if there are one or more passing results for `entityPair` and `testName`, `True` otherwise.
    return not check_diag_result_pass(response, entityPair, testName)

def check_diag_result_non_failing(response, entityPair, testName):
# Returns `False` if there are one or more failing results for `entityPair` and `testName`, `True` otherwise.
    return not check_diag_result_fail(response, entityPair, testName)

def check_diag_result_non_running(response, entityPair, testName):
    # Returns `False` if there are one or more tests running for `entityPair` and `testName`, `True` otherwise.
    # "nonrunning" in this sense matches [ SKIP, NOT_RUN ]
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if test.name == testName:
            break
    if test.name == testName:
        # The test was found, so any result should be skipped or not run.
        notRunResults = {dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN, dcgm_structs.DCGM_DIAG_RESULT_SKIP}
        if test.result in notRunResults:
            return True
        if next(filter(lambda cur: cur.result not in notRunResults and cur.entity == entityPair,
                   map(lambda resIdx: response.results[resIdx], test.resultIndices[:min(test.numResults, dcgm_structs.DCGM_DIAG_TEST_RUN_RESULTS_MAX)])), None):
            return False
    return True

def GetEntityCount(response, entityGroupId):
    # Returns the count of the specified entity group associated with the response.
    if hasattr(response, 'entities'):
        return sum(1 for entity in response.entities[:min(response.numEntities, dcgm_structs.DCGM_DIAG_RESPONSE_ENTITIES_MAX)] \
                   if entity.entity.entityGroupId == entityGroupId)
    elif hasattr(response, 'gpuCount') and entityGroupId == dcgm_fields.DCGM_FE_GPU:
        return response.gpuCount
    else:
        raise NotImplementedError("Entity type {} not handled for version {}.".format(entityGroupId, response.version))

def GetGpuCount(response):
    return GetEntityCount(response, dcgm_fields.DCGM_FE_GPU)

def ResultToString(result):
    # Return a string that reflects the specified `result`.
    # This may exist elsewhere, but wasn't found when needed. Replace/remove as makes sense.
    # Well suited to structural match, but pylint doesn't seem to like this.

    if result == dcgm_structs.DCGM_DIAG_RESULT_PASS:
        return "Pass"
    elif result == dcgm_structs.DCGM_DIAG_RESULT_FAIL:
        return "Fail"
    elif result ==  dcgm_structs.DCGM_DIAG_RESULT_WARN:
        return "Warn"
    elif result ==  dcgm_structs.DCGM_DIAG_RESULT_SKIP:
        return "Skip"
    elif result == dcgm_structs.DCGM_DIAG_RESULT_NOT_RUN:
        return "Not Run"
    else:
        raise ValueError("Invalid result {} specified" % result)

def DumpTestResults(logger, response, testNames=[]):
    # Utility for debugging. Dump test results from the response, optionally for specified tests.
    for test in response.tests[:min(response.numTests, dcgm_structs.DCGM_DIAG_RESPONSE_TESTS_MAX)]:
        if not testNames or test.name in testNames:
            logger.info("Results for {}, overall: {}".format(test.name, ResultToString(test.result)))
            for result in map(lambda resIdx: response.results[resIdx], test.resultIndices[:test.numResults]):
                logger.info("   testId: {} ent grp:{}, id:{}, result: {}".format(result.testId, result.entity.entityGroupId,
                                                                                 result.entity.entityId,
                                                                                 ResultToString(result.result)))
