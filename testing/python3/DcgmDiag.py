# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import dcgm_agent

class DcgmDiag:

    # Maps version codes to simple version values for range comparisons
    _versionMap = {
        dcgm_structs.dcgmRunDiag_version: 5
    }

    def __init__(self, gpuIds=None, testNamesStr='', paramsStr='', verbose=True, train=False, forceTrain=False, 
                 version=dcgm_structs.dcgmRunDiag_version):
        # Make sure version is valid
        if version not in DcgmDiag._versionMap:
            raise ValueError("'%s' is not a valid version for dcgmRunDiag." % version)
        self.version = version

        if self.version == dcgm_structs.dcgmRunDiag_version7:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_v7()
        else:
            self.runDiagInfo = dcgm_structs.c_dcgmRunDiag_t()

        self.numTests = 0
        self.numParams = 0
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
        else:
            # Make sure no number other that 1-3 were submitted
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

        if train == True:
            self.runDiagInfo.flags = dcgm_structs.DCGM_RUN_FLAGS_TRAIN
            if forceTrain == True:
                self.runDiagInfo.flags |= dcgm_structs.DCGM_RUN_FLAGS_FORCE_TRAIN

        if gpuIds:
            first = True
            for gpu in gpuIds:
                if first:
                    self.runDiagInfo.gpuList = str(gpu)
                    first = False
                else:
                    self.runDiagInfo.gpuList = "%s,%s" % (self.runDiagInfo.gpuList, str(gpu))
    
    def SetVerbose(self, val):
        if val == True:
            self.runDiagInfo.flags |= dcgm_structs.DCGM_RUN_FLAGS_VERBOSE
        else:
            self.runDiagInfo.flags &= ~dcgm_structs.DCGM_RUN_FLAGS_VERBOSE

    def UseFakeGpus(self):
        self.runDiagInfo.fakeGpuList = self.runDiagInfo.gpuList

    def GetStruct(self):
        return self.runDiagInfo

    def AddParameter(self, parameterStr):
        if len(parameterStr) >= dcgm_structs.DCGM_MAX_TEST_PARMS_LEN:
            err = 'DcgmDiag cannot add parameter \'%s\' because it exceeds max length %d.' % \
                  (parameterStr, dcgm_structs.DCGM_MAX_TEST_PARMS_LEN)
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

    def SetThrottleMask(self, value):
        if DcgmDiag._versionMap[self.version] < 3:
            raise ValueError("Throttle mask requires minimum version 3 for dcgmRunDiag.")
        if isinstance(value, str) and len(value) >= dcgm_structs.DCGM_THROTTLE_MASK_LEN:
            raise ValueError("Throttle mask value '%s' exceeds max length %d." 
                             % (value, dcgm_structs.DCGM_THROTTLE_MASK_LEN - 1))
        
        self.runDiagInfo.throttleMask = str(value)
    
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
        if len(logFileName) >= dcgm_structs.DCGM_FILE_LEN:
            raise ValueError("Cannot set debug file to '%s' because it exceeds max length %d."\
                % (logFileName, dcgm_structs.DCGM_FILE_LEN))

        self.runDiagInfo.debugLogFile = logFileName

    def SetDebugLevel(self, debugLevel):
        if debugLevel < 0 or debugLevel > 5:
            raise ValueError("Cannot set debug level to %d. Debug Level must be a value from 0-5 inclusive.")

        self.runDiagInfo.debugLevel = debugLevel
