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
import subprocess
import json

import dcgm_structs
import dcgm_agent
import dcgm_fields
import nvidia_smi_utils
import test_utils

def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]

logFile = "nvvs_diag.log"

NAME_FIELD = "name"
RESULTS_FIELD = "results"
WARNINGS_FIELD = "warnings"
STATUS_FIELD = "status"
TEST_SUMMARY_FIELD = "test_summary"
INFO_FIELD = "info"
GPU_FIELD = "gpu_ids"
RUNTIME_ERROR_FIELD = "runtime_error"

TEST_STATUS_FAIL = 'Fail'

DIAG_CLOCKS_EVENT_WARNING = "Clocks are being optimized for"
DIAG_DBE_WARNING = "ecc_dbe_volatile_total"
DIAG_ECC_MODE_WARNING = "Skipping test because ECC is not enabled on this GPU"
DIAG_INFOROM_WARNING = "nvmlDeviceValidateInforom for nvml device"
DIAG_THERMAL_WARNING = "Thermal violations totaling "

DIAG_CLOCKS_EVENT_SUGGEST = "A GPU's clocks are being optimized due to a cooling issue. Please make sure your GPUs are properly cooled."
DIAG_DBE_SUGGEST = "This GPU needs to be drained and reset to clear the non-recoverable double bit errors."
DIAG_ECC_MODE_SUGGEST = "Run nvidia-smi -i <gpu id> -e 1 and then reboot to enable."
DIAG_INFOROM_SUGGEST = "A GPU's inforom is corrupt. You should re-flash it with iromflash or replace the GPU. run nvidia-smi without arguments to see which GPU."
DIAG_THERMAL_SUGGEST = "A GPU has thermal violations happening. Please make sure your GPUs are properly cooled."
DIAG_VARY_SUGGEST = "Please check for transient conditions on this machine that can disrupt consistency from run to run"

errorTuples = [(dcgm_fields.DCGM_FI_DEV_CLOCKS_EVENT_REASONS, DIAG_CLOCKS_EVENT_WARNING, DIAG_CLOCKS_EVENT_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, DIAG_DBE_WARNING, DIAG_DBE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_ECC_CURRENT, DIAG_ECC_MODE_WARNING, DIAG_ECC_MODE_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_INFOROM_CONFIG_VALID, DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST),
               (dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST)
               ]


################################################################################
class FailedTestInfo():
    ################################################################################
    def __init__(self, testname, warnings, info=None, entityPair=None):
        self.m_warning = warnings
        self.m_testname = testname
        self.m_info = info
        self.m_entityPair = entityPair
        self.m_isAnError = True
        self.m_fieldId = None
        self.m_suggestion = ''
        self.m_evaluatedMsg = ''
        if self.m_warning:
            for errorTuple in errorTuples:
                for warning in self.m_warning:
                    if warning['warning'].find(errorTuple[1]) != -1:
                        # Matched, record field ID and suggestion
                        self.m_fieldId = errorTuple[0]
                        self.m_suggestion = errorTuple[2]

    ################################################################################
    def SetInfo(self, info):
        self.m_info = info

    ################################################################################
    def GetFullError(self):
        if self.m_evaluatedMsg:
            return self.m_evaluatedMsg
        if not self.m_warning:
            full = "%s is reported as failed but has no warning message" % self.m_testname
        else:
            full = "%s failed: '%s'" % (self.m_testname, self.m_warning)
        if self.m_info:
            full += "\n%s" % self.m_info
        if self.m_entityPair:
            full += f"\n for entity {self.m_entityPair}"
        return full

    ################################################################################
    def GetFieldId(self):
        return self.m_fieldId

    ################################################################################
    def GetGpuId(self):
        (group, id) = self.m_entityPair
        if group.lower() != 'gpu':
            raise ValueError(f'Expected entity of type GPU, found {group}')
        return id

    ################################################################################
    def GetEntityPair(self):
        return self.m_entityPair

    ################################################################################
    def GetWarning(self):
        return self.m_warning

    ################################################################################
    def GetTestname(self):
        return self.m_testname

    ################################################################################
    def SetFailureMessage(self, val, correct_val):
        fieldName = dcgm_fields.DcgmFieldGetTagById(self.m_fieldId)
        if fieldName is None:
            fieldName = "Cannot find field id %d" % self.m_fieldId
        if val is None:
            # Our Nvidia-smi checker doesn't support this value yet
            self.m_evaluatedMsg = "%s\nOur nvidia-smi checker doesn't support evaluating field %s yet." % \
                                  (self.GetFullError(), fieldName)
        elif val != correct_val:
            self.m_evaluatedMsg = None
            self.m_isAnError = False # nvidia-smi reports an error in this field, so this is not a DCGM mistake
            if (self.m_fieldId):
                self.m_evaluatedMsg = "%s\nnvidia-smi found a value of %s for field %s instead of %s" % \
                                      (self.GetFullError(), str(val), fieldName, str(correct_val))
            else:
                self.m_evaluatedMsg = self.GetFullError()
        else:
            self.m_evaluatedMsg = "%s\nnvidia-smi found the correct value %s for field %s" %\
                                  (self.GetFullError(), str(val), fieldName)

    ################################################################################
    def IsAnError(self):
        """
        Simply return the error status in m_isAnError.
        """
        return self.m_isAnError


################################################################################
class DcgmiDiag:

    ################################################################################
    def __init__(self, gpuIds=None, testNamesStr='', paramsStr='', verbose=True,  
                 dcgmiPrefix='', runMode=0, configFile='', debugLevel=0, debugFile=''):
        #gpuList is expected to be a string. Convert it if it was provided
        self.gpuList = None
        if gpuIds is not None:
            if isinstance(gpuIds, str):
                self.gpuList = gpuIds
            else:
                self.gpuList = ",".join(map(str,gpuIds))
        
        self.testNamesStr = testNamesStr
        self.paramsStr = paramsStr
        self.verbose = verbose
        self.dcgmiPrefix = dcgmiPrefix
        self.runMode = runMode
        self.configFile = configFile
        self.debugLevel = debugLevel
        self.debugFile = debugFile

    ################################################################################
    def BuildDcgmiCommand(self):
        cmd = []

        if self.dcgmiPrefix:
            cmd.append("%s/dcgmi" % self.dcgmiPrefix)
        else:
            cmd.append("dcgmi")

        cmd.append("diag")

        if self.runMode == 0:
            # Use the test names string if a run mode was not specified
            cmd.append('-r')

            if self.testNamesStr:
                cmd.append(self.testNamesStr)
            else:
                # default to running level 3 tests
                cmd.append('3')
        else:
            # If the runMode has been specified, then use that over the test names string
            cmd.append('-r')
            cmd.append(str(self.runMode))

        if self.paramsStr:
            cmd.append('-p')
            cmd.append(self.paramsStr)

        if self.debugFile:
            cmd.append('--debugLogFile')
            cmd.append(self.debugFile)

            if self.debugLevel:
                cmd.append('-d')
                cmd.append(test_utils.DebugLevelToString(self.debugLevel))

        cmd.append('-j')
        
        if self.verbose:
            cmd.append('-v')

        if self.configFile:
            cmd.append('-c')
            cmd.append(self.configFile)

        if self.gpuList is not None:
            cmd.append('-i')
            cmd.append(self.gpuList)

        return cmd
    
    ################################################################################
    def AddGpuList(self, gpu_list):
        self.gpuList = gpu_list
    
    ################################################################################
    def FindFailedTests(self, jsondict, failed_list):
        ENTITY_ID_FIELD = 'entity_id'
        ENTITY_GROUP_FIELD = 'entity_group'

        def findFailuresInResults(testName, results, failed_list):
            # { status, warnings[], info[] }
            for result in results:
                if result[STATUS_FIELD] == TEST_STATUS_FAIL:
                    if WARNINGS_FIELD in result:
                        if ENTITY_ID_FIELD in result:
                            entityPair = (result[ENTITY_GROUP_FIELD], result[ENTITY_ID_FIELD])
                        info = '\n'.join(result.get(INFO_FIELD) or [])
                        failed_list.append(FailedTestInfo(testName, result.get(WARNINGS_FIELD), info, entityPair))
                        return True
            return False

        def findFailuresInTestSummary(testName, summary, failed_list):
            # { status, warnings[], info[] }
            if summary[STATUS_FIELD] == TEST_STATUS_FAIL:
                if WARNINGS_FIELD in summary:
                    info = '\n'.join(summary.get(INFO_FIELD) or [])
                    failed_list.append(FailedTestInfo(testName, summary.get(WARNINGS_FIELD), info))
                    return True
                if not failed_list:
                    failed_list.append(FailedTestInfo(testName, [{ 'warning' : f'Status in \'{TEST_SUMMARY_FIELD}\' is ' \
                                                                  f'\'{summary[STATUS_FIELD]}\'' }]))
            return False

        if not isinstance(jsondict, dict):
            # Only inspect dictionaries
            return

        if RESULTS_FIELD in jsondict:
            # This is a test entry.
            testName = jsondict[NAME_FIELD]
            assert isinstance(jsondict[RESULTS_FIELD], list)
            findFailuresInResults(testName, jsondict[RESULTS_FIELD], failed_list)

            assert NAME_FIELD in jsondict
            assert isinstance(jsondict[NAME_FIELD], str)

            assert TEST_SUMMARY_FIELD in jsondict
            assert isinstance(jsondict[TEST_SUMMARY_FIELD], dict)
            findFailuresInTestSummary(testName, jsondict[TEST_SUMMARY_FIELD], failed_list)
        elif RUNTIME_ERROR_FIELD in jsondict:
            # Experienced a complete failure while trying to run the diagnostic. No need
            # to parse for further errors because there will be no other json entries.
            # failInfo wants something that looks like a 'warnings' array
            failed_list.append(FailedTestInfo('System_Failure', [{ 'warning' : jsondict.get(RUNTIME_ERROR_FIELD) }] ))
        else:
            for key in jsondict:
                if isinstance(jsondict[key], list):
                    for item in jsondict[key]:
                        self.FindFailedTests(item, failed_list)
                else:
                    self.FindFailedTests(jsondict[key], failed_list)

    ################################################################################
    def IdentifyFailingTests(self, jsondict, nsc):
        failed_list = []

        self.FindFailedTests(jsondict, failed_list)
        for failInfo in failed_list:
            fieldId = failInfo.GetFieldId()
            if fieldId:
                val, correct_val = nsc.GetErrorValue(failInfo.GetGpuId(), fieldId)
                failInfo.SetFailureMessage(val, correct_val)

        return failed_list
    
    ################################################################################
    def SetAndCheckOutput(self, stdout, stderr, ret=0, nsc=None):
        self.lastStdout = stdout
        self.lastStderr = stderr
        self.diagRet = ret
        if not nsc:
            nsc = nvidia_smi_utils.NvidiaSmiJob()
        return self.CheckOutput(nsc)

    ################################################################################
    def CheckOutput(self, nsc):
        failed_list = []
        
        if self.lastStdout:
            try:
                jsondict = json.loads(trimJsonText(self.lastStdout))
                test_utils.diag_verify_json(jsondict)
            except ValueError as e:
                print(("Couldn't parse json from '%s'" % self.lastStdout))
                return None, 1

            failed_list = self.IdentifyFailingTests(jsondict, nsc)

            # Saves diag stdout into a log file - use append to get multiple runs in 
            # the same file if we're called repeatedly.
            with open(logFile, "a") as f:
                f.seek(0)
                f.write(str(self.lastStdout))

        if self.lastStderr:
            # Saves diag stderr into a log file - use append to get multiple runs in 
            # the same file if we're called repeatedly.
            with open(logFile, "a") as f:
                f.seek(0)
                f.write(str(self.lastStderr))

        return failed_list, self.diagRet

    ################################################################################
    def __RunDcgmiDiag__(self, cmd):
        self.lastCmd = cmd
        self.lastStdout = ''
        self.lastStderr = ''

        nsc = nvidia_smi_utils.NvidiaSmiJob()
        nsc.start()
        runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdout_buf, stderr_buf) = runner.communicate()
        self.lastStdout = stdout_buf and stdout_buf.decode('utf-8')
        self.lastStderr = stderr_buf and stderr_buf.decode('utf-8')
        self.diagRet = runner.returncode
        nsc.m_shutdownFlag.set()
        nsc.join()

        return self.CheckOutput(nsc)
    
    ################################################################################
    def DidIFail(self):
        if self.failed_list:
            for failure in self.failed_list:
                if failure.IsAnError():
                    return True
                    break

        if self.diagRet != 0:
            return True

        return False
    
    ################################################################################
    def RunDcgmiDiag(self, config_file, runMode=0):
        oldConfig = self.configFile
        oldRunMode = self.runMode

        if config_file:
            self.configFile = config_file
        else:
            self.configFile = ''

        if runMode:
            self.runMode = runMode

        cmd = self.BuildDcgmiCommand()
        self.failed_list, self.diagRet = self.__RunDcgmiDiag__(cmd)

        self.configFile = oldConfig
        self.runMode = oldRunMode

        return self.DidIFail()

    ################################################################################
    def RunAtLevel(self, runMode, configFile=None):
        if runMode < 1 or runMode > 3:
            return dcgm_structs.DCGM_ST_BADPARAM

        return self.RunDcgmiDiag(configFile, runMode)

    ################################################################################
    def Run(self):
        cmd = self.BuildDcgmiCommand()
        self.failed_list, self.diagRet = self.__RunDcgmiDiag__(cmd)
        return self.DidIFail()

    ################################################################################
    def SetConfigFile(self, config_file):
        self.configFile = config_file
    
    ################################################################################
    def SetRunMode(self, run_mode):
        self.runMode = run_mode

    ################################################################################
    def PrintFailures(self):
        for failure in self.failed_list:
            print(failure.GetFullError())

    ################################################################################
    def PrintLastRunStatus(self):
        print("Ran '%s' and got return code %d" % (self.lastCmd, self.diagRet))
        print("stdout: \n\n%s" % self.lastStdout)
        if self.lastStderr:
            print("\nstderr: \n\n%s" % self.lastStderr)
        else:
            print("\nNo stderr output")
        self.PrintFailures()

def main():
    dd = DcgmiDiag()
    failed = dd.Run()
    dd.PrintLastRunStatus()

if __name__ == '__main__':
    main()
