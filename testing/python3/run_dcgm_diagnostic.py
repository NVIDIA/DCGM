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


import os
import sys
import copy
import test_utils
import utils
import pydcgm
import option_parser
import time
import glob
import shutil
import json
import logger
import argparse
import nvidia_smi_utils
import DcgmiDiag
import dcgm_structs
import dcgm_fields
import version

from subprocess import Popen, STDOUT, PIPE, check_output

logFile = "nvvs_diag.log"
debugFile = "nvvs_debug.log"
goldenValuesFile = "/tmp/golden_values.yml"
LOAD_TOO_HIGH_TO_TRAIN = 1010101010
PASSED_COUNT = 0
FAILED_COUNT = 0
WAIVED_COUNT = 0


################################################################################
def remove_file_yolo(filename):
    '''
    Try to remove a file, not caring if any error occurs
    '''
    try:
        os.remove(filename)
    except:
        pass


################################################################################
def setupEnvironment(cmdArgs):
    """
    Function to prepare the test environment
    """
    message = ''

    # Set variable indicating we are running tests
    os.environ['__DCGM_TESTING_FRAMEWORK_ACTIVE'] = '1'

    # Verify if GPUs are free before running the tests
    if not nvidia_smi_utils.are_gpus_free():
        print("Some GPUs are in use, please check the workload and try again")
        sys.exit(1)

    if test_utils.is_framework_compatible() is False:
        print("run_dcgm_diagnostic.py found to be a different version than DCGM. Exiting")
        sys.exit(1)
    else:
        print("Running against Git Commit %s" % version.GIT_COMMIT)

    # Enable persistence mode or the tests will fail
    print("\nEnabling persistence mode")
    (message, error) = nvidia_smi_utils.enable_persistence_mode()
    if message:
        print(message)
    if error:
        print(error)
        print("\nWarning! Please make sure to enable persistence mode")
        time.sleep(1)

    # Collects the output of "nvidia-smi -q" and prints it out on the screen for debugging
    print("\n###################### NVSMI OUTPUT FOR DEBUGGING ONLY ##########################")

    (message, error) = nvidia_smi_utils.get_output()
    if message:
        print(message)
    if error:
        print(error)

    print("\n###################### NVSMI OUTPUT FOR DEBUGGING ONLY ##########################\n\n")

    # Tries to remove older log files
    remove_file_yolo(logFile)
    remove_file_yolo(debugFile)

    print("============= TEST CONFIG ==========")
    print("TEST CYLES:   {}".format(cmdArgs.cycles))
    print("DEVICE LIST:  {}".format(cmdArgs.device_id))
    print("TRAINING:     {}".format(not cmdArgs.notrain))
    print("====================================")


def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]


NAME_FIELD = "name"
RESULTS_FIELD = "results"
WARNING_FIELD = "warnings"
STATUS_FIELD = "status"
INFO_FIELD = "info"
GPU_FIELD = "gpu_ids"
RUNTIME_ERROR_FIELD = "runtime_error"

DIAG_THROTTLE_WARNING = "Clocks are being throttled for"
DIAG_DBE_WARNING = "ecc_dbe_volatile_total"
DIAG_ECC_MODE_WARNING = "because ECC is not enabled on GPU"
DIAG_INFOROM_WARNING = "Error calling NVML API nvmlDeviceValidateInforom"
DIAG_THERMAL_WARNING = "Thermal violations totaling "
DIAG_MIG_INCOMPATIBLE_WARNING = "is incompatible with the diagnostic because it prevents access to the entire GPU."
DIAG_MIG_MULTIPLE_GPU_WARNING = "Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode enabled"

DIAG_THROTTLE_SUGGEST = "A GPU's clocks are being throttled due to a cooling issue. Please make sure your GPUs are properly cooled."
DIAG_DBE_SUGGEST = "This GPU needs to be drained and reset to clear the non-recoverable double bit errors."
DIAG_ECC_MODE_SUGGEST = "Run 'nvidia-smi -i <gpu id> -e 1' and then reboot to enable ECC memory."
DIAG_INFOROM_SUGGEST = "A GPU's inforom is corrupt. You should re-flash it with iromflsh or replace the GPU. Run nvidia-smi without arguments to see which GPU."
DIAG_THERMAL_SUGGEST = "A GPU has thermal violations happening. Please make sure your GPUs are properly cooled."
DIAG_MIG_INCOMPATIBLE_SUGGEST = "You must disable MIG mode or configure instances that use the entire GPU to run the diagnostic."
DIAG_MIG_MULTIPLE_GPU_SUGGEST = "You must run on only one GPU at a time when MIG is configured."

class TestRunner():

    ################################################################################
    def __init__(self, cycles, dcgmiDiag, notrain, verbose):
        self.cycles = int(cycles)
        self.dcgmiDiag = dcgmiDiag 
        self.notrain = notrain
        self.verbose = verbose
        self.failed_runs = 0
        self.failing_tests = {}
        # The exclusion list is a list of [textToSearchFor, whatToPrintIfFound] entries
        self.exclusions = [
            [DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST],
            [DIAG_THROTTLE_WARNING, DIAG_THROTTLE_SUGGEST],
            [DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST],
            [DIAG_MIG_INCOMPATIBLE_WARNING, DIAG_MIG_INCOMPATIBLE_SUGGEST],
            [DIAG_MIG_MULTIPLE_GPU_WARNING, DIAG_MIG_MULTIPLE_GPU_SUGGEST],
        ]

    ################################################################################
    def matchesExclusion(self, warning):
        for exclusion in self.exclusions:
            if warning.find(exclusion[0]) != -1:
                return exclusion[1]

        return None

    def getErrorMessage(self, failureInfo, runIndex, recommendation):
        msg = ''
        if recommendation:
            msg = "Iteration %d test '%s' is ignoring error '%s' : %s" % \
                (runIndex, failureInfo.GetTestname(), failureInfo.GetFullError(), recommendation)
        else:
            msg = "Iteration %d test '%s' failed: '%s'" % \
                (runIndex, failureInfo.GetTestname(), failureInfo.GetFullError())

        return msg

    ################################################################################
    def checkForErrors(self):
        '''
        Check the NVVS JSON output for errors, filtering out any errors that are environmental rather
        than NVVS bugs. Returns a count of the number of errors. Anything > 0 will result in bugs against
        NVVS.

        Returns a tuple of [numErrors, numExclusions]
        '''
        numErrors = 0
        numExclusions = 0
        failureDetails = []
        for key in self.failing_tests:
            runFailures = 0
            for failureInfo in self.failing_tests[key]:
                recommendation = self.matchesExclusion(failureInfo.GetWarning())
                if recommendation:
                    print(self.getErrorMessage(failureInfo, key, recommendation))
                    numExclusions += 1
                else:
                    failureDetails.append(self.getErrorMessage(failureInfo, key, None))
                    runFailures += 1

            if runFailures > 0:
                self.failed_runs += 1
                numErrors += runFailures

        for failure in failureDetails:
            print("%s\n" % failure)

        return [numErrors, numExclusions]

    ################################################################################
    def run_command(self, cycles):
        """
        Helper method to run a give command
        """

        global WAIVED_COUNT
        print("Running command: %s " % " ".join(self.dcgmiDiag.BuildDcgmiCommand()))
        ret = 0
        for runIndex in range(cycles):
            self.dcgmiDiag.Run()
            self.failing_tests[runIndex] = self.dcgmiDiag.failed_list
            if self.dcgmiDiag.diagRet and not ret:
                ret = self.dcgmiDiag.diagRet

        # Get the number of actual errors in the output
        failCount, exclusionCount = self.checkForErrors()

        if self.verbose:
            print(self.dcgmiDiag.lastStdout)
            if self.dcgmiDiag.lastStderr:
                print(self.dcgmiDiag.lastStderr)

        if (failCount != 0):
            if self.failed_runs > 0:
                print("%d of %d runs Failed. Please attach %s and %s to your bug report."
                      % (self.failed_runs, cycles, logFile, debugFile))
            print("ExclusionCount: %d" % exclusionCount)
            print("FailCount: %d" % failCount)
            print("&&&& FAILED")
            print("Diagnostic test failed with code %d.\n" % ret)

            # Popen returns 0 even if diag test fails, so failing here
            return 1
        elif exclusionCount > 0:
            WAIVED_COUNT += 1
        else:
            print("Success")

        return 0

    ################################################################################
    def run_without_training(self):
        self.dcgmiDiag.SetTraining(False)
        self.dcgmiDiag.SetRunMode(3)
        self.dcgmiDiag.SetConfigFile(None)
        ret = self.run_command(self.cycles)
        return ret

    ################################################################################
    def run_with_training(self):
        retries = 0
        maxRetries = 180
        # Sleep for up to three minutes to see if we can train
        while os.getloadavg()[0] > .5:
            if retries >= maxRetries:
                print("Skipping training! Cannot train because the load is above .5 after %d seconds\n" % maxRetries)
                print("This is not a bug, please make sure that no other workloads are using up the system's resources")
                return LOAD_TOO_HIGH_TO_TRAIN
            retries += 1
            time.sleep(1)

        print("\nTraining once to generate golden values... This may take a while, please wait...\n")

        self.dcgmiDiag.SetRunMode(None)
        self.dcgmiDiag.SetTraining(True)

        ret = self.run_command(1)  # Always train only once
        return ret

    ################################################################################
    def run_with_golden_values(self):
        print("Running tests using existing golden values from %s" % goldenValuesFile)
        self.dcgmiDiag.SetConfig(goldenValuesFile)
        self.dcgmiDiag.SetRunMode(3)
        self.dcgmiDiag.SetTraining(False)
        print("Generated golden values file: \n")
        try:
            with open(goldenValuesFile, 'r') as f:
                print(f.read())
        except IOError as e:
            print("Error attempting to dump the golden values file '%s' : '%s'" % (goldenValuesFile, str(e)))
            print("Attempting to continue...")

        if os.path.isfile(goldenValuesFile):
            ret = self.run_command(1)  # Run against the golden values only once
            return ret
        else:
            print("Unable to find golden values file")
            return -1


################################################################################
def checkCmdLine(cmdArgs, settings):

    if cmdArgs.device_id:
        # Verify devices have been specified correctly
        if len(cmdArgs.device_id) > 1 and ("," in cmdArgs.device_id):
            gpuIds = cmdArgs.device_id.split(",")
            for gpuId in gpuIds:
                if not gpuId.isdigit():  # despite being named isdigit(), ensures the string is a valid unsigned integer
                    print("Please specify a comma separated list of device IDs.")
                    sys.exit(1)
        elif len(cmdArgs.device_id) > 1 and ("," not in cmdArgs.device_id):
            print("Please specify a comma separated list of device IDs.")
            sys.exit(1)
        elif len(cmdArgs.device_id) == 1:
            if not cmdArgs.device_id[0].isdigit():
                print("\"{}\" is not a valid device ID, please provide a number instead.".format(cmdArgs.device_id[0]))
                sys.exit(1)
        else:
            print("Device list validated successfully")

    if cmdArgs.vulcan or cmdArgs.verbose:
        settings['verbose'] = True
    else:
        settings['verbose'] = False

    settings['dev_id'] = cmdArgs.device_id
    settings['cycles'] = cmdArgs.cycles
    settings['notrain'] = True # Don't train until we are decided to support this feature

################################################################################
def getGoldenValueDebugFiles():
    # This method copies debug files for golden values to the current directory.
    gpuIdMetricsFileList = glob.glob('/tmp/dcgmgd_withgpuids*')
    gpuIdMetricsFile = None
    allMetricsFileList = glob.glob('/tmp/dcgmgd[!_]*')
    allMetricsFile = None

    if gpuIdMetricsFileList:
        # Grab latest debug file
        gpuIdMetricsFileList.sort()
        gpuIdMetricsFile = gpuIdMetricsFileList[-1]
    if allMetricsFileList:
        # Grab latest debug file
        allMetricsFileList.sort()
        allMetricsFile = allMetricsFileList[-1]

    fileList = []
    try:
        if gpuIdMetricsFile is not None:
            shutil.copyfile(gpuIdMetricsFile, './dcgmgd_withgpuids.txt')
            fileList.append('dcgmgd_withgpuids.txt')
        if allMetricsFile is not None:
            shutil.copyfile(allMetricsFile, './dcgmgd_allmetrics.txt')
            fileList.append('dcgmgd_allmetrics.txt')
        if os.path.isfile(goldenValuesFile):
            shutil.copyfile(goldenValuesFile, './golden_values.yml')
            fileList.append('golden_values.yml')
    except (IOError, OSError) as e:
        print("There was an error copying the debug files to the current directory %s" % e)

    if fileList:
        print("Please attach %s to the bug report." % fileList)
    else:
        print("No debug files were copied.")


################################################################################
def parseCommandLine():

    parser = argparse.ArgumentParser(description="DCGM DIAGNOSTIC TEST FRAMEWORK")
    parser.add_argument("-c", "--cycles", required=True, help="Number of test cycles to run, all tests are one cycle.")
    parser.add_argument("-v", "--vulcan", action="store_true", help="Deprecated flag for running in the eris environment")
    parser.add_argument("--verbose", action="store_true", help="Sets verbose mode")
    parser.add_argument("-d", "--device-id", help="Comma separated list of nvml device ids.")
    parser.add_argument("-n", "--notrain", action="store_true", help="Disable training for golden values. Enabled by default.")

    args = parser.parse_args()

    return args

################################################################################
def main(cmdArgs):

    settings = {}
    checkCmdLine(cmdArgs, settings)

    # Prepare the test environment and setup step
    option_parser.initialize_as_stub()
    setupEnvironment(cmdArgs)
    prefix = utils.verify_binary_locations()

    # Build a nvvs command list. Each element is an argument
    current_location = os.path.realpath(sys.path[0])

    # Get a list of gpus to run against
    gpuIdStr = ''
    if settings['dev_id'] is None:
        # None specified on the command line. Build compatible lists of GPUs
        dcgmHandle = pydcgm.DcgmHandle(ipAddress=None)
        gpuIds = dcgmHandle.GetSystem().discovery.GetAllSupportedGpuIds()
        gpuGroups = test_utils.group_gpu_ids_by_sku(dcgmHandle.handle, gpuIds)
        if len(gpuGroups) > 1:
            print("This system has more than one GPU SKU; DCGM Diagnostics is defaulting to just GPU(s) %s" %
                  gpuGroups[0])
        gpuGroup = gpuGroups[0]
        gpuIdStr = ",".join(map(str, gpuGroup))
        del(dcgmHandle)
        dcgmHandle = None
    else:
        gpuIdStr = settings['dev_id']

    dcgmiDiag = DcgmiDiag.DcgmiDiag(gpuIds=gpuIdStr, dcgmiPrefix=prefix, runMode=3, debugLevel=5, \
                debugFile=debugFile)

    # Start tests
    run_test = TestRunner(settings['cycles'], dcgmiDiag, settings['notrain'], settings['verbose'])

    if settings['notrain']:
        print("\nRunning with training mode disabled... This may take a while, please wait...\n")
        ret = run_test.run_without_training()
        if ret != 0:
            print("&&&& FAILED")
            return ret
    else:
        ret = run_test.run_without_training()
        if ret != 0:
            print("Exiting early after a failure in the initial runs of the diagnostic")
            print("&&&& FAILED")
            return ret

        ret = run_test.run_with_training()
        if ret == LOAD_TOO_HIGH_TO_TRAIN:
            return 0
        elif ret != 0:
            getGoldenValueDebugFiles()
            print("Exiting early after a failure while training the diagnostic")
            print("&&&& FAILED")
            return ret

        # Disable running against the generated golden values for a time until we fix DCGM-1134
        # ret = run_test.run_with_golden_values()
        # if ret != 0:
        #     getGoldenValueDebugFiles()
        #     print("Failed when running against the generated golden values")
        #     return ret

    return ret

if __name__ == "__main__":
    cmdArgs = parseCommandLine()
    ret = main(cmdArgs)

    if os.path.isfile(logFile):
        with open(logFile, "r") as f:
            log_content = f.readlines()
            for log in log_content:
                if "Pass" in log:
                    PASSED_COUNT += 1
                elif "Fail" in log:
                    FAILED_COUNT += 1
                elif "Skip" in log:
                    WAIVED_COUNT += 1

        # QA uses these to count the tests passed
        if FAILED_COUNT:
            print('&&&& FAILED')
        elif PASSED_COUNT == 0:
            print('&&&& SKIPPED')
        else:
            print('&&&& PASSED')

        logger.info("\n========== TEST SUMMARY ==========\n")
        logger.info("Passed: {}".format(PASSED_COUNT))
        logger.info("Failed: {}".format(FAILED_COUNT))
        logger.info("Waived: {}".format(WAIVED_COUNT))
        logger.info("Total:  {}".format(PASSED_COUNT + FAILED_COUNT + WAIVED_COUNT))
        logger.info("Cycles: {}".format(cmdArgs.cycles))
        logger.info("==================================\n\n")
    else:
        print("&&&& SKIPPED")
        print("Unable to provide test summary due to missing log file")

    sys.exit(ret)
