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
PASSED_COUNT = 0
base_test_name = "run_dcgm_diag"
test_name = ""

################################################################################
def print_parseable_status(phase_name, iteration):
    if iteration:
        print("&&&& %s %s_%d" % (phase_name, test_name, iteration))
    else:
        print("&&&& %s %s_%d" % (phase_name, test_name, iteration))

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
    print("====================================")


def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]

DIAG_CLOCKS_EVENT_WARNING = "Clocks are being optimized for"
DIAG_INFOROM_WARNING = "Error calling NVML API nvmlDeviceValidateInforom"
DIAG_THERMAL_WARNING = "Thermal violations totaling "
DIAG_MIG_INCOMPATIBLE_WARNING = "is incompatible with the diagnostic because it prevents access to the entire GPU."
DIAG_MIG_MULTIPLE_GPU_WARNING = "Cannot run diagnostic: CUDA does not support enumerating GPUs with MIG mode enabled"

DIAG_CLOCKS_EVENT_SUGGEST = "A GPU's clocks are being optimized due to a cooling issue. Please make sure your GPUs are properly cooled."
DIAG_INFOROM_SUGGEST = "A GPU's inforom is corrupt. You should re-flash it with iromflsh or replace the GPU. Run nvidia-smi without arguments to see which GPU."
DIAG_THERMAL_SUGGEST = "A GPU has thermal violations happening. Please make sure your GPUs are properly cooled."
DIAG_MIG_INCOMPATIBLE_SUGGEST = "You must disable MIG mode or configure instances that use the entire GPU to run the diagnostic."
DIAG_MIG_MULTIPLE_GPU_SUGGEST = "You must run on only one GPU at a time when MIG is configured."

class TestRunner():

    ################################################################################
    def __init__(self, cycles, dcgmiDiag, verbose):
        self.cycles = int(cycles)
        self.dcgmiDiag = dcgmiDiag 
        self.verbose = verbose
        self.failed_runs = 0
        self.failing_tests = {}
        # The exclusion list is a list of [textToSearchFor, whatToPrintIfFound] entries
        self.exclusions = [
            [DIAG_INFOROM_WARNING, DIAG_INFOROM_SUGGEST],
            [DIAG_CLOCKS_EVENT_WARNING, DIAG_CLOCKS_EVENT_SUGGEST],
            [DIAG_THERMAL_WARNING, DIAG_THERMAL_SUGGEST],
            [DIAG_MIG_INCOMPATIBLE_WARNING, DIAG_MIG_INCOMPATIBLE_SUGGEST],
            [DIAG_MIG_MULTIPLE_GPU_WARNING, DIAG_MIG_MULTIPLE_GPU_SUGGEST],
        ]

    ################################################################################
    def matchesExclusion(self, warnings):
        for exclusion in self.exclusions:
            for warning in warnings:
                if warning['warning'].find(exclusion[0]) != -1:
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

        print("Running command: %s " % " ".join(self.dcgmiDiag.BuildDcgmiCommand()))
        ret = 0
        fail_total = 0
        exclusion_total = 0
        for runIndex in range(cycles):
            print_parseable_status("RUNNING", runIndex)
            self.dcgmiDiag.Run()
            if self.dcgmiDiag.failed_list:
                self.failing_tests[runIndex] = self.dcgmiDiag.failed_list
            else:
                self.failing_tests[runIndex] = []
            if self.dcgmiDiag.diagRet and not ret:
                ret = self.dcgmiDiag.diagRet

            # Get the number of actual errors in the output
            failCount, exclusionCount = self.checkForErrors()

            if failCount > fail_total:
                print_parseable_status("FAILED", runIndex)
            else:
                print_parseable_status("PASSED", runIndex)

            fail_total = fail_total + failCount
            exclusion_total = exclusion_total + exclusionCount


        if self.verbose:
            print(self.dcgmiDiag.lastStdout)
            if self.dcgmiDiag.lastStderr:
                print(self.dcgmiDiag.lastStderr)

        if (failCount != 0):
            if self.failed_runs > 0:
                print("%d of %d runs Failed. Please attach %s and %s to your bug report."
                      % (self.failed_runs, cycles, logFile, debugFile))
            print("ExclusionCount: %d" % exclusion_total)
            print("FailCount: %d" % fail_total)
            from ctypes import c_int8
            print("Diagnostic test failed with code %d.\n" % c_int8(ret).value)

            # Popen returns 0 even if diag test fails, so failing here
            return [fail_total, exclusion_total]

        return [0, exclusionCount]

    ################################################################################
    def run(self):
        self.dcgmiDiag.SetConfigFile(None)
        failCount, exclusionCount = self.run_command(self.cycles)
        return [failCount, exclusionCount]

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

    global test_name
    if cmdArgs.test_names is not "":
        settings['run_mode'] = 0
        settings['test_names'] = cmdArgs.test_names
        test_name = "%s_%s" % (base_test_name, cmdArgs.test_names)
    else:
        settings['run_mode'] = int(cmdArgs.run_mode)
        test_name = "%s_%d" % (base_test_name, int(cmdArgs.run_mode))
        settings['test_names'] = cmdArgs.test_names

    settings['dev_id'] = cmdArgs.device_id
    settings['cycles'] = cmdArgs.cycles

################################################################################
def parseCommandLine():

    parser = argparse.ArgumentParser(description="DCGM DIAGNOSTIC TEST FRAMEWORK")
    parser.add_argument("-c", "--cycles", required=True, help="Number of test cycles to run, all tests are one cycle.")
    parser.add_argument("-v", "--vulcan", action="store_true", help="Deprecated flag for running in the eris environment")
    parser.add_argument("--verbose", action="store_true", help="Sets verbose mode")
    parser.add_argument("-d", "--device-id", help="Comma separated list of nvml device ids.")
    parser.add_argument("-r", "--run-mode", default=4, help="Specify the tests to run: (1,2,3, or 4")
    parser.add_argument("-t", "--test-names", default="", help="Specify a comma-separated list of test names. Will override run mode")

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
    
    # Skip checks for nvlinks and pci width to avoid QA bugs
    paramsStr = "pcie.test_nvlink_status=false"
    paramsStr += ";pcie.h2d_d2h_single_unpinned.min_pci_width=1"
    paramsStr += ";pcie.h2d_d2h_single_pinned.min_pci_width=1"

    dcgmiDiag = DcgmiDiag.DcgmiDiag(gpuIds=gpuIdStr, testNamesStr=settings['test_names'], paramsStr=paramsStr,
                dcgmiPrefix=prefix, runMode=settings['run_mode'], debugLevel=5, debugFile=debugFile)

    # Start tests
    run_test = TestRunner(settings['cycles'], dcgmiDiag, settings['verbose'])

    print("\nRunning with the diagnostic... This may take a while, please wait...\n")
    failedCount, exclusionCount = run_test.run()

    return [failedCount, exclusionCount]

if __name__ == "__main__":
    cmdArgs = parseCommandLine()
    failedCount, exclusionCount = main(cmdArgs)
    ret = 0

    if os.path.isfile(logFile):
        with open(logFile, "r") as f:
            log_content = f.readlines()
            for log in log_content:
                if "Pass" in log:
                    PASSED_COUNT += 1

        logger.info("\n========== TEST SUMMARY ==========\n")
        logger.info("Passed: {}".format(PASSED_COUNT))
        logger.info("Failed: {}".format(failedCount))
        logger.info("Waived: {}".format(exclusionCount))
        logger.info("Total:  {}".format(PASSED_COUNT + failedCount + exclusionCount))
        logger.info("Cycles: {}".format(cmdArgs.cycles))
        logger.info("==================================\n\n")
    else:
        print_parseable_status("SKIPPED", None)
        print("Unable to provide test summary due to missing log file")

    sys.exit(ret)
