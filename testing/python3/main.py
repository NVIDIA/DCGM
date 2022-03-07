#! /usr/bin/env python3

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

print('Starting DCGM test framework')

# Early check to make sure we're running on a supported version of Python
import sys
print('Python version: {}'.format(sys.version))
def _version_check():
    version = sys.version.split()[0] # Discard compilation information
    version_tuple = tuple(map(int, version.split('.')))
    if version_tuple < (3, 5):
        print('DCGM Testing framework requires Python 3.5+')
        sys.exit(1)
_version_check()

import os
import platform
import test_utils
import option_parser
import logger
import utils
import shutil
import nvidia_smi_utils

from subprocess import check_output, Popen, CalledProcessError
from run_tests import run_tests
from run_tests import print_test_info

def is_file_binary(FileName):
    """ Checks for binary files and skips logging if True """
    try:
        with open(FileName, 'rb') as f:
            # Files with null bytes are binary
            if b'\x00' in f.read():
                print("\n=========================== " + FileName + " ===========================\n")
                print("File is binary, skipping log output!")
                return True
            else:
                return False
    except IOError:
        pass

def _summarize_tests():

    test_root = test_utils.SubTest.get_all_subtests()[0]
    tests_ok_count = test_root.stats[test_utils.SubTest.SUCCESS]
    tests_fail_count = test_root.stats[test_utils.SubTest.FAILED]
    tests_waived_count = test_root.stats[test_utils.SubTest.SKIPPED]
    tests_count = tests_ok_count + tests_fail_count

    # Dump all log output in Eris
    if tests_fail_count > 0 and option_parser.options.eris:
        logPath = os.path.join(logger.default_log_dir, logger.log_dir)
        logFiles = os.listdir(logPath)
        for logFile in logFiles:
            logFilePath = os.path.join(logPath,logFile)
            if not is_file_binary(logFilePath) and not os.path.isdir(logFilePath):
                print("\n=========================== " + logFile + " ===========================\n")
                with open(logFilePath, "r", encoding="utf-8-sig") as f:
                    print(f.read())
    
    logger.info("\n========== TEST SUMMARY ==========\n")
    logger.info("Passed: {}".format(tests_ok_count))
    logger.info("Failed: {}".format(tests_fail_count))
    logger.info("Waived: {}".format(tests_waived_count))
    logger.info("Total:  {}".format(tests_count))

    tests_completed_ratio = 0.0
    if tests_count > 0.0:
        tests_completed_ratio = float(tests_ok_count) / (float(tests_count) - (float(tests_fail_count / 2)))
    logger.info("Score:  %.2f" % (100.0 * tests_completed_ratio))
    logger.info("==================================\n\n")

    warnings_count = logger.messages_level_counts[logger.WARNING]
    if warnings_count > 0:
        logger.warning("Framework encountered %d warning%s" % (warnings_count, utils.plural_s(warnings_count)))

    if tests_ok_count < tests_count:
        logger.info()
        logger.info("Bug filing instructions:")
        logger.info(" * In bug description please include first and last error")
        logger.info(" * Also attach %s file (it already contains nvml trace logs, nvidia-bug report and stdout)" % (logger.log_archive_filename))

def _run_burn_in_tests():
    file_name = "burn_in_stress.py"
    if os.path.exists(file_name):
        logger.info("\nRunning a single iteration of Burn-in Stress Test! \nPlease wait...\n")

        #updates environment for the child process
        env = os.environ.copy()

        #remove env. variables below to prevent log file locks
        if "__DCGM_DBG_FILE" in env: del env["__DCGM_DBG_FILE"]
        if "__NVML_DBG_FILE" in env: del env["__NVML_DBG_FILE"]

        burn = Popen([sys.executable, file_name, "-t", "3"], stdout=None, stderr=None, env = env)
        
        if burn.pid == None:
            assert False, "Failed to launch Burn-in Tests" 
        burn.wait()
    else:
        logger.warning("burn_in_stress.py script not found!")

class TestFrameworkSetup(object):
    def __enter__(self):
        '''Initialize the test framework or exit on failure'''

        os.environ['__DCGM_TESTING_FRAMEWORK_ACTIVE'] = '1'
        # Make sure that the MPS server is disabled before running the test-suite
        if utils.is_mps_server_running():
            print('DCGM Testing framework is not interoperable with MPS server. Please disable MPS server.')
            sys.exit(1)
        
        # Various setup steps
        option_parser.parse_options() 
        utils.verify_user_file_permissions()
        utils.verify_localhost_dns()
        if not option_parser.options.use_running_hostengine:
            utils.verify_hostengine_port_is_usable()

        if not test_utils.noLogging:
            logger.setup_environment()

            if logger.log_dir:
                logger.close()
            
        option_parser.validate()
        
        if not test_utils.is_framework_compatible():
            logger.fatal("The test framework and dcgm versions are incompatible. Exiting Test Framework.")
            sys.exit(1)

        # Directory where DCGM test*.py files reside
        test_utils.set_tests_directory('tests')

        # Verify that package architecture matches python architecture
        if utils.is_64bit():
            # ignore this check on ppc64le and armv8 for now
            if not (platform.machine() == "ppc64le" or platform.machine() == "aarch64"):
                if not os.path.exists(os.path.join(utils.script_dir, "apps/amd64")):
                    print("Testing package is missing 64bit binaries, are you sure you're using package of correct architecture?")
                    sys.exit(1)
        else:
            if not os.path.exists(os.path.join(utils.script_dir, "apps/x86")):
                print("Testing package is missing 32bit binaries, are you sure you're using package of correct architecture?")
                sys.exit(1)

        # Stops the framework if running python 32bits on 64 bits OS
        if utils.is_windows():
            if os.name == "nt" and "32 bit" in sys.version and platform.machine() == "AMD64":
                print("Running Python 32-bit on a 64-bit OS is not supported. Please install Python 64-bit")
                sys.exit(1)

        if utils.is_linux():
            python_exec = str(sys.executable)
            python_arch = check_output(["file", "-L", python_exec])

            if "32-bit" in python_arch.decode('utf-8') and utils.is_64bit() == True:
                print("Running Python 32-bit on a 64-bit OS is not supported. Please install Python 64-bit")
                sys.exit(1)

        #Tell DCGM how to find our testing package's NVVS
        test_utils.set_nvvs_bin_path()

    def __exit__(self, type, value, traceback):
        logger.close()
        del os.environ['__DCGM_TESTING_FRAMEWORK_ACTIVE']
        pass

def main():

    with TestFrameworkSetup():

        if not option_parser.options.no_env_check:
            if not test_utils.is_test_environment_sane():
                logger.warning("The test environment does not seem to be healthy, test framework cannot continue.")
                sys.exit(1)

        if not option_parser.options.no_process_check:
            if not nvidia_smi_utils.are_gpus_free():
                sys.exit(1)
        else:
            logger.warning("Not checking for other processes using the GPU(s), test failures may occur.")

        if option_parser.options.test_info:
            print_test_info()
            return

        testdir = os.path.dirname(os.path.realpath(__file__))
        os.environ['GCOV_PREFIX'] = os.path.join(testdir, "_coverage/python")
        os.environ['GCOV_PREFIX_STRIP'] = '5'

        if test_utils.noLogging:
            run_tests()
        else:
            logger.run_with_coverage(run_tests())

        _summarize_tests()

        # Runs a single iteration of burn_in_stress test
        if option_parser.options.burn:
            _run_burn_in_tests()

if __name__ == '__main__':
    main()
