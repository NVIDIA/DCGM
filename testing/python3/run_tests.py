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
import getpass
import os
import re
import string
import apps
import shlex
import dcgm_structs
import dcgm_agent
import dcgm_agent_internal
import utils
import test_utils
import logger
import option_parser
import time
import subprocess
import nvidia_smi_utils
from sys import version as python_version

from tests.nvswitch_tests import test_nvswitch_utils

def log_environment_info():
    if utils.is_linux():
        logger.info("Xorg running:        %s" % test_utils.is_xorg_running())
    logger.info("Python version:      %s" % python_version.split(None, 1)[0])
    logger.info("Platform identifier: %s" % utils.platform_identifier)
    logger.info("Bare metal:          %s" % utils.is_bare_metal_system())
    logger.info("Running as user:     %s" % getpass.getuser())

    rawVersionInfo = dcgm_agent.dcgmVersionInfo()
    logger.info("Build info:          %s" % rawVersionInfo.rawBuildInfoString)
    logger.info("Mig:                 %s" % test_utils.is_mig_mode_enabled())
    logger.debug("ENV : %s" % "\n".join(list(map(str, sorted(os.environ.items())))))
        
##################################################################################
### Kills the specified processes. If murder is specified, then they are kill -9'ed
### instead of nicely killed.
##################################################################################
def kill_process_ids(process_ids, murder):
    running = False
    for pid in process_ids:
        if not pid:
            break
        running = True
        if murder:
            cmd = 'kill -9 %s' % pid
        else:
            cmd = 'kill %s' % pid
            runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output, error = runner.communicate()

    return running

##################################################################################
### Cleans up the hostengine if needed. If we can't clean it up, then we will 
### abort the testing framework.
##################################################################################
def kill_hostengine_if_needed():
    running = False
    need_to_validate = False
    for i in range(0,2):
        process_ids = test_utils.check_for_running_hostengine_and_log_details(True)
        running = kill_process_ids(process_ids, False)

        if running == False:
            break
        need_to_validate = True
        time.sleep(.5)

    if running:
        for i in range(0,2):
            process_ids = test_utils.check_for_running_hostengine_and_log_details(True)
            running = kill_process_ids(process_ids, True)

        msg = "Cannot run test! An instance of nv-hostengine is running and cannot be killed."
        msg += " Ensure nv-hostengine is stopped before running the tests."
        pids = test_utils.check_for_running_hostengine_and_log_details(False)
        assert not pids, msg

def run_tests():
    '''
    testDir: Subdirectory to look for tests in. For example: "tests" for NVML

    '''
    with test_utils.SubTest("Main"):

        log_environment_info()

        test_utils.RestoreDefaultEnvironment.restore_env()
        try:
            dcgm_structs._dcgmInit(utils.get_testing_framework_library_path())
            
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_LIBRARY_NOT_FOUND):
            logger.warning("DCGM Library hasn't been found in the system, is the driver correctly installed?")

            if utils.is_linux() and utils.is_32bit() and utils.is_system_64bit():
                # 32bit test on 64bit system
                logger.warning("Make sure that you've installed driver with both 64bit and 32bit binaries (e.g. not -internal.run or -no-compact32.run)")
            raise

        if option_parser.options.use_running_hostengine:
            with test_utils.RunStandaloneHostEngine() as handle:
                dcgmGpuCount = test_utils.log_gpu_information(handle)
                if dcgmGpuCount < 1:
                    logger.error("No DCGM-whitelisted GPUs found. Skipping tests.")
                    return
        else:
            with test_utils.RunEmbeddedHostEngine() as handle:
                dcgmGpuCount = test_utils.log_gpu_information(handle)
                if dcgmGpuCount < 1:
                    logger.error("No DCGM-whitelisted GPUs found. Skipping tests.")
                    return

        # Persistence mode is required
        (_, error) = nvidia_smi_utils.enable_persistence_mode()
        if error:
            logger.error(error)
            return
        
        with test_utils.SubTest("restore state", quiet=True):
            test_utils.RestoreDefaultEnvironment.restore() # restore the nvml settings

        test_content = test_utils.get_test_content()
        nvswitchModuleCounter = 0
        try:
            for module in test_content:
                # Attempt to clean up stranded processes instead of aborting
                kill_hostengine_if_needed()

                with test_utils.SubTest("module %s" % module[0].__name__):
                    for function in module[1]:
                        test_utils.run_subtest(function)
                        with test_utils.SubTest("%s - restore state" % (function.__name__), quiet=True):
                            test_utils.RestoreDefaultEnvironment.restore()
        finally:
            # SubTest might return KeyboardInterrupt exception. We should try to restore
            # state before closing
            with test_utils.SubTest("restore state", quiet=True):
                test_utils.RestoreDefaultEnvironment.restore()
            #dcgm_structs.dcgmShutdown()

_test_info_split_non_verbose = re.compile("\n *\n") # Matches empty line that separates short from long version of function_doc
_test_info_split_verbose_first_newlines = re.compile("^[\n ]*\n") # Matches empty lines at the beginning of string
_test_info_split_verbose_last_newlines = re.compile("[\n ]*$") # Matches empty lines at the end of the string
def print_test_info():
    """
    testDir: Subdirectory to look for tests in
    """
    #Convert module subdirectory into module dot path like tests/nvvs/x => tests.nvvs.x
    testDirWithDots = test_utils.test_directory.replace("/", ".")

    test_content = test_utils.get_test_content()
    for module in test_content:
        module_name = module[0].__name__
        module_name = module_name.replace("%s." % testDirWithDots, "", 1) # all tests are in testDir. module there's no use in printing that
        for function in module[1]:
            function_name = function.__name__
            function_doc = function.__doc__
            if function_doc is None:
                # verbose output uses indentation of the original string
                function_doc = "    Missing doc"
            
            if option_parser.options.verbose:
                # remove new lines at the beginning and end of the function_doc
                function_doc = _test_info_split_verbose_first_newlines.sub("", function_doc)
                function_doc = _test_info_split_verbose_last_newlines.sub("", function_doc)
                print("%s.%s:\n%s\n" % (module_name, function_name, function_doc))
            else:
                # It's non verbose output so just take the first part of the description (up to first double empty line) 
                function_doc = _test_info_split_non_verbose.split(function_doc)[0]
                # remove spaces at beginning of each line (map strip), remove empty lines (filter bool) and make it one line (string join)
                function_doc = " ".join(list(filter(bool, list(map(string.strip, function_doc.split("\n"))))))
                print("%s.%s:\n\t%s" % (module_name, function_name, function_doc))
