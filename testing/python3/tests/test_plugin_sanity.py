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
"""
This module runs all plugins (and the short,medium,long suites) through dcgmi and logs the JSON output to a file in
OUTPUT_DIR (test_plugin_sanity_out folder in the testing folder). The module does not actually perform any verification
(except for the implicit verification that dcgmi returns with status 0 for each of the plugins/test suite runs when
there are no errors inserted) - you must *manually* verify the output in the output directory makes sense.

Since this module runs all plugins and the short,medium,long suites, all tests in this plugin are marked developer tests
(the total runtime for this plugin is 1 - 2 hours). It is *NOT* recommended to run this suite on a system with
GPUs that have different SKUs because each test is run once for every SKU on the machine.

The main purpose of this module is to provide a very basic sanity test for large changes made to nvvs plugins.

Note: When DCGM and NVVS have been updated to use error codes, these tests can be updated to perform automatic
validation of the output based on the error codes. (The tests will need to be updated to use the API call instead of
running dcgmi).

When debugging or there is a need to run only a subset of the tests, the following filter can be helpful:
`test_plugin_sanity.(test_create|<TEST_NAME_HERE>).*`
-> Replace <TEST_NAME_HERE> with the test to run. In case of multiple tests, use the `|` separator.
-> All existing files in the output directory will be deleted - to prevent this, remove "test_create" from the filter
"""

from functools import wraps
import ctypes
import os
import shutil
import signal
import time

from apps.dcgmi_app import DcgmiApp
from apps.nv_hostengine_app import NvHostEngineApp
from dcgm_internal_helpers import InjectionThread, check_nvvs_process

import dcgm_internal_helpers
import dcgm_fields
import dcgm_structs
import logger
import option_parser
import test_utils


### Constants
OUTPUT_DIR = "./test_plugin_sanity_out"
DEV_MODE_MSG = "Manual test for verifying plugin output. Use developer mode to enable."

### Helpers
@test_utils.run_first()
def test_create_output_dir():
    """
    Ensure we have a new results directory on every run. This "test" is called first when no filters are used
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

def log_app_output_to_file(app, filename):
    with open(filename, 'w') as f:
        for line in app.stdout_lines:
            f.write(line + "\n")
        for line in app.stderr_lines:
            f.write(line + "\n")

def log_app_output_to_stdout(app):
    logger.info("app output:")
    for line in app.stdout_lines:
        logger.info(line)
    for line in app.stderr_lines:
        logger.info(line)

def copy_nvvs_log(nvvsLogFile, outputLogFile):
    """
    Copy nvvs log file to the output dir. This method is needed because of path length limitations when using DCGM
    to run NVVS.
    """
    try:
        if os.path.exists(nvvsLogFile):
            shutil.copyfile(nvvsLogFile, outputLogFile)
    except IOError as e:
        logger.error("Could not copy nvvs log to output dir.")
        logger.error(e)


# Main test helper methods
def no_errors_run(handle, gpuIds, name, testname, parms=None):
    """
    Runs the given test (testname) without inserting errors, and ensures that dcgmi returns with a exit code of 0.
    name is the name of the plugin in nvvs (e.g. constant_perf)
    """
    output_file = OUTPUT_DIR + "/dcgmi_%s_no_err_%s.json" % (name, gpuIds[0])
    log_file = OUTPUT_DIR + "/nvvs_%s_no_err_%s.log" % (name, gpuIds[0])
    gpu_list = ",".join(map(str, gpuIds))

    # Note: Although using the dcgmActionValidate api (via DcgmDiag.Execute()) would allow for some automatic
    # verification, we use dcgmi diag and log output to a file for easier debugging when something goes wrong.
    args = ["diag", "-r", "%s" % testname, "-i", gpu_list, "-j", "-v", "-d", "5", "--debugLogFile", "/tmp/nvvs.log"]
    if parms != None:
        args.extend(["-p", "%s" % parms])
    dcgmi = DcgmiApp(args=args)

    dcgmi.start(timeout=1500) # 25min timeout
    logger.info("Started diag with args: %s" % args)

    retcode = dcgmi.wait()
    copy_nvvs_log("/tmp/nvvs.log", log_file)
    if retcode != 0:
        logger.error("dcgmi_%s_no_err failed with retcode: %s" % (name, retcode))
        copy_nvvs_log("/tmp/nvvs.log", log_file)

    log_app_output_to_file(dcgmi, output_file)


def with_error_run(handle, gpuIds, name, testname, parms=None):
    """
    Runs the given test (testname) and inserts throttling / REPLAY_COUNTER errors depending on the test.
    name is the name of the plugin in nvvs (e.g. constant_perf)

    Logs an error (but does not fail the test) if the dcgmi return code is not 226 (lower 8 bits of
    -30/DCGM_ST_NVVS_ERROR) which is expected since the test should fail due to inserted errors.

    Since busgrind/PCIe does a diff for the REPLAY_COUNTER field we need to insert errors after busgrind has read 
    some zero values for the field. As a result, the hardcoded delay of 15 seconds must be adjusted on different
    systems (currently a delay of 15 seconds works for the bstolle-dgx machine).
    """
    output_file = OUTPUT_DIR + "/dcgmi_%s_with_err_%s.json" % (name, gpuIds[0])
    log_file = OUTPUT_DIR + "/nvvs_%s_with_err_%s.log" % (name, gpuIds[0])
    gpu_list = ",".join(map(str, gpuIds))
    
    args = ["diag", "-r", "%s" % testname, "-i", gpu_list, "-j", "-v", "-d", "5", "--debugLogFile", "/tmp/nvvs.log"]
    if parms != None:
        args.extend(["-p", "%s" % parms])
    dcgmi = DcgmiApp(args=args)
    
    field_id = dcgm_fields.DCGM_FI_DEV_GPU_TEMP
    value = 1000
    delay = 0
    if name == "busgrind":
        field_id = dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER
        value = 1000
        delay = 15

    inject_error = InjectionThread(handle, gpuIds[0], field_id, value)
    if delay == 0:
        inject_error.start()
        logger.info("Injecting errors now (field %s, value %s)" % (field_id, value))
        assert dcgm_internal_helpers.verify_field_value(gpuIds[0], field_id, value)

    start = time.time()
    dcgmi.start(timeout=1500) # 25min timeout
    logger.info("Started diag with args: %s" % args)

    # Some tests do a diff test for the field values so we must let them see 0 values first
    if delay != 0:
        running, _ = dcgm_internal_helpers.check_nvvs_process(want_running=True)
        assert running, "nvvs did not start"
        logger.info("Nvvs started after %.1fs" % (time.time() - start))
        time.sleep(delay)
        logger.info("Injecting errors now (field %s, value %s)" % (field_id, value))
        inject_error.start()
        assert dcgm_internal_helpers.verify_field_value(gpuIds[0], field_id, value, maxWait=3)

    retcode = dcgmi.wait()
    
    inject_error.Stop()
    inject_error.join()
    assert inject_error.retCode == dcgm_structs.DCGM_ST_OK
    
    copy_nvvs_log("/tmp/nvvs.log", log_file)
    expected_retcode = ctypes.c_uint8(dcgm_structs.DCGM_ST_NVVS_ERROR).value
    if retcode != expected_retcode:
        logger.error("Expected retcode to be %s, but retcode of dcgmi is %s" % (expected_retcode, retcode))
    dcgmi.validate() # Validate because dcgmi returns non zero when the diag fails (expected)
    log_app_output_to_file(dcgmi, output_file)


### Tests
# busgrind
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_busgrind_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "busgrind", "PCIe")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_busgrind_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "busgrind", "PCIe")


# constant perf / targeted stress
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_constant_perf_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "constant_perf", "targeted stress")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_constant_perf_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "constant_perf", "targeted stress")


# constant power / targeted power
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_constant_power_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "constant_power", "targeted power")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_constant_power_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "constant_power", "targeted power")


# context create
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_context_create_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "context_create", "context create")


# gpuburn / diagnostic
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_gpuburn_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "gpuburn", "diagnostic", "diagnostic.test_duration=60")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_gpuburn_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "gpuburn", "diagnostic", "diagnostic.test_duration=60")


# memory
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_memory_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "memory", "memory")

# No run for memory plugin with inserted errors - memory test completes too quickly for reliably simulating a DBE
# For manual verification, a good WaR is to add a sleep(5) just before the memory plugin performs a memory allocation,
# create a temp build with this change, and then try inserting a DBE after launching the diag.


# memory bandwidth
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_memory_bandwidth_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "memory_bandwidth", "memory bandwidth", "memory bandwidth.is_allowed=true")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_memory_bandwidth_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "memory_bandwidth", "memory bandwidth", "memory bandwidth.is_allowed=true")


# sm perf / sm stress
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sm_perf_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "sm_perf", "sm stress")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_sm_perf_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "sm_perf", "sm stress")


# short
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_short_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "short", "short")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_short_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "short", "short")


# medium
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_medium_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "medium", "medium")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_medium_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "medium", "medium")


# long
@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_long_no_errors(handle, gpuIds):
    no_errors_run(handle, gpuIds, "long", "long")

@test_utils.run_with_developer_mode(msg=DEV_MODE_MSG)
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_long_with_error(handle, gpuIds):
    with_error_run(handle, gpuIds, "long", "long")
