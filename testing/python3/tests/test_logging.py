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
from contextlib import closing
import pydcgm
import dcgm_structs
import logger
import test_utils
import apps
import time

@test_utils.run_only_on_linux()
@test_utils.run_only_on_bare_metal()
@test_utils.run_with_logging_on()
def test_logging_env_var():
    """
    Verifies that we log to the supplied env var
    """

    if test_utils.loggingLevel != 'DEBUG':
        test_utils.skip_test("Detected logLevel != DEBUG. This test requires DEBUG. Likely cause: --eris option")

    passed = False

    # Env var is automatically set in NvHostEngineApp
    nvhost_engine = apps.NvHostEngineApp()
    nvhost_engine.start(timeout=10)
    contents = None

    # Try for 5 seconds
    for i in range(25):
        time.sleep(0.2)
        with closing(open(nvhost_engine.dcgm_trace_fname, encoding='utf-8')) as f:
            # pylint: disable=no-member
            contents = f.read()
            logger.debug("Read %d bytes from %s" % (len(contents), nvhost_engine.dcgm_trace_fname))
            # This is checking two things:
            #   - that we are logging to the file specified in ENV
            #   - that we are setting severity according to ENV (DEBUG)
            if 'DEBUG' in contents:
                passed = True
                break

    # Cleaning up
    nvhost_engine.terminate()
    nvhost_engine.validate()

    errorString = ""
    if (not passed):
        if contents is not None:
            errorString = "Unable to find 'DEBUG' in log file"
        else:
            errorString = "log file %s was never read" % nvhost_engine.dcgm_trace_fname

    assert passed, errorString

@test_utils.run_with_logging_on()
def test_logging_modules():
    """
    Verifies that module logging is functional
    """

    if test_utils.loggingLevel != 'DEBUG':
        test_utils.skip_test("Detected logLevel != DEBUG. This test requires DEBUG. Likely cause: --eris option")

    PASSED = True
    FAILED = False

    result = FAILED

    nvhost_engine = apps.NvHostEngineApp()
    nvhost_engine.start(timeout=10)
    contents = None

    # Try for 5 seconds
    for i in range(25):
        time.sleep(0.2)
        with closing(open(nvhost_engine.dcgm_trace_fname, encoding="utf-8")) as f:
            # pylint: disable=no-member
            contents = f.read()
            logger.debug("Read %d bytes from %s" % (len(contents), nvhost_engine.dcgm_trace_fname))

            # NVSwitch module is loaded on startup. So we check for records from that module
            test_string = "Initialized logging for module 1"

            # Note that if --eris is passsed, we only log at WARNING level
            # If logging is not at DEBUG level, then skip the test
            if test_string in contents:
                result = PASSED
                break

    # Cleaning up
    nvhost_engine.terminate()
    nvhost_engine.validate()

    errorString = ""
    if (result != PASSED):
        if contents is not None:
            errorString = "Unable to find $test_string in log file"
        else:
            errorString = "log file %s was never read" % nvhost_engine.dcgm_trace_fname

    assert result == PASSED, errorString
