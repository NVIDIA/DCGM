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
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
import dcgm_structs
import logger
import subprocess

import os
import time
import threading
import sys

from apps.app_runner import AppRunner
from DcgmReader import DcgmReader
from dcgm_field_injection_helpers import inject_value


class FieldReader(DcgmReader):

    def __init__(self, expectedValue, desiredNumMatches, *args, **kwargs):
        super(FieldReader, self).__init__(*args, **kwargs)
        self._expectedValue = expectedValue
        self._desiredNumMatches = desiredNumMatches
        self._mostRecentTs = 0
        self.numMatchesSeen = 0
        self.passed = False

    def CustomFieldHandler(self, gpuId, fieldId, fieldTag, val):
        """
        This method is called once for each field for each GPU each
        time that its Process() method is invoked, and it will be skipped
        for blank values and fields in the ignore list.

        fieldTag is the field name, and val is a dcgm_field_helpers.DcgmFieldValue instance.
        """
        if val.ts > self._mostRecentTs:
            self._mostRecentTs = val.ts
        else:
            return
        if val.value == self._expectedValue:
            self.numMatchesSeen += 1
            if self.numMatchesSeen == self._desiredNumMatches:
                self.passed = True
                return

STANDALONE_DENYLIST_SCRIPT_NAME = "denylist_recommendations.py"
def createDenylistApp(numGpus=None, numSwitches=None, testNames=None, instantaneous=False):
    args = ["./%s" % STANDALONE_DENYLIST_SCRIPT_NAME]
    if numGpus == None or numSwitches == None:
        args.append("-d")
    else:
        args.append("-g")
        args.append(str(numGpus))
        args.append("-s")
        args.append(str(numSwitches))

    if instantaneous:
        args.append("-i")
    elif testNames:
        args.append("-r")
        args.append(testNames)
    else:
        args.append("-r")
        args.append("memory bandwidth")

    return AppRunner(sys.executable, args)


## Helper for verifying inserted values
STANDALONE_VALUE_VERIFICATION_SCRIPT_NAME = "verify_field_value.py"
def verify_field_value(gpuId, fieldId, expectedValue, maxWait=2, checkInterval=0.1, numMatches=3):
    """
    Verify that DCGM sees the expected value for the specified field ID. Waits a maximum of maxWait seconds to see
    the given value.

    - numMatches is the number of times the expected value must be seen before the verification is considered successful.
    - checkInterval is the update interval for the given fieldId in seconds (this is also roughly the interval
      at which the value test is performed).

    Returns True on successful verifcation and False otherwise.
    """
    interval_in_usec = int(checkInterval * 1000000)
    fr = FieldReader(expectedValue, numMatches, fieldIds=[fieldId], updateFrequency=interval_in_usec, gpuIds=[gpuId])

    start = time.time()
    while (time.time() - start) < maxWait:
        fr.Process()
        if fr.passed:
            return True
        time.sleep(checkInterval)

    # If we were unable to see the expected value, log how many times we did see it before failing
    logger.info("Saw expected value %s (for field %s) %s times" % (expectedValue, fieldId, fr.numMatchesSeen))
    return False

# Verifies that the NVVS process is running. This is used to make tests more deterministic instead of sleeping
# and hoping that the NVVS process has started at the end of the sleep
def check_nvvs_process(want_running, delay=0.5, attempts=20):
    """
    Checks status of nvvs process.
    If want_running is True, method returns True if nvvs is running.
    If want_running is False, method returns True if nvvs is NOT running.
    """
    retry_count = 0
    debug_output = ''
    while retry_count < attempts:
        retry_count += 1
        time.sleep(delay) # delay for a bit before trying again
        try:
            # If pgrep is run via a shell, there will be extraneous output caused by the shell command itself
            debug_output = subprocess.check_output(["pgrep", "-l", "-f", "apps/nvvs/nvvs"])
            if want_running:
                return True, debug_output
        except subprocess.CalledProcessError:
            if not want_running:
                return True, debug_output

    return False, debug_output
