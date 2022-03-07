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


class InjectionThread(threading.Thread):
    """
    Thread for injecting values into DCGM.
    """
    def __init__(self, handle, gpuId, fieldId, value, offset=0, interval=0.1, iterations=0, isInt=True):
        """
        Initialize the thread.
        If 'iterations' is 0, thread inserts values (every 'interval' seconds) until the 'Stop()' method
        is called (the thread may take up to 'interval' seconds to stop after the method is called).
        Otherwise, thread inserts values (every 'interval' seconds) until values have been inserted
        'iterations' number of times.

        :param dcgmHandle_t* handle: Handle to the host engine
        :param int gpuId: The GPU ID for which the values should be inserted.
        :param int fieldId: The field for which the values should be inserted.
        :param (float or int) value: The value to insert.
        :param float offset: Offset (in seconds) for the inserted values, defaults to 0, optional
        :param float interval: The interval (in seconds) at which values should be inserted, defaults to 0.1, optional
        :param int iterations: The number of values to insert, defaults to 0 (insert until Stop() is called), optional
        :param bool isInt: Whether the value to insert is an integer value, defaults to True, optional
        """
        super(InjectionThread, self).__init__()
        self._handle = handle
        self._gpuId = gpuId
        self._injections = []
        self._injections.append((fieldId, value))
        self._offset = offset
        self._interval = interval
        if self._interval < 0.05:
            self._interval = 0.05
        self._iterations = iterations
        self._isInt = isInt
        self._stopRequested = threading.Event()
        self.retCode = None

    def AddInjectedValue(self, fieldId, value):
        self._injections.append((fieldId, value))
        self._interval = 0.04

    def Stop(self):
        """
        Requests the thread to stop inserting values. Thread may continue running for up to 'interval' seconds
        depending on when it checks for the stop request.
        """
        self._stopRequested.set()

    def ShouldStop(self):
        return self._stopRequested.is_set()

    def run(self):
        i = 0
        while i < self._iterations or not self.ShouldStop():
            for pair in self._injections:
                self.retCode = inject_value(self._handle, self._gpuId, pair[0], pair[1], self._offset,
                                             self._isInt, False)
                if self.retCode != dcgm_structs.DCGM_ST_OK:
                    return
            time.sleep(self._interval)
            i += 1


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


## Injection helpers
def inject_value(handle, entityId, fieldId, value, offset, isInt=True, verifyInsertion=True,
                 entityType=dcgm_fields.DCGM_FE_GPU):
    if isInt:
        ret = inject_field_value_i64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)
    else:
        ret = inject_field_value_fp64(handle, entityId, fieldId, value, offset, entityGroupId=entityType)

    if verifyInsertion:
        assert ret == dcgm_structs.DCGM_ST_OK, "Could not inject value %s in field id %s" % (value, fieldId)
    return ret

def inject_field_value_i64(handle, entityId, fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()+offset) * 1000000.0)
    field.value.i64 = value

    return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, entityGroupId, entityId, field)

def inject_field_value_fp64(handle, entityId, fieldId, value, offset, entityGroupId=dcgm_fields.DCGM_FE_GPU):
    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_DOUBLE)
    field.ts = int((time.time()+offset) * 1000000.0)
    field.value.dbl = value

    return dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, entityGroupId, entityId, field)

STANDALONE_BLACKLIST_SCRIPT_NAME = "blacklist_recommendations.py"
def createBlacklistApp(numGpus=None, numSwitches=None, testNames=None, instantaneous=False):
    args = ["./%s" % STANDALONE_BLACKLIST_SCRIPT_NAME]
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
