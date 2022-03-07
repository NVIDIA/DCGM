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
from DcgmReader import DcgmReader
import dcgm_fields

import argparse
import sys
import time

##############################################################################
# Parse arguments
parser = argparse.ArgumentParser(
    description="""
Verifies that DCGM reports the expected value for the specified field id and GPU.
Waits a maximum of maxWait seconds to see the expectedValue for the given field. 
The expectedValue must be reported by DCGM at least numMatches times for the check to be considered successful.
Returns 0 on success and prints "Passed" to stdout.
Returns 20 on failure and prints "Failed" to stdout.
"""
)
parser.add_argument('-f', '--fieldId', type=int, required=True)
parser.add_argument('-v', '--expectedValue', type=int, required=True, help='The expected value for the field')
parser.add_argument('-i', '--gpuId', type=int, required=True)
parser.add_argument('-w', '--maxWait', type=float, required=True, 
                    help='The maximum number of seconds the script should wait for the expected value before failing')
parser.add_argument('--checkInterval', type=float, required=False, default=0.5, 
                    help='How often the field value should be updated in seconds')
parser.add_argument('-n', '--numMatches', type=int, required=False, default=3, 
                    help='The number of occurences of expected value to look for before treating it as a success')

args = parser.parse_args()

##############################################################################
# Constants
RET_VALUE_PASS = 0
RET_VALUE_FAILED = 20

# Global vars
MOST_RECENT_TS = 0
NUM_MATCHES = 0
PASSED = False

class FieldReader(DcgmReader):
    def CustomFieldHandler(self, gpuId, fieldId, fieldTag, val):
        '''
        This method is called once for each field for each GPU each 
        time that its Process() method is invoked, and it will be skipped
        for blank values and fields in the ignore list.

        fieldTag is the field name, and val is a dcgm_field_helpers.DcgmFieldValue instance.
        '''
        global MOST_RECENT_TS
        global NUM_MATCHES
        global PASSED
        if val.ts > MOST_RECENT_TS:
            MOST_RECENT_TS = val.ts
        else:
            return
        if val.value == args.expectedValue:
            NUM_MATCHES += 1
            if NUM_MATCHES == args.numMatches:
                PASSED = True
                return

def main():
    interval_in_usec = int(args.checkInterval * 1000000)
    fr = FieldReader(fieldIds=[args.fieldId], updateFrequency=interval_in_usec, gpuIds=[args.gpuId])

    start = time.time()
    while True:
        fr.Process()
        if PASSED:
            print("Passed")
            return RET_VALUE_PASS
        if (time.time() - start > args.maxWait):
            print("Failed")
            return RET_VALUE_FAILED
        time.sleep(args.checkInterval)

if __name__ == "__main__":
    sys.exit(main())
