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
import dcgm_field_injection_helpers
import dcgm_fields
import dcgm_structs
import pydcgm

import argparse
import sys
import time


##############################################################################

# NOTE: Although DCGM supports injecting floating point field values, the argument parser currently only accepts
# integer values for the fieldValue argument

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fieldId', dest='fieldId', type=int, required=True)
parser.add_argument('-i', dest='gpuId', type=int, default=0)
parser.add_argument('-o', '--offset', dest='offset', type=float, default=0)
parser.add_argument('-v', '--value', dest='fieldValue', type=int, required=True)
parser.add_argument('-l', '--loop', dest='loop', action='store_true')
parser.add_argument('--interval', dest='interval', type=float)
parser.add_argument('--iterations', dest='iterations', type=int, 
                    help='Set to 0 to insert the given value until stopped via SIGINT')
args = parser.parse_args()

if args.loop and (args.interval is None or args.iterations is None):
    print("Must specify interval and iterations when looping")
    sys.exit(-1)

if args.iterations is not None and args.iterations < 0:
    print("iterations must be >= 0")
    sys.exit(-1)

handle = pydcgm.DcgmHandle(None, 'localhost', dcgm_structs.DCGM_OPERATION_MODE_AUTO)

if not args.loop:
    dcgm_field_injection_helpers.inject_value(handle.handle, args.gpuId, args.fieldId, args.fieldValue, args.offset)
    sys.exit(0)

# loop
try:
    i = 0
    while args.iterations == 0 or i < args.iterations:
        dcgm_field_injection_helpers.inject_value(handle.handle, args.gpuId, args.fieldId, args.fieldValue, args.offset)
        time.sleep(args.interval)
        i += 1
except KeyboardInterrupt:
    print("Exiting")

sys.exit(0)
