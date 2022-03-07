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
# Find all of the function symbols in the passed-in ptx file and append them as variables to 
# the passed in output file
import sys

if len(sys.argv) < 3:
    print "USAGE: find_ptx_symbols.py <input.ptx> <output.h>\nThere must be two arguments supplied to this script"
    sys.exit(1)

ptxFilename = sys.argv[1]
outFilename = sys.argv[2]

ptxFp = open(ptxFilename, "rt")
outFp = open(outFilename, "at")

outFp.write("\n\n")

for line in ptxFp.readlines():
    if line.find(".entry") < 0:
        continue

    lineParts = line.split()
    funcName = lineParts[2][0:-1]

    outFp.write("const char *%s_func_name = \"%s\";\n" % (funcName, funcName))


