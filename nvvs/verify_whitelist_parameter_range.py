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
import re
import sys

def VerifyWhitelistParameterRanges(whitelistFile):
    pattern = r"(?P<value>[\d\.]+), (?P<min>[\d\.]+), (?P<max>[\d\.]+)"
    f = open(whitelistFile)
    lines = f.readlines()

    errorCount = 0
    print("Verifying parameter ranges in whitelist file...")
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            val = float(match.group('value'))
            min_val = float(match.group('min'))
            max_val = float(match.group('max'))
            if val < min_val or val > max_val:
                errorCount += 1
                print("Line %s: invalid range or value: %s" % (i+1, line.rstrip()))
    
    if errorCount:
        print("Errors found. Please fix errors before committing.")
        sys.exit(1)
    print("Success!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Script called with args: %s" % (sys.argv[1:]))
        print("Invalid arguments. Script should be called with path to whitelist file only.")
        sys.exit(1)
    VerifyWhitelistParameterRanges(sys.argv[1])
