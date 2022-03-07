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
import fileinput 

whitelist = 'whitelist.txt'
tests = ['quick', 'long']
template = 'template.txt'

f = open(whitelist, 'r')
tuples = list()

for line in f:
    line = line.replace("\n", "")
    tuples.append(line)

f.close()

for test in tests:
    for tuple in tuples:
        splitTuple = tuple.split(", ")
        outFileName = splitTuple[0] + "_" + test + ".conf"
        outFileName = outFileName.replace(" ", "_")
        try:
            outFile = open(outFileName, 'w')
        except IOError as e:
            print "Unable to open %s for writing. Skipping." % outFileName
            continue

        for line in fileinput.input(template):
            if '%DEVICE%' in line:
                outFile.write (line.replace('%DEVICE%', splitTuple[0]))
            elif '%SETNAME%' in line:
                outFile.write (line.replace('%SETNAME%', "All " + splitTuple[0]))
            elif '%ID%' in line:
                outFile.write (line.replace('%ID%', splitTuple[1]))
            elif '%TEST%' in line:
                outFile.write (line.replace('%TEST%', test.capitalize()))
            else:
                outFile.write (line)
        outFile.close()


        
    
