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
import json
import csv
import sys
import getopt
import string

inputfile = ''
outputfile = ''
keys = ''

def printUsage():
    print str(sys.argv[0]) + ' [-i <inputfile>] [-o <outputfile>] -k <keys (comma separated)>'

def parseArgs(argv):
    global inputfile
    global outputfile
    global keys
    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=","ofile=","keys="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    keyArg = False
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-k", "--keys"):
            keys = arg
            keyArg = True
    if not keyArg:
        printUsage()
        sys.exit()

def cleanup():
    global jsonFile
    global outHandle

    if jsonFile is not sys.stdin:
        jsonFile.close()

    if outHandle is not sys.stdout:
        outHandle.close()

if __name__ == "__main__":
   parseArgs(sys.argv[1:])

jsonFile  = open(inputfile) if inputfile is not "" else sys.stdin
jsonData = json.load(jsonFile)

outHandle = open(outputfile, 'wb') if outputfile is not ""  else sys.stdout 
csvWriter = csv.writer(outHandle, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=",")

keyList = keys.split(",")

gpusData = jsonData["gpus"]

header = ["GPU#", "time"]
for key in keyList:
    header.append(str(key))

csvWriter.writerow(header)

for gpu in gpusData:
    try:
        key = keyList[0]
        for i in range(len(gpusData[gpu][key])):
            row = [gpu]
            row.append(str(i))
            for key in keyList:
                entry = gpusData[gpu][key][i]
                row.append(str(entry["value"]))
            csvWriter.writerow(row)

    except KeyError:
        print 'Key \"' + key + '\" not found in JSON file.'

cleanup()


