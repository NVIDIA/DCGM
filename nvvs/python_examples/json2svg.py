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
import pygal
from pygal.style import Style # pylint: disable=import-error
import sys
import getopt
import string
from pprint import pprint

inputfile = ''
outputfile = ''
keys = ''

def printUsage():
    print str(sys.argv[0]) + ' [-i <inputfile>] -o <outputfile> -k <keys (comma separated)>'

def parseArgs(argv):
    global inputfile
    global outputfile
    global keys
    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=","ofile=","keys="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    outputArg = False
    keysArg = False
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            outputArg = True
        elif opt in ("-k", "--keys"):
            keysArg = True
            keys = arg
    if not outputArg or not keysArg:
        printUsage()
        sys.exit()

def cleanup():
    global jsonFile

    if jsonFile is not sys.stdin:
        jsonFile.close()

if __name__ == "__main__":
   parseArgs(sys.argv[1:])

jsonFile  = open(inputfile) if inputfile is not "" else sys.stdin
jsonData = json.load(jsonFile)

keyList = keys.split(",")

gpusData = jsonData["gpus"]

custom_style = Style(
    colors=('#76B900', '#feed6c', '#8cedff', '#9e6ffe',
            '#899ca1', '#f8f8f2', '#bf4646', '#516083', '#f92672',
            '#82b414', '#fd971f', '#56c2d6', '#808384', '#8c54fe',
            '#465457'))

lineChart = pygal.Line(x_labels_major_every=10, show_minor_x_labels=False, show_dots=False, legend_font_size=10, \
                       legend_at_bottom=True, style=custom_style, include_x_axis=False)

try: 
    key = keyList[0]
    lineChart.x_labels = map(str, range(0, len(gpusData["0"][key])))
except KeyError:
    print 'Key \"' + key + '\" not found in JSON file.'

for gpu in gpusData:
    try:
        for key in keyList:
            line = list()
            secondaryAxis = False

            for entry in gpusData[gpu][key]:
                line.append(entry["value"]);
            if key == "gpu_temperature" or key == "power_violation":
                secondaryAxis = True

            lineChart.add(str(key) + ' ' + str(gpu), line, secondary=secondaryAxis)
    except KeyError:
        print 'Key \"' + key + '\" not found in JSON file.'

lineChart.render_to_file(outputfile)

cleanup()


