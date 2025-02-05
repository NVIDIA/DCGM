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
#
# Filter out lines in text files that are version dependant
# This is for filtering text in man pages, bindings, etc
#
import getopt
import sys
import datetime

# globals
hIn = None
hOut = None
vMajor = None
vMinor = None

now = datetime.datetime.now()

def CompareField(v1, v2):
    isNum1 = v1.isdigit()
    isNum2 = v2.isdigit()

    # if both are numbers, higher numbers win
    if isNum1 and isNum2:
        return cmp(int(v1), int(v2))

    # if one is a string and one is a number, the number wins
    if isNum1 and not isNum2:
        return 1
    if not isNum1 and isNum2:
        return -1

    # otherwise, use strcmp()
    return cmp(v1, v2)

def VersionCompare(v1, v2):
    v1Fields = v1.split('.')
    v2Fields = v2.split('.')
    for (v1Field, v2Field) in zip(v1Fields, v2Fields):
        diff = CompareField(v1Field, v2Field)
        if diff != 0:
            return diff
    # If one version is longer than the other, the longer one wins
    return len(v1) - len(v2)

def VersionMatch(opStr, v1, v2):
    vDiff = VersionCompare(v1, v2)
    if '==' == opStr:
        return vDiff == 0
    elif '>=' == opStr:
        return vDiff >= 0
    elif '<=' == opStr:
        return vDiff <= 0
    elif '<' == opStr:
        return vDiff < 0
    elif '>' == opStr:
        return vDiff > 0
    else:
        print '"%s": operation string is unknown [==, <=, >=,  <, >]' % opStr
        exit(1)

# determine if lines should be shown or hidden
def GetLineMode(line):
    parts = line.split("&&&") # ['', '_DCGM_VERSION_IF_', '>= 2.0']
    stmt = parts[2].strip().split() # ['>=', '2.0']
    op = stmt[0]
    lineVersion = stmt[1]
    if (VersionMatch(op, version, lineVersion)):
        return 'show'
    else:
        return 'hide'

def ShowLine(line):
    # convert all DCGM_VERSION_DEFINEs to the version number
    truncatedVersion = ".".join(version.split('.')[:2])
    line = line.replace("&&&_DCGM_VERSION_DEFINE_&&&", truncatedVersion)

    # convert all DATE to the current date
    line = line.replace("&&&_CURRENT_DATE_&&&", ("%d/%d/%d" % (now.year, now.month, now.day)))
    line = line.replace("&&&_CURRENT_YEAR_&&&", ("%d" % (now.year)))
    
    # Error on any DCGM_VERSION_ERRORs
    if line.strip().startswith("&&&_DCGM_VERSION_ERROR_&&&"): # &&&_DCGM_VERSION_ERROR %%
        print line
        exit(1)
    
    hOut.write(line)
    

def CheckForUnexpectedTokens(line):
    if ("&&&_" in line) and ("_&&&" in line) and (not ("&&&_DCGM_VERSION_DEFINE_&&&" in line)) and (not ("&&&_DCGM_VERSION_ERROR_&&&" in line)) \
            and (not ("&&&_CURRENT_DATE_&&&" in line)) and (not ("&&&_CURRENT_YEAR_&&&" in line)):
        print '"%s": looks like a version filter token, but is malformed or misused.' % line
        exit(1)

# Read and filter according to the versioning tags
def FilterFile():
    lineMode = 'default'
    for line in hIn:
        # determine if the line should be printed
        if 'default' == lineMode:
            if line.strip().startswith("&&&_DCGM_VERSION_IF_&&&"): # &&&_DCGM_VERSION_IF %% >= 2.0
                lineMode = GetLineMode(line)
            else:
                CheckForUnexpectedTokens(line)
                ShowLine(line) # the default behavior is to show the line
        else:
            # inside a DCGM_VERSION_IF block
            if line.strip().startswith("&&&_DCGM_VERSION_ELSE_IF_&&&"): # &&&_DCGM_VERSION_ELSE_IF %% >= 2.0
                if ('show' == lineMode or 'ifdone' == lineMode):
                    lineMode = 'ifdone' # already shown, ignore rest of if block
                else:
                    lineMode = GetLineMode(line)
            elif line.strip().startswith("&&&_DCGM_VERSION_ELSE_&&&"): # &&&_DCGM_VERSION_ELSE %%
                # The else shows lines when the if has not been completed (linemode = !ifdone) and all previous
                # conditionals have not shown lines (linemode = hide)
                if ('hide' == lineMode):
                    lineMode = 'show'
                else:
                    lineMode = 'ifdone'
            elif line.strip().startswith("&&&_DCGM_VERSION_END_IF_&&&"): # &&&_DCGM_VERSION_END_IF %%
                # exit the block
                lineMode = 'default'
            elif 'show' == lineMode:
                CheckForUnexpectedTokens(line)
                ShowLine(line)
            elif 'hide' == lineMode or 'ifdone' == lineMode:
                CheckForUnexpectedTokens(line)
                # ignore this line
            else:
                print '"%s": is not a valid mode. [default, show, hide, ifdone]' % lineMode
                exit(1)

def usage(code=1):
    print "python version_filter.py -v 2 infile.txt outfile.txt"
    exit(code)

def main():
    global hIn
    global hOut
    global version

    opts, args = getopt.getopt(sys.argv[1:], 'v:')

    version = ""

    if len(args) != 2:
        usage()
    else:
        inFile = args[0]
        outFile = args[1]
    
    for o, a in opts:
        if "-v" == o:
            version = a
        else:
            usage()

    if "" == version:
        usage()

    hIn = open(inFile, 'r')
    hOut = open(outFile, 'w')
    
    FilterFile()

#
main()
