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

import os
import subprocess

FIND_PTX_FILENAME = "find_ptx_symbols.py"
CUDA_IMAGE = "nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04"

###################################
"""
This function gets the template defined for each build script
"""
def getBuildScriptTemplate(arch, cuFileName, buildPTXName, buildHeaderName, addPythonLine=0, pythonLineArg="", compute=""):
    pythonLine = ""
    if addPythonLine==1:
        if pythonLineArg != "":
            pythonLine = pythonLineArg.lstrip()
        else:
            pythonLine = "python find_ptx_symbols.py"

    computeValue = ""
    if compute!="":
        computeValue = "--gpu-architecture " + compute

    buildScriptContent = """
#!/bin/bash

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

# This script generates {1} and {2}
# {2} is then converted to {3} as a hexified string
# Last, symbol names are added to {3} by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer

CUDA_IMAGE=nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04
docker run \\
    --rm \\
    -v $(pwd):/work \\
    -w /work \\
    ${{CUDA_IMAGE}} \\
    /bin/bash -c "/usr/local/cuda/bin/nvcc -ptx -m64 {5} -arch={0} -o {2} {1} || die 'Failed to compile {1}'; \\
                    /usr/local/cuda/bin/bin2c {2} --padd 0 --name {3} > {3}.h; chmod a+w {3}.h"
{4}
    """.format(arch, cuFileName, buildPTXName, buildHeaderName, pythonLine, computeValue)

    return buildScriptContent

###################################
"""
returns all the paths for given extension
"""
def getPaths(rootDir, extension):
    cuDir = []
    folderName = rootDir.split("/")[-1]
    for root, _, files in os.walk(rootDir):
        # ignore certain folders
        if "_out" not in root and "testing" not in root:
            if any(file.endswith(extension) for file in files):
                cuDir.append(folderName + "/" + os.path.relpath(root, rootDir))
    return cuDir

###################################
"""
return the cuda filename given in a directory
"""
def getCuFileName(dir):
    files = os.listdir(dir)
    filteredFiles = [file for file in files if file.endswith(".cu")]
    return filteredFiles[0]
    
###################################
"""
returns the path of the build script given in a directory
"""
def getPathOfBuildScript(dir):
    path = ""

    # List all files in the directory
    files = os.listdir(dir)

    # Search for .sh file with "build" in its name
    for file in files:
        if file.endswith(".sh") and "build" in file:
             path = os.path.join(dir, file)

    return path

###################################
"""
check if the dictionary has empty values and if yes add default values
"""
def checkAndUpdateToDefault(parsedDic, cuFileName, arch):
    if "arch" not in parsedDic.keys():
        parsedDic["arch"] = arch
    if "ptx" not in parsedDic.keys():
        parsedDic["ptx"] = cuFileName[:-3] + ".ptx"
    if "cu" not in parsedDic.keys():
        parsedDic["ptx"] = cuFileName
    if "pythonLine" not in parsedDic.keys():
        parsedDic["pythonLine"] = ""
    if "compute" not in parsedDic.keys():
        parsedDic["compute"] = ""

###################################
"""
reformat existing build script for uniformity 
"""
def reformatBuildScript(path, cuFileName, arch):

    bin2cExists = 0
    addPythonLine = 0
    
    with open(path, "r") as file:
        lines = [line.rstrip() for line in file.readlines()]

    parsedDic = {}

    for line in lines:
        # ignoring all comments in the line

        if not line.startswith("#"):
            if "nvcc" in line:
                curLine = line.split(" ")
                
                # get the arch value, ptx filname and .cu filename
                for l in curLine:
                    if "arch=" in l and "arch" not in parsedDic.keys():
                        parsedDic["arch"] = l.split("=")[-1]
                        if "dcgmproftester" in cuFileName.lower():
                            parsedDic["arch"] = arch
                    elif ".ptx" in l and "ptx" not in parsedDic.keys():
                        parsedDic["ptx"] = l
                    elif ".cu" in l and "cu" not in parsedDic.keys():
                        parsedDic["cu"] = l
                    elif "compute" in l.lower() and "compute" not in parsedDic.keys():
                        parsedDic["compute"] = l.split("=")[-1]
                
            elif "bin2c" in line:
                # get the .h file
                curLine = line.split(" ")
                headerName = [word for word in curLine if ".h" in word]

                # handle if header file not parsed correctly or if the name not found in the existing script 
                if len(headerName) != 0:
                    headerName = headerName[0]
                    if not headerName.endswith(".h"):
                        headerName = headerName[:headerName.rfind(".")]
                    else:
                        headerName = headerName[:-2]
                else: 
                    headerName = parsedDic["cu"] + "_ptx_string"
                
                parsedDic["header"] = headerName
                bin2cExists = 1

            elif "python" in line:
                addPythonLine = 1
                parsedDic["pythonLine"] = line

    if bin2cExists == 0:
        # add header file name
        parsedDic["header"] = parsedDic["cu"][:-3] + "_ptx_string"

    # check if parsedDic was updated, and if not append with default values
    checkAndUpdateToDefault(parsedDic, cuFileName, arch)

    # get the content from template
    buildScriptContent = getBuildScriptTemplate(parsedDic["arch"], parsedDic["cu"], parsedDic["ptx"], parsedDic["header"], addPythonLine, parsedDic["pythonLine"])

    # update the script
    with open(path, 'w') as buildScriptFile:
        buildScriptFile.write(buildScriptContent)
    
    return parsedDic["ptx"], parsedDic["header"], addPythonLine


###################################
"""
check if ptx and header exists
"""
def filesExists(dir, filename):
    value = os.path.exists(os.path.join(dir, filename))
    return value


###################################
def generatePTXFile(dir, arch, cuFileName, ptxFileName): 
    # get initial dir
    initialDir = os.getcwd()

    # change dir to current dir
    os.chdir(dir)

    command = "docker run --rm -v \"$(pwd)\":/work -w /work {0} /bin/bash -c  '/usr/local/cuda/bin/nvcc -ptx -m64 -arch={1} -o {2} {3}' ".format(CUDA_IMAGE, arch, ptxFileName, cuFileName)
    result = subprocess.run(command, shell=True, check=False, stderr=subprocess.PIPE)

    # change back to original dir
    os.chdir(initialDir)

    if result.returncode == 0:
        print("Command output:")
        print(result.stdout.decode("utf-8"))
        return 0
    else:
        print("Error executing command:")
        print(result.stderr.decode("utf-8"))
        return 1


###################################
def generateHeaderFile(dir, ptxFileName, headerFileName):
    # get initial dir
    initialDir = os.getcwd()

    # change dir to current dir
    os.chdir(dir)

    command = "/usr/local/cuda/bin/bin2c {0} --padd 0 --name {1} > {1}.h; chmod a+w {1}.h".format(ptxFileName, headerFileName)
    result = subprocess.run(command, shell=True, check=False, stderr=subprocess.PIPE)

    # change back to original dir
    os.chdir(initialDir)

    if result.returncode == 0:
        print("Command run successfully")
        return 0
    else:
        print("Error executing command:")
        print(result.stderr.decode("utf-8"))
        return 1

###################################
"""
generate new build scripts
"""
def generateBuildScripts(dir, cuFileName, arch="sm_30"):
    pathOfBuildScript = ""

    # file names
    buildScriptName = "build_ptx_string.sh"
    buildPTXName = cuFileName[:-3] + ".ptx"
    buildHeaderName = cuFileName[:-3] + "_ptx_string"

    # get the code template
    buildScriptContent = getBuildScriptTemplate(arch, cuFileName, buildPTXName, buildHeaderName)

    # get the path of for the build script
    pathOfBuildScript = os.path.join(dir, buildScriptName)

    # write the code to the file
    with open(pathOfBuildScript, 'w') as buildScriptFile:
        buildScriptFile.write(buildScriptContent)

    return pathOfBuildScript, buildPTXName, buildHeaderName

###################################
"""
run given .sh file
"""
def runBuildScript(dir, filePath, buildPTXName, buildHeaderName):
    generatedPaths = {
        "ptx":"",
        "header": ""
    }

    # get initial dir
    initialDir = os.getcwd()

    # change dir to current dir
    os.chdir(dir)

    # execution permission
    fileName = filePath.split("/")[-1]
    os.chmod(fileName, 0o755)

    # execution of script
    result = subprocess.run(["sh", fileName], stderr=subprocess.PIPE)

    # change back to original dir
    os.chdir(initialDir)

    # script executed successfully 
    if result.returncode == 0 and result.stderr==b'':
        # get the ptx file and header file paths
        files = os.listdir(dir)
        for file in files:
            if file.endswith(".ptx") and buildPTXName in file:
                generatedPaths["ptx"] = os.path.join(dir, file)
            elif file.endswith(".h") and buildHeaderName in file:
                generatedPaths["header"] = os.path.join(dir, file)
    else:
        print(result.stderr)
        print("\n######################### Build failed due to compilation error ##########################")
        

    return generatedPaths
    
###################################
"""
parse the generate ptx file and extract values
"""
def parsePTXFile(ptxFilePath):
    # read the file
    ptxFp = open(ptxFilePath, "rt")

    # get the lines/parsed value
    parsedValue = []
    for line in ptxFp.readlines():
        if line.find(".entry") < 0:
            continue

        lineParts = line.split()
        funcName = lineParts[2][0:-1]
        parsedValue.append("const char *%s_func_name = \"%s\";\n" % (funcName, funcName))
    
    # close the file
    ptxFp.close()

    return parsedValue

###################################
"""
update the generated header file with the values from the parsed ptx
"""
def updateHeaderFile(parsedValue, headerFilePath):
    # open the header file
    headerFp = open(headerFilePath, "at")

    # add the lines from parsedValue to the .h file
    headerFp.write("\n\n")
    for value in parsedValue:
        headerFp.write(value)

    # close the file 
    headerFp.close()

###################################
"""
generate a py file which can be used independently to parse ptx file and update header
"""
def createPythonParserForPtxFile(dir, buildPTXName, buildHeaderName):
    fileName = FIND_PTX_FILENAME

    # code to be written to the python file
    code = """
ptxFilename = "{0}"
outFilename = "{1}.h"
ptxFp = open(ptxFilename, "rt")
outFp = open(outFilename, "at")

outFp.write("\\n\\n")

for line in ptxFp.readlines():
    if line.find(".entry") < 0:
        continue

    lineParts = line.split()
    funcName = lineParts[2][0:-1]

    outFp.write("const char *%s_func_name = \\"%s\\";\\n" % (funcName, funcName))
    """.format(buildPTXName, buildHeaderName)
    
    # open the file in write mode
    with open(dir + "/" + fileName, "w") as f:
        f.write(code)

###################################
"""
update the python run cmd in build script
"""
def updateBuildScript(path):
    newLine = "python find_ptx_symbols.py"
    try:
        # Open the .sh file in append mode
        with open(path, 'a') as scriptFile:
            # Add the new line
            scriptFile.write(newLine + '\n')
        print("New line added to the build script successfully.")
    except FileNotFoundError:
        print(f"File not found at {path}. Please provide a valid path.")
    except Exception as e:
        print(f"An error occurred: {e}")

###################################
"""
main function to iterate and check all build scripts
"""
def normaliseAllFolders():
    # local vars
    cuDir = []
    rootDirectory = os.getcwd() + "/dcgm"

    # search for all directories which have .cu file
    cuDir = getPaths(rootDirectory, ".cu")

    for dir in cuDir:
        cuFileName = ""
        buildPresent = 0
        generatedPaths = {
            "ptx":"",
            "header": ""
        }

        # getting the .cu file name
        cuFileName = getCuFileName(dir)
        print("1) Cuda file: " + cuFileName)

        # setting archtecture value based on condition - DCGM-3733
        if "dcgmproftester" in cuFileName.lower():
            arch = "sm_70"
        else:
            arch = "sm_30"

        # check if build.sh present in the folders
        # generate it if does not exist
        # get path if exists
        if (getPathOfBuildScript(dir) == ""):
            # if build.sh not present, generate it
            print("2) Build script does not exist, generating it")
            pathOfBuild, buildPTXName, buildHeaderName = generateBuildScripts(dir, cuFileName, arch)
            print("Generated script path: " + pathOfBuild)

        else:
            print("2) Build script exists, parsing it and reformating")
            buildPresent = 1
            pathOfBuild = getPathOfBuildScript(dir)

            # parse the build script file, compare with the template and reparse it
            buildPTXName, buildHeaderName, pythonAdded = reformatBuildScript(pathOfBuild, cuFileName, arch)
            print("Script path: " + pathOfBuild)

        # check if .ptx and .h do not exist 
        print("3)")
        if not filesExists(dir, buildPTXName) and filesExists(dir, buildHeaderName+".h"):
            # generate .ptx file
            successRun = generatePTXFile(dir, arch, cuFileName, buildPTXName)
            if successRun == 0:
                generatedPaths["ptx"] = dir + "/" + buildPTXName
                generatedPaths["header"] = dir + "/" + buildHeaderName + ".h"

        elif filesExists(dir, buildPTXName) and not filesExists(dir, buildHeaderName+".h"):
            # generate .h file
            successRun = generateHeaderFile(dir, buildPTXName, buildHeaderName)
            if successRun == 0:
                generatedPaths["ptx"] = dir + "/" + buildPTXName
                generatedPaths["header"] = dir + "/" + buildHeaderName + ".h"

        elif not filesExists(dir, buildPTXName) and not filesExists(dir, buildHeaderName+".h"):
            # run the build.sh and generate .ptx file and .h file
            print("Running the build file")
            generatedPaths = runBuildScript(dir, pathOfBuild, buildPTXName, buildHeaderName)

        else:
            print("PTX Path: " + dir + "/" + buildPTXName)
            print("Header Path: " + dir + "/" + buildHeaderName + ".h")
            generatedPaths["ptx"] = dir + "/" + buildPTXName
            generatedPaths["header"] = dir + "/" + buildHeaderName + ".h"

        if (buildPresent == 0 or (buildPresent == 1 and pythonAdded == 0)) and (generatedPaths["ptx"] != "" or generatedPaths["header"] != ""):
            print("4) Parsing the PTX file to update header")
            print("PTX file: " + buildPTXName)
            print("Header file: " + buildHeaderName)

            # parse the .ptx file, update the .h file
            parsedValue = parsePTXFile(generatedPaths["ptx"])
            updateHeaderFile(parsedValue, generatedPaths["header"])

            # create .py file to parse ptx file
            createPythonParserForPtxFile(dir, buildPTXName, buildHeaderName)

            # append ptx file parser to build script
            updateBuildScript(pathOfBuild)

        print("################################################################################################")
    print("Done!")

    
if __name__ == '__main__':
    normaliseAllFolders()