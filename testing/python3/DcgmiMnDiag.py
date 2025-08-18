#!/usr/bin/env python3
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

import subprocess
import nvidia_smi_utils
import utils
import test_utils

def trimJsonText(text):
    return text[text.find('{'):text.rfind('}') + 1]

logFile = "nvvs_diag.log"

class DcgmiMnDiag:
    def __init__(self, hostList=None, hostEngineAddress=None, testName=None, parameters=[], 
                 verbose=False, debugLevel=None, debugLogFile=None, dcgmiPrefix=""):
        self.hostList = hostList
        self.hostEngineAddress = hostEngineAddress
        self.testName = testName
        self.parameters = parameters
        self.verbose = verbose
        self.debugLevel = debugLevel
        self.debugLogFile = debugLogFile
        self.dcgmiPrefix = dcgmiPrefix

    def BuildDcgmiCommand(self):
        cmd = []

        if self.dcgmiPrefix:
            cmd.append("%s/dcgmi" % self.dcgmiPrefix)
        else:
            cmd.append("dcgmi")

        cmd.append("mndiag")

        # Add required host list if specified
        if self.hostList:
            cmd.append('--hostList')
            cmd.append(self.hostList)

        # Add test name if specified
        if self.testName:
            cmd.append('-r')
            cmd.append(self.testName)

        # Add host engine address if specified
        if self.hostEngineAddress:
            cmd.append('--hostEngineAddress')
            cmd.append(self.hostEngineAddress)

        # Add parameters if any
        for param in self.parameters:
            cmd.append('-p')
            cmd.append(param)

        # Add verbose flag
        if self.verbose:
            cmd.append("-v")

        # Add debug level if specified
        if self.debugLevel:
            cmd.append('-d')
            cmd.append(test_utils.DebugLevelToString(self.debugLevel))

        # Add debug log file if specified
        if self.debugLogFile:
            cmd.append('--debugLogFile')
            cmd.append(self.debugLogFile)

        return cmd

    def Run(self):
        cmd = self.BuildDcgmiCommand()
        self.__RunDcgmiMnDiag__(cmd)
    

    #####################################
    def __RunDcgmiMnDiag__(self, cmd):
        self.lastCmd = cmd
        self.lastStdout = ''
        self.lastStderr = ''

        nsc = nvidia_smi_utils.NvidiaSmiJob()
        nsc.start()

        print(f"Running command: {cmd}")
        runner = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdout_buf, stderr_buf) = runner.communicate()
        self.lastStdout = stdout_buf and stdout_buf.decode('utf-8')
        self.lastStderr = stderr_buf and stderr_buf.decode('utf-8')
        self.diagRet = runner.returncode
        nsc.m_shutdownFlag.set()
        nsc.join()

    def PrintLastRunStatus(self):
        print("Ran '%s' and got return code %d" % (self.lastCmd, self.diagRet))
        print("stdout: \n\n%s" % self.lastStdout)
        if self.lastStderr:
            print("\nstderr: \n\n%s" % self.lastStderr)
        else:
            print("\nNo stderr output")
