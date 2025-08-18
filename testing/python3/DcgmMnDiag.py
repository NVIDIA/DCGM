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

from ctypes import *
from dcgm_structs import *
import dcgm_agent
import dcgm_structs

class DcgmMnDiag:
    """
    Python wrapper for DCGM Multi-Node Diagnostics
    """
    def __init__(self, hostList=None, hostEngineAddress=None, testName=None, parameters=[], 
                 handle=None, version=dcgm_structs.dcgmRunMnDiag_version1):
        """
        Constructor for DcgmMnDiag
        
        :param handle: DCGM Handle to use for API calls
        :param version: Version of the API to use
        """
        self.version = version
        self.handle = handle
        self.runMnDiagInfo = dcgm_structs.c_dcgmRunMnDiag_v1()
        self.hostEngineAddress = hostEngineAddress
        self.numHosts = 0
        self.numParams = 0
        self.runMnDiagInfo.version = version
        
        if hostList:
            for i, host in enumerate(hostList):
                if i >= dcgm_structs.DCGM_MAX_NUM_HOSTS:
                    break
                host_bytes = host.encode('utf-8')
                # Zero-fill the array first
                self.runMnDiagInfo.hostList[i][:] = b'\0' * dcgm_structs.DCGM_MAX_STR_LENGTH
                # Copy the host string
                self.runMnDiagInfo.hostList[i][:len(host_bytes)] = host_bytes
            self.numHosts = len(hostList)
            if self.numHosts > dcgm_structs.DCGM_MAX_NUM_HOSTS:
                raise ValueError(f"Cannot add more than {dcgm_structs.DCGM_MAX_NUM_HOSTS} hosts, got {self.numHosts}")
        
        if testName:
            self.runMnDiagInfo.testName = testName
        
        if parameters:
            for i, param in enumerate(parameters):
                if i >= dcgm_structs.DCGM_MAX_TEST_PARMS:
                    break
                param_bytes = param.encode('utf-8')
                self.runMnDiagInfo.testParms[i][:] = b'\0' * dcgm_structs.DCGM_MAX_TEST_PARMS_LEN_V2
                self.runMnDiagInfo.testParms[i][:len(param_bytes)] = param_bytes
            self.numParams = len(parameters)
            if self.numParams > dcgm_structs.DCGM_MAX_TEST_PARMS:
                raise ValueError(f"Cannot add more than {dcgm_structs.DCGM_MAX_TEST_PARMS} parameters, got {self.numParams}")

    def AddHost(self, hostname):
        """Add a host to the diagnostic run"""
        if len(hostname) >= dcgm_structs.DCGM_MAX_STR_LENGTH:
            raise ValueError(f"Hostname exceeds maximum length of {dcgm_structs.DCGM_MAX_STR_LENGTH}")
        
        # Convert string to bytes and copy to the correct host slot
        host_bytes = hostname.encode('utf-8')
        for i, b in enumerate(host_bytes):
            self.runMnDiagInfo.hostList[self.numHosts][i] = b
        
        # Null terminate the string
        self.runMnDiagInfo.hostList[self.numHosts][len(host_bytes)] = 0
        self.numHosts += 1

        if self.numHosts > dcgm_structs.DCGM_MAX_NUM_HOSTS:
            raise ValueError(f"Cannot add more than {dcgm_structs.DCGM_MAX_NUM_HOSTS} hosts, got {self.numHosts}")
        
    def SetTestName(self, testName):
        """Set the test name"""
        if len(testName) >= dcgm_structs.DCGM_MAX_STR_LENGTH:
            raise ValueError(f"Test name exceeds maximum length of {dcgm_structs.DCGM_MAX_STR_LENGTH}")
        self.runMnDiagInfo.testName = testName

    def AddParameter(self, parameterStr):
        """Add a parameter to the diagnostic run"""
        if len(parameterStr) >= dcgm_structs.DCGM_MAX_TEST_PARMS_LEN:
            raise ValueError(f"Parameter exceeds maximum length of {dcgm_structs.DCGM_MAX_TEST_PARMS_LEN}")

        # Convert string to bytes and copy to the correct parameter slot
        param_bytes = parameterStr.encode('utf-8')
        for i, b in enumerate(param_bytes):
            self.runMnDiagInfo.testParms[self.numParams][i] = b
            
        # Null terminate the string
        self.runMnDiagInfo.testParms[self.numParams][len(param_bytes)] = 0
        self.numParams += 1

        if self.numParams > dcgm_structs.DCGM_MAX_TEST_PARMS:
            raise ValueError(f"Cannot add more than {dcgm_structs.DCGM_MAX_TEST_PARMS} parameters, got {self.numParams}")

    def GetStruct(self):
        return self.runMnDiagInfo
    
    def Execute(self, handle):
        """Execute the diagnostic"""
        if self.numHosts == 0:
            raise ValueError("No hosts specified for diagnostic")
        return dcgm_agent.dcgmRunMnDiagnostic(handle, self.runMnDiagInfo)

