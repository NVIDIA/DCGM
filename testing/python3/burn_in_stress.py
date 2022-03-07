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
import time
import datetime
import inspect
import ctypes
import platform
import string
import os
import sys
import argparse
import _thread
import threading
import test_utils
import utils
import option_parser
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import dcgmvalue
import dcgm_fields
import DcgmiDiag
import nvidia_smi_utils

import subprocess
import shlex
import pydcgm
import apps
import logger
import json
from datetime import date, timedelta
from subprocess import Popen, PIPE, STDOUT, check_output, check_call, CalledProcessError
import version

# Initializing logger
logger.log_dir = os.path.join(os.getcwd(), "_out_runLogs")
script_dir = os.path.realpath(sys.path[0])

# Global variables
DEFAULT_POWER_LIMIT = 0
MAX_POWER_LIMIT = 1
MIN_POWER_LIMIT = 2

TOTAL_TEST_PASSED = 0
TOTAL_TEST_FAILED = 0
TOTAL_TEST_WAIVED = 0
TOTAL_TEST_COUNT = 0
TOTAL_TEST_CYCLES = 0

def get_dcgmi_bin_directory():
    """
    Function to return the directory where dcgmi is expected 

    Example: apps/amd64
    """
    path = ""

    if platform.machine() == 'x86_64' and platform.system() == 'Linux':
        path = "apps/amd64"

    elif platform.machine() == 'aarch64' and platform.system() == 'Linux':
        path = "apps/aarch64"

    elif platform.machine() == 'ppc64le' and platform.system() == 'Linux':
        path = "apps/ppc64le"
    else:
        print("Unsupported platform. Please modify get_dcgmi_bin_directory()")
        sys.exit(1)

    return path

def get_dcgmi_bin_path():
    """
    Function to figure out what dcgmi binary to use based on the platform
    """

    # Including future supported architectures
    return get_dcgmi_bin_directory() + "/dcgmi"

dcgmi_absolute_path = os.path.join(script_dir, get_dcgmi_bin_path())

def get_current_field_group_id(hostIP):
    data = {}
    field_group_id = ""
    try:
        cmd = "%s fieldgroup --host %s -l" % (dcgmi_absolute_path, hostIP)
        output_buf = check_output(cmd, shell=True)
        output = output_buf.decode('utf-8')
        field_group_id = output.splitlines()[-4].split()[-2]  # Parser output to get the latest fieldgroup id
    except CalledProcessError:
        print("Unable to get fieldgroup ID\n")
    return field_group_id


def get_field_ids(hostIP, fieldGroupID):
    data = {}
    field_ids = ""
    field_group_name = None
    try:
        cmd = "%s fieldgroup --host %s -i -j -g %s" % (dcgmi_absolute_path, hostIP, fieldGroupID)
        output = check_output(cmd, shell=True)
        data = json.loads(output)
        field_ids = data["body"]["Field IDs"]["value"]
        field_group_name = data["body"]["Name"]["value"]
    except Exception:
        print("Failed to gather fieldgroup json data")        
    return field_ids.replace(" ", ""), field_group_name


def updateTestResults(result):
    """
    Helper to function to update test results count
    """

    global TOTAL_TEST_PASSED
    global TOTAL_TEST_FAILED
    global TOTAL_TEST_WAIVED
    global TOTAL_TEST_COUNT
    global TOTAL_TEST_CYCLES

    if "PASSED" in result:
        TOTAL_TEST_PASSED += 1
    elif "FAILED" in result:
        TOTAL_TEST_FAILED += 1
    elif "WAIVED" in result:
        TOTAL_TEST_WAIVED += 1
    elif "COUNT" in result:
        TOTAL_TEST_COUNT += 1
    elif "CYCLE" in result:
        TOTAL_TEST_CYCLES += 1

def getTestSummary():
    """
    Function to print out final results
    """

    print("\n============ SUMMARY TEST RESULTS ============\n")
    print("Total number of Tests PASS: %d " % TOTAL_TEST_PASSED)
    print("Total number of Tests FAILED: %d " % TOTAL_TEST_FAILED)
    print("Total number of Tests WAIVED: %d " % TOTAL_TEST_WAIVED)
    print("Total number of Tests: %d " % TOTAL_TEST_COUNT)
    print("Total number of Cycles: %d " % TOTAL_TEST_CYCLES)
    print("\n===============================================\n")


def setupEnvironment():
    """
    Function to prepare the test environment
    """

    # Set variable indicating we are running tests
    os.environ['__DCGM_TESTING_FRAMEWORK_ACTIVE'] = '1'

    # Verify if GPUs are free before running the tests
    if not nvidia_smi_utils.are_gpus_free():
        print("Some GPUs are in use, please make sure that GPUs are free and try again")
        sys.exit(1)
    
    if test_utils.is_framework_compatible() == False:
        print("burn_in_stress.py found to be a different version than DCGM. Exiting")
        sys.exit(1)
    else:
        print(("Running against Git Commit %s" % version.GIT_COMMIT))
    
    test_utils.set_nvvs_bin_path()

    # Collects the output of "nvidia-smi -q" and prints it out on the screen for debugging
    print("\n###################### NVSMI OUTPUT FOR DEBUGGING ONLY ##########################")

    (message, error) = nvidia_smi_utils.get_output()
    if message:
        print(message)
    if error:
        print(error)

    print("\n###################### NVSMI OUTPUT FOR DEBUGGING ONLY ##########################\n\n")

    print("---------> Enabling persistence mode <------------")
    # Enable persistence mode or the tests will fail
    (message, error) = nvidia_smi_utils.enable_persistence_mode()
    if message:
        print(message)
    if error:
        print(error)
        sys.exit(1)

# class to enable printing different colors for terminal output
class bcolors:
    PURPPLE = '\033[95m'     # purpple
    BLUE = '\033[94m'        # blue
    GREEN = '\033[92m'      # green
    YELLOW = '\033[93m'      # yellow
    RED = '\033[91m'         # red
    ENDC = '\033[0m'         # ends coloring from this point on
    BOLD = '\033[1m'         # bold
    UNDERLINE = '\033[4m'    # underline

    """
     Usage:
            print bcolors.YELLOW + "Warning: No active frommets remain. Continue?" + bcolors.ENDC
            The above would print a nice yellow warning
    """


class Logger(object):

    def __init__(self):
        import socket
        self.terminal = sys.stdout
        self.timestamp = str(time.strftime('%Y-%m-%d'))
        self.logName = "DCGM-BURN-IN_%s_%s.log" % (socket.gethostname(), self.timestamp)

        # Early attempt to clean up the old log file
        if os.path.exists(self.logName):
            try:
                os.remove(self.logName)
            except IOError:
                print("\nUnable to remove older logs file.\n")

        self.log = open(self.logName, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class RunHostEngine(apps.NvHostEngineApp):
    def __init__(self, writeDebugFile=False):
        self.timestamp = str(time.strftime('%Y-%m-%d'))
        self.memlog = "HOST_ENGINE_MEMORY_USAGE_%s.log" % self.timestamp
        self.cpulog = "HOST_ENGINE_CPU_USAGE_%s.log" %  self.timestamp
        super(RunHostEngine, self).__init__()
        self.writeDebugFile = writeDebugFile

    def mem_usage(self, timeout):
        """
        Monitors memory usage of the hostEngine
        """
        pid = self.getpid()
        vmem = []
        rmem = []
        hm = open(self.memlog, "a") 
        loopTime = 3  # Seconds between loops

        timeout_start = time.time()
        while time.time() < timeout_start + float(timeout):
            filename = "/proc/%d/status" % pid
            try:
                fp = open(filename)
                lines = fp.readlines()
            except IOError as e:
                print("Unable to read process file for pid %d: %d msg=%s fn=%s" % (pid, e.errno, e.message, e.filename))
                time.sleep(loopTime)
                continue

            for line in lines:
                if "VmPeak" in line:
                    vmem.append(float(line.split()[1])/1024)

                if "VmRSS" in line:
                    rmem.append(float(line.split()[1])/1024)

            if len(vmem) < 1 or len(rmem) < 1:
                print("VmPeak or VmRSS not found in %d lines of %s" % (len(lines), filename))
                time.sleep(loopTime)
                continue

            virtual_min = min(vmem)
            virtual_max = max(vmem)
            virtual_avg = sum(vmem) / len(vmem)

            resident_min = min(rmem)
            resident_max = max(rmem)
            resident_avg = sum(rmem) / len(rmem)

            hm.write("%s\n" % time.asctime())
            hm.write("Virtual Memory info in MB, Min: %.4f, Max: %.4f, Avg: %.4f\n" % (virtual_min, virtual_max, virtual_avg))
            hm.write("Resident Memory info in MB, Min: %.4f, Max: %.4f, Avg: %.4f\n" % (resident_min, resident_max, resident_avg))
            hm.write("\n.........................................................\n\n")
            time.sleep(loopTime)

        hm.close()

    def cpu_usage(self, timeout):
        """
        Monitors cpu usage of the hostEngine
        """
        pid = self.getpid()
        cpu = []
        hc = open(self.cpulog, "a")

        timeout_start = time.time()
        while time.time() < timeout_start + float(timeout):
            cmd = check_output(shlex.split("ps -p %d -o %%cpu" % pid)).decode('utf-8')
            cpu.append(float(cmd.split("\n")[1]))
            time.sleep(3)

            cpu_min = min(cpu)
            cpu_max = max(cpu)
            cpu_avg = sum(cpu) / len(cpu)

            hc.write("%s\n" % time.asctime())
            hc.write("CPU %% used for HostEngine, Min: %.4f, Max: %.4f, Avg: %.4f\n" % (cpu_min, cpu_max, cpu_avg))
            hc.write("\n.........................................................\n\n")
        hc.close()


class BurnInHandle(object):
    """
    This class is used to communicate with the host engine from the burn-in tests

    hostEngineIp is the IP address of the running host engine. None=start embedded.
    burnInCfg is the parsed command-line parameters. Note that we're not using the IP address from these
    """
    def __init__(self, hostEngineIp, burnInCfg):
        self.dcgmHandle = None
        self.dcgmSystem = None
        self.burnInCfg = burnInCfg
        self.hostEngineIp = hostEngineIp
        self.Connect()

    def Connect(self):
        self.dcgmHandle = pydcgm.DcgmHandle(ipAddress=self.hostEngineIp, opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO)
        self.dcgmSystem = self.dcgmHandle.GetSystem()

    def __del__(self):
        if self.dcgmSystem is not None:
            del(self.dcgmSystem)
            self.dcgmSystem = None
        if self.dcgmHandle is not None:
            del(self.dcgmHandle)
            self.dcgmHandle = None

    def GetGpuIds(self):
        """
        Function to get a list of all GPU IDs. We are looking at the default group's GPU IDs rather than
        dcgmGetAllDevices()'s since the default group includes GPU IDs DCGM cares about
        """
        dcgmGroup = self.dcgmSystem.GetDefaultGroup()
        groupGpuIds = dcgmGroup.GetGpuIds()

        assert len(groupGpuIds) > 0, "DCGM doesn't see any enabled GPUs. Set __DCGM_WL_BYPASS=1 in your environment to bypass DCGM's whitelist"

        # See if the user provided the GPU IDs they care about. If so, return those
        if len(self.burnInCfg.onlyGpuIds) < 1:
            return groupGpuIds

        for gpuId in self.burnInCfg.onlyGpuIds:
            assert(gpuId in groupGpuIds), "User-specified GPU ID %d is not known by the system. System GPU IDs: %s" % (gpuId, str(groupGpuIds))

        return self.burnInCfg.onlyGpuIds

    def GetGpuAttributes(self, gpuIds=None):
        """
        Get an array of dcgm_structs.c_dcgmDeviceAttributes_v1 entries for the passed in gpuIds. None=all devices DCGM knows about
        """
        retList = []
        if gpuIds is None:
            gpuIds = self.GetGpuIds()

        for gpuId in gpuIds:
            retList.append(self.dcgmSystem.discovery.GetGpuAttributes(gpuId))

        return retList

    def GetBusIds(self):
        """
        Get the PCI-E Bus IDs of the attached devices.

        Returns an array of strings like ["0:1;0", "0:2:0"]
        """
        busIdList = []
        attributeList = self.GetGpuAttributes()
        for attributeElem in attributeList:
            busIdList.append(attributeElem.identifiers.pciBusId)

        return busIdList

    def GetValueForFieldId(self, gpuId, fieldId):
        '''
        Watch and get the value of a fieldId

        Returns a dcgmFieldValue_v1 instance
        '''
        dcgm_agent_internal.dcgmWatchFieldValue(self.dcgmHandle.handle, gpuId, fieldId, 60000000, 3600.0, 0)
        self.dcgmSystem.UpdateAllFields(1)
        values = dcgm_agent_internal.dcgmGetLatestValuesForFields(self.dcgmHandle.handle, gpuId, [fieldId, ])
        return values[0]

    def GpuSupportsEcc(self, gpuId):
        '''
        Returns whether (True) or not (False) a gpu supports ECC
        '''
        value = self.GetValueForFieldId(gpuId, dcgm_fields.DCGM_FI_DEV_ECC_CURRENT)
        if dcgmvalue.DCGM_INT64_IS_BLANK(value.value.i64):
            return False
        else:
            return True

    def GetGpuIdsGroupedBySku(self):
        '''
        Get the GPU IDs DCGM supports, grouped by SKU. The return value will be a list of lists like:
        [[gpu0, gpu1], [gpu2, gpu3]]
        In the above example, gpu0 and gpu1 are the same sku, and gpu2 and gpu3 are the same sku
        '''
        gpuIds = self.GetGpuIds()
        listBySku = test_utils.group_gpu_ids_by_sku(self.dcgmHandle.handle, gpuIds)
        return listBySku


class RunCudaCtxCreate:
    def __init__(self, gpuIds, burnInHandle, runTimeSeconds, timeoutSeconds):
        self.burnInHandle = burnInHandle
        self.runTimeSeconds = runTimeSeconds
        self.timeoutSeconds = timeoutSeconds

        # Gets gpuId list
        # gpuIds = self.burnInHandle.GetGpuIds()

        self._apps = []

        if gpuIds is None or len(gpuIds) < 1:
            gpuIds = self.burnInHandle.GetGpuIds()

        deviceAttribs = self.burnInHandle.GetGpuAttributes(gpuIds)

        for deviceAttrib in deviceAttribs:
            busId = deviceAttrib.identifiers.pciBusId
            assert len(busId) > 0, ("Failed to get busId for device %d" % deviceAttrib.identifiers.gpuId)
            args = ["--ctxCreate", busId, "--busyGpu", busId, str(self.runTimeSeconds * 1000)]
            app = apps.CudaCtxCreateAdvancedApp(args)
            app.busId = busId
            self._apps.append(app)

    def start(self):
        for app in self._apps:
            print("Generating Cuda Workload for GPU %s " % app.busId + " at %s \n" % time.asctime())
            app.start(timeout=self.timeoutSeconds)

    def wait(self):
        for app in self._apps:
            app.wait()

    def terminate(self):
        for app in self._apps:
            app.terminate()
            app.validate()

    def getpids(self):
        pids = []
        for app in self._apps:
            pids.append(app.getpid())
        return pids


def get_host_ip(burnInCfg):
    if burnInCfg.remote:
        return burnInCfg.srv
    return "127.0.0.1"


# Helper Class to do group operations
class GroupsOperationsHelper:

    def __init__(self, burnInHandle):
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.dcgmi_path = get_dcgmi_bin_path()
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def __safe_checkcall(self, args):
        try:
            check_call([self.dcgmi_path]+args)
        except CalledProcessError:
            pass
        return True

    def __safe_checkoutput(self, args):
        rc = None
        try:
            rc = check_output([self.dcgmi_path]+args)
        except CalledProcessError:
            pass
        return rc

    def create_group(self, groupName, gpuIds):
        if groupName is None:
            groupName = "Group"
        args = ["group", "--host", self.host_ip, "-c", groupName]
        output = self.__safe_checkoutput(args)
        self.group_id = int(output.strip().split()[-1]) if output is not None else -1

        for gpuId in gpuIds:
            self.add_device(gpuId)

        return self.group_id

    def list_group(self):
        args = ["group", "--host", self.host_ip, "-l"]
        self.__safe_checkcall(args)

    def get_group_info(self):
        assert self.group_id is not None
        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-i"]
        return self.__safe_checkcall(args)

    def add_device(self, gpuId):
        assert self.group_id is not None
        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-a", str(gpuId)]
        return self.__safe_checkcall(args)

    def remove_device(self, gpuId):
        assert self.group_id is not None

        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-r", str(gpuId)]
        return self.__safe_checkcall(args)

    def delete_group(self):
        assert self.group_id is not None 
        args = ["group", "--host", self.host_ip, "-d", str(self.group_id)]
        return self.__safe_checkcall(args)


# Run the GROUPS subsystem tests
class GroupTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

        print("The hostEngine IP is %s\n" % self.host_ip)


    def test1_create_group(self, gpuIds):
        """ Test to create groups using the subsystem groups """
        #Intentionally create the group as empty so that test5_add_device_to_group doesn't fail
        self.group_id = self.groups_op.create_group(None, [])
        print("Creating a default test group: %s" % self.group_id)
        return self.group_id > 0

    def test2_list_group(self, gpuIds):
        """ Test to list existing groups using the subsystem groups """

        args = ["group", "--host", self.host_ip, "-l"]
        print("Listing the test group: %s" % args)

        return args

    def test3_get_group_info(self, gpuIds):
        """ Test to get info about exiting groups using the subsystem groups """

        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-i"]
        print("Showing info about the test group: %s" % args)

        return args

    def test5_remove_device_from_group(self, gpuIds):
        """ Test to remove a GPU from a group using the subsystem groups """

        # Removes a GPU from the test group
        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-r", str(gpuIds[0])]
        print("Removing GPU %s from the test group: %s" % (gpuIds[0], args))

        return args

    def test4_add_device_to_group(self, gpuIds):
        """ Test to add a GPU to a groups using the subsystem groups """
        args = ["group", "--host", self.host_ip, "-g", str(self.group_id), "-a", str(gpuIds[0])]
        print("Adding GPU %s to the test group: %s" % (gpuIds[0], args))

        return args

    def test6_delete_group(self, gpuIds):
        """ Test deleting existing groups using the subsystem groups """

        print("Deleting the default test group: %s" % self.group_id)

        return self.groups_op.delete_group()


# Run the CONFIG subsystem tests
class ConfigTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem config """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def _set_compute_mode_helper(self, gpuIds, comp):
        """ Test --set compute mode values on "config" subsystem """

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--set", "-c", comp]
        print("Setting different compute modes: %s" % args)
        return args

    # Runs with each possible compute mode
    @test_utils.run_only_if_mig_is_disabled()
    def test2_set_compute_mode_value_0(self, gpuIds):
        return self._set_compute_mode_helper(gpuIds, "2")

    @test_utils.run_only_if_mig_is_disabled()
    def test2_set_compute_mode_value_1(self, gpuIds):
        return self._set_compute_mode_helper(gpuIds, "1")

    @test_utils.run_only_if_mig_is_disabled()
    def test2_set_compute_mode_value_2(self, gpuIds):
        return self._set_compute_mode_helper(gpuIds, "0")

    def get_power_power_limit(self, pLimitType, gpuIds):
        """
        Helper function to get power limit for the device
        """

        deviceAttrib = self.burnInHandle.dcgmSystem.discovery.GetGpuAttributes(gpuIds[0])
        if pLimitType == DEFAULT_POWER_LIMIT:
            pwrLimit = str(deviceAttrib.powerLimits.defaultPowerLimit)
        elif pLimitType == MAX_POWER_LIMIT:
            pwrLimit = str(deviceAttrib.powerLimits.maxPowerLimit)
        elif pLimitType == MIN_POWER_LIMIT:
            pwrLimit = str(deviceAttrib.powerLimits.minPowerLimit)

        return pwrLimit

    def _set_power_limit_helper(self, kind, pwr):
        """ Test --set power limit values on "config" subsystem """

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--set", "-P", pwr]
        print("Setting %s power limit for devices: %s" % (kind, args))
        return args

    # Runs and tries to set the minimum power limit supported
    def test3_set_min_power_limit(self, gpuIds):

        defPowerLimit = self.get_power_power_limit(DEFAULT_POWER_LIMIT, gpuIds)
        minPowerLimit = self.get_power_power_limit(MIN_POWER_LIMIT, gpuIds)
        if defPowerLimit == minPowerLimit:
            print("Only the default power limit is available for this device, skipping minimum power limit test")
            print(bcolors.PURPPLE + "&&&& SKIPPED" + bcolors.ENDC)
            return
        else:
            minPowerLimit = int(minPowerLimit) + 1  # +1 to address fractions lost in data type conversion

        return self._set_power_limit_helper("minimum", str(minPowerLimit))

    def test4_set_max_power_limit(self, gpuIds):

        defPowerLimit = self.get_power_power_limit(DEFAULT_POWER_LIMIT, gpuIds)
        maxPowerLimit = self.get_power_power_limit(MAX_POWER_LIMIT, gpuIds)
        maxPowerLimit = int(maxPowerLimit) - 1  # -1 to address fractions lost in data type conversion

        return self._set_power_limit_helper("maximum", str(maxPowerLimit))

    def test5_set_default_power_limit(self, gpuIds):
        defPowerLimit = self.get_power_power_limit(DEFAULT_POWER_LIMIT, gpuIds)
        return self._set_power_limit_helper("default", defPowerLimit)

    def _set_application_clocks_helper(self, mem, sm):
        """ Test --set application clocks on "config" subsystem """

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--set", "-a", "%s,%s" % (mem, sm)]
        print("Setting application clocks values for \"mem,proc\": %s" % args)
        return args

    # Runs and tries to set the application clocks to 900W
    def test6_set_application_clocks_values(self, gpuIds):
        deviceAttrib = self.burnInHandle.dcgmSystem.discovery.GetGpuAttributes(gpuIds[0])
        sm_clk = str(deviceAttrib.clockSets.clockSet[0].smClock)
        mem_clk = str(deviceAttrib.clockSets.clockSet[0].memClock)

        return self._set_application_clocks_helper(mem_clk, sm_clk)


    def test7_enforce_values(self, gpuIds):
        """ Test on "config" subsystem using the "enforce" operation """

        # Trying to enforce previous "--set" configurations for each device
        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--enforce"]
        print("Trying to enforce last configuration used via \"--set\": %s" % args)
        time.sleep(1) 

        return args


    def _set_sync_boost_helper(self, gpuIds, val):
        """ Test --set syncboost on "config" subsystem """

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--set", "-s", val]
        print("Enable/Disable syncboost feature on device: %s" % args)
        return args

    #These tests aren't valid as long as we're running on one GPU at a time
    #Runs and tries to enable and disable sync_boost
    #def test8_set_sync_boost_value_0(self, dev): return self._set_sync_boost_helper(dev, "0")
    #def test8_set_sync_boost_value_1(self, dev): return self._set_sync_boost_helper(dev, "1")

    def _set_ecc_helper(self, gpuIds, val):
        """ Test --set ecc on "config" subsystem """

        for gpuId in gpuIds:
            if not self.burnInHandle.GpuSupportsEcc(gpuId):
                print("Skipping ECC tests for GPU %d that doesn't support ECC" % gpuId)
                return None

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--set", "-e", val]
        print("Enable/Disable ecc on device: %s" % args)
        return args

    # Runs and tries to enable and disable ecc / DISABLED until GPU Reset issue can be resolved
    # def test9_set_ecc_value_0(self, gpuIds): return self._set_ecc_helper(gpuIds, "0")
    # def test9_set_ecc_value_1(self, gpuIds): return self._set_ecc_helper(gpuIds, "1")


    def test11_get_values(self, gpuIds):
        """ Test getting values on "config" subsystem using the get operation """

        args = ["config", "--host", self.host_ip, "-g", str(self.group_id), "--get"]
        print("Getting subsystem \"config\" information for a group: %s" % args)
        time.sleep(1)
        return args

    def test12_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the DISCOVERY subsystem tests
class DiscoveryTests():

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem discovery """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_discovery_gpus(self, gpuIds):
        """ Test to list existing GPUs """

        args = ["discovery", "--host", self.host_ip, "-l"]
        print("Querying existing gpus: %s" % args)

        return args

    def _discovery_device_info_helper(self, gpuIds, flag):
        """ Test to get discovery info for device per feature """

        args = ["discovery", "--host", self.host_ip, "-g", str(self.group_id), "-i", flag]
        print("Querying info in group: %s" % args)

        return args

    def test3_discovery_device_info_a(self, gpuIds): return self._discovery_device_info_helper(gpuIds, "a")
    def test3_discovery_device_info_p(self, gpuIds): return self._discovery_device_info_helper(gpuIds, "t")
    def test3_discovery_device_info_t(self, gpuIds): return self._discovery_device_info_helper(gpuIds, "p")
    def test3_discovery_device_info_c(self, gpuIds): return self._discovery_device_info_helper(gpuIds, "c")

    def test4_discovery_device(self, gpuIds):
        """ Test to discovery info of each GPU """

        args = ["discovery", "--host", self.host_ip, "--gpuid", str(gpuIds[0]), "-i", "aptc"]
        print("Querying info for GPU %s: %s" % (gpuIds[0], args))
        return args

    def test5_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the HEALTH subsystem tests
class HealthTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle

        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem health """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1


    def _set_health_watches_helper(self, gpuIds, flag):
        """ Test to set health watches """

        args = ["health", "--host", self.host_ip, "-g", str(self.group_id), "--set", flag]
        print("Setting health watch %s: %s" % (flag, args))

        return args

    def test2_health_set_watches_t(self, gpuIds): return self._set_health_watches_helper(gpuIds, "a")
    def test2_health_set_watches_p(self, gpuIds): return self._set_health_watches_helper(gpuIds, "p")
    def test2_health_set_watches_m(self, gpuIds): return self._set_health_watches_helper(gpuIds, "m")
    def test2_health_set_watches_a(self, gpuIds): return self._set_health_watches_helper(gpuIds, "t")
    def test2_health_set_watches_i(self, gpuIds): return self._set_health_watches_helper(gpuIds, "i")
    def test2_health_set_watches_n(self, gpuIds): return self._set_health_watches_helper(gpuIds, "n")

    def test3_heath_fetch_watchers_status(self, gpuIds):
        """ Test to fetch watcher list """

        args = ["health", "--host", self.host_ip, "-g", str(self.group_id), "--fetch"]
        print("Fetching Health Watches: %s" % args)

        return args

    def test4_health_check(self, gpuIds):
        """ Test to check the overall health """

        args = ["health", "--host", self.host_ip, "-g", str(self.group_id), "--check"]
        print("Checking overall health: %s" % args)

        return args

    def test5_health_clear_watches(self, gpuIds):
        """ Test to clear all watches """

        args = ["health", "--host", self.host_ip, "-g", str(self.group_id), "--clear"]
        print("Clearing all Health Watches: %s" % args)

        return args

    def test6_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Runs the DIAG subsystem tests
class DiagnosticsTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle

        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem diag """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def _set_diag_helper(self, gpuIds, flag):
        """ Test to run diag tests """
        args = ["diag", "--host", self.host_ip, "-g", str(self.group_id), "--run", flag]
        print("Running Diagnostic Test : %s" % args)

        return args

    def test2_diag1_short(self, gpuIds): return self._set_diag_helper(gpuIds, "1")
    def test2_diag2_medium(self, gpuIds): return self._set_diag_helper(gpuIds, "2")
    def test2_diag3_long(self, gpuIds): return self._set_diag_helper(gpuIds, "3")


    def test3_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the TOPO subsystem tests
class TopologyTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle

        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem diag """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1


    def test2_query_topology_by_groupId(self, gpuIds):
        """ Test to read topology by group id """

        args = ["topo", "--host", self.host_ip, "-g", str(self.group_id)]
        print("Reading topology by group Id")
        return args

    def test3_query_topology_by_gpuId(self, gpuIds):
        """ Test to read topology by gpu id """

        args = ["topo", "--host", self.host_ip, "--gpuid", str(gpuIds[0])]
        print("Reading topology by GPU Id %s: %s" % (gpuIds[0], args))
        return args


    def test4_delete_group(self, dev):
        """ Removes group used for testing """
        return self.groups_op.delete_group()


# Run the POLICY subsystem tests
class PolicyTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle

        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)


    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem policy """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_get_current_policy_violation_by_groupId(self, gpuIds):
        """ Get the current violation policy by group id """

        args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), "--get"]
        print("Getting current violation policy list: %s " % args)

        return args

    def test3_set_pcierrors_policy_violation(self, gpuIds):
        """ Set the policy violation by group id """

        actions = ["0","1"] # 0->None, 1->GPU Reset
        validations = ["0","1","2","3"] # 0->None, 1-> NVVS(short), 2-> NVVS(medium), 3-> NVVS(long)

        for action in actions:
            for val in validations:
                args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), \
                          "--set", "%s,%s" % (action,val), "-p"]
                print("Setting PCI errors sviolation policy list action %s, validation %s: %s " % (action, val, args))

        return args

    def test4_set_eccerrors_policy_violation(self, gpuIds):
        """ Set the policy violation by group id """

        actions = ["0","1"] # 0->None, 1->GPU Reset
        validations = ["0","1","2","3"] # 0->None, 1-> NVVS(short), 2-> NVVS(medium), 3-> NVVS(long)

        for action in actions:
            for val in validations:
                args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), \
                          "--set", "%s,%s" % (action,val), "-e"]
                print("Setting ECC errors violation policy list action %s, validation %s: %s " % (action, val, args))

        return args


    def test5_set_power_temperature_values_policy(self, gpuIds):
        """ Get the violation policy for all devices at once """

        deviceAttrib = self.burnInHandle.dcgmSystem.discovery.GetGpuAttributes(gpuIds[0])

        max_temp = str(deviceAttrib.thermalSettings.slowdownTemp)
        max_pwr = str(deviceAttrib.powerLimits.maxPowerLimit)
        max_pages = str(60)

        args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), "--set", "1,1", "-T", max_temp, "-P", max_pwr, "-M", max_pages]
        print("Setting max power and max temperature for policy list: %s " % args)

        return args

    def test6_get_detailed_policy_violation(self, gpuIds):
        """ Get the violation policy for all devices at once """

        args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), "--get", "-v"]
        print("Getting detailed violation policy list: %s " % args)

        return args

    def test7_clear__policy_values(self, gpuIds):
        """ Get the violation policy for all devices at once """

        args = ["policy", "--host", self.host_ip, "-g", str(self.group_id), "--clear"]
        print("Clearing settings for policy list: %s " % args)

        return args


    def test8_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the STATS subsystem tests
class ProcessStatsTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem diag """
        self.gpuIds = gpuIds
        self.group_id = self.groups_op.create_group(None, self.gpuIds)

        return self.group_id > 1

    @test_utils.run_only_if_mig_is_disabled()
    def test2_enable_system_watches(self, gpuIds):
        """ Enable watches for process stats """

        args = ["stats", "--host", self.host_ip, "-g", str(self.group_id), "--enable"]
        print("Enabling system watches for process stats: %s" % args)

        return args

    @test_utils.run_only_if_mig_is_disabled()
    def test3_get_pid_stats(self, gpuIds):
        """ Gets the process stats using the stats subsystem """

        app = RunCudaCtxCreate(self.gpuIds, self.burnInHandle, runTimeSeconds=10, timeoutSeconds=(20 * len(gpuIds)))
        app.start()

        pids = app.getpids()
        for pid in pids:
            stats_msg = "--> Generating Data for Process Stats -  PID %d <--" % pid
            updateTestResults("CYCLE")
            print(bcolors.PURPPLE + stats_msg + bcolors.ENDC)
            args = ["stats", "--host", self.host_ip, "-g", str(self.group_id), "--pid", str(pid), "-v"]
            print("Collecting process stats information: %s" % args)
            sys.stdout.flush()

        app.wait()
        app.terminate()

        return args

    @test_utils.run_only_if_mig_is_disabled()
    def test4_disable_system_watches(self, gpuIds):
        """ Enable watches for process stats """

        args = ["stats", "--host", self.host_ip, "-g", str(self.group_id), "--disable"]
        print("Disabling system watches for process stats: %s" % args)

        return args

    def test5_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the NVLINK subsystem tests
class NvlinkTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem nvlink """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_check_nvlink_status(self, gpuIds):
        """ Gets the nvlink error counts for various links and GPUs on the system  """

        args = ["nvlink", "--host", self.host_ip, "-g", str(self.group_id), "-s"]
        print("Reporting current nvlink status")
        return args

    def test3_query_nvlink_errors(self, gpuIds):
        for gpuId in gpuIds:
            args = ["nvlink", "--host", self.host_ip, "-g", str(gpuId), "-e"]
            print("Running queries to get errors for various nvlinks on the system: %s" % args)
        return args

    def test4_query_nvlink_errors_json_output(self, gpuIds):
        for gpuId in gpuIds:
            args = ["nvlink", "--host", self.host_ip, "-g", str(gpuId), "-e", "-j"]
            print("Running queries to get errors for various nvlinks on the system in Json format: %s" % args)
        return args

    def test5_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the Introspection subsystem tests
class IntrospectionTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)


    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem nvlink """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_enable_introspection_feature(self, gpuIds):
        """ Enables the introspection feature """

        args = ["introspect", "--host", self.host_ip, "--enable"]
        print("Enabling the introspection features for collecting various stats: %s" % args)

        return args

    def test3_show_hostengine_stats(self, gpuIds):
        """ Prints out hostengine statistics """

        args = ["introspect", "--host", self.host_ip, "--show", "--hostengine"]
        print("Showing introspection information for hostengine: %s" % args)

        return args

    def test4_show_all_fields_stats(self, gpuIds):
        """ Prints out all-fields statistics """

        args = ["introspect", "--host", self.host_ip, "--show", "--all-fields"]
        print("Showing introspection information for all fields: %s" % args)

        return args


    def test5_enable_watches_for_introspection(self, gpuIds):
        """ Set watches for device introspection information """

        args = ["stats", "--host", self.host_ip, "-g", str(self.group_id), "-e"]
        print("Enabling watches for collecting device introspection information: %s" % args)

        return args


    def test6_show_field_group_stats(self, gpuIds):
        """ Prints out all field-group stats """

        args = ["introspect", "--host", self.host_ip, "--show", "--field-group", "all"]
        print("Showing all field-group introspection information: %s" % args)

        return args

    def test7_disable_introspection_feature(self, gpuIds):
        """ Disables the introspection feature """

        args = ["introspect", "--host", self.host_ip, "--disable"]
        print("Disabling the introspection feature: %s" % args)

        return args

    def test8_delete_group(self, gpuIds):
        """ Removes group used for testing """

        return self.groups_op.delete_group()


# Run the Fieldgroups subsystem tests
class FieldGroupsTests():

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_create_fieldgroup(self, gpuIds):
        all_field_ids = []
        for i in range(1,4): # There are 3 fieldGroups DCGM_INTERNAL_30SEC, HOURLY, JOB
            field_Ids, field_group_name  = get_field_ids(self.host_ip, i)
            if "," in field_Ids[-1]:
                field_Ids = field_Ids[:-1] # removes unwanted comma from last field id
            all_field_ids.append(field_Ids)

        all_fields = "".join(all_field_ids)
        args = ["fieldgroup", "--host", self.host_ip, "-c", "testFg", "-f", "%s" % all_fields]
        print("Creating a field groups: %s" % args)
        return args

    def test3_list_existing_field_groups(self, gpuIds):
        args = ["fieldgroup", "--host", self.host_ip, "-l"]
        print("Listing available field IDs: %s" % args)
        return args

    def test4_list_json_format_field_groups(self, gpuIds):
        args = ["fieldgroup", "--host", self.host_ip, "-l", "-j"]
        print("Listing available field IDs in Json format: %s" % args)
        return args

    def test5_get_fieldgroup_info(self, gpuIds):
        fieldGroupId = get_current_field_group_id(self.host_ip)
        args = ["fieldgroup", "--host", self.host_ip, "-i", "-g", "%s" % fieldGroupId]
        print("Listing field ID information: %s" % args)
        return args

    def test6_delete_fieldgroup(self, gpuIds):
        fieldGroupId = get_current_field_group_id(self.host_ip)
        args = ["fieldgroup", "--host", self.host_ip, "-d", "-g", "%s" % fieldGroupId]
        print("Deleting a field group: %s" % args)
        return args

    def test7_delete_group(self, gpuIds):
        """ Removes group used for testing """
        return self.groups_op.delete_group()


# Run the Modules subsystem tests
class ModulesTests():

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_list_modules(self, gpuIds):
        args = ["modules", "--host", self.host_ip, "-l"]
        print("Listing existing modules: %s" % args)
        return args

    def test3_list_modules_json(self, gpuIds):
        args = ["modules", "--host", self.host_ip, "-l", "-j"]
        print("Listing existing modules Json format: %s" % args)
        return args

    def test4_delete_group(self, gpuIds):
        """ Removes group used for testing """
        return self.groups_op.delete_group()


# Run the dmon subsystem tests
class DmonTests:

    def __init__(self, burnInHandle):
        self.groups_op = GroupsOperationsHelper(burnInHandle)
        self.group_id = None
        self.burnInHandle = burnInHandle
        self.host_ip = get_host_ip(self.burnInHandle.burnInCfg)

    def test1_create_group(self, gpuIds):
        """ Creates a group for testing the subsystem dmon """
        self.group_id = self.groups_op.create_group(None, gpuIds)
        return self.group_id > 1

    def test2_list_long_short_name_field_ids_dmon(self, gpuIds):
        """ Prints out list of available items for dmon"""
        args = ["dmon", "--host", self.host_ip, "-l"]
        print("Listing available items for dmon to monitor: %s" % args)
        return args

    def test3_dmon_with_various_field_ids_per_device(self, gpuIds):
        """ Runs dmon to get field group info for each device """

        for i in range(1,4): # There are 3 fieldGroups DCGM_INTERNAL_30SEC, HOURLY, JOB
            field_Ids, field_group_name  = get_field_ids(self.host_ip, i)
            if "," in field_Ids[-1]:
                field_Ids = field_Ids[:-1] # removes unwanted comma from last field id

            for gpuId in gpuIds:
                cmd = "%s dmon --host %s -i %s -e %s -c 10 -d 100" % (dcgmi_absolute_path, self.host_ip, str(gpuId), field_Ids)
                print("Running dmon on a single GPU with field group %s: %s" % (field_group_name, shlex.split(cmd)))
                try:
                    check_output(cmd, shell=True)
                    print(bcolors.BLUE + "&&&& PASSED" + bcolors.ENDC)
                    updateTestResults("PASSED")
                    updateTestResults("COUNT") 
                except CalledProcessError:
                    print("Failed to get dmon data for GPU %s " % str(gpuId))
                    print(bcolors.RED + "&&&& FAILED" + bcolors.ENDC)
                    updateTestResults("FAILED")
                    updateTestResults("COUNT")

        return

    def test4_dmon_field_group_dcgm_internal_30sec(self, gpuIds):
        args = ["dmon", "--host", self.host_ip, "-f", "1", "-c", "10", "-d", "100"]
        print("Running dmon on a group to monitor data on field group DCGM_INTERNAL_30SEC:  %s" % args)
        return args

    def test5_dmon_field_group_dcgm_internal_hourly(self, gpuIds):
        "dcgmi dmon -f 3 -c 10 -d 100"
        args = ["dmon", "--host", self.host_ip, "-f", "2", "-c", "10", "-d", "100"]
        print("Running dmon on a group to monitor data on field group DCGM_INTERNAL_HOURLY:  %s" % args)
        return args

    def test6_dmon_field_group_dcgm_internal_job(self, gpuIds):
        "dcgmi dmon -f 3 -c 10 -d 100"
        args = ["dmon", "--host", self.host_ip, "-f", "3", "-c", "10", "-d", "100"]
        print("Running dmon on a group to monitor data on field group DCGM_INTERNAL_JOB:  %s" % args)
        return args

    def test5_delete_group(self, gpuIds):
        """ Removes group used for testing """
        return self.groups_op.delete_group()

def IsDiagTest(testname):
    if testname == 'test2_diag1_short':
        return 1
    elif testname == 'test2_diag2_medium':
        return 2
    elif testname == 'test2_diag3_long':
        return 3

    return 0

class RunDcgmi():
    """
    Class to launch an instance of the dcgmi client for each gpu  
    and control its execution
    """ 

    forbidden_strings = [
        # None of this error codes should be ever printed by dcgmi
        "Unknown Error",
        "Uninitialized",
        "Invalid Argument",
        "Already Initialized",
        "Insufficient Size",
        "Driver Not Loaded",
        "Timeout",
        "DCGM Shared Library Not Found",
        "Function Not Found",
        "(null)", # e.g. from printing %s from null ptr
        ]

    def __init__(self, burnInHandle):
        self.burnInHandle = burnInHandle

        self.dcgmi_path = get_dcgmi_bin_path()
        self.timestamp = str(time.strftime('%Y-%m-%d'))
        self._timer = None              # to implement timeout
        self._subprocess = None
        self._retvalue = None           # stored return code or string when the app was terminated
        self._lock = threading.Lock()   # to implement thread safe timeout/terminate
        self.log = open('DCGMI-CLIENT_%s.log' % self.timestamp, 'a+')
        self.stdout_lines = []
        self.stderr_lines = []
        self.group_tests = GroupTests(burnInHandle)
        self.config_tests = ConfigTests(burnInHandle)
        self.health_tests = HealthTests(burnInHandle)
        self.policy_tests = PolicyTests(burnInHandle)
        self.discovery_tests = DiscoveryTests(burnInHandle)
        self.diag_tests = DiagnosticsTests(burnInHandle)
        self.stats_tests = ProcessStatsTests(burnInHandle)
        self.topo_tests = TopologyTests(burnInHandle)
        self.nvlink_tests = NvlinkTests(burnInHandle)
        self.introspection_tests = IntrospectionTests(burnInHandle)
        self.field_groups_tests = FieldGroupsTests(burnInHandle)
        self.modules_tests = ModulesTests(burnInHandle)
        self.dmon_tests = DmonTests(burnInHandle)

    def _get_sorted_tests(self, obj):
        """ Helper function to get test elements from each class and sort them in ascending order

            Lambda breakdown: 
            sorted(filter(lambda x:x[0].startswith("test"), inspect.getmembers(obj)) -> Filters and sorts each members from the class that startswith "test"
            key=lambda x:int(x[0][4:x[0].find('_')])) -> key modifies the object to compare the items by their integer value found after the "_"          
        """
        return sorted([x for x in inspect.getmembers(obj) if x[0].startswith("test")], key=lambda x:int(x[0][4:x[0].find('_')]))
        
    def get_group_tests(self):
        return self._get_sorted_tests(self.group_tests)

    def get_config_tests(self):
        return self._get_sorted_tests(self.config_tests)

    def get_discovery_tests(self):
        return self._get_sorted_tests(self.discovery_tests)

    def get_health_tests(self):
        return self._get_sorted_tests(self.health_tests)

    def get_diag_tests(self):
        return self._get_sorted_tests(self.diag_tests)

    def get_stats_tests(self):
        return self._get_sorted_tests(self.stats_tests)

    def get_policy_tests(self):
        return self._get_sorted_tests(self.policy_tests)

    def get_topo_tests(self):
        return self._get_sorted_tests(self.topo_tests)

    def get_nvlink_tests(self):
        return self._get_sorted_tests(self.nvlink_tests)

    def get_introspection_tests(self):
        return self._get_sorted_tests(self.introspection_tests)

    def get_fieldgroups_tests(self):
        return self._get_sorted_tests(self.field_groups_tests)

    def get_modules_tests(self):
        return self._get_sorted_tests(self.modules_tests)

    def get_dmon_tests(self):
        return self._get_sorted_tests(self.dmon_tests)

    def print_test_header(self, testName):
        print(("&&&& RUNNING " + testName))

    def print_test_footer(self, testName, statusText, color):
        #Don't include colors for eris
        if option_parser.options.eris or option_parser.options.dvssc_testing:
            print(("&&&& " + statusText + " " + testName))
        else:
            print((color + "&&&& " + statusText + " " + testName + bcolors.ENDC))

    def start(self, timeout=None, server=None):
        """
        Launches dcgmi application.
        """
        assert self._subprocess is None

        # checks if the file has executable permission
        if os.path.exists(self.dcgmi_path):
            assert os.access(self.dcgmi_path, os.X_OK), "Application binary %s is not executable! Make sure that the testing archive has been correctly extracted." % (self.dcgmi_path)

        timeout_start = time.time()
        while time.time() < timeout_start + timeout:
            # Gets gpuId list
            gpuIdLists = self.burnInHandle.GetGpuIdsGroupedBySku()


            # Starts a process to run dcgmi 
            for gpuIds in gpuIdLists:
                fout = open('DCGMI-RUN_%s.log' % self.timestamp, 'a+')

                # Creates a list of lists from the return of each function
                all_tests = [
                                self.get_group_tests(),
                                self.get_config_tests(),
                                self.get_discovery_tests(),
                                self.get_health_tests(),
                                self.get_stats_tests(),
                                self.get_topo_tests(),
                                self.get_policy_tests(),
                                self.get_nvlink_tests(),
                                self.get_introspection_tests(),
                                self.get_fieldgroups_tests(),
                                self.get_modules_tests(),
                                self.get_dmon_tests(),
                                self.get_diag_tests()
                            ]
                if not self.burnInHandle.burnInCfg.eud:
                    all_tests.pop()

                for test in all_tests:
                    for (testName, testMethod) in test:
                        try:
                            self.print_test_header(testName)

                            exec_test = testMethod(gpuIds)
                            if type(exec_test) == bool or exec_test is None:
                                continue

                            # We need to check pass / fail differently for the diag. If it finds a legitimate problem
                            # we want to consider the test waived.
                            dd = None
                            diagPassed = False
                            diagTest = IsDiagTest(testName)
                            if diagTest:
                                nsc = nvidia_smi_utils.NvidiaSmiJob()
                                nsc.start()
                                dd = DcgmiDiag.DcgmiDiag(dcgmiPrefix=get_dcgmi_bin_directory(), runMode=diagTest, gpuIds=gpuIds)
                                diagPassed = not dd.Run()
                                nsc.m_shutdownFlag.set()
                                nsc.join()
                                fout.write(str(dd.lastStdout))
                                fout.write(str(dd.lastStderr))
                                rc = dd.diagRet
                            else:
                                self._subprocess = Popen([self.dcgmi_path]+exec_test, stdout=fout, stderr=fout)
                                self._subprocess.wait()
                                rc = self._subprocess.returncode

                            #DCGM returns an undeflowed int8 as the status. So -3 is returned as 253. Convert it to the negative value
                            if rc != 0:
                                rc -= 256
                            print("Got rc %d" % rc)

                            if rc == 0:
                                self.print_test_footer(testName, "PASSED", bcolors.BLUE)
                                updateTestResults("PASSED")
                                updateTestResults("COUNT")
                            elif rc == dcgm_structs.DCGM_ST_NOT_SUPPORTED:
                                self.print_test_footer(testName, "WAIVED", bcolors.YELLOW)
                                updateTestResults("WAIVED")
                                updateTestResults("COUNT")
                            elif diagPassed == True:
                                # If we reach here, it means the diag detected a problem we consider legitimate.
                                # Mark this test as waived instead of failed
                                self.print_test_footer(testName, "WAIVED", bcolors.YELLOW)
                                print("Waiving test due to errors we believe to be legitimate detected by the diagnostic")
                                if dd is not None:
                                    dd.PrintFailures()
                                updateTestResults("WAIVED")
                                updateTestResults("COUNT")
                            else:
                                self.print_test_footer(testName, "FAILED", bcolors.RED)
                                if dd is not None:
                                    dd.PrintFailures()
                                updateTestResults("FAILED")
                                updateTestResults("COUNT")
                        except test_utils.TestSkipped as err:
                                self.print_test_footer(testName, "SKIPPED", bcolors.PURPPLE)
                                print(err)
                                updateTestResults("SKIPPED")

        return ("\n".join(self.stdout_lines), self._subprocess.returncode), str(gpuIds)

    def retvalue(self):
        """
        Returns code/string if application finished or None otherwise.
        """
        if self._subprocess.poll() is not None:
            self.wait()
            return self._retvalue


    def _process_finish(self):
        # if still alive, kill process
        if self._subprocess.returncode is None:
            self._subprocess.terminate()
            self._subprocess.wait()

        # Check if child process has terminated.
        if self._subprocess.returncode is not None:
            if self._subprocess.returncode == 0:
                message = ("PASSED - DCGMI is running normally, returned %d\n" % self._subprocess.returncode)
                self.log.writelines(message)
            else:
                message = ("FAILED - DCGMI failed with returned non-zero %s \n") % self._subprocess.returncode
                self.log.writelines(message)


        # Verify that dcgmi doesn't print any strings that should never be printed on a working system
        stdout = "\n".join(self.stdout_lines)
        for forbidden_text in RunDcgmi.forbidden_strings:
            assert stdout.find(forbidden_text) == -1, "dcgmi printed \"%s\", this should never happen!" % forbidden_text

        return self._retvalue

    def wait(self):
        """
        Wait for application to finish and return the app's error code/string

        """
        if self._retvalue is not None:
            return self._retvalue

        with self._lock:                   # set ._retvalue in thread safe way. Make sure it wasn't set by timeout already
            if self._retvalue is None:
                self._retvalue = self._subprocess.returncode
                self._process_finish()

        return self._retvalue

    def _trigger_timeout(self):
        """
        Function called by timeout routine. Kills the app in a thread safe way.

        """
        with self._lock: # set ._retvalue in thread safe way. Make sure that app wasn't terminated already
            if self._retvalue is not None:
                return self._retvalue

            self._subprocess.kill()
            self._process_finish()

            return self._retvalue

# Start host egine on headnode
def run_local_host_engine(burnInCfg):

    #Starting HostEngine
    host_engine = RunHostEngine(burnInCfg.writeHostEngineDebugFile)
    host_engine.start(int(burnInCfg.runtime)+5)
    try:
        _thread.start_new_thread(lambda: host_engine.mem_usage(burnInCfg.runtime), ())
        _thread.start_new_thread(lambda: host_engine.cpu_usage(burnInCfg.runtime), ())
    except:
        print("Error: unable to create thread") 
    time.sleep(1)

    return host_engine

# Start cuda workload
def run_cuda_workload(burnInCfg):
    burnInHandle = BurnInHandle(burnInCfg.ip, burnInCfg)

    #Creates cuda workload
    cuda_workload = RunCudaCtxCreate(None, burnInHandle, int(burnInCfg.runtime)+4, timeoutSeconds=10)
    cuda_workload.start()
    print("\nGenerating Cuda Workload on Clients...\n")
    cuda_workload.wait()


# Start dcgmi testing
def run_dcgmi_client(burnInCfg):
    burnInHandle = BurnInHandle(burnInCfg.ip, burnInCfg)
    dcgmi_client = RunDcgmi(burnInHandle)
    dcgmi_client.start(int(burnInCfg.runtime)+1, burnInCfg.srv)
    time.sleep(2)

# Copy packages to test nodes
def copy_files_to_targets(ip):

    # Gets current user name
    user = os.getlogin()

    # Compress current directory, copies the package to target systems
    print("Creating burning-package.tar.gz package...")
    time.sleep(2)
    package = "burning-package.tar.gz"
    abscwd=os.path.abspath(os.getcwd())
    os.system("tar zcvf /tmp/%s -C %s testing" % (package, os.path.dirname(abscwd)))

    print("\n...................................................\n")
    print("Copying package to test systems... %s" % ip)

    for address in ip:
        # Sends package to remote's /home/$USER folder
        os.system("scp -rp /tmp/" + package + " " + user + "@" + address + ":")
        time.sleep(1)

        # Unpacking package on remote nodes
        os.system("ssh " + user + "@" +  address + " tar zxf " + package)
        time.sleep(1)

# Run the tests on the remote nodes
def run_remote(runtime, address, srv, nodes):

    # Gets current user name
    user = os.getlogin()

    # Loads python 2.7.x cluster module as target system only has python 2.6.x 
    # (may vary on different cluster configurations)
    module = "module load python/2.7/2.7.8"
    py_cmd = "python2 burn_in_stress.py"

    # Run the tests on remote systems
    cmd="ssh %s@%s \"MODULEPATH=%s %s;cd testing; LD_LIBRARY_PATH=~/testing %s -t %s -s %s \"" % (user, address, os.environ["MODULEPATH"], module, py_cmd, runtime, srv)
    return Popen(cmd.split())

# Run the tests on the local (single) node
def run_tests(burnInCfg):

    color = bcolors()

    # Start an embedded host engine first to make sure we even have GPUs configured
    burnInHandle = BurnInHandle(None, burnInCfg)

    # Gets gpuId list
    gpuIds = burnInHandle.GetGpuIds()

    if len(gpuIds) < 1:
        print("\n...................................................\n")
        print("At least one GPU is required to run the burn_in stress test on a single node.\n")
        print("Set __DCGM_WL_BYPASS=1 in your environment to bypass the DCGM whitelist\n")
        sys.exit(1)
    else:
        for gpuId in gpuIds:
            print("The available devices are: " + color.BOLD + "GPU %d" % gpuId + color.ENDC)

    # Disconnect from embedded host engine
    del(burnInHandle)
    burnInHandle = None

    start_timestamp = time.asctime()
    host_engine = None
    if not test_utils.is_hostengine_running():
        # Starting HostEngine
        host_engine = RunHostEngine(burnInCfg.writeHostEngineDebugFile)
        host_engine.start(int(burnInCfg.runtime)+5)
        try:
            _thread.start_new_thread(lambda: host_engine.mem_usage(burnInCfg.runtime), ())
            _thread.start_new_thread(lambda: host_engine.cpu_usage(burnInCfg.runtime), ())
        except:
            print("Error: unable to create thread") 
        time.sleep(1)
    else:
        print("\nHostengine detected, using existing nv-hostegine...")

    # We're now running a host engine daemon. Connect to it using TCP/IP
    burnInHandle = BurnInHandle("127.0.0.1", burnInCfg)
    dcgmi_client = RunDcgmi(burnInHandle)
    time.sleep(2)
    dcgmi_client.start(int(burnInCfg.runtime)+1)
    time.sleep(2)

    print("\nStart timestamp: %s" % start_timestamp)

    # Finish when done
    dcgmi_client.wait()
    if host_engine is not None:
        host_engine.wait()
        host_engine.terminate()
        host_engine.validate()

    #Disconnect from host engine
    del(burnInHandle)
    burnInHandle = None

def validate_ip(s):
    # Function to validate ip addresses
    a = s.split('.')
    if len(a) != 4:
        return False
    for x in a:
        if not x.isdigit():
            return False
        i = int(x)
        if i < 0 or i > 255:
            return False
    color = bcolors()
    print(color.GREEN + "IP Validate successfully..." + color.ENDC)
    return True

# Class for holding global configuration information for this test module
class BurnInGlobalConfig:
    def __init__(self):
        self.remote = False  #Are we connecting to a remote server? True = Yes. False = No
        self.eud = True      #Should we run the EUD? True = Yes
        self.runtime = 0     #How long to run the tests in seconds
        self.onlyGpuIds = []      #Only GPU IDs this framework should run on

        #Undocumented globals I'm pulling out of the global namespace for sanity's sake
        self.srv = None
        self.ip = None
        self.nodes = []
        self.server = []

def parseCommandLine():
    burnInCfg = BurnInGlobalConfig()

    color = bcolors()

    # Script supports following args
    # -t time to run in seconds
    # -s (after that, one or more server names/IPs, default is localhost)
    # -a (after that one or more client names, default is localhost) or
    # -n (one or more client names from a text file)
    # -i GPU ids separated by commas

    # Parsing arguments
    parser = argparse.ArgumentParser(description="BURN-IN STRESS TEST")
    parser.add_argument("-t", "--runtime", required=True, help="Number of seconds to keep the test running")
    #nargs="+" means no args or any number of arguments
    parser.add_argument("-a", "--address", nargs="+", help="One or more IP addresses separated by spaces where DCGMI Clients will run on ")
    parser.add_argument("-n", "--nodesfile", help="File with list of IP address or hostnames where DCGMI Clients will run on")
    parser.add_argument("-s", "--server", help="Server IP address where DCGM HostEngines will running on")
    parser.add_argument("-ne", "--noeud",  action="store_true", help="Runs without the EUD Diagnostics test")
    parser.add_argument("-i", "--indexes", nargs=1, help="One or more GPU IDs to run the burn-in tests on, separated by commas. These come from 'dcgmi discovery -l'")
    parser.add_argument("-d", "--debug", action="store_true", help="Write debug logs from nv-hostengine")

    args = parser.parse_args()

    # passes stdout to the Logger Class
    sys.stdout = Logger()

    # Parsing time to run argument
    if args.runtime:
        burnInCfg.runtime = str(args.runtime)
        if args.runtime.isdigit():
            print("\n...................................................")
            print(color.YELLOW + " ##### DCGM BURN-IN STRESS TEST ##### " + color.ENDC)
            print("...................................................\n")
        else:
            print(color.RED + "\nPlease enter a decimal number for the time option.\n" + color.ENDC)
            sys.exit(1)

    # Runs without EUD tests
    if args.noeud:
        burnInCfg.eud = False
        print(color.PURPPLE + "## Running without Diagnostics Tests ##\n" + color.ENDC)

    # Parsing server arguments
    burnInCfg.server = []
    if args.server:
        burnInCfg.remote = True
        burnInCfg.srv = args.server
        if validate_ip(burnInCfg.srv):
            burnInCfg.server.append(burnInCfg.srv)
        else:
            print(color.RED + "\n%s is invalid. Please enter a valid IP address.\n" + color.ENDC)
            sys.exit(1)

    # Parsing IP address arguments
    burnInCfg.ip = []
    burnInCfg.nodes = []
    if args.address:
        if not args.server:
            print(color.RED + "The server IP was not informed" + color.ENDC)
            sys.exit(1)

        if "," in args.address:
            print(color.RED + "Comma separator is not allowed ok" + color.ENDC)
            sys.exit(1)

        for add in args.address:
            if validate_ip(add):
                burnInCfg.ip.append(add)
            else:
                print(color.RED + "\nFailed to validate IP Address %s\n" % add + color.ENDC)
                sys.exit(1)

        print(color.PURPPLE + "IP address to run on: %s " % str(burnInCfg.ip) + color.ENDC)
        print("\n...................................................\n")

    else:
        if args.nodesfile:
            burnInCfg.remote = True
            if not args.server:
                print(color.RED + "The server IP was not informed" + color.ENDC)
                sys.exit(1)
            # Parsing hostfile argument
            location = os.path.abspath(os.path.join(args.nodesfile))

            if os.path.exists(location):
                print(color.PURPPLE + "Nodes list file used \"%s\"\n" % str(location).split() + color.ENDC)

                # Reads the node list file and removes newlines from each line
                f = open(location, 'r')
                for lines in f.readlines():
                    burnInCfg.nodes.append(lines[:-1])
            print(color.PURPPLE + "IP address to run on: %s " % burnInCfg.nodes + color.ENDC)
            print("\n...................................................\n")

    if args.indexes:
        burnInCfg.onlyGpuIds = []
        for gpuId in args.indexes[0].split(','):
            if not gpuId.isdigit():
                print(color.RED + "GPU ID '%s' must be a number" % gpuId + color.ENDC)
                sys.exit(1)
            burnInCfg.onlyGpuIds.append(int(gpuId))

    if args.debug:
        burnInCfg.writeHostEngineDebugFile = True
    else:
        burnInCfg.writeHostEngineDebugFile = False

    return burnInCfg


def cleanup():
    '''
    Clean up our environment before exit
    '''
    apps.AppRunner.clean_all()

def main_wrapped():
    # Initialize the framework's option parser so we can use framework classes
    option_parser.initialize_as_stub()
    
    if not utils.is_root():
        sys.exit("\nOnly root can run this script\n")

    #Parse the command line
    burnInCfg = parseCommandLine()

    setupEnvironment()

    #initialize the DCGM library globally ONCE
    try:
        dcgm_structs._dcgmInit(utils.get_testing_framework_library_path())
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_LIBRARY_NOT_FOUND):
        print("DCGM Library hasn't been found in the system, is the driver correctly installed?", file=sys.stderr)
        sys.exit(1)

    if not burnInCfg.remote:
        run_tests(burnInCfg)
    elif len(burnInCfg.ip)==0:
        run_cuda_workload(burnInCfg)
        run_dcgmi_client(burnInCfg)
    else:
        copy_files_to_targets(burnInCfg.ip)
        run_local_host_engine(burnInCfg)
        remotes=[]
        for address in burnInCfg.ip:
            remotes.append(run_remote(burnInCfg.runtime, address, burnInCfg.srv, burnInCfg.nodes))

        for remote in remotes:
            remote.wait()

    getTestSummary()

def main():
    try:
        main_wrapped()
    except Exception as e:
        print(("Got exception " + str(e)))
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()

