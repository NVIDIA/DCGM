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
# Sample script to test python bindings for DCGM

import os
import re
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import option_parser
import utils
import dcgm_fields
import apps
import inspect
import dcgmvalue
import pydcgm
import time
import shlex
from dcgm_structs import dcgmExceptionClass
from subprocess import check_output, check_call, Popen, CalledProcessError

# Helper function to find out if DCGM is actually installed
def is_dcgm_package_installed():
    """ Find out if DCGM package is already installed """ 

    # Looks for the nv-hostengine in the $PATH and hides possible error output
    with open(os.devnull, "w") as fnull:
        nv_host_find = Popen(["which","nv-hostengine"], stdout=fnull, stderr=fnull)
        nv_host_find.wait()

    if nv_host_find.returncode == 0:
        return True
    else:
        return False

# Helper function to get the path to libdcgm.so.2
# Returns the path to libdcgm.so.2 as a string on success
# Returns None on failure
def get_libdcgm_path():
    """ 
    Returns relative path to libdcgm.so.2
    """
    return "../../lib/libdcgm.so.2"

@test_utils.run_only_as_root()
@test_utils.run_only_on_bare_metal()
@test_utils.run_with_embedded_host_engine()
def test_dcgm_stub_library(handle):
    """ 
    Verifies that DCGM fails gracefully using the stub library
    if a proper DCGM installation is not present or shared 
    libraries are not included in the library search path
    """

    if utils.is_esx_hypervisor_system():
        test_utils.skip_test("Compute Mode tests are not supported in VMware ESX Environments")

    if is_dcgm_package_installed():
        test_utils.skip_test("A DCGM package is already installed on this machine")

    # Checks if libdcgm.so.2 is set within LD_LIBRARY_PATH
    libdcgm_path = get_libdcgm_path()
    assert libdcgm_path is not None

    if libdcgm_path is not None:
        # Verify is stub library is present
        if not (os.path.isfile(libdcgm_path + "/libdcgm_stub.a")):
            test_utils.skip_test("Unable to find \"libdcgm_stub.a\" in %s" % libdcgm_path)
        else:
            dcgm_lib_original = libdcgm_path + "/libdcgm.so.2"
            dcgm_lib_modified = dcgm_lib_original + "_modified"
    else:
        # Tear down the environment by finding and renaming "libdcgm.so.2" to "libdcgm.so.2_orig"
        # gets the path to libdcgm.so.2, like: /usr/lib/libdcgm.so.2
        try:
            ldconfig_out_buf = check_output(["ldconfig","-p"])
            ldconfig_out = ldconfig_out_buf.decode('utf-8')
            dcgm_lib = [x for x in ldconfig_out.split("\n") if "libdcgm.so.2" in x]
            dcgm_lib_original = [x for x in dcgm_lib[0].split("=>")[-1] if x[0]!=" "]
            dcgm_lib_modified = [x for x in dcgm_lib_original + "_modified" if x[0]!=" "]
        except:
            test_utils.skip_test("Unable to find libdcgm.so.2 library")

    # Renaming the file
    try:
        os.rename(dcgm_lib_original,dcgm_lib_modified)
    except:
        test_utils.skip_test("Unable to rename libdcgm.so.2 library")

    try:
        stub_app = apps.DcgmStubRunnerApp()
        stub_app.start()
        pid = stub_app.getpid()
        stub_app.wait()
    finally:
        # Restore environment
        os.rename(dcgm_lib_modified,dcgm_lib_original)
        logger.info("stub_library_tet PID was %d" % pid)

    assert "!!!!!!!!" in stub_app.stdout_lines[1], "Failed to collect stub library output"
    assert "WARNING:" in stub_app.stdout_lines[2], "Failed to collect stub library output"


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.skip_blacklisted_gpus(["GeForce GT 640"])
def test_dcgm_agent_get_values_for_fields(handle, gpuIds):
    """
    Verifies that DCGM Engine can be initialized successfully
    """
    # Watch field so we can fetch it
    fieldId = dcgm_fields.DCGM_FI_DEV_NAME
    gpuId = gpuIds[0]

    ret = dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, 10000000, 86400.0, 0)
    assert(ret == dcgm_structs.DCGM_ST_OK)

    # wait for at least one update of the field before trying to read it
    ret = dcgm_agent.dcgmUpdateAllFields(handle, True)
    assert(ret == dcgm_structs.DCGM_ST_OK)

    values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, [fieldId,])
    assert values[0].status == dcgm_structs.DCGM_ST_OK
    assert chr(values[0].fieldType) == dcgm_fields.DCGM_FT_STRING, "Wrong field type: %s" % values[0].fieldType
    assert len(values[0].value.str) > 0
    logger.debug("Brand of GPU %u is %s" % (gpuId, values[0].value.str))


@test_utils.run_with_embedded_host_engine()
def test_dcgm_engine_watch_field_values(handle):
    """
    Verifies that cache manager can watch a field value
    """
    
    # Watch field so we can fetch it
    fieldId = dcgm_fields.DCGM_FI_DEV_NAME
    gpuId = 0
    
    try:
        fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fieldId)
        numWatchersBefore = fieldInfo.numWatchers
    except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_WATCHED) as e:
        numWatchersBefore = 0
    
    ret = dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, 10000000, 86400.0, 0)
    assert(ret == dcgm_structs.DCGM_ST_OK)

    fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fieldId)
    assert fieldInfo.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED, "Expected watch. got flags %08X" % fieldInfo.flags

    numWatchersAfter = fieldInfo.numWatchers
    assert numWatchersAfter == numWatchersBefore + 1, "Expected 1 extra watcher. Before %d. After %d" % (numWatchersBefore, numWatchersAfter)
    
@test_utils.run_with_embedded_host_engine()
def test_dcgm_engine_unwatch_field_value(handle):
    """
    Verifies that the cache manager can unwatch a field value
    """
    
    # Watch field so we can fetch it
    fieldId = dcgm_fields.DCGM_FI_DEV_NAME
    gpuId = 0

    ret = dcgm_agent_internal.dcgmWatchFieldValue(handle, gpuId, fieldId, 10000000, 86400.0, 0)
    assert(ret == dcgm_structs.DCGM_ST_OK)

    fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fieldId)
    numWatchersBefore = fieldInfo.numWatchers

    # Unwatch field 
    clearCache = 1
    ret = dcgm_agent_internal.dcgmUnwatchFieldValue(handle, gpuId, fieldId, clearCache)
    assert(ret == dcgm_structs.DCGM_ST_OK)

    fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, fieldId)
    numWatchersAfter = fieldInfo.numWatchers

    assert numWatchersAfter == numWatchersBefore - 1, "Expected 1 fewer watcher. Before %d. After %d" % (numWatchersBefore, numWatchersAfter)
    assert (numWatchersAfter > 0) or (0 == (fieldInfo.flags & dcgm_structs_internal.DCGM_CMI_F_WATCHED)), "Expected no watch. got flags %08X" % fieldInfo.flags

def helper_unwatch_field_values_public(handle, gpuIds):
    """
    Verifies that dcgm can unwatch a field value
    """
    fieldId = dcgm_fields.DCGM_FI_DEV_NAME
    fieldIds = [fieldId, ]
    
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('mygroup', gpuIds)
    fieldGroup = pydcgm.DcgmFieldGroup(handleObj, "myfieldgroup", fieldIds)

    updateFreq = 10000000
    maxKeepAge = 86400
    maxKeepSamples = 0

    #These are all gpuId -> watcher count
    numWatchersBefore = {}   
    numWatchersWithWatch = {}
    numWatchersAfter = {}

    #Get watch info before our test begins
    for gpuId in gpuIds:
        try:
            fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, fieldId)
            numWatchersBefore[gpuId] = fieldInfo.numWatchers
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_WATCHED) as e:
            numWatchersBefore[gpuId] = 0

    #Now watch the fields
    groupObj.samples.WatchFields(fieldGroup, updateFreq, maxKeepAge, maxKeepSamples)

    #Get watcher info after our watch and check it against before our watch
    for gpuId in gpuIds:
        fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, fieldId)
        numWatchersWithWatch[gpuId] = fieldInfo.numWatchers
        assert numWatchersWithWatch[gpuId] == numWatchersBefore[gpuId] + 1,\
               "Watcher mismatch at gpuId %d, numWatchersWithWatch[gpuId] %d != numWatchersBefore[gpuId] %d + 1" %\
                (gpuId, numWatchersWithWatch[gpuId], numWatchersBefore[gpuId])

    #Unwatch fields
    groupObj.samples.UnwatchFields(fieldGroup)

    #Get watcher count after our unwatch. This should match our original watch count
    for gpuId in gpuIds:
        fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, fieldId)
        numWatchersAfter[gpuId] = fieldInfo.numWatchers

    assert numWatchersBefore == numWatchersAfter, "Expected numWatchersBefore (%s) to match numWatchersAfter %s" %\
           (str(numWatchersBefore), str(numWatchersAfter))
    

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_unwatch_field_values_public_embedded(handle, gpuIds):
    helper_unwatch_field_values_public(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_unwatch_field_values_public_remote(handle, gpuIds):
    helper_unwatch_field_values_public(handle, gpuIds)

def helper_promote_field_values_watch_public(handle, gpuIds):
    """
    Verifies that dcgm can update a field value watch
    """
    fieldId = dcgm_fields.DCGM_FI_DEV_NAME
    fieldIds = [fieldId, ]
    
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetGroupWithGpuIds('mygroup', gpuIds)
    fieldGroup = pydcgm.DcgmFieldGroup(handleObj, "myfieldgroup", fieldIds)

    updateFreq = 100000 #100 msec
    maxKeepAge = 3600
    maxKeepSamples = 0

    #Track the number of watchers to make sure our watch promotion doesn't create another sub-watch
    #but rather updates the existing one
    numWatchersWithWatch = {}
    numWatchersAfter = {}

    #Watch the fields
    groupObj.samples.WatchFields(fieldGroup, updateFreq, maxKeepAge, maxKeepSamples)

    #Get watcher info after our watch and verify that the updateFrequency matches
    for gpuId in gpuIds:
        fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, fieldId)
        numWatchersWithWatch[gpuId] = fieldInfo.numWatchers
        assert fieldInfo.monitorFrequencyUsec == updateFreq, "after watch: fieldInfo.monitorFrequencyUsec %d != updateFreq %d" % \
               (fieldInfo.monitorFrequencyUsec, updateFreq)

    #Update the watch with a faster update frequency
    updateFreq = 50000 #50 msec
    groupObj.samples.WatchFields(fieldGroup, updateFreq, maxKeepAge, maxKeepSamples)

    #Get watcher info after our second watch and verify that the updateFrequency matches
    for gpuId in gpuIds:
        fieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handleObj.handle, gpuId, fieldId)
        numWatchersAfter[gpuId] = fieldInfo.numWatchers
        assert fieldInfo.monitorFrequencyUsec == updateFreq, "after watch: fieldInfo.monitorFrequencyUsec %d != updateFreq %d" % \
               (fieldInfo.monitorFrequencyUsec, updateFreq)

    assert numWatchersWithWatch == numWatchersAfter, "numWatchersWithWatch (%s) != numWatchersAfter (%s)" % \
           (str(numWatchersWithWatch), str(numWatchersAfter))
    

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_promote_field_values_watch_public_embedded(handle, gpuIds):
    helper_promote_field_values_watch_public(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_promote_field_values_watch_public_remote(handle, gpuIds):
    helper_promote_field_values_watch_public(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_engine_update_all_fields(handle):
    """
    Verifies that the cache manager can update all fields
    """

    waitForUpdate = True
    ret = dcgm_agent.dcgmUpdateAllFields(handle, waitForUpdate)
    assert(ret == dcgm_structs.DCGM_ST_OK)

@test_utils.run_only_on_linux()
@test_utils.run_only_as_root()
def test_dcgm_cgroups_device_block():
    """
    Test whether the correct device uuid is found when a
    device is blocked by cgroups.
    """
    try:
        cgsetPath = check_output(["which", "cgset"])
        cgclearPath = check_output(["which", "cgclear"])
    except CalledProcessError as e:
        logger.debug("Exception was: %s" % e)
        test_utils.skip_test("Unable to find cgset or gclear, skipping test.")

    if (not os.path.exists(cgsetPath.strip())) or (not os.path.exists(cgclearPath.strip())):
        test_utils.skip_test("Unable to find cgset or gclear, skipping test.")

    dcgmHandle = pydcgm.DcgmHandle()
    dcgmSystem = dcgmHandle.GetSystem()
    allDcgmGpuIds = dcgmSystem.discovery.GetAllSupportedGpuIds()
    if len(allDcgmGpuIds) > 0:
        # Mounting a new cgroups hierarchy
        try:
            os.system("mkdir devices")
            os.system("mount -t cgroup -o devices dcgm devices")
            os.system("cgcreate -g devices:/cgroup/dcgm_group1")
        except Exception as msg:
            logger.debug("Failed to mount cgroup with: %s" %  msg)
            test_utils.skip_test("Unable to create cgroups mount point, skipping test.")

        try:
            PrevGpuUuid = []
            for gpuId in allDcgmGpuIds:
                # Recording first GPU UUID seen
                PrevGpuUuid.append(dcgmSystem.discovery.GetGpuAttributes(gpuId).identifiers.uuid)
                
                logger.info("Blocking access to device %s using cgroups..." % dcgmSystem.discovery.GetGpuAttributes(gpuId).identifiers.deviceName)
                os.system("%s -r devices.deny='c 195:%d rwm' /" % (cgsetPath.strip(), gpuId))
            
            GpuUuid = []
            for gpuId in allDcgmGpuIds:
                # Release the cgroups restriction
                logger.info("Freeing access device %s using cgroups..." % dcgmSystem.discovery.GetGpuAttributes(gpuId).identifiers.deviceName)
                os.system("%s -r devices.allow='c 195:%d rwm' /" % (cgsetPath.strip(), gpuId))
                
                # Getting current GPU UUID
                GpuUuid.append(dcgmSystem.discovery.GetGpuAttributes(gpuId).identifiers.uuid)
            
            assert PrevGpuUuid == GpuUuid, "Previous UUIDs %s should have been the same as current GPU UUID %s" % (PrevGpuUuid, GpuUuid)

        finally:
            #This will always bring GPUs back out of cgroups
            os.system("umount dcgm")
            os.system("cgclear")
            os.system("rm -rf devices")

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_entity_api_sanity(handle, gpuIds):
    '''
    Test that the basic entity APIs behave sanely
    '''
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    entityList = systemObj.discovery.GetEntityGroupEntities(dcgm_fields.DCGM_FE_GPU, True)
    assert entityList == gpuIds, "entityList %s != gpuIds %s" % (str(entityList), str(gpuIds))

    #Now check unsupported GPU IDs. This will only behave differently if you have an old GPU
    gpuIds = systemObj.discovery.GetAllGpuIds()
    entityList = systemObj.discovery.GetEntityGroupEntities(dcgm_fields.DCGM_FE_GPU, False)
    assert entityList == gpuIds, "entityList %s != gpuIds %s" % (str(entityList), str(gpuIds))

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_all_supported_gpus()
@test_utils.skip_blacklisted_gpus(["GeForce GT 640"])
@test_utils.run_with_injection_nvswitches(2)
def test_dcgm_nvlink_link_state(handle, gpuIds, switchIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Will throw an exception on API error
    linkStatus = systemObj.discovery.GetNvLinkLinkStatus()

    assert linkStatus.version == dcgm_structs.dcgmNvLinkStatus_version2, "Version mismatch %d != %d" % (linkStatus.version, dcgm_structs.dcgmNvLinkStatus_version2)

    if len(systemObj.discovery.GetAllGpuIds()) == len(gpuIds):
        assert linkStatus.numGpus == len(gpuIds), "Gpu count mismatch: %d != %d" % (linkStatus.numGpus, len(gpuIds))

    allSwitchIds = test_utils.get_live_nvswitch_ids(handle)
    totalSwitchCount = len(allSwitchIds)
    assert linkStatus.numNvSwitches == totalSwitchCount, "NvSwitch count mismatch: %d != %d" % (linkStatus.numNvSwitches, totalSwitchCount)

    #Check for unset/duplicate GPU IDs
    if len(gpuIds) > 1:
        assert linkStatus.gpus[0].entityId != linkStatus.gpus[1].entityId, "Got same GPU entity ID"
    if len(switchIds) > 1:
        assert linkStatus.nvSwitches[0].entityId != linkStatus.nvSwitches[1].entityId, "Got same switch entity ID"

    #Further sanity checks
    for i in range(len(gpuIds)):
        assert linkStatus.gpus[i].entityId in gpuIds, "GPU index %d id %d missing from %s" % (i, linkStatus.gpus[i].entityId, str(gpuIds))
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU):
            ls = linkStatus.gpus[i].linkState[j]
            assert ls >= dcgm_structs.DcgmNvLinkLinkStateNotSupported and ls <= dcgm_structs.DcgmNvLinkLinkStateUp, "Invalid GPU linkState %d at i %d, j %d" % (ls, i, j) 
    
    for i in range(len(switchIds)):
        assert linkStatus.nvSwitches[i].entityId in allSwitchIds, "Switch index %d id %d missing from %s" % (i, linkStatus.nvSwitches[i].entityId, str(switchIds))
        for j in range(dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH):
            ls = linkStatus.nvSwitches[i].linkState[j]
            assert ls >= dcgm_structs.DcgmNvLinkLinkStateNotSupported and ls <= dcgm_structs.DcgmNvLinkLinkStateUp, "Invalid NvSwitch linkState %d at i %d, j %d" % (ls, i, j) 



