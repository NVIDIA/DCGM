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
import test_utils
import utils
import pydcgm
import dcgm_structs 
import dcgm_fields
import logger
import time
import dcgm_agent
import datetime

import os
import signal

def test_connection_disconnect_error_after_shutdown():
    '''
    Test that DCGM_ST_BADPARAM is returned when the dcgm API is used after
    a call to dcgmShutdown has been made.
    '''
    handle = pydcgm.DcgmHandle()
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')
    
    gpudIds = group.GetGpuIds()
    
    handle.Shutdown()
    
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        gpuIds = group.GetGpuIds()
        
@test_utils.run_with_standalone_host_engine(passAppAsArg=True)
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_connection_disconnect_error_after_hostengine_terminate(handle, hostengineApp):
    '''
    Test that DCGM_ST_CONNECTION_NOT_VALID is returned when the dcgm API is used after 
    the hostengine process is terminated via `nv-hostengine --term`.
    '''
    
    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')
    gpudIds = group.GetGpuIds()
    
    hostengineApp.terminate()
    hostengineApp.validate()
    
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        gpuIds = group.GetGpuIds()

# When fabric manager is enabled and the hostengine is killed via SIGKILL, the apprunner framework is unable to detect
# that the hostengine has actually stopped. In the app runner's retvalue() method, subprocess.poll() returns None
# which implies that the hostengine is still running. As a temporary WaR, we do not enable the fabric manager for this
# test. (It is possible that there is a race condition once SIGKILL is sent which causes subprocess.poll() 
# to return None - I did not get a chance to investigate it further).
@test_utils.run_with_standalone_host_engine(passAppAsArg=True)
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_connection_disconnect_error_after_hostengine_murder(handle, hostengineApp):
    '''
    Test that DCGM_ST_CONNECTION_NOT_VALID is returned when the dcgm API is used after 
    the hostengine process is killed via a `SIGKILL` signal.
    '''
    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')
    
    gpudIds = group.GetGpuIds()
    
    pid = hostengineApp.getpid()
    os.kill(pid, signal.SIGKILL)
    utils.wait_for_pid_to_die(pid)
    
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        gpuIds = group.GetGpuIds()
        
@test_utils.run_only_as_root()
def test_dcgm_connection_error_when_no_hostengine_exists():
    if not utils.is_bare_metal_system():
        test_utils.skip_test("Virtualization Environment not supported")

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        # use a TEST-NET (rfc5737) addr instead of loopback in case a local hostengine is running
        handle = pydcgm.DcgmHandle(ipAddress='192.0.2.0', timeoutMs=100)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_connection_client_cleanup(handle, gpuIds):
    '''
    Make sure that resources that were allocated by a client are cleaned up
    '''
    fieldGroupFieldIds = [dcgm_fields.DCGM_FI_DEV_GPU_TEMP, ]
    
    #Get a 2nd connection which we'll check for cleanup. Use the raw APIs so we can explicitly cleanup
    connectParams = dcgm_structs.c_dcgmConnectV2Params_v1()
    connectParams.version = dcgm_structs.c_dcgmConnectV2Params_version
    connectParams.persistAfterDisconnect = 0
    cleanupHandle = dcgm_agent.dcgmConnect_v2('localhost', connectParams)
    
    groupName = 'clientcleanupgroup'
    groupId = dcgm_agent.dcgmGroupCreate(cleanupHandle, dcgm_structs.DCGM_GROUP_EMPTY, groupName)
    
    fieldGroupName = 'clientcleanupfieldgroup'
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(cleanupHandle, fieldGroupFieldIds, fieldGroupName)
    
    #Disconnect our second handle. This should cause the cleanup to occur
    dcgm_agent.dcgmDisconnect(cleanupHandle)

    time.sleep(1.0) #Allow connection cleanup to occur since it's asynchronous
    
    #Try to retrieve the field group info. This should throw an exception
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NO_DATA)):
        fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(handle, fieldGroupId)
    
    #Try to retrieve the group info. This should throw an exception
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_connection_versions(handle):
    '''
    Test that different versions of dcgmConnect_v2 work
    '''
    localhostStr = "127.0.0.1"

    v1Struct = dcgm_structs.c_dcgmConnectV2Params_v1()
    v1Struct.version = dcgm_structs.c_dcgmConnectV2Params_version1
    #These APIs throw exceptions on error
    v1Handle = dcgm_agent.dcgmConnect_v2(localhostStr, v1Struct, dcgm_structs.c_dcgmConnectV2Params_version1)
    
    v2Struct = dcgm_structs.c_dcgmConnectV2Params_v2()
    v2Struct.version = dcgm_structs.c_dcgmConnectV2Params_version2
    #These APIs throw exceptions on error
    v2Handle = dcgm_agent.dcgmConnect_v2(localhostStr, v2Struct, dcgm_structs.c_dcgmConnectV2Params_version2)

    #Do a basic request with each handle
    gpuIds = dcgm_agent.dcgmGetAllSupportedDevices(v1Handle)
    gpuIds2 = dcgm_agent.dcgmGetAllSupportedDevices(v2Handle)

    #Clean up the handles
    dcgm_agent.dcgmDisconnect(v1Handle)
    dcgm_agent.dcgmDisconnect(v2Handle)


def _test_connection_helper(domainSocketName):
    #Make sure the library is initialized
    dcgm_agent.dcgmInit()
    #First, try the raw method of using the dcgm_agent API directly
    v2Struct = dcgm_structs.c_dcgmConnectV2Params_v2()
    v2Struct.version = dcgm_structs.c_dcgmConnectV2Params_version2
    v2Struct.addressIsUnixSocket = 1
    v2Handle = dcgm_agent.dcgmConnect_v2(domainSocketName, v2Struct, dcgm_structs.c_dcgmConnectV2Params_version2)
    #Use the handle, which will throw an exception on error
    gpuIds2 = dcgm_agent.dcgmGetAllSupportedDevices(v2Handle)
    dcgm_agent.dcgmDisconnect(v2Handle)

    #Now use the DcgmHandle method
    dcgmHandle = pydcgm.DcgmHandle(unixSocketPath=domainSocketName)
    dcgmSystem = dcgmHandle.GetSystem()

    gpuIds = dcgmSystem.discovery.GetAllGpuIds()

    #Try to disconnect cleanly from our domain socket
    del(dcgmHandle)
    dcgmHandle = None


# Add a date-based extension to the path to prevent having trouble when the framework is run as root
# and then again as non-root
domainSocketFilename = '/tmp/dcgm_test%s' % (datetime.datetime.now().strftime("%j%f"))

@test_utils.run_with_standalone_host_engine(20, heArgs=['-d', domainSocketFilename])
def test_dcgm_connection_domain_socket():
    '''
    Test that DCGM can listen on a unix domain socket, you can connect to it,
    and you can do basic queries against it
    '''
    _test_connection_helper(domainSocketFilename)


defaultSocketFilename = '/tmp/nv-hostengine'

@test_utils.run_only_as_root()
@test_utils.run_with_standalone_host_engine(20, heArgs=['-d'])
def test_dcgm_connection_domain_socket_default():
    '''
    Test that DCGM can listen on the default unix domain socket, you can connect to it,
    and you can do basic queries against it
    '''
    _test_connection_helper(defaultSocketFilename)
