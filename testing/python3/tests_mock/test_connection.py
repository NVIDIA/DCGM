# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from tests.test_connection import _connect_v3_helper
from tests.test_connection import _test_connection_helper
import test_utils
import utils
import pydcgm
import dcgm_structs
import dcgm_fields
import logger
import apps
import time
import dcgm_agent
import datetime

import os
import signal

from _test_helpers import skip_test_if_no_dcgm_nvml, maybe_dcgm_nvml


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_connection_disconnect_error_after_shutdown(handle, gpuIds):
    '''
    Test that DCGM_ST_BADPARAM is returned when the dcgm API is used after
    a call to dcgmShutdown has been made.
    '''
    handle = pydcgm.DcgmHandle()
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')
    gpuIds = group.GetGpuIds()

    handle.Shutdown()

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_BADPARAM)):
        gpuIds = group.GetGpuIds()


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(passAppAsArg=True)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_standalone_connection_disconnect_error_after_hostengine_terminate(
        handle, gpuIds, hostengineApp):
    '''
    Test that DCGM_ST_CONNECTION_NOT_VALID is returned when the dcgm API is used after
    the hostengine process is terminated via `nv-hostengine --term`.
    '''

    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')
    gpuIds = group.GetGpuIds()

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


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(passAppAsArg=True)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_standalone_connection_disconnect_error_after_hostengine_murder(
        handle,
        gpuIds,
        hostengineApp):
    '''
    Test that DCGM_ST_CONNECTION_NOT_VALID is returned when the dcgm API is used after
    the hostengine process is killed via a `SIGKILL` signal.
    '''
    handle = pydcgm.DcgmHandle(handle)
    group = pydcgm.DcgmGroup(handle, groupName='test-connection')

    gpuIds = group.GetGpuIds()

    pid = hostengineApp.getpid()
    os.kill(pid, signal.SIGKILL)
    utils.wait_for_pid_to_die(pid)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        gpuIds = group.GetGpuIds()


@test_utils.run_only_as_root()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_connection_error_when_no_ip4_hostengine_exists(handle, gpuIds):
    if not utils.is_bare_metal_system():
        test_utils.skip_test("Virtualization Environment not supported")

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        # use a TEST-NET (rfc5737) addr instead of loopback in case a local
        # hostengine is running
        handle = pydcgm.DcgmHandle(ipAddress='192.0.2.0', timeoutMs=100)


@test_utils.run_only_as_root()
@test_utils.run_with_ipv6_enabled()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_connection_error_when_no_ipv6_hostengine_exists(handle, gpuIds):
    if not utils.is_bare_metal_system():
        test_utils.skip_test("Virtualization Environment not supported")

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(
                                  dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID)):
        # use a ULA (rfc4193) instead of loopback in case a local hostengine is
        # running
        pydcgm.DcgmHandle(ipAddress='[fd00::7fff:42]', timeoutMs=100)


@test_utils.run_only_as_root()
@test_utils.run_with_ipv6_enabled()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_ipv6_loopback(handle, gpuIds):
    nvHe = apps.NvHostEngineApp(['-b', '[::1]'])
    nvHe.start(timeout=90)

    handle = pydcgm.DcgmHandle(ipAddress='[::1]', timeoutMs=90)
    assert handle
    dcgmSystem = handle.GetSystem()
    gpuIds = dcgmSystem.discovery.GetAllGpuIds()

    # Try to disconnect cleanly
    del (handle)
    handle = None

    nvHe.terminate()
    nvHe.validate()


@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_connection_versions(handle, gpuIds):
    '''
    Test that different versions of dcgmConnect_v2 work
    '''
    localhostStr = "127.0.0.1"

    v1Struct = dcgm_structs.c_dcgmConnectV2Params_v1()
    v1Struct.version = dcgm_structs.c_dcgmConnectV2Params_version1
    # These APIs throw exceptions on error
    v1Handle = dcgm_agent.dcgmConnect_v2(
        localhostStr, v1Struct, dcgm_structs.c_dcgmConnectV2Params_version1)

    v2Struct = dcgm_structs.c_dcgmConnectV2Params_v2()
    v2Struct.version = dcgm_structs.c_dcgmConnectV2Params_version2
    # These APIs throw exceptions on error
    v2Handle = dcgm_agent.dcgmConnect_v2(
        localhostStr, v2Struct, dcgm_structs.c_dcgmConnectV2Params_version2)

    # Do a basic request with each handle
    gpuIds = dcgm_agent.dcgmGetAllSupportedDevices(v1Handle)
    gpuIds2 = dcgm_agent.dcgmGetAllSupportedDevices(v2Handle)

    # Clean up the handles
    dcgm_agent.dcgmDisconnect(v1Handle)
    dcgm_agent.dcgmDisconnect(v2Handle)


# Add a date-based extension to the path to prevent having trouble when the framework is run as root
# and then again as non-root
# domainSocketFilename = '/tmp/dcgm_test%s' % (datetime.datetime.now().strftime("%j%f"))

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.get_domainSocketFilename_and_heArgs()
@test_utils.run_with_standalone_host_engine(20,
                                            heArgs=[],
                                            initializedClient=True)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_connection_domain_socket(handle, gpuIds, domainSocketFilename):
    '''
    Test that DCGM can listen on a unix domain socket, you can connect to it,
    and you can do basic queries against it
    '''
    _test_connection_helper(domainSocketFilename)


defaultSocketFilename = '/tmp/nv-hostengine'


@test_utils.run_only_as_root()
@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku('A100.yaml')
@test_utils.run_with_standalone_host_engine(20,
                                            ipAddress='/tmp/nv-hostengine',
                                            domainSocketFilename='/tmp/nv-hostengine',
                                            heArgs=[
                                                '-d', '/tmp/nv-hostengine'],
                                            initializedClient=True)
@test_utils.run_with_nvml_injected_gpus()
@test_utils.run_only_with_nvml()
def test_dcgm_connection_domain_socket_default(handle, gpuIds):
    '''
    Test that DCGM can listen on the default unix domain socket, you can connect to it,
    and you can do basic queries against it
    '''
    _test_connection_helper(defaultSocketFilename)


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_multiple_hostengine_connections(handle, gpuIds):

    he2app = apps.NvHostEngineApp(args=["-p 7777"], pid_dir="/tmp")
    he2app.start(timeout=10)

    remotehostStr = "127.0.0.1:7777"

    he2 = dcgm_structs.c_dcgmConnectV2Params_v2()
    he2.version = dcgm_structs.c_dcgmConnectV2Params_version2
    he2handle = dcgm_agent.dcgmConnect_v2(
        remotehostStr, he2, dcgm_structs.c_dcgmConnectV2Params_version2)

    # Do multiple basic requests with each handle to verify handle consistency
    he1gpus = dcgm_agent.dcgmGetAllSupportedDevices(handle)
    he2gpus = dcgm_agent.dcgmGetAllSupportedDevices(he2handle)
    assert len(he1gpus) > len(he2gpus), "number of gpus should be greater"

    he1gpus = dcgm_agent.dcgmGetAllSupportedDevices(handle)
    he2gpus = dcgm_agent.dcgmGetAllSupportedDevices(he2handle)
    assert len(he1gpus) > len(he2gpus), "number of gpus should be greater"

    he1gpus = dcgm_agent.dcgmGetAllSupportedDevices(handle)
    he2gpus = dcgm_agent.dcgmGetAllSupportedDevices(he2handle)
    assert len(he1gpus) > len(he2gpus), "number of gpus should be greater"

    he2app.terminate()


@test_utils.run_with_standalone_host_engine(20,
                                            ipAddress='localHost',
                                            heArgs=['--port', '5545'],
                                            initializedClient=True,
                                            port=5545)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_connect_v3_ip4(handle, gpuIds):
    _connect_v3_helper(
        'localhost:5545', dcgm_structs.c_dcgmConnectV3Params_v1())


@test_utils.run_with_ipv6_enabled()
@test_utils.run_with_standalone_host_engine(20,
                                            ipAddress='tcp://[::1]',
                                            heArgs=['-b',
                                                    '[::1]',
                                                    '--port',
                                                    '5545'],
                                            initializedClient=True,
                                            port=5545)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_connect_v3_ip6(handle, gpuIds):
    _connect_v3_helper('tcp://[::1]:5545',
                       dcgm_structs.c_dcgmConnectV3Params_v1())


@test_utils.run_with_standalone_host_engine(20,
                                            ipAddress='unix:///tmp/nv-hostengine_test',
                                            heArgs=['-d',
                                                    '/tmp/nv-hostengine_test'],
                                            initializedClient=True)
@test_utils.run_with_injection_gpus(1)
@test_utils.run_only_with_nvml()
def test_dcgm_connect_v3_domain_socket(handle, gpuIds):
    _connect_v3_helper('unix:///tmp/nv-hostengine_test',
                       dcgm_structs.c_dcgmConnectV3Params_v1())
