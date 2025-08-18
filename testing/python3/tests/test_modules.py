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

import pydcgm
import dcgm_structs
import test_utils
import utils
import dcgm_agent
import os

@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_get_statuses(handle):
    '''
    Do a basic sanity check of the DCGM module statuses returned
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    ms = dcgmSystem.modules.GetStatuses()

    assert ms.numStatuses == dcgm_structs.DcgmModuleIdCount, "%d != %d" % (ms.numStatuses, dcgm_structs.DcgmModuleIdCount)
    assert ms.statuses[0].id == dcgm_structs.DcgmModuleIdCore, "%d != %d" % (ms.statuses[0].id, dcgm_structs.DcgmModuleIdCore)
    assert ms.statuses[0].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[0].status, dcgm_structs.DcgmModuleStatusLoaded)

    for i in range(1, ms.numStatuses):
        #.id == index
        assert ms.statuses[i].id == i, "%d != %d" % (ms.statuses[i].id, i)
        #Assert all non-core modules aren't loaded besides NvSwitch. This one can be loaded
        #because creating default groups causes a RPC to the NvSwitch manager
        if ms.statuses[i].id != dcgm_structs.DcgmModuleIdNvSwitch:
            assert ms.statuses[i].status == dcgm_structs.DcgmModuleStatusNotLoaded, "%d != %d" % (ms.statuses[i].status, dcgm_structs.DcgmModuleStatusNotLoaded)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_in_use_introspection(handle):
    '''
    Make sure that the introspection module cannot be added to denylist after it's loaded
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect

    #Lazy load the introspection module
    bytesUsed = dcgmSystem.introspect.memory.GetForHostengine().bytesUsed

    #Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    assert ms.statuses[moduleId].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[moduleId].status, dcgm_structs.DcgmModuleStatusLoaded)
    
    #Make sure we can't add the module to the denylist after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Denylist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_denylist_introspection(handle):
    '''
    Make sure that the introspection module can be added to the denylist
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect
    
    dcgmSystem.modules.Denylist(moduleId)

    #Try to lazy load the introspection module which is on the denylist
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        bytesUsed = dcgmSystem.introspect.memory.GetForHostengine().bytesUsed


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_in_use_health(handle):
    '''
    Make sure that the health module cannot be added to the denylist after it's loaded
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth

    #Lazy load the health module
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)

    #Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    assert ms.statuses[moduleId].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[moduleId].status, dcgm_structs.DcgmModuleStatusLoaded)
    
    #Make sure we can't add the module to the denylist after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Denylist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_denylist_health(handle):
    '''
    Make sure that the health module can be added to the denylist
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth
    
    dcgmSystem.modules.Denylist(moduleId)

    #Try to lazy load the introspection module which is on the denylist
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_paused(handle):
    """
    Make sure that a module is loaded in the paused state if the DCGM is paused
    And that it is resumed when DCGM is resumed
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth

    # First make sure the module is not loaded
    ms = dcgmSystem.modules.GetStatuses()
    status = ms.statuses[moduleId].status
    assert status == dcgm_structs.DcgmModuleStatusNotLoaded, "{} != {}".format(status,
                                                                               dcgm_structs.DcgmModuleStatusNotLoaded)

    dcgmSystem.PauseTelemetryForDiag()

    # Lazy load the health module
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)

    # Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    status = ms.statuses[moduleId].status
    assert status == dcgm_structs.DcgmModuleStatusPaused, "{} != {}".format(status,
                                                                            dcgm_structs.DcgmModuleStatusPaused)

    dcgmSystem.ResumeTelemetryForDiag()

    # Make sure the module was resumed
    ms = dcgmSystem.modules.GetStatuses()
    status = ms.statuses[moduleId].status
    assert status == dcgm_structs.DcgmModuleStatusLoaded, "{} != {}".format(status,
                                                                            dcgm_structs.DcgmModuleStatusLoaded)


@test_utils.run_only_if_checking_libraries()
def test_dcgm_library_existence():
    libraries = [
        'libdcgmmoduleconfig.so.4',
        'libdcgmmodulehealth.so.4',
        'libdcgmmodulenvswitch.so.4',
        'libdcgm_cublas_proxy11.so.4',
        'libdcgm_cublas_proxy12.so.4',
        'libdcgmmodulediag.so.4',
        'libdcgmmodulemndiag.so.4',
        'libdcgmmoduleintrospect.so.4',
        'libdcgmmodulepolicy.so.4',
    ]

    name_to_found = {}

    for library in libraries:
        name_to_found[library] = False

    lib_path = utils.get_testing_framework_library_path()

    file_list = os.listdir(lib_path)
    for filename in file_list:
        if filename in name_to_found:
            name_to_found[filename] = True

    for name in name_to_found:
        assert name_to_found[name] == True, "Didn't find required library '%s' in library path '%s'" % (name, lib_path)
