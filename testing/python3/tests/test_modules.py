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
    Make sure that the introspection module cannot be blacklisted after it's loaded
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect

    #Lazy load the introspection module
    dcgmSystem.introspect.state.toggle(dcgm_structs.DCGM_INTROSPECT_STATE.ENABLED)

    #Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    assert ms.statuses[moduleId].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[moduleId].status, dcgm_structs.DcgmModuleStatusLoaded)
    
    #Make sure we can't blacklist the module after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Blacklist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_blacklist_introspection(handle):
    '''
    Make sure that the introspection module can be blacklisted
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect
    
    dcgmSystem.modules.Blacklist(moduleId)

    #Try to lazy load the blacklisted introspection module
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        dcgmSystem.introspect.state.toggle(dcgm_structs.DCGM_INTROSPECT_STATE.ENABLED)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_in_use_health(handle):
    '''
    Make sure that the health module cannot be blacklisted after it's loaded
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
    
    #Make sure we can't blacklist the module after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Blacklist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_blacklist_health(handle):
    '''
    Make sure that the health module can be blacklisted
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth
    
    dcgmSystem.modules.Blacklist(moduleId)

    #Try to lazy load the blacklisted introspection module
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)

@test_utils.run_only_if_checking_libraries()
def test_dcgm_library_existence():
    libraries = [
        'libdcgmmoduleconfig.so',
        'libdcgmmodulehealth.so',
        'libdcgmmodulenvswitch.so',
        'libdcgm_cublas_proxy11.so',
        'libdcgmmodulediag.so',
        'libdcgmmoduleintrospect.so',
        'libdcgmmodulepolicy.so',
    ]

    name_to_found = {}

    for library in libraries:
        name_to_found[library] = False

    lib_path = utils.get_testing_framework_library_path()

    # Only check for the older proxy libraries if we aren't on aarch64
    if lib_path[-8:] != 'aarch64/':
        name_to_found['libdcgm_cublas_proxy10.so'] = False
        name_to_found['libdcgm_cublas_proxy9.so'] = False

    file_list = os.listdir(lib_path)
    for filename in file_list:
        if filename in name_to_found:
            name_to_found[filename] = True

    for name in name_to_found:
        assert name_to_found[name] == True, "Didn't find required library '%s' in library path '%s'" % (name, lib_path)

