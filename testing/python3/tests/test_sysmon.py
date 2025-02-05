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
# test the policy manager for DCGM

import pydcgm
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgm_structs
from ctypes import *
import sys
import time
import os
import pprint
import re
import DcgmGroup
import DcgmReader

import dcgm_field_injection_helpers

# The value for this environment variable doesn't matter
from test_globals import DCGM_SKIP_SYSMON_HARDWARE_CHECK

def helper_dcgm_sysmon_cpu_hierarchy(hierarchy):
    numa_hierarchy = test_utils.helper_read_numa_hierarchy()

    assert len(numa_hierarchy) == hierarchy.numCpus
    for i in range(hierarchy.numCpus):
        node_id = hierarchy.cpus[i].cpuId
        local_node = numa_hierarchy[node_id]
        for cpu_index in range(pydcgm.dcgm_structs.DCGM_MAX_NUM_CPUS):
            expected_bool = cpu_index in local_node['cpus']

            # TODO(aalsudani) replace with (sizeof(hierarchy.cpus[i].ownedCores.bitmask[0]) * CHAR_BIT)
            # The above was not working because sizeof(int) throws in Python
            bitmask_index = int(cpu_index / 64)
            bitmask_offset = int(cpu_index % 64)
            bitmask_mask = 1 << bitmask_offset;

            bitmask_value = int(hierarchy.cpus[i].ownedCores.bitmask[bitmask_index]) & bitmask_mask;
            bitmask_bool = bool(bitmask_value)

            assert bitmask_bool == expected_bool, f'''node {node_id} cpu {cpu_index} expected {expected_bool} actual {bitmask_bool} at index {bitmask_index} offset {bitmask_offset}'''

            # Also test dcgmCpuHierarchyCpuOwnsCore implementation
            cpu_owns_core = bool(pydcgm.dcgm_agent.dcgmCpuHierarchyCpuOwnsCore(cpu_index, hierarchy.cpus[node_id].ownedCores))
            assert cpu_owns_core == expected_bool, f'''node {node_id} cpu {cpu_index} cpu_owns_core {cpu_owns_core} expected {expected_bool}'''

@test_utils.run_only_on_numa_systems()
@test_utils.run_with_standalone_host_engine(5, heEnv={ DCGM_SKIP_SYSMON_HARDWARE_CHECK : "Nomad"})
def test_dcgm_sysmon_cpu_hierarchy(handle):
    """
    Verifies that we can read the CPU hierarchy
    """
    hierarchy = dcgm_agent.dcgmGetCpuHierarchy(handle)
    helper_dcgm_sysmon_cpu_hierarchy(hierarchy)

@test_utils.run_only_on_numa_systems()
@test_utils.run_with_standalone_host_engine(5, heEnv={ DCGM_SKIP_SYSMON_HARDWARE_CHECK : "Nomad"})
def test_dcgm_sysmon_cpu_hierarchy_v2(handle):
    """
    Verifies that we can read the CPU hierarchy with the v2 API
    """
    hierarchy = dcgm_agent.dcgmGetCpuHierarchy_v2(handle)
    helper_dcgm_sysmon_cpu_hierarchy(hierarchy)

@test_utils.run_with_standalone_host_engine(5)
@test_utils.run_only_with_live_cpus()
def test_dcgm_sysmon_cpu_hierarchy_serial_number(handle, cpuIds):
    """
    Verifies that we can read the serial number of NVIDIA CPUs with the CPU hierarchy 
    """
    hierarchy = dcgm_agent.dcgmGetCpuHierarchy_v2(handle)
    for i in range(hierarchy.numCpus):
        serial_number = hierarchy.cpus[i].serial
        assert serial_number, "Read empty serial number"

@test_utils.run_with_embedded_host_engine(heEnv={ DCGM_SKIP_SYSMON_HARDWARE_CHECK : "Sigzil"})
@test_utils.run_with_injection_cpus(1)
@test_utils.run_with_injection_cpu_cores(1)
def test_sysmon_reading_injected_values(handle, cpuIds, coreIds):
    entityPair               = dcgm_structs.c_dcgmGroupEntityPair_t()
    entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU_CORE
    entityPair.entityId      = coreIds[0]
    offset                   = 5

    injection_info = [ [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_TOTAL, 100.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_USER, 65.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_NICE, 66.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_SYS, 25.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_IRQ, 10.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_TEMP_CURRENT, 71.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_CLOCK_CURRENT, 1294 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_POWER_LIMIT, 150.0 ],
                       [ dcgm_fields.DCGM_FI_DEV_CPU_POWER_UTIL_CURRENT, 113.1 ],
    ]

    fieldIds = []
    for ii in injection_info:
        fieldIds.append(ii[0])
    
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    fieldGroup = pydcgm.DcgmFieldGroup(dcgmHandle, "my_field_group", fieldIds)
    dcgmGroup = DcgmGroup.DcgmGroup(dcgmHandle, groupName='cpuGroup')
    dcgmGroup.AddEntity(entityPair.entityGroupId, entityPair.entityId)
    dcgmGroup.samples.WatchFields(fieldGroup, 1000000, 3600.0, 0)

    for ii in injection_info:
        dcgm_field_injection_helpers.inject_value(handle, entityPair.entityId, ii[0],
                               ii[1], offset, verifyInsertion=True,
                               entityType=entityPair.entityGroupId)

    entities = [ entityPair ]
    values = dcgm_agent.dcgmEntitiesGetLatestValues(handle, entities, fieldIds, 0)
    for value in values:
        index = [x[0] for x in injection_info].index(value.fieldId)
        assert value.status == dcgm_structs.DCGM_ST_OK, "Couldn't read field %d: '%s'" % (value.fieldId, str(dcgm_structs.DCGMError(value.status)))

        if value.fieldId == dcgm_fields.DCGM_FI_DEV_CPU_CLOCK_CURRENT:
            assert value.value.i64 == injection_info[index][1], "Expected %d but read %d" % (injection_info[index][1], value.value.i64)
        else:
            assert value.value.dbl == injection_info[index][1], "Expected %f but read %f" % (injection_info[index][1], value.value.dbl)

@test_utils.run_only_on_numa_systems()
@test_utils.run_with_standalone_host_engine(5, heEnv={ DCGM_SKIP_SYSMON_HARDWARE_CHECK : "The Sunlit Man"})
def test_dcgm_sysmon_fields_with_dcgmreader(handle):
    """
    Read Sysmon data through dcgmreader
    """
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()

    cpuIds = dcgm_agent.dcgmGetEntityGroupEntities(handle, dcgm_fields.DCGM_FE_CPU, 0)
    cpuEntities = []
    for cpu in cpuIds:
        entityPair = dcgm_structs.c_dcgmGroupEntityPair_t()
        entityPair.entityGroupId = dcgm_fields.DCGM_FE_CPU
        entityPair.entityId = cpu
        cpuEntities.append(entityPair)


    dcgmGroup = dcgmSystem.GetGroupWithEntities('cpugroup', cpuEntities)

    fieldIds = [ dcgm_fields.DCGM_FI_DEV_CPU_UTIL_USER,
               ]

    updateFrequencyUsec = 200000 # 200ms
    sleepTime = updateFrequencyUsec / 1000000 * 2 # Convert to seconds and sleep twice as long; ensures fresh sample

    dr = DcgmReader.DcgmReader(fieldIds=fieldIds, updateFrequency=updateFrequencyUsec, maxKeepAge=30.0, entities=cpuEntities)
    dr.SetHandle(handle)

    for i in range(5):
        time.sleep(sleepTime)

        cpuLatest = dr.GetLatestEntityValuesAsFieldIdDict()[dcgm_fields.DCGM_FE_CPU]

        for cpuId in cpuIds:
            if len(cpuLatest[cpuId]) != len(fieldIds):
                missingFieldIds = []
                extraFieldIds = []
                for fieldId in fieldIds:
                    if fieldId not in cpuLatest[cpuId]:
                        missingFieldIds.append(fieldId)

                for fieldId in cpuLatest[cpuId]:
                    if fieldId not in fieldIds:
                        extraFieldIds.append(fieldId)

                errmsg = "i=%d, cpuId %d, len %d != %d" % (i, cpuId, len(cpuLatest[cpuId]), len(fieldIds))
                if len(missingFieldIds) > 0:
                    errmsg = errmsg + " GPU is missing entries for fields %s" % str(missingFieldIds)
                if len(extraFieldIds) > 0:
                    errmsg = errmsg + " GPU has extra entries for fields %s" % str(extraFieldIds)

                assert len(cpuLatest[cpuId]) == len(fieldIds), errmsg
