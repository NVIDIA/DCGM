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
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
from dcgm_structs import dcgmExceptionClass
import test_utils
import time
import os
import sys

# Set up the environment for the DcgmCollectd class before importing
os.environ['DCGM_TESTING_FRAMEWORK'] = 'True'
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['DCGMLIBPATH'] = os.environ['LD_LIBRARY_PATH']

stubspath  = os.path.dirname(os.path.realpath(__file__)) + '/stubs/'
if stubspath not in sys.path:
     sys.path.insert(0, stubspath)

import collectd_tester_globals
import dcgm_collectd_plugin


class Config:
    """
    pseudo collectd Config class.
    """

    def __init__(self, key = None, values = None):
        self.key = key
        self.values = values
        self.children = []

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_collectd_basic_integration(handle, gpuIds):
    """ 
    Verifies that we can inject specific data and get that same data back
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()

    specificFieldIds = [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION]

    fieldValues = [1,
                   5,
                   1000,
                   9000]

    for gpuId in gpuIds:    
        for i in range(0, len(specificFieldIds)):
            field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
            field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
            field.fieldId = specificFieldIds[i]
            field.status = 0
            field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
            field.ts = int((time.time()+10) * 1000000.0) # set the injected data into the future
            field.value.i64 = fieldValues[i]
            ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
            assert (ret == dcgm_structs.DCGM_ST_OK)

    gvars = collectd_tester_globals.gvars

    assert 'config' in gvars
    gvars['config']()

    assert 'init' in gvars
    gvars['init']()

    assert 'read' in gvars
    gvars['read']()

    assert 'out' in gvars
    outDict = gvars['out']

    assert 'shutdown' in gvars
#    gvars['shutdown']()

    # Verify that we can read back the fields we watch.
    for gpuId in gpuIds:
        assert str(gpuId) in outDict

        gpuDict = outDict[str(gpuId)]

        for i in range(0, len(specificFieldIds)):
            fieldTag = dcgmSystem.fields.GetFieldById(specificFieldIds[i]).tag
            assert fieldTag in gpuDict
            assert gpuDict[fieldTag] == fieldValues[i]


@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_collectd_config_integration(handle, gpuIds):
    """
    Verifies that we can parse config and get specified fields back.
    """
    config = Config()

    config.children = [Config('Interval', ['2']), Config('FieldIds', ['(100,memory_clock):5,video_clock:.1'])]
    gvars = collectd_tester_globals.gvars

    assert 'config' in gvars
    gvars['config'](config)

    assert 'init' in gvars
    gvars['init']()

    assert 'read' in gvars
    gvars['read']()

    assert 'out' in gvars
    outDict = gvars['out']

    assert 'shutdown' in gvars
    gvars['shutdown']()

    # Verify that we can read back the fields we watch.
    fieldTags = [ 'sm_clock', 'memory_clock', 'video_clock' ]

    for gpuId in gpuIds:
        assert str(gpuId) in outDict

        gpuDict = outDict[str(gpuId)]

        for fieldTag in fieldTags:
            assert fieldTag in gpuDict
            # We don't actually verify the value here, just the field tag name.
            # This verifies that we parsed the fields properly, set the
            # watches, and actually retrieves values for those fields. The
            # value will likely be zero on an initial read, but we can't
            # guarantee this. The basic test checks reading back expected
            # values.
#           assert gpuDict[fieldTag] == fieldValues[i]


