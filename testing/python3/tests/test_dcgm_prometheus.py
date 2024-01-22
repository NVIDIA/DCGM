# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

FUTURE_INSERT_TIME = 2

# Set up the environment for the DcgmPrometheus class before importing
os.environ['DCGM_TESTING_FRAMEWORK'] = 'True'
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['DCGMLIBPATH'] = os.environ['LD_LIBRARY_PATH']

stubspath  = os.path.dirname(os.path.realpath(__file__)) + '/stubs/'
if stubspath not in sys.path:
     sys.path.insert(0, stubspath)

import dcgm_prometheus
import prometheus_tester_globals

@test_utils.run_with_standalone_host_engine(90)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_prometheus_basic_integration(handle, gpuIds):
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

    dcgm_prometheus.initialize_globals()
    dcgm_prometheus.g_settings['publishFieldIds'] = specificFieldIds
    dcgm_prometheus.g_settings['prometheusPublishInterval'] = 10
    dcgm_prometheus.g_settings['sendUuid'] = False
    dcgm_prometheus.g_settings['dcgmHostName'] = "localhost"
    dcgmPrometheus = dcgm_prometheus.DcgmPrometheus()
    dcgmPrometheus.Init()
    dcgmPrometheus.LogBasicInformation()

    for gpuId in gpuIds:    
        for i in range(0, len(specificFieldIds)):
            field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
            field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
            field.fieldId = specificFieldIds[i]
            field.status = 0
            field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
            field.ts = int((time.time()+FUTURE_INSERT_TIME) * 1000000.0) # set the injected data into the future 
            field.value.i64 = fieldValues[i]
            ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
            assert (ret == dcgm_structs.DCGM_ST_OK)

    # Verify that we can read back the fields we watch.

    time.sleep(FUTURE_INSERT_TIME)
    dcgmPrometheus.Scrape(dcgmPrometheus)
        
    for gpuId in gpuIds:
        for i in range(0, len(specificFieldIds)):
            fieldTag = dcgmSystem.fields.GetFieldById(specificFieldIds[i]).tag
            label = prometheus_tester_globals.gvars['fields']["dcgm_" + fieldTag]

            foundGpuId = False

            for uniqueGpuId, value in label.values.items():
                if gpuId == value.id:
                    foundGpuId = True
                    assert (fieldValues[i] == value.get())
                    
            assert(foundGpuId == True)
