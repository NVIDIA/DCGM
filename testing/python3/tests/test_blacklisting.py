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
import dcgm_agent_internal
import dcgm_agent
import logger
import test_utils
import dcgm_fields
import dcgm_internal_helpers
import option_parser
import DcgmDiag
import DcgmHandle
import blacklist_recommendations

import threading
import time
import sys
import os
import signal
import utils
import json

from apps.app_runner import AppRunner

def helper_test_blacklist_briefly():
    # Run a basic test of the blacklist script to make sure we don't break compatibility
    blacklistApp = dcgm_internal_helpers.createBlacklistApp(instantaneous=True)
    try:
        blacklistApp.run()
    except Exception as e:
        assert False, "Exception thrown when running the blacklist app: '%s'" % str(e)

    try:
        output = ""
        for line in blacklistApp.stdout_lines:
            output += "%s\n" % line
        jo = json.loads(output)
    except Exception as e:
        assert False, "Couldn't parse the json output by the blacklist. Got exception: %s\noutput\n:%s" % (str(e), output)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_basic_blacklisting_script(handle, gpuIds):
    helper_test_blacklist_briefly()

def helper_test_blacklist_checks(handle, gpuIds):
    handleObj = DcgmHandle.DcgmHandle(handle=handle)
    settings = {}
    settings['instant'] = True
    settings['entity_get_flags'] = 0
    settings['testNames'] = '3'
    settings['hostname'] = 'localhost'
    settings['watches'] = dcgm_structs.DCGM_HEALTH_WATCH_MEM | dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    error_list = []
    
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                        dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 0, -50)
    blacklist_recommendations.check_health(handleObj, settings, error_list)

    # Make sure the GPUs pass a basic health test before running this test
    for gpuObj in blacklist_recommendations.g_gpus:
        if gpuObj.IsHealthy() == False:
            test_utils.skip_test("Skipping because GPU %d is not healthy. " % gpuObj.GetEntityId())

    # Inject a memory error and verify that we fail
    blacklist_recommendations.g_gpus = [] # Reset g_gpus
    
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                       dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, 1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    blacklist_recommendations.check_health(handleObj, settings, error_list)
    for gpuObj in blacklist_recommendations.g_gpus:
        if gpuObj.GetEntityId() == gpuIds[0]:
            assert gpuObj.IsHealthy() == False, "Injected error didn't trigger a failure on GPU %d" % gpuIds[0]
        else:
            assert gpuObj.IsHealthy(), "GPU %d reported unhealthy despite not having an inserted error: '%s'" % (gpuIds[0], gpuObj.WhyUnhealthy())
    
    # Remove the memory monitor and make sure we pass our checks
    blacklist_recommendations.g_gpus = [] # Reset g_gpus
    settings['watches'] = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    blacklist_recommendations.check_health(handleObj, settings, error_list)
    for gpuObj in blacklist_recommendations.g_gpus:
        if gpuObj.GetEntityId() == gpuIds[0]:
            assert gpuObj.IsHealthy(), "Injected error wasn't ignored for GPU %d: %s" % (gpuIds[0], gpuObj.WhyUnhealthy())
        else:
            assert gpuObj.IsHealthy(), "GPU %d reported unhealthy despite not having an inserted error: '%s'" % (gpuIds[0], gpuObj.WhyUnhealthy())

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_blacklist_checks(handle, gpuIds):
    helper_test_blacklist_checks(handle, gpuIds)
