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
# test the health module for DCGM

import pydcgm
import dcgm_structs
import dcgm_agent_internal
import test_utils
import time
from ctypes import *
import apps
import nvml_injection
from _test_helpers import skip_test_if_no_dcgm_nvml

def _generate_xid_43_with_cuda_assert(handle, busId, appTimeout, gpuId):
    app = apps.RunCudaAssert(["--ctxCreate", busId,
                              "--assertGpu", busId, str(appTimeout),
                              "--ctxDestroy", busId], env=test_utils.get_cuda_visible_devices_env(handle, gpuId))
    
    app.start(appTimeout*2)
    return app

@skip_test_if_no_dcgm_nvml()
@test_utils.run_with_additional_fatal_kmsg_xid('43')
@test_utils.run_with_current_system_injection_nvml()
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_kmsg_fatal_xid_parsing(handle, gpuIds):
    """
    This test verifies that DcgmCacheManager skips NVML driver calls on
    detecting a fatal XID in /dev/kmsg.
    By default, DcgmCacheManager assumes only XIDs 79 and 119 are fatal XIDs.
    This test first sets an environment variable to add XID 43 to this list,
    and generates XID 43 using cuda assert. It then verifies that no new NVML
    calls are made.
    """
    funcCallCounts = nvml_injection.c_injectNvmlFuncCallCounts_t()
    ret = dcgm_agent_internal.dcgmGetNvmlInjectFuncCallCount(handle, funcCallCounts)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    assert funcCallCounts.numFuncs > 0, "numFuncs value: %u" % funcCallCounts.numFuncs

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    assert len(gpuIds) > 0, "number of GPUs: %u" % len(gpuIds)
    gpuAttrib = systemObj.discovery.GetGpuAttributes(gpuIds[0])
    gpuPciBusId = gpuAttrib.identifiers.pciBusId

    app = _generate_xid_43_with_cuda_assert(handle, gpuPciBusId, 1000, gpuIds[0])
    app.wait()
    app.terminate()
    app.validate()
    time.sleep(10)

    ret = dcgm_agent_internal.dcgmResetNvmlInjectFuncCallCount(handle)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    time.sleep(10)
    funcCallCounts = nvml_injection.c_injectNvmlFuncCallCounts_t()
    ret = dcgm_agent_internal.dcgmGetNvmlInjectFuncCallCount(handle, funcCallCounts)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    assert funcCallCounts.numFuncs == 0, "numFuncs value: %u" % funcCallCounts.numFuncs
