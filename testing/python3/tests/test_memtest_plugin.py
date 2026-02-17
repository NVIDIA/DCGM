# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import DcgmDiag
import test_utils
import dcgm_structs
import dcgm_fields


@test_utils.run_with_standalone_host_engine(120, heEnv=test_utils.smallFbModeEnv)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_memtest_plugin_skip_if_free_mem_less_than_threshold(handle, gpuIds):
    dd = DcgmDiag.DcgmDiag(gpuIds=[gpuIds[0]], testNamesStr="memtest",
                           paramsStr="memtest.is_allowed=True;memtest.minimum_allocation_percentage=100")

    response = test_utils.diag_execute_wrapper(dd, handle)
    assert response.numTests == 2

    memtestId = 1
    assert response.tests[memtestId].result == dcgm_structs.DCGM_DIAG_RESULT_SKIP

    gpuResult = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId ==
                     gpuIds[0] and cur.testId == memtestId, response.results[:min(response.numResults, dcgm_structs.DCGM_DIAG_RESPONSE_RESULTS_MAX)]), None)
    assert gpuResult, f"Expected to find a result for gpu {gpuIds[0]} with testId {memtestId}"
    assert gpuResult.result == dcgm_structs.DCGM_DIAG_RESULT_SKIP, f"Actual result is {gpuResult.result}"

    gpuInfo = next(filter(lambda cur: cur.entity.entityGroupId == dcgm_fields.DCGM_FE_GPU and cur.entity.entityId ==
                   gpuIds[0] and cur.testId == memtestId, response.info[:min(response.numInfo, dcgm_structs.DCGM_DIAG_RESPONSE_INFO_MAX_V2)]), None)
    assert gpuInfo, f"Expected to find a info for gpu {gpuIds[0]} with testId {memtestId}"
    assert gpuInfo.msg == "Free memory is less than 100% of total memory. Skipping memtest.", f"Actual info is {gpuInfo.msg}"
