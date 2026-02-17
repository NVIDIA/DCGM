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

import time
import dcgm_structs
import dcgm_fields
import DcgmDiag
import test_utils
import logger


def gpu_is_healthy_and_support_memtest(handle, gpuId):
    func = gpu_is_healthy_and_support_memtest
    if not hasattr(func, "result_cache"):
        func.result_cache = {}

    if gpuId in func.result_cache:
        return func.result_cache[gpuId]

    testName = "memtest"
    dd = DcgmDiag.DcgmDiag(
        gpuIds=[gpuId], testNamesStr=testName, paramsStr="memtest.test_duration=2")
    start_time = time.time()
    response = test_utils.diag_execute_wrapper(dd, handle)
    end_time = time.time()
    logger.info(
        f"{end_time - start_time:.2f} seconds spent on checking whether the GPU is healthy/supported.")
    entityPair = dcgm_structs.c_dcgmGroupEntityPair_t(
        dcgm_fields.DCGM_FE_GPU, gpuId)
    if not DcgmDiag.check_diag_result_pass(response, entityPair, testName):
        func.result_cache[gpuId] = False
        return False
    func.result_cache[gpuId] = True
    return True

