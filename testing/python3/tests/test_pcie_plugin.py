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

"""
Tests for PCIe diagnostic plugin functionality.

This module contains tests for various aspects of the PCIe diagnostic plugin.
"""

import dcgm_structs
import DcgmDiag
import test_utils


@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.for_all_same_sku_gpus()
def test_pcie_libnuma_error_handling(handle, gpuIds):
    """
    Test PCIe diagnostic libnuma error handling.
    
    - Run PCIe test
    - If no libnuma related error messages in stderr, we are good
    - If there are libnuma related error messages, verify they contain our solution string
    """
    testGpuId = gpuIds[0]
    testName = "pcie"
    
    # Run PCIe diagnostic
    dd = DcgmDiag.DcgmDiag(
        gpuIds=[testGpuId], 
        testNamesStr=testName,
        paramsStr="pcie.test_duration=5;pcie.is_allowed=true;pcie.test_with_gemm=false",
        version=dcgm_structs.dcgmRunDiag_version10
    )
    
    response = test_utils.diag_execute_wrapper(dd, handle)

    if response.numErrors > 0:
        for i in range(response.numErrors):
            error = response.errors[i]
            error_msg = error.msg.decode('utf-8') if isinstance(error.msg, bytes) else str(error.msg)
            
            # Check if this is a libnuma-related error
            if "libnuma" in error_msg.lower():
                # Verify the error contains our solution message
                expected_solution = "Install the libnuma1 package to resolve this issue"
                assert expected_solution in error_msg, \
                    f"Libnuma error found but missing solution message. Error: {error_msg}"
    
    # If no libnuma errors found, test passes (normal case)
    # If libnuma errors found, they must contain our solution message (verified above)