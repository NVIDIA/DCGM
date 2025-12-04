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

import dcgm_structs
import logger
import test_utils

def validate_pulse_test_response(response, expected_results=None, require_test_found=True):
    """
    Helper method to validate pulse_test diagnostic response.
    
    Args:
        response: The diagnostic response object from action_validate_wrapper
        expected_results: List of acceptable test results (defaults to [PASS, FAIL, SKIP])
        require_test_found: Whether to assert if pulse_test is not found in response
    """
    if expected_results is None:
        expected_results = [
            dcgm_structs.DCGM_DIAG_RESULT_PASS,
            dcgm_structs.DCGM_DIAG_RESULT_FAIL,
            dcgm_structs.DCGM_DIAG_RESULT_SKIP
        ]
    
    # Verify we got a response
    assert response is not None, "Should have received a response from pulse_test"
    logger.info(f"Received response with {response.numTests} tests")
    
    # Look for the pulse_test in the response
    pulse_test_found = False
    for i in range(response.numTests):
        test_name = response.tests[i].name
        logger.info(f"Test {i}: {test_name}")
        
        if test_name == "pulse_test":
            pulse_test_found = True
            test_result = response.tests[i]
            
            # Check that the test completed with an acceptable result
            assert test_result.result in expected_results, \
                f"Unexpected pulse_test result: {test_result.result}"
            
            logger.info(f"pulse_test completed with result: {test_result.result}")
            
            # Log any info messages for debugging
            if hasattr(test_result, 'info') and test_result.info:
                for j in range(len(test_result.info)):
                    if test_result.info[j]:
                        logger.info(f"pulse_test info: {test_result.info[j]}")
            
            break
    
    if require_test_found:
        assert pulse_test_found, "pulse_test should have been found in the diagnostic response"

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_if_mig_is_disabled()
def test_pulse_test_basic_sanity(handle, gpuIds):
    """
    Basic sanity check for pulse_test to ensure it can launch and run properly.
    This test runs for a very short duration (2 seconds) to verify functionality
    without performing extensive testing.
    
    Equivalent to: dcgmi diag -r pulse_test --patterns 0 --test-duration 2
    """
    logger.info("Running pulse_test basic sanity check")
    
    # Use only the first GPU for this sanity check
    gpuId = gpuIds[0]
    
    # Create the diagnostic run structure
    runDiagInfo = dcgm_structs.c_dcgmRunDiag_v10()
    runDiagInfo.version = dcgm_structs.dcgmRunDiag_version10
    runDiagInfo.groupId = dcgm_structs.DCGM_GROUP_NULL
    runDiagInfo.entityIds = str(gpuId)
    runDiagInfo.flags = dcgm_structs.DCGM_RUN_FLAGS_FAIL_EARLY
    runDiagInfo.failCheckInterval = 1  # 1 second
    
    # Set test name to pulse_test
    testNameBytes = b"pulse_test"
    for i, byte_val in enumerate(testNameBytes):
        runDiagInfo.testNames[0][i] = byte_val
    
    # Set a very short test duration for sanity check (2 seconds) and patterns=0
    # This aligns with: dcgmi diag -r pulse_test --patterns 0 --test-duration 2
    testDurationBytes = b"pulse_test.test_duration=2;pulse_test.patterns=0"
    for i, byte_val in enumerate(testDurationBytes):
        runDiagInfo.testParms[0][i] = byte_val
    
    # Run the pulse_test
    logger.info(f"Starting pulse_test sanity check on GPU {gpuId} with 2-second duration and patterns=0")
    
    try:
        response = test_utils.action_validate_wrapper(runDiagInfo, handle, dcgm_structs.dcgmRunDiag_version10)
        validate_pulse_test_response(response)
        logger.info("pulse_test sanity check completed successfully")
    except Exception as e:
        logger.error(f"pulse_test sanity check failed with exception: {str(e)}")
        raise
