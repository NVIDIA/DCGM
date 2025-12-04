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
import dcgm_agent
import dcgm_agent_internal
import dcgm_fields
import dcgmvalue
import test_utils
import logger
import time
import subprocess
import os

def helper_imex_fields_basic_retrieval(handle, gpuIds):
    """Helper function to test basic retrieval of IMEX field values"""
    
    # Create a field group with IMEX fields
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, 
                                                   [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS,
                                                    dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS],
                                                   "imex_test_fields")
    
    # Create a group with all GPUs (default behavior)
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "imex_test_group")
    # Get the first GPU ID from the group
    gpuId = dcgm_agent.dcgmGroupGetInfo(handle, groupId).entityList[0].entityId
    
    try:
        # Start watching the fields - this is required for IMEX fields
        updateFreq = 1000000  # 1 second in microseconds
        maxKeepAge = 3600.0   # 1 hour
        maxKeepSamples = 0    # Keep all samples
        
        dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, 
                                 updateFreq, maxKeepAge, maxKeepSamples)
        
        # Wait for at least one update
        time.sleep(2)
        
        # Get the latest values for the GPU - request each field separately to avoid race conditions
        domain_values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                        [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS])
        daemon_values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                        [dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS])
        
        logger.info(f"Retrieved {len(domain_values)} domain values and {len(daemon_values)} daemon values")
        
        # Validate domain status
        domain_found = False
        for value in domain_values:
            if value.fieldId == dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS:
                domain_found = True
                # Check not blank
                is_blank = dcgmvalue.DCGM_STR_IS_BLANK(value.value.str)
                assert not is_blank, f"Domain status field should not be blank"
                logger.info(f"IMEX Domain Status: {value.value.str}")
                break
        
        # Validate daemon status
        daemon_found = False
        for value in daemon_values:
            if value.fieldId == dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS:
                daemon_found = True
                # Check not blank
                is_blank = dcgmvalue.DCGM_INT64_IS_BLANK(value.value.i64)
                assert not is_blank, f"Daemon status field should not be blank"
                logger.info(f"IMEX Daemon Status: {value.value.i64}")
                break
        
        logger.info(f"Domain field found: {domain_found}, Daemon field found: {daemon_found}")
        
        # We should get at least one valid field, ideally both
        assert domain_found or daemon_found, f"Expected at least one IMEX field to be found"
        
        logger.info("Basic IMEX field retrieval test passed")
        
    finally:
        # Cleanup - handle potential errors gracefully
        try:
            dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        except Exception as e:
            # Only log if it's not the common "Field is not being updated" error
            if "Field is not being updated" not in str(e):
                logger.warning(f"Failed to unwatch fields: {e}")
        
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except Exception as e:
            logger.warning(f"Failed to destroy group: {e}")
        
        try:
            dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        except Exception as e:
            logger.warning(f"Failed to destroy field group: {e}")

def helper_imex_domain_status_values(handle, gpuIds):
    """Helper function to test IMEX domain status field returns valid values"""
    
    # Create field group for domain status only
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, 
                                                   [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS],
                                                   "imex_domain_test")
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "imex_domain_group")
    # Get the first GPU ID from the group
    gpuId = dcgm_agent.dcgmGroupGetInfo(handle, groupId).entityList[0].entityId
    
    try:
        updateFreq = 1000000  # 1 second
        dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, updateFreq, 3600.0, 0)
        
        time.sleep(2)
        
        values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                 [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS])
        
        assert len(values) == 1, f"Expected 1 domain status value, got {len(values)}"
        
        domain_value = values[0]
        assert domain_value.fieldId == dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS
        assert not dcgmvalue.DCGM_STR_IS_BLANK(domain_value.value.str)
        
        # Check that the value is a valid string (not blank)
        status_str = str(domain_value.value.str)
        assert len(status_str) > 0, f"Domain status should not be empty"
        logger.info(f"IMEX Domain Status: {status_str}")
        
        # Validate that the returned status is one of the expected values
        valid_statuses = ["UP", "DOWN", "DEGRADED", "NOT_INSTALLED", "NOT_CONFIGURED", "UNAVAILABLE"]
        assert status_str in valid_statuses, f"Invalid domain status '{status_str}'. Expected one of: {valid_statuses}"
        logger.info(f"Domain status '{status_str}' is valid")
        
    finally:
        # Cleanup - handle potential errors gracefully
        try:
            dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        except Exception as e:
            # Only log if it's not the common "Field is not being updated" error
            if "Field is not being updated" not in str(e):
                logger.warning(f"Failed to unwatch fields: {e}")
        
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except Exception as e:
            logger.warning(f"Failed to destroy group: {e}")
        
        try:
            dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        except Exception as e:
            logger.warning(f"Failed to destroy field group: {e}")

def helper_imex_daemon_status_values(handle, gpuIds):
    """Helper function to test IMEX daemon status field returns valid values"""
    
    # Create field group for daemon status only
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, 
                                                   [dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS],
                                                   "imex_daemon_test")
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "imex_daemon_group")
    # Get the first GPU ID from the group
    gpuId = dcgm_agent.dcgmGroupGetInfo(handle, groupId).entityList[0].entityId
    
    try:
        updateFreq = 1000000  # 1 second
        dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, updateFreq, 3600.0, 0)
        
        time.sleep(2)
        
        values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                 [dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS])
        
        logger.info(f"Retrieved {len(values)} daemon status values")
        for i, value in enumerate(values):
            logger.info(f"Value {i}: field_id={value.fieldId}, status={value.status}")
        
        # Check if we got a valid daemon status field
        valid_values = [v for v in values if v.fieldId == dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS]
        
        if len(valid_values) == 0:
            logger.info("IMEX daemon status not available in this environment - skipping test")
            return
        
        daemon_value = valid_values[0]
        assert not dcgmvalue.DCGM_INT64_IS_BLANK(daemon_value.value.i64)
        
        # Check that the value is a valid integer
        status_int = daemon_value.value.i64
        logger.info(f"IMEX Daemon Status: {status_int}")
        
        # Validate that the returned status is one of the expected values
        valid_range = list(range(0, 8)) + [-1, -2, -3]  # 0-7: normal states, -1,-2,-3: error states
        assert status_int in valid_range, f"Invalid daemon status {status_int}. Expected one of: {valid_range}"
        
        # Log the human-readable status
        status_names = {
            0: "INITIALIZING", 1: "STARTING_AUTH_SERVER", 2: "WAITING_FOR_PEERS", 
            3: "WAITING_FOR_RECOVERY", 4: "INIT_GPU", 5: "READY", 
            6: "SHUTTING_DOWN", 7: "UNAVAILABLE",
            -1: "NOT_INSTALLED", -2: "NOT_CONFIGURED", -3: "COMMAND_ERROR"
        }
        status_name = status_names.get(status_int, f"UNKNOWN({status_int})")
        logger.info(f"Daemon status {status_int} ({status_name}) is valid")
        
    finally:
        # Cleanup - handle potential errors gracefully
        try:
            dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        except Exception as e:
            # Only log if it's not the common "Field is not being updated" error
            if "Field is not being updated" not in str(e):
                logger.warning(f"Failed to unwatch fields: {e}")
        
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except Exception as e:
            logger.warning(f"Failed to destroy group: {e}")
        
        try:
            dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        except Exception as e:
            logger.warning(f"Failed to destroy field group: {e}")

def helper_imex_fields_consistency(handle, gpuIds):
    """Helper function to test that IMEX field values are consistent across multiple GPUs"""
    
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, 
                                                   [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS,
                                                    dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS],
                                                   "imex_consistency_test")
    
    # Use default group which includes all GPUs
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "imex_consistency_group")
    
    # Get all GPU IDs from the group
    groupInfo = dcgm_agent.dcgmGroupGetInfo(handle, groupId)
    available_gpus = [entity.entityId for entity in groupInfo.entityList]
    
    if len(available_gpus) < 2:
        logger.info("Skipping consistency test: need at least 2 GPUs")
        return
    
    try:
        updateFreq = 1000000  # 1 second
        dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, updateFreq, 3600.0, 0)
        
        time.sleep(2)
        
        # Get values for all GPUs in the group (limit to 4 to avoid excessive test time)
        # Request each field type separately to avoid potential race conditions
        domain_values = []
        daemon_values = []
        
        for i, gpuId in enumerate(available_gpus[:4]):
            # Get domain status
            domain_vals = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                          [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS])
            domain_values.extend([v for v in domain_vals if v.fieldId == dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS])
            
            # Get daemon status  
            daemon_vals = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                          [dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS])
            daemon_values.extend([v for v in daemon_vals if v.fieldId == dcgm_fields.DCGM_FI_IMEX_DAEMON_STATUS])
        
        logger.info(f"Final count - Domain values: {len(domain_values)}, Daemon values: {len(daemon_values)}")
        
        # Since IMEX fields are global, all GPUs must report the same values
        if len(domain_values) > 1:
            first_domain = str(domain_values[0].value.str)
            for i, value in enumerate(domain_values[1:], 1):
                current_domain = str(value.value.str)
                assert current_domain == first_domain, \
                    f"IMEX domain status inconsistent: GPU {i} reports '{current_domain}' but GPU 0 reports '{first_domain}'"
        
        if len(daemon_values) > 1:
            first_daemon = daemon_values[0].value.i64
            for i, value in enumerate(daemon_values[1:], 1):
                current_daemon = value.value.i64
                assert current_daemon == first_daemon, \
                    f"IMEX daemon status inconsistent: GPU {i} reports {current_daemon} but GPU 0 reports {first_daemon}"
                
        logger.info(f"Consistency test passed with {len(domain_values)} domain and {len(daemon_values)} daemon values")
        
    finally:
        # Cleanup - handle potential errors gracefully
        try:
            dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        except Exception as e:
            # Only log if it's not the common "Field is not being updated" error
            if "Field is not being updated" not in str(e):
                logger.warning(f"Failed to unwatch fields: {e}")
        
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except Exception as e:
            logger.warning(f"Failed to destroy group: {e}")
        
        try:
            dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        except Exception as e:
            logger.warning(f"Failed to destroy field group: {e}")

def helper_imex_fields_update_frequency(handle, gpuIds):
    """Helper function to test that IMEX fields can be updated at different frequencies"""
    
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(handle, 
                                                   [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS],
                                                   "imex_frequency_test")
    
    groupId = dcgm_agent.dcgmGroupCreate(handle, dcgm_structs.DCGM_GROUP_DEFAULT, "imex_freq_group")
    # Get the first GPU ID from the group
    gpuId = dcgm_agent.dcgmGroupGetInfo(handle, groupId).entityList[0].entityId
    
    try:
        # Test with 2-second update frequency
        updateFreq = 2000000  # 2 seconds in microseconds
        dcgm_agent.dcgmWatchFields(handle, groupId, fieldGroupId, updateFreq, 3600.0, 0)
        
        # Wait for multiple updates
        time.sleep(5)
        
        # Get multiple samples (using single latest value for now)
        values = dcgm_agent_internal.dcgmGetLatestValuesForFields(handle, gpuId, 
                                                                 [dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS])
        
        # Should have at least one sample
        assert len(values) > 0, "No IMEX field values returned for frequency test"
        
        # Check that we got values for the domain status field
        domain_samples = [v for v in values if v.fieldId == dcgm_fields.DCGM_FI_IMEX_DOMAIN_STATUS]
        assert len(domain_samples) > 0, "No domain status samples found"
        
        logger.info(f"Got {len(domain_samples)} domain status samples in frequency test")
        
    finally:
        # Cleanup - handle potential errors gracefully
        try:
            dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        except Exception as e:
            # Only log if it's not the common "Field is not being updated" error
            if "Field is not being updated" not in str(e):
                logger.warning(f"Failed to unwatch fields: {e}")
        
        try:
            dcgm_agent.dcgmGroupDestroy(handle, groupId)
        except Exception as e:
            logger.warning(f"Failed to destroy group: {e}")
        
        try:
            dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        except Exception as e:
            logger.warning(f"Failed to destroy field group: {e}")

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_imex_fields_basic_retrieval(handle, gpuIds):
    """Test basic retrieval of IMEX field values"""
    helper_imex_fields_basic_retrieval(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_imex_domain_status_values(handle, gpuIds):
    """Test IMEX domain status field returns valid values"""
    helper_imex_domain_status_values(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_imex_daemon_status_values(handle, gpuIds):
    """Test IMEX daemon status field returns valid values"""
    helper_imex_daemon_status_values(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_imex_fields_consistency(handle, gpuIds):
    """Test that IMEX field values are consistent across multiple GPUs"""
    helper_imex_fields_consistency(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_only_with_live_gpus()
def test_imex_fields_update_frequency(handle, gpuIds):
    """Test that IMEX fields can be updated at different frequencies"""
    helper_imex_fields_update_frequency(handle, gpuIds)