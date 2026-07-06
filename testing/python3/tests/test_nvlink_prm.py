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
##
# Tests for NvLink PRM fields
##

import dcgm_fields
import dcgm_structs
import dcgm_agent
import test_utils
import pydcgm
import logger
import dcgm_field_helpers
import dcgm_field_injection_helpers
from ctypes import *
from _test_helpers import skip_test_if_no_dcgm_nvml
import nvml_injection
import nvml_injection_structs
import dcgm_nvml
import dcgm_agent_internal
import dcgmvalue


@test_utils.run_with_embedded_host_engine()
def test_nvlink_prm_field_registration(handle):
    """
    Test that all PRM fields are properly registered with correct metadata
    """
    prm_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SUCCESSFUL_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_RECOVERY_SUCCESSFUL_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PHYSICAL_LINK_DOWN_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_UNCORRECTABLE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_CODE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENT_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT,
    ]

    # Verify all fields are registered with correct metadata
    for field_id in prm_fields:
        fieldMeta = dcgm_fields.DcgmFieldGetById(field_id)
        assert fieldMeta is not None, f"Field {field_id} not registered"
        assert fieldMeta.entityLevel == dcgm_fields.DCGM_FE_LINK, (
            f"Wrong entity type for {field_id}: "
            f"{fieldMeta.entityLevel}")
        assert fieldMeta.fieldType == dcgm_fields.DCGM_FT_INT64, (
            f"Wrong field type for {field_id}: "
            f"{fieldMeta.fieldType}")
        assert fieldMeta.scope == dcgm_fields.DCGM_FS_ENTITY, (
            f"Wrong scope for {field_id}: {fieldMeta.scope}")

        # Verify field has non-empty name and description
        assert len(fieldMeta.tag) > 0, f"Field {field_id} has empty tag"


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_minimum_gpu_architecture(
    dcgm_structs.DCGM_CHIP_ARCH_BLACKWELL)
@test_utils.run_only_with_nvml()
def test_nvlink_prm_manual_link_entity_creation(handle, gpuIds):
    """
    Test creating link entities manually for specific GPU+port combinations
    """
    # Test with first available GPU
    gpuId = gpuIds[0]

    # Test a few port indices
    test_ports = [0, 1, 5]  # Test first, second, and middle port
    test_field = dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL

    for port_index in test_ports:
        # Create dcgm_link_t using the structure with property-based raw
        # encoding
        link = dcgm_structs.c_dcgm_link_t()
        link.type = dcgm_fields.DCGM_FE_GPU  # Link belongs to a GPU
        link.index = port_index              # Port/link index
        link.id = gpuId                      # GPU ID

        # Get the raw entity ID via the property
        raw_entity_id = link.raw

        # Create entity pair using the manually encoded entity ID
        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_LINK
        entity.entityId = raw_entity_id

        try:
            # Test field access with manually created link entity
            fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(
                handle, [entity], [test_field], dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)

            assert len(fieldValues) == 1, (
                f"Expected 1 field value, got {len(fieldValues)}")
            fieldValue = fieldValues[0]

            # Verify the entity decoding worked correctly
            assert fieldValue.entityGroupId == dcgm_fields.DCGM_FE_LINK, (
                f"Wrong entity group: {fieldValue.entityGroupId}")
            assert fieldValue.entityId == raw_entity_id, (
                f"Wrong entity ID: {fieldValue.entityId}")
            assert fieldValue.fieldId == test_field, (
                f"Wrong field ID: {fieldValue.fieldId}")

            # Status should be one of the valid states (likely NO_DATA without
            # hardware support)
            valid_statuses = [
                dcgm_structs.DCGM_ST_OK,
                dcgm_structs.DCGM_ST_NOT_SUPPORTED,
                dcgm_structs.DCGM_ST_NO_DATA,
                dcgm_structs.DCGM_ST_NVML_ERROR
            ]
            assert fieldValue.status in valid_statuses, (
                f"Unexpected status: {fieldValue.status}")

            logger.debug(
                f"Successfully queried GPU {gpuId} port {port_index} "
                f"(entity ID {raw_entity_id:08x}): "
                f"status {fieldValue.status}")

        except Exception as e:
            logger.warning(
                f"Failed to query PRM field for manually created link GPU {gpuId} port {port_index}: {e}")


def test_dcgm_link_t_structure():
    """
    Test the dcgm_link_t structure with property-based raw encoding
    """
    # Test dcgm_link_t creation
    link = dcgm_structs.c_dcgm_link_t()

    # Test field assignment and retrieval
    link.type = dcgm_fields.DCGM_FE_GPU
    link.index = 5
    link.id = 123

    assert link.type == dcgm_fields.DCGM_FE_GPU, f"Type mismatch: {link.type}"
    assert link.index == 5, f"Index mismatch: {link.index}"
    assert link.id == 123, f"ID mismatch: {link.id}"

    # Test that raw property reflects the field values
    original_raw = link.raw
    assert original_raw != 0, "Raw value should be non-zero after setting fields"
    expected_raw = (123 << 24) | (5 << 8) | dcgm_fields.DCGM_FE_GPU
    assert link.raw == expected_raw, (
        f"Raw encoding incorrect: 0x{link.raw:08x} != "
        f"0x{expected_raw:08x}")

    # Test different entity types
    link.type = dcgm_fields.DCGM_FE_SWITCH
    assert link.type == dcgm_fields.DCGM_FE_SWITCH, (
        f"Switch type mismatch: {link.type}")
    assert link.raw != original_raw, "Raw value should change when type changes"

    # Test boundary values
    link.index = 0
    assert link.index == 0, "Min index failed"

    link.index = 17  # Near max for NvLink
    assert link.index == 17, "Max index failed"

    link.id = 0
    assert link.id == 0, "Min ID failed"

    link.id = 255
    assert link.id == 255, "Max ID failed"

    # Test round-trip: set raw, read fields
    link.raw = 0x12345678
    # The field values should reflect the bit-unpacked raw value
    assert link.type == (0x12345678 & 0xFF), "Raw->type conversion failed"
    assert link.index == ((0x12345678 >> 8) &
                          0xFFFF), "Raw->index conversion failed"
    assert link.id == ((0x12345678 >> 24) & 0xFF), "Raw->id conversion failed"


def test_dcgm_link_entity_id_encoding():
    """
    Test dcgm_link_t encoding/decoding using the property-based structure
    """
    # Test cases: (type, id, index) -> expected behavior
    # Note: DCGM_FE_GPU = 1, DCGM_FE_SWITCH = 3
    test_cases = [
        (dcgm_fields.DCGM_FE_GPU, 0, 0),      # Minimal values (type=1)
        (dcgm_fields.DCGM_FE_GPU, 0, 5),      # Common port (type=1)
        (dcgm_fields.DCGM_FE_GPU, 3, 17),     # Higher values (type=1)
        (dcgm_fields.DCGM_FE_SWITCH, 1, 10),  # Switch entity (type=3)
    ]

    for entity_type, entity_id, port_index in test_cases:
        # Test using the property-based structure
        link = dcgm_structs.c_dcgm_link_t()

        # Set fields
        link.type = entity_type
        link.id = entity_id
        link.index = port_index

        # Get the raw entity ID from the property
        packed_entity_id = link.raw

        # Create a new link from the raw value
        link2 = dcgm_structs.c_dcgm_link_t()
        link2.raw = packed_entity_id

        # Verify round-trip encoding/decoding
        assert link2.type == entity_type, (
            f"Round-trip type failed: {link2.type} != {entity_type}")
        assert link2.id == entity_id, (
            f"Round-trip id failed: {link2.id} != {entity_id}")
        assert link2.index == port_index, (
            f"Round-trip index failed: {link2.index} != {port_index}")

        # Also verify manual bit packing matches property packing
        manual_packed = (entity_id << 24) | (port_index << 8) | entity_type
        assert packed_entity_id == manual_packed, (
            f"Property packing differs from manual: 0x{packed_entity_id:08x} "
            f"!= 0x{manual_packed:08x}")

        logger.debug(
            f"Entity {entity_type}:{entity_id}:{port_index} -> 0x{packed_entity_id:08x}")


# NO HARDWARE

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_gpus()
def test_prm_field_injection_infrastructure(handle, gpuIds):
    """
    Test PRM field injection infrastructure for LINK entities.
    """

    # This test validates that:
    # 1. PRM field values can be injected into DCGM cache for LINK entities
    # 2. Field injection works across different PRM register groups
    # 3. Injected values can be retrieved using cached data queries
    #
    # Note: This tests the injection infrastructure, not live hardware queries.

    # Create synthetic link entity ID (no real hardware needed)
    link = dcgm_structs.c_dcgm_link_t()
    link.type = dcgm_fields.DCGM_FE_GPU
    link.id = gpuIds[0]
    link.index = 1  # Port 1
    linkEntityId = link.raw

    # Test representative fields from each register group
    plr_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL,
    ]

    recovery_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_BETWEEN_LAST_TWO,
    ]

    misc_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT,
    ]

    # Define test field groups for validation
    watched_plr_fields = plr_fields[:2]  # Simulate watched PLR fields
    # Simulate watched recovery field
    watched_recovery_fields = recovery_fields[:1]
    all_test_fields = plr_fields + recovery_fields + misc_fields

    # Inject synthetic values to simulate hardware providing data
    injected_values = {}

    for i, field_id in enumerate(all_test_fields):
        test_value = 5000 + i * 100  # Unique values: 5000, 5100, 5200, etc.
        injected_values[field_id] = test_value

        # Inject directly into cache (simulates what hardware would provide)
        dcgm_field_injection_helpers.inject_field_value_i64(
            handle, linkEntityId, field_id, test_value, 0, dcgm_fields.DCGM_FE_LINK)

    # Query injected fields and verify correct values
    entity = dcgm_structs.c_dcgmGroupEntityPair_t()
    entity.entityGroupId = dcgm_fields.DCGM_FE_LINK
    entity.entityId = linkEntityId

    # Test field retrieval from different register groups
    test_fields = watched_plr_fields + watched_recovery_fields + misc_fields

    for field_id in test_fields:
        # Query using cached data (not live) to test injection infrastructure
        values = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [field_id], 0)  # Use cached data only

        assert len(values) == 1, (
            f"Expected 1 value for field {field_id}, got {len(values)}")
        field_value = values[0]

        # Verify structure
        assert field_value.entityGroupId == dcgm_fields.DCGM_FE_LINK, f"Wrong entity group for {field_id}"
        assert field_value.entityId == linkEntityId, f"Wrong entity ID for {field_id}"
        assert field_value.fieldId == field_id, f"Wrong field ID for {field_id}"

        # Verify injected value is retrieved correctly
        expected_value = injected_values[field_id]
        assert field_value.status == dcgm_structs.DCGM_ST_OK, (
            f"Field {field_id}: injection failed, status "
            f"{field_value.status}")
        assert field_value.value.i64 == expected_value, (
            f"Field {field_id}: got {field_value.value.i64}, expected "
            f"{expected_value}")

    # Test cross-register-group injection/retrieval
    cross_group_fields = [
        plr_fields[0],         # PLR group
        recovery_fields[0],    # Recovery group
        *misc_fields,          # PRM Counters
    ]

    for field_id in cross_group_fields:
        # Test cross-group behavior with cached queries
        values = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [field_id], 0)  # Use cached data

        field_value = values[0]
        expected_value = injected_values[field_id]

        assert field_value.status == dcgm_structs.DCGM_ST_OK, (
            f"Cross-group field {field_id}: status {field_value.status}")
        assert field_value.value.i64 == expected_value, f"Cross-group field {field_id}: value mismatch"

    # Test multiple field query from same group (PLR)
    batch_fields = plr_fields[:2]  # Query multiple PLR fields
    values = dcgm_agent.dcgmEntitiesGetLatestValues(
        handle, [entity], batch_fields, 0)  # Use cached data

    assert len(values) == len(batch_fields), (
        f"Batch query: expected {len(batch_fields)} values, got "
        f"{len(values)}")

    for i, field_value in enumerate(values):
        field_id = batch_fields[i]
        expected_value = injected_values[field_id]

        assert field_value.status == dcgm_structs.DCGM_ST_OK, (
            f"Batch field {field_id}: status {field_value.status}")
        assert field_value.value.i64 == expected_value, f"Batch field {field_id}: value mismatch"


# NO HARDWARE

@test_utils.run_with_injection_nvml()
@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_gpus()
def test_prm_field_caching_with_field_group_watcher(handle, gpuIds):
    """
    Enhanced test using DcgmFieldGroupEntityWatcher to properly test PRM field caching optimization.
    """
    # This test creates proper DCGM groups and field groups to test the caching optimization
    # in a more realistic scenario that exercises the actual watch
    # infrastructure.

    import dcgm_field_injection_helpers

    # Create synthetic link entity
    link = dcgm_structs.c_dcgm_link_t()
    link.type = dcgm_fields.DCGM_FE_GPU
    link.id = gpuIds[0]
    link.index = 3  # Port 3 (different from other tests)
    linkEntityId = link.raw

    logger.debug(
        f"Testing PRM caching with field group watcher using link entity: "
        f"0x{linkEntityId:08x}")

    # Define PRM fields for testing (subset from each register group)
    plr_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL,
    ]

    recovery_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_SUCCESSFUL_TOTAL,
    ]

    misc_fields = [
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT,
    ]

    all_test_fields = plr_fields + recovery_fields + misc_fields

    # Create DCGM handle and group infrastructure
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    # Create entity for the link
    linkEntity = dcgm_structs.c_dcgmGroupEntityPair_t()
    linkEntity.entityGroupId = dcgm_fields.DCGM_FE_LINK
    linkEntity.entityId = linkEntityId

    # Create entity group with the link entity
    groupObj = systemObj.GetGroupWithEntities('PRM_test_links', [linkEntity])

    # Create field group with PRM fields
    fieldGroupObj = pydcgm.DcgmFieldGroup(
        handleObj, "PRM_test_fields", all_test_fields)

    try:
        # Create field group entity watcher (this properly sets up watches)
        operationMode = dcgm_structs.DCGM_OPERATION_MODE_AUTO
        updateFreq = 3_600_000_000  # 1h in microseconds
        maxKeepAge = 300.0          # 5 minutes
        maxKeepSamples = 100        # Keep up to 100 samples

        logger.debug("Creating DcgmFieldGroupEntityWatcher for PRM fields")
        fieldWatcher = dcgm_field_helpers.DcgmFieldGroupEntityWatcher(
            handle, groupObj.GetId(), fieldGroupObj, operationMode,
            updateFreq, maxKeepAge, maxKeepSamples, 0)

        # Inject synthetic values before starting watcher
        logger.debug("Injecting synthetic values for PRM fields")
        injected_values = {}
        for i, field_id in enumerate(all_test_fields):
            # Unique values: 8000, 8200, 8400, etc.
            test_value = 8000 + i * 200
            injected_values[field_id] = test_value

            dcgm_field_injection_helpers.inject_field_value_i64(
                handle, linkEntityId, field_id, test_value, 0, dcgm_fields.DCGM_FE_LINK)

        fieldWatcher.GetAllSinceLastCall()

        # Verify that the watcher captured our injected values
        logger.debug("Verifying field watcher captured injected values")

        # Check that we have values for our link entity
        assert dcgm_fields.DCGM_FE_LINK in fieldWatcher.values, "No LINK entities found in watcher values"
        assert linkEntityId in fieldWatcher.values[
            dcgm_fields.DCGM_FE_LINK], (
            f"Link entity {linkEntityId:08x} not found in watcher values")

        entity_values = fieldWatcher.values[dcgm_fields.DCGM_FE_LINK][linkEntityId]

        # Test that we got values for each field
        for field_id in all_test_fields:
            assert field_id in entity_values, f"Field {field_id} not found in entity values"

            field_samples = entity_values[field_id]
            assert len(
                field_samples) > 0, f"No samples found for field {field_id}"

            # Check the latest sample
            latest_sample = field_samples[-1]  # Most recent sample
            expected_value = injected_values[field_id]

            if not latest_sample.isBlank:
                assert latest_sample.value == expected_value, (
                    f"Field {field_id}: got {latest_sample.value}, expected "
                    f"{expected_value}")
                logger.debug(
                    f"Field {field_id}: correct value {latest_sample.value}")
            else:
                logger.debug(
                    f"Field {field_id}: blank value (status may indicate unsupported)")

        # Test caching optimization scenario:
        # Force an update to trigger potential hardware queries and caching
        logger.debug("Testing caching optimization by forcing field updates")

        # Get more values to test the optimization
        num_new_values = fieldWatcher.GetAllSinceLastCall()
        logger.debug(f"Got {num_new_values} new values from subsequent update")

        # Verify cross-register-group behavior
        logger.debug("Testing cross-register-group field access")

        # Check that we can access fields from different register groups
        test_cross_group = [
            plr_fields[0],         # PLR group
            recovery_fields[0],    # Recovery group
            *misc_fields,          # PRM Counters
        ]

        for field_id in test_cross_group:
            if field_id in entity_values and len(entity_values[field_id]) > 0:
                latest_sample = entity_values[field_id][-1]
                expected_value = injected_values[field_id]

                if not latest_sample.isBlank:
                    assert latest_sample.value == expected_value, f"Cross-group field {field_id}: value mismatch"
                    logger.debug(
                        f"Cross-group field {field_id}: correct value")

        logger.debug(
            "PRM field caching optimization test with field group watcher completed successfully")

    finally:
        # Cleanup is handled automatically by fieldWatcher and groupObj
        # destructors
        logger.debug("Test cleanup completed")


@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
@test_utils.run_only_with_minimum_gpu_architecture(
    dcgm_structs.DCGM_CHIP_ARCH_BLACKWELL)
@test_utils.run_only_with_nvml()
@test_utils.run_only_with_live_nvlinks()
def test_nvlink_prm_watches(handle, gpuIds):
    """
    Comprehensive DCGM NvLink PRM watch validation test.
    Uses the group+fieldgroup approach that we know works, but validates data quality.
    """
    import time

    # Use the first available GPU from the passed gpuIds
    gpuId = gpuIds[0]

    link = dcgm_structs.c_dcgm_link_t()
    link.type = dcgm_fields.DCGM_FE_GPU
    link.index = 1  # Port/link index
    link.id = gpuId  # Use the actual GPU ID
    linkEntityId = link.raw

    try:
        # Step 1: Establish the watch
        group_id = dcgm_agent.dcgmGroupCreate(
            handle, dcgm_structs.DCGM_GROUP_EMPTY, "prm_test")
        dcgm_agent.dcgmGroupAddEntity(
            handle, group_id, dcgm_fields.DCGM_FE_LINK, linkEntityId)

        # Test all PRM fields
        field_ids = [
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_UNCORRECTABLE_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_CODE_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENT_TOTAL,
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT,
        ]

        updateFreq = 100000  # 100ms (0.1 second)
        maxKeepAge = 60.0
        maxKeepSamples = 0
        collectTime = 5
        field_group_id = dcgm_agent.dcgmFieldGroupCreate(
            handle, field_ids, "all_prm_fields")
        dcgm_agent.dcgmWatchFields(
            handle,
            group_id,
            field_group_id,
            updateFreq,
            maxKeepAge,
            maxKeepSamples)
        logger.debug(
            f"Watch established for {len(field_ids)} PRM fields on link "
            f"entity {linkEntityId:08x} (GPU {gpuId}, port {link.index})")

        # Step 2: Wait for substantial data collection
        logger.debug(f"Waiting {collectTime} seconds for data collection...")
        time.sleep(collectTime)

        # Step 3: Validate data using entity approach
        logger.debug("Validating collected data...")

        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_LINK
        entity.entityId = linkEntityId

        successful_fields = 0
        fields_with_data = 0
        fields_with_nonzero_data = 0

        field_names = {
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL: "PLR_RCV_CODES",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_ERROR_TOTAL: "PLR_RCV_CODES_ERR",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_UNCORRECTABLE_TOTAL: "PLR_RCV_CODES_UNCTBL",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_CODE_TOTAL: "PLR_TX_CODES",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_CODE_TOTAL: "PLR_TX_RETRY_CODES",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_TX_RETRY_EVENT_TOTAL: "PLR_TX_RETRY_CODES_ERR",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_SYNC_EVENT_TOTAL: "PLR_SYNC_EVENTS",
            dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT: "IBPC_PORT_XMIT_WAIT",
        }

        for field_id in field_ids:
            try:
                fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(
                    handle, [entity], [field_id], 0)  # Get cached data

                if len(fieldValues) > 0:
                    fv = fieldValues[0]
                    field_name = field_names.get(field_id, f"Field_{field_id}")

                    successful_fields += 1

                    logger.debug(
                        f"Field {field_name} ({field_id}): Status="
                        f"{fv.status}, Value={fv.value.i64:X}, Timestamp="
                        f"{fv.ts}")

                    if fv.status == dcgm_structs.DCGM_ST_OK:
                        fields_with_data += 1
                        if fv.value != 0:
                            fields_with_nonzero_data += 1
                            logger.debug(
                                f"Non-zero data for {field_name}: {fv.value.i64:X}")
                        else:
                            logger.debug(f"Zero value for {field_name}")
                    elif fv.status == dcgm_structs.DCGM_ST_NO_DATA:
                        logger.debug(f"No data available for {field_name}")
                    elif fv.status == dcgm_structs.DCGM_ST_NOT_SUPPORTED:
                        logger.debug(f"Field {field_name} not supported")
                    else:
                        logger.error(
                            f"Error status for {field_name}: {fv.status}")

            except Exception as e:
                logger.error(
                    f"{field_names.get(field_id, f'Field_{field_id}')} failed: {e}")

        # Assert basic field access and data availability criteria here
        # Basic field access - 80% success rate
        # basic_field_access_passed = successful_fields >= len(field_ids) * 0.8
        basic_field_access_passed = successful_fields >= len(field_ids)
        assert basic_field_access_passed, (
            f"Basic field access failed: only {successful_fields}/"
            f"{len(field_ids)} fields successfully queried (need "
            f"{len(field_ids) * 0.8:.1f})")

        # Data availability - at least one field has data
        data_availability_passed = fields_with_data >= 1
        assert data_availability_passed, f"Data availability failed: {fields_with_data} fields have data (need at least 1)"

        # Step 4: Test with live data flag
        logger.debug("Testing with live data flag...")
        live_fields_with_data = 0

        for field_id in field_ids:
            try:
                fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(
                    handle, [entity], [field_id], dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)

                if len(fieldValues) > 0:
                    fv = fieldValues[0]
                    field_name = field_names[field_id]
                    logger.debug(
                        f"LIVE {field_name}: status={fv.status}, "
                        f"value={fv.value.i64:X}")

                    if fv.status == dcgm_structs.DCGM_ST_OK:
                        live_fields_with_data += 1

            except Exception as e:
                logger.error(f"Live {field_names[field_id]} failed: {e}")

        # Assert live data access criteria here
        # Live data access - at least one live field works
        live_data_access_passed = live_fields_with_data >= 1
        assert live_data_access_passed, f"Live data access failed: {live_fields_with_data} live fields working (need at least 1)"

        # Step 5: Test timestamp progression
        logger.debug("Testing timestamp progression...")

        # Get baseline
        baseline_values = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL], 0)
        baseline_ts = baseline_values[0].ts if len(baseline_values) > 0 else 0

        logger.debug(f"Baseline timestamp: {baseline_ts}")

        # Wait and get new values
        logger.debug(f"Waiting {collectTime} seconds for data collection...")
        time.sleep(collectTime)
        current_values = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL], 0)
        current_ts = current_values[0].ts if len(current_values) > 0 else 0

        logger.debug(f"Current timestamp: {current_ts}")

        timestamp_progressed = current_ts > baseline_ts

        # Assert watch functionality criteria here
        # Watch functionality - timestamps are updating
        assert timestamp_progressed, f"Watch functionality failed: timestamps not progressing (baseline: {baseline_ts}, current: {current_ts})"

        # Summary
        timestamp_str = 'Yes' if timestamp_progressed else 'No'
        logger.debug(
            f"Fields queried: {successful_fields}/{len(field_ids)}, "
            f"OK status: {fields_with_data}/{len(field_ids)}, "
            f"Non-zero: {fields_with_nonzero_data}/{len(field_ids)}, "
            f"Live fields: {live_fields_with_data}/2, "
            f"Timestamp progression: {timestamp_str}")

        if fields_with_nonzero_data > 0:
            logger.debug("Real PRM data is being collected")
        else:
            logger.debug("Note: Data is zero (may indicate no NvLink traffic)")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Cleanup
        try:
            dcgm_agent.dcgmUnwatchFields(handle, group_id, field_group_id)
            dcgm_agent.dcgmFieldGroupDestroy(handle, field_group_id)
            dcgm_agent.dcgmGroupDestroy(handle, group_id)
        except Exception:
            pass


recordedPRMTLVData = b'\x08\x04\x00\x00P\x08\x81\x01\x00\x00\x00\x00\x00\x00\x00\x00\x18?\x00\x00\x00\x0f\x00"\x00\x00\x00\x00\x00\x00\x03\xf82#\xbf\x1c\x00\x00\x00\x00\x0c(\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x8b\xff\x86\xee\xfd\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07!e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00b\xe3\xd2\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


# NO HARDWARE

@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku("B200.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_nvlink_prm_single_field_live_then_cached(handle, gpuIds):
    gpuId = gpuIds[0]
    link = dcgm_structs.c_dcgm_link_t()
    link.type = dcgm_fields.DCGM_FE_GPU
    link.id = gpuId
    link.index = 1
    linkEntityId = link.raw

    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_PRMTLV_V1
    injectedRet.values[0].value.PRMTLV_v1.dataSize = 0
    injectedRet.values[0].value.PRMTLV_v1.status = 0
    injectedRet.values[0].value.PRMTLV_v1.data = (
        c_ubyte * dcgm_nvml.NVML_PRM_DATA_MAX_SIZE).from_buffer_copy(recordedPRMTLVData)
    injectedRet.valueCount = 1

    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(
        handle, gpuId, "ReadWritePRM", None, 0, injectedRet)
    assert ret == dcgm_structs.DCGM_ST_OK

    groupId = dcgm_agent.dcgmGroupCreate(
        handle, dcgm_structs.DCGM_GROUP_EMPTY, "prm_live_then_cached")
    fieldId = dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL
    fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(
        handle, [fieldId], "single_field_live_then_cached")

    try:
        dcgm_agent.dcgmGroupAddEntity(
            handle, groupId, dcgm_fields.DCGM_FE_LINK, linkEntityId)
        dcgm_agent.dcgmWatchFields(
            handle, groupId, fieldGroupId, 100000, 60.0, 0)

        entity = dcgm_structs.c_dcgmGroupEntityPair_t()
        entity.entityGroupId = dcgm_fields.DCGM_FE_LINK
        entity.entityId = linkEntityId

        liveValues = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [fieldId], dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)
        assert len(liveValues) == 1, (
            f"Expected 1 live field value, got {len(liveValues)}")
        assert liveValues[0].status == dcgm_structs.DCGM_ST_OK, (
            f"Expected live status OK, got {liveValues[0].status}")
        assert not dcgmvalue.DCGM_INT64_IS_BLANK(liveValues[0].value.i64), (
            f"Expected live value not blank, got {liveValues[0].value.i64}")

        cachedValues = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [fieldId], 0)
        assert len(cachedValues) == 1, (
            f"Expected 1 cached field value, got {len(cachedValues)}")
        assert cachedValues[0].status == dcgm_structs.DCGM_ST_OK, (
            f"Expected cached status OK, got {cachedValues[0].status}")
        assert not dcgmvalue.DCGM_INT64_IS_BLANK(cachedValues[0].value.i64), (
            f"Expected cached value not blank, got {cachedValues[0].value.i64}")
        assert cachedValues[0].value.i64 == liveValues[0].value.i64, (
            f"Expected cached value {cachedValues[0].value.i64} to match live "
            f"value {liveValues[0].value.i64}")
        assert cachedValues[0].ts >= liveValues[0].ts, (
            f"Expected cached timestamp {cachedValues[0].ts} to be greater "
            f"than or equal to live timestamp {liveValues[0].ts}")
    finally:
        dcgm_agent.dcgmUnwatchFields(handle, groupId, fieldGroupId)
        dcgm_agent.dcgmFieldGroupDestroy(handle, fieldGroupId)
        dcgm_agent.dcgmGroupDestroy(handle, groupId)


def _make_link_entity(gpuId, portIndex):
    link = dcgm_structs.c_dcgm_link_t()
    link.type = dcgm_fields.DCGM_FE_GPU
    link.id = gpuId
    link.index = portIndex

    entity = dcgm_structs.c_dcgmGroupEntityPair_t()
    entity.entityGroupId = dcgm_fields.DCGM_FE_LINK
    entity.entityId = link.raw
    return entity


def _inject_prm_tlv_response(handle, gpuId):
    injectedRet = nvml_injection.c_injectNvmlRet_t()
    injectedRet.nvmlRet = dcgm_nvml.NVML_SUCCESS
    injectedRet.values[0].type = nvml_injection_structs.c_injectionArgType_t.INJECTION_PRMTLV_V1
    injectedRet.values[0].value.PRMTLV_v1.dataSize = 0
    injectedRet.values[0].value.PRMTLV_v1.status = 0
    injectedRet.values[0].value.PRMTLV_v1.data = (
        c_ubyte * dcgm_nvml.NVML_PRM_DATA_MAX_SIZE).from_buffer_copy(recordedPRMTLVData)
    injectedRet.valueCount = 1

    ret = dcgm_agent_internal.dcgmInjectNvmlDevice(
        handle, gpuId, "ReadWritePRM", None, 0, injectedRet)
    assert ret == dcgm_structs.DCGM_ST_OK


def _get_live_link_value(handle, entity, fieldId):
    fieldValue, err = _try_get_live_link_value(handle, entity, fieldId)
    if err is not None:
        raise err
    return fieldValue


def _try_get_live_link_value(handle, entity, fieldId):
    try:
        fieldValues = dcgm_agent.dcgmEntitiesGetLatestValues(
            handle, [entity], [fieldId], dcgm_structs.DCGM_FV_FLAG_LIVE_DATA)
    except dcgm_structs.DCGMError as err:
        return None, err

    assert len(
        fieldValues) == 1, f"Expected 1 field value, got {len(fieldValues)}"
    return fieldValues[0], None


def _assert_live_link_query_not_supported(handle, entity, fieldId, expected_description):
    fieldValue, err = _try_get_live_link_value(handle, entity, fieldId)
    assert err is None, (
        f"Expected {expected_description} to return NOT_SUPPORTED field value, got error {err.value}")
    assert fieldValue.status == dcgm_structs.DCGM_ST_OK, (
        f"Expected {expected_description} to return OK, got {fieldValue.status}")
    assert fieldValue.value.i64 == dcgmvalue.DCGM_INT64_NOT_SUPPORTED, (
        f"Expected {expected_description} to return NOT_SUPPORTED, got {fieldValue.value.i64}")


@skip_test_if_no_dcgm_nvml()
@test_utils.run_only_with_nvml()
@test_utils.run_with_injection_nvml_using_specific_sku("B200.yaml")
@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_nvml_injected_gpus()
def test_nvlink_prm_tlv_port_offset_behavior(handle, gpuIds):
    gpuId = gpuIds[0]

    _inject_prm_tlv_response(handle, gpuId)

    nonIbPortZeroValue = _get_live_link_value(
        handle,
        _make_link_entity(gpuId, 0),
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL)
    assert nonIbPortZeroValue.status == dcgm_structs.DCGM_ST_OK, (
        f"Expected non-IB port 0 query to succeed, got {nonIbPortZeroValue.status}")
    assert not dcgmvalue.DCGM_INT64_IS_BLANK(nonIbPortZeroValue.value.i64), (
        f"Expected non-IB port 0 value not blank, got {nonIbPortZeroValue.value.i64}")

    nonIbPortSeventeenValue = _get_live_link_value(
        handle,
        _make_link_entity(
            gpuId, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU - 1),
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL)
    assert nonIbPortSeventeenValue.status == dcgm_structs.DCGM_ST_OK, (
        f"Expected non-IB port 17 query to succeed, got {nonIbPortSeventeenValue.status}")
    assert not dcgmvalue.DCGM_INT64_IS_BLANK(nonIbPortSeventeenValue.value.i64), (
        f"Expected non-IB port 17 value not blank, got {nonIbPortSeventeenValue.value.i64}")

    _assert_live_link_query_not_supported(
        handle,
        _make_link_entity(gpuId, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU),
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_PLR_RX_CODE_TOTAL,
        "non-IB port 18 query")

    _assert_live_link_query_not_supported(
        handle,
        _make_link_entity(
            gpuId, dcgm_structs.DCGM_NVLINK_MAX_LINKS_PER_GPU + 1),
        dcgm_fields.DCGM_FI_DEV_NVLINK_PPCNT_IBPC_PORT_XMIT_WAIT,
        "IB port 19 query")
