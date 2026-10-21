/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! # Common Plugin Utilities
//!
//! This file is used to implement the common plugin utilities.

use crate::{
    DCGM_INT32_BLANK, dcgmDiagEntityResults_v2, dcgmDiagError_v1, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL,
    dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP, dcgmError_t,
    dcgmErrorCategory_t, dcgmErrorSeverity_t, dcgmGroupEntityPair_t,
};
use dcgm_logging::error;
use dcgm_utils::write_c_char_buf;

/// Set the result to pass for the skipped entity.
///
/// * It will set the result to pass for the skipped entity.
/// * It will not overwrite the result if it is already set to fail.
///
/// # Arguments
///
/// * `result` - The result structure.
/// * `entity` - The entity to set the result to pass.
///
/// # Output
///
/// Returns `()` after updating the matching skipped result, or after logging that no match exists.
pub fn set_pass_to_skipped_entity(result: &mut dcgmDiagEntityResults_v2, entity: dcgmGroupEntityPair_t) {
    for (index, _) in (0..std::cmp::min(result.numResults as usize, result.results.len())).enumerate() {
        if result.results[index].entity.entityGroupId == entity.entityGroupId
            && result.results[index].entity.entityId == entity.entityId
        {
            // We cannot overwrite fail, warn as pass.
            if result.results[index].result == dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP {
                result.results[index].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS;
            }
            return;
        }
    }
    error!("No result found for entity: {:?}", entity);
}

/// Set the result to pass for all skipped entities.
///
/// * It will set the result to pass for all skipped entities.
/// * It will not overwrite the result if it is already set to fail.
///
/// # Arguments
///
/// * `result` - The result structure.
///
/// # Output
///
/// Returns `()` after converting every initialized skipped result to pass.
pub fn set_pass_to_all_skipped_entities(result: &mut dcgmDiagEntityResults_v2) {
    for (index, _) in (0..std::cmp::min(result.numResults as usize, result.results.len())).enumerate() {
        // We cannot overwrite fail, warn as pass.
        if result.results[index].result == dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP {
            result.results[index].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS;
        }
    }
}

/// Set the result to fail for the entity.
///
/// * It will set the result to fail for the entity.
///
/// # Arguments
///
/// * `result` - The result structure.
/// * `entity` - The entity to set the result to fail.
///
/// # Output
///
/// Returns `()` after updating the matching result, or after logging that no match exists.
pub fn set_fail_to_entity(result: &mut dcgmDiagEntityResults_v2, entity: dcgmGroupEntityPair_t) {
    for (index, _) in (0..std::cmp::min(result.numResults as usize, result.results.len())).enumerate() {
        if result.results[index].entity.entityGroupId == entity.entityGroupId
            && result.results[index].entity.entityId == entity.entityId
        {
            result.results[index].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL;
            return;
        }
    }
    error!("No result found for entity: {:?}", entity);
}

/// Set the result to fail for all entities.
///
/// * It will set the result to fail for all entities.
///
/// # Arguments
///
/// * `result` - The result structure.
///
/// # Output
///
/// Returns `()` after marking every initialized entity result as failed.
pub fn set_fail_to_all_entities(result: &mut dcgmDiagEntityResults_v2) {
    for (index, _) in (0..std::cmp::min(result.numResults as usize, result.results.len())).enumerate() {
        result.results[index].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL;
    }
}

/// Add an error to the entity.
///
/// * It will add an error to the entity.
///
/// # Arguments
///
/// * `result` - The result structure.
/// * `entity` - The entity to add the error to.
/// * `code` - The error code.
/// * `category` - The error category.
/// * `severity` - The error severity.
/// * `error` - The error message.
///
/// # Output
///
/// Returns `()` after appending the error, or after logging that the fixed-capacity error array is full.
pub fn add_error_to_entity(
    result: &mut dcgmDiagEntityResults_v2,
    entity: dcgmGroupEntityPair_t,
    code: dcgmError_t,
    category: dcgmErrorCategory_t,
    severity: dcgmErrorSeverity_t,
    error: &str,
) {
    if usize::from(result.numErrors) >= result.errors.len() {
        error!(
            "Too many errors for entity: {:?}, code: {}, category: {}, severity: {}, error: {}",
            entity, code, category, severity, error
        );
        return;
    }
    let index = usize::from(result.numErrors);
    result.errors[index] = dcgmDiagError_v1::default();
    result.errors[index].entity = entity;
    result.errors[index].code = code;
    result.errors[index].category = category;
    result.errors[index].severity = severity;
    result.errors[index].testId = DCGM_INT32_BLANK;
    write_c_char_buf(&mut result.errors[index].msg, error);
    result.numErrors += 1;
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{
        dcgm_field_entity_group_t_DCGM_FE_CPU, dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED,
        dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC, dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE,
    };
    use dcgm_utils::c_char_buf_to_string;

    /// Creates a CPU entity pair for the supplied entity identifier.
    ///
    /// Input: `entity_id`, the CPU identifier. Output: a CPU [`dcgmGroupEntityPair_t`].
    fn cpu_entity(entity_id: u32) -> dcgmGroupEntityPair_t {
        dcgmGroupEntityPair_t {
            entityGroupId: dcgm_field_entity_group_t_DCGM_FE_CPU,
            entityId: entity_id,
        }
    }

    /// Creates results containing one skipped CPU and one failed CPU.
    ///
    /// Input: none. Output: an initialized [`dcgmDiagEntityResults_v2`] test fixture.
    fn result_with_two_cpus() -> dcgmDiagEntityResults_v2 {
        let mut result = dcgmDiagEntityResults_v2 {
            numResults: 2,
            ..Default::default()
        };
        result.results[0].entity = cpu_entity(0);
        result.results[0].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP;
        result.results[1].entity = cpu_entity(1);
        result.results[1].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL;
        result
    }

    #[test]
    /// Verifies that only the matching skipped entity becomes a pass.
    /// Input: none. Output: `()` when fixture assertions pass.
    fn set_pass_to_skipped_entity_updates_only_matching_skip() {
        let mut result = result_with_two_cpus();

        set_pass_to_skipped_entity(&mut result, cpu_entity(0));
        set_pass_to_skipped_entity(&mut result, cpu_entity(1));

        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
    }

    #[test]
    /// Verifies that converting skips to passes preserves existing failures.
    /// Input: none. Output: `()` when fixture assertions pass.
    fn set_pass_to_all_skipped_entities_preserves_failures() {
        let mut result = result_with_two_cpus();

        set_pass_to_all_skipped_entities(&mut result);

        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
    }

    #[test]
    /// Verifies that failing one entity does not alter other entity results.
    /// Input: none. Output: `()` when fixture assertions pass.
    fn set_fail_to_entity_marks_only_matching_entity() {
        let mut result = result_with_two_cpus();
        result.results[1].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP;

        set_fail_to_entity(&mut result, cpu_entity(0));

        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP);
    }

    #[test]
    /// Verifies that all initialized entity results can be marked failed.
    /// Input: none. Output: `()` when fixture assertions pass.
    fn set_fail_to_all_entities_marks_initialized_results() {
        let mut result = result_with_two_cpus();

        set_fail_to_all_entities(&mut result);

        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
    }

    #[test]
    /// Verifies that an entity error retains its metadata and message.
    /// Input: none. Output: `()` when fixture assertions pass.
    fn add_error_to_entity_records_error_metadata_and_message() {
        let mut result = result_with_two_cpus();

        add_error_to_entity(
            &mut result,
            cpu_entity(1),
            dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED,
            dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC,
            dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE,
            "cpu mismatch detected",
        );

        assert_eq!(result.numErrors, 1);
        assert_eq!(result.errors[0].entity.entityId, 1);
        assert_eq!(result.errors[0].code, dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED);
        assert_eq!(
            result.errors[0].category,
            dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC
        );
        assert_eq!(result.errors[0].severity, dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE);
        assert_eq!(
            c_char_buf_to_string(&result.errors[0].msg).expect("error message should decode"),
            "cpu mismatch detected"
        );
    }
}
