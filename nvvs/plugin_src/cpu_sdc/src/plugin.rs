/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

use crate::common_plugin::{
    add_error_to_entity, set_fail_to_all_entities, set_fail_to_entity, set_pass_to_all_skipped_entities,
    set_pass_to_skipped_entity,
};
use crate::payload::{ProtocolMessage, TestResult};
use crate::{
    dcgm_field_entity_group_t_DCGM_FE_CPU, dcgm_field_entity_group_t_DCGM_FE_NONE, dcgmDiagAuxData_version1,
    dcgmDiagAuxDataType_JSON_VALUE_AUX_DATA_TYPE, dcgmDiagEntityResults_v2, dcgmDiagPluginEntityList_v1,
    dcgmDiagPluginTestParameter_t, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP,
    dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED, dcgmError_t, dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC,
    dcgmErrorCategory_t, dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE, dcgmErrorSeverity_t, dcgmGroupEntityPair_t,
};
use child_process::{ChildProcessConfig, ChildProcessRunner, LogPaths, RealChildProcessRunner, check_binary_fd};
use dcgm_logging::{debug, error, info, warn};
use dcgm_utils::c_char_buf_to_string;
use std::collections::HashMap;
use std::ffi::{OsString, c_char, c_void};
use std::fs::File;
use std::num::NonZeroU64;
use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;
use std::time::Duration;

#[cfg(not(test))]
unsafe extern "C" {
    /// The NVVS process sets this when plugin work should stop.
    ///
    /// Input: none. Output: nonzero when the plugin should stop, or zero when work may continue.
    fn nvvs_main_should_stop_load() -> i32;
}

#[cfg(test)]
static MAIN_SHOULD_STOP_FOR_TESTS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

const ENV_CPU_SDC_SKIP_BIN_PERMISSION_CHECK: &str = "DCGM_CPU_SDC_SKIP_BIN_PERMISSION_CHECK";
const ENV_CPU_SDC_BIN_PATH: &str = "DCGM_CPU_SDC_BIN_PATH";
const ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED: &str = "DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED";
const ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS: &str = "DCGM_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS";
const ENV_DCGM_HOME_DIR: &str = "DCGM_HOME_DIR";
const ENV_MOCKED_SDC_RETURN_PATH: &str = "DCGM_MOCKED_SDC_RETURN_PATH";
const ENV_MOCKED_SDC_EXIT_CODE: &str = "DCGM_MOCKED_SDC_EXIT_CODE";
const ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC: &str = "DCGM_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC";
const ENV_PASSTHROUGH: &[&str] = &[
    ENV_MOCKED_SDC_RETURN_PATH,
    ENV_MOCKED_SDC_EXIT_CODE,
    ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC,
];
const SDC_STDOUT_LOG_FILENAME: &str = "dcgm_sdc_stdout.log";
const SDC_STDERR_LOG_FILENAME: &str = "dcgm_sdc_stderr.log";
const SDC_BOOLEAN_TRUE: &str = "true";
const SDC_BOOLEAN_TRUE_NUMERIC: &str = "1";
const SDC_BOOLEAN_FALSE: &str = "false";
const SDC_BOOLEAN_FALSE_NUMERIC: &str = "0";
const SDC_DEVICE_TYPE_CPU: &str = "cpu";
const SDC_DEVICE_ID_UNDEFINED: &str = "undefined";
const SDC_TEST_STATUS_PASS: &str = "pass";
const SDC_TEST_STATUS_FAIL: &str = "fail";
const SDC_ARG_ENABLE_HEARTBEAT: &str = "--enable-heartbeat";
const SDC_ARG_CONFIG_PATH: &str = "--config-path";
const SDC_ERROR_DETAILS_PREFIX: &str = "Error: ";
const DEFAULT_HEARTBEAT_TIMEOUT_SECONDS: u64 = 60;

pub const DEFAULT_SDC_BINARY_PATH: &str = "/path/to/real/sdc";

/// Reports whether NVVS has requested that the plugin stop its current work.
///
/// Input: none. Output: `true` when the NVVS stop flag is set; otherwise `false`.
fn should_stop_main_thread() -> bool {
    #[cfg(test)]
    {
        MAIN_SHOULD_STOP_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed) != 0
    }

    #[cfg(not(test))]
    {
        // SAFETY: NVVS exports this C ABI shim for the lifetime of the loaded plugin.
        unsafe { nvvs_main_should_stop_load() != 0 }
    }
}

/// Get the path to the SDC binary. The environment variable `DCGM_CPU_SDC_BIN_PATH` can be used to override the default path.
///
/// # Returns
///
/// * `String` - The path to the SDC binary.
///
/// Input: none; the optional path override is read from the process environment.
#[must_use]
pub fn get_sdc_binary_path() -> String {
    std::env::var(ENV_CPU_SDC_BIN_PATH).unwrap_or_else(|_| DEFAULT_SDC_BINARY_PATH.to_string())
}

#[must_use]
/// Reports whether the configured SDC binary path names a regular file.
///
/// Input: none; the configured path is resolved by [`get_sdc_binary_path`]. Output: whether that path is a file.
pub fn is_sdc_binary_available() -> bool {
    Path::new(&get_sdc_binary_path()).is_file()
}

/// Open and check if the SDC binary is valid.
///
/// * It will check if the binary is a regular file and owned by root.
/// * It will also check if the binary is executable and not writable by anyone but the root user.
///
/// Note: The environment variable `DCGM_CPU_SDC_SKIP_BIN_PERMISSION_CHECK` can be used to skip the check.
///
/// # Arguments
///
/// * `sdc_binary_path` - The path to the SDC binary.
///
/// # Returns
///
/// * `Result<File, String>` - The open binary file, or the error message if the check fails.
///
fn open_checked_sdc_binary(sdc_binary_path: &str) -> Result<File, String> {
    let sdc_binary =
        File::open(sdc_binary_path).map_err(|err| format!("failed to open SDC binary [{sdc_binary_path}]: {err}"))?;
    check_sdc_binary_valid(sdc_binary.as_fd(), Path::new(sdc_binary_path))?;
    Ok(sdc_binary)
}

/// Validates the opened SDC binary's type, ownership, and permissions.
///
/// Inputs: the open `sdc_binary_fd` and its `sdc_binary_path`. Output: `Ok(())` when valid, or a descriptive error.
fn check_sdc_binary_valid(sdc_binary_fd: BorrowedFd<'_>, sdc_binary_path: &Path) -> Result<(), String> {
    let skip_bin_permission_check = std::env::var(ENV_CPU_SDC_SKIP_BIN_PERMISSION_CHECK)
        .unwrap_or_else(|_| String::from(SDC_BOOLEAN_FALSE_NUMERIC))
        .parse::<i32>()
        .unwrap_or(0);
    let skip_bin_permission_check = skip_bin_permission_check == 1;
    if skip_bin_permission_check {
        info!(
            "Skip checking the permission of binary [{}].",
            sdc_binary_path.display()
        );
    }

    check_binary_fd(sdc_binary_fd, sdc_binary_path, skip_bin_permission_check).map_err(|err| err.to_string())
}

/// Copies test-only SDC behavior overrides into the child environment.
///
/// Input: mutable child-process `config`. Output: `()` after adding every configured pass-through variable.
fn pass_cheating_env_to_child(config: &mut ChildProcessConfig) {
    config.environment.extend(
        ENV_PASSTHROUGH
            .iter()
            .filter_map(|name| std::env::var_os(name).map(|value| (OsString::from(*name), value))),
    );
}

/// Reads the heartbeat timeout override, falling back to the default on invalid input.
///
/// Input: none; reads the timeout environment variable. Output: the configured or default [`Duration`].
fn heartbeat_timeout() -> Duration {
    let default = Duration::from_secs(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS);
    match std::env::var(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS) {
        Ok(value) => match value.parse::<NonZeroU64>() {
            Ok(seconds) => Duration::from_secs(seconds.get()),
            Err(err) => {
                warn!(
                    "Invalid {ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS} value [{value}]: {err}; using the default of {DEFAULT_HEARTBEAT_TIMEOUT_SECONDS} seconds"
                );
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => {
            warn!(
                "Unable to read {ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS}: {err}; using the default of {DEFAULT_HEARTBEAT_TIMEOUT_SECONDS} seconds"
            );
            default
        }
    }
}

/// Reads whether child-process heartbeat timeout enforcement is disabled.
///
/// Input: none; reads the disable environment variable. Output: the parsed flag, defaulting to `false`.
fn heartbeat_disabled() -> bool {
    match std::env::var(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED) {
        Ok(value) if value.eq_ignore_ascii_case(SDC_BOOLEAN_TRUE) || value == SDC_BOOLEAN_TRUE_NUMERIC => true,
        Ok(value) if value.eq_ignore_ascii_case(SDC_BOOLEAN_FALSE) || value == SDC_BOOLEAN_FALSE_NUMERIC => false,
        Ok(value) => {
            warn!("Invalid {ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED} value [{value}]; using the default of false");
            false
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(err) => {
            warn!("Unable to read {ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED}: {err}; using the default of false");
            false
        }
    }
}

/// Applies the configured heartbeat timeout and disabled state to a child process.
///
/// Input: mutable child-process `config`. Output: `()` after updating its heartbeat fields.
fn configure_heartbeat(config: &mut ChildProcessConfig) {
    config.heartbeat_disabled = heartbeat_disabled();
    config.heartbeat_timeout = heartbeat_timeout();
}

/// Get the default DCGM log folder.
///
/// # Returns
///
/// * `String` - The default DCGM log folder.
///
/// Input: none; reads the optional DCGM home override from the environment.
fn get_default_dcgm_log_folder() -> String {
    std::env::var(ENV_DCGM_HOME_DIR).unwrap_or_else(|_| "/var/log/nvidia-dcgm".to_string())
}

/// The configuration for the CPU SDC plugin.
///
/// * `is_allowed` - Whether the test is allowed to run.
/// * `config_path` - The path to the SDC binary config file. This path will be passed to the SDC binary.
///
struct CpuSdcPluginConfig {
    /// Whether the test is allowed to run.
    is_allowed: bool,
    /// The path to the SDC config file passed to the SDC binary.
    config_path: String,
}

/// The CPU SDC plugin.
///
/// * `results` - The results of the test.
/// * `configs` - The configurations for the CPU SDC plugin.
///
pub struct CpuSdcPlugin {
    /// Per-test entity results accumulated from SDC protocol messages.
    results: HashMap<String, dcgmDiagEntityResults_v2>,
    /// Serialized per-test auxiliary data retained to back returned raw pointers.
    aux_data: HashMap<String, Vec<u8>>,

    /// Per-test configuration parsed from NVVS test parameters.
    configs: HashMap<String, CpuSdcPluginConfig>,
    /// Runner used to execute and supervise the SDC child process.
    child_runner: Option<Box<dyn ChildProcessRunner>>,
}

/// The CPU SDC plugin.
impl CpuSdcPlugin {
    /// Initialize the CPU SDC plugin.
    ///
    /// # Returns
    ///
    /// * `CpuSdcPlugin` - The CPU SDC plugin.
    ///
    /// Input: none.
    pub fn new() -> Self {
        info!("Initialized CPU SDC plugin");
        Self {
            results: HashMap::new(),
            aux_data: HashMap::new(),
            configs: HashMap::new(),
            child_runner: Some(Box::new(RealChildProcessRunner)),
        }
    }

    #[cfg(test)]
    #[cfg(not(miri))]
    /// Creates a plugin that delegates child-process execution to a test runner.
    ///
    /// Input: `child_runner`, which owns mocked process behavior. Output: a plugin configured to use that runner.
    fn with_child_runner(child_runner: Box<dyn ChildProcessRunner>) -> Self {
        info!("Initialized CPU SDC plugin");
        Self {
            results: HashMap::new(),
            aux_data: HashMap::new(),
            configs: HashMap::new(),
            child_runner: Some(child_runner),
        }
    }

    /// Initialize the results for the test.
    ///
    /// * It will create a new result struct for the test.
    /// * It will set the result to skip for all entities.
    ///
    /// Note: This function should be called before we can set the result for the test.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `entity_info` - The entity info.
    ///
    /// # Output
    ///
    /// Returns `Ok(())` after initializing results, or `Err(())` after recording an initialization error.
    fn init_result(&mut self, test_name: &str, entity_info: &dcgmDiagPluginEntityList_v1) -> Result<(), ()> {
        self.results
            .insert(test_name.to_string(), dcgmDiagEntityResults_v2::default());
        let Some(result) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return Err(());
        };
        for index in 0..entity_info.numEntities as usize {
            if index >= result.results.len() {
                let error_msg = format!(
                    "Entity count exceeds maximum per-test result slots: {} > {}",
                    entity_info.numEntities,
                    result.results.len()
                );
                error!("{error_msg}");
                self.add_general_error(test_name, &error_msg);
                return Err(());
            }
            if index >= entity_info.entities.len() {
                let error_msg = format!(
                    "Entity index exceeds entity list size: {} > {}",
                    index, entity_info.numEntities
                );
                error!("{error_msg}");
                self.add_general_error(test_name, &error_msg);
                return Err(());
            }
            result.results[index].entity = entity_info.entities[index].entity;
            result.results[index].result = dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP;
            result.numResults += 1;
        }
        Ok(())
    }

    /// Parse the C char array to String.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `c_char_arr` - The C char array.
    ///
    /// # Returns
    ///
    /// * `Result<String, ()>` - The String if the conversion is successful.
    ///
    fn parse_c_char_arr_to_string(&mut self, test_name: &str, c_char_arr: &[c_char]) -> Result<String, ()> {
        match c_char_buf_to_string(c_char_arr) {
            Ok(s) => Ok(s),
            Err(e) => {
                let error_msg = format!("Failed to convert C character buffer to string: {e}");
                error!("{}", error_msg);
                self.add_general_error(test_name, &error_msg);
                Err(())
            }
        }
    }

    /// Initialize the configurations for the test.
    ///
    /// * It will create a new configuration struct for the test.
    /// * It will parse the test parameters and set the configurations.
    ///
    /// Note: This function should be called before we can read the config of the test.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `test_parameters` - The test parameters.
    ///
    /// # Returns
    ///
    /// * `Result<(), ()>` - The result of the initialization.
    ///
    fn init_config(&mut self, test_name: &str, test_parameters: &[dcgmDiagPluginTestParameter_t]) -> Result<(), ()> {
        let mut is_allowed = true;
        let mut config_path = String::new();

        for parameter in test_parameters {
            let parameter_name = self.parse_c_char_arr_to_string(test_name, &parameter.parameterName)?;
            if parameter_name.eq_ignore_ascii_case(crate::CPU_SDC_PARAMETER_IS_ALLOWED) {
                let parameter_value = self.parse_c_char_arr_to_string(test_name, &parameter.parameterValue)?;
                is_allowed = parameter_value.eq_ignore_ascii_case(SDC_BOOLEAN_TRUE)
                    || parameter_value.eq_ignore_ascii_case(SDC_BOOLEAN_TRUE_NUMERIC);
            } else if parameter_name.eq_ignore_ascii_case(crate::CPU_SDC_PARAMETER_CONFIG_PATH) {
                let parameter_value = self.parse_c_char_arr_to_string(test_name, &parameter.parameterValue)?;
                config_path = parameter_value;
            }
        }
        self.configs.insert(
            test_name.to_string(),
            CpuSdcPluginConfig {
                is_allowed,
                config_path,
            },
        );
        Ok(())
    }

    /// Set the result to pass for the skipped entity.
    ///
    /// * It will set the result to pass for the skipped entity.
    /// * It will not overwrite the result if it is already set to fail.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `entity` - The entity to set the result to pass.
    ///
    /// # Output
    ///
    /// Returns `()` after updating the result or logging that the test has no result storage.
    fn set_pass_to_skipped_entity(&mut self, test_name: &str, entity: dcgmGroupEntityPair_t) {
        let Some(result_struct) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return;
        };
        set_pass_to_skipped_entity(result_struct, entity);
    }

    /// Set the result to pass for all skipped entities.
    ///
    /// * It will set the result to pass for all skipped entities.
    /// * It will not overwrite the result if it is already set to fail.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    ///
    /// # Output
    ///
    /// Returns `()` after updating results or logging that the test has no result storage.
    fn set_pass_to_all_skipped_entities(&mut self, test_name: &str) {
        let Some(result_struct) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return;
        };
        set_pass_to_all_skipped_entities(result_struct);
    }

    /// Set the result to fail for the entity.
    ///
    /// * It will set the result to fail for the entity.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `entity` - The entity to set the result to fail.
    ///
    /// # Output
    ///
    /// Returns `()` after updating the result or logging that the test has no result storage.
    fn set_fail_to_entity(&mut self, test_name: &str, entity: dcgmGroupEntityPair_t) {
        let Some(result_struct) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return;
        };
        set_fail_to_entity(result_struct, entity);
    }

    /// Set the result to fail for all entities.
    ///
    /// * It will set the result to fail for all entities.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    ///
    /// # Output
    ///
    /// Returns `()` after updating results or logging that the test has no result storage.
    fn set_fail_to_all_entities(&mut self, test_name: &str) {
        let Some(result_struct) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return;
        };
        set_fail_to_all_entities(result_struct);
    }

    /// Add an error to the entity.
    ///
    /// * It will add an error to the entity.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `entity` - The entity to add the error to.
    /// * `code` - The error code.
    /// * `category` - The error category.
    /// * `severity` - The error severity.
    /// * `error` - The error message.
    ///
    /// # Output
    ///
    /// Returns `()` after appending the error or logging that the test has no result storage.
    fn add_error_to_entity(
        &mut self,
        test_name: &str,
        entity: dcgmGroupEntityPair_t,
        code: dcgmError_t,
        category: dcgmErrorCategory_t,
        severity: dcgmErrorSeverity_t,
        error: &str,
    ) {
        let Some(result_struct) = self.results.get_mut(test_name) else {
            error!("Failed to get result for test {}", test_name);
            return;
        };
        add_error_to_entity(result_struct, entity, code, category, severity, error);
    }

    /// Add a general error to the test.
    ///
    /// * It will add a general error to the test.
    /// * It will set the result to fail for all entities.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `error` - The error message.
    ///
    /// # Output
    ///
    /// Returns `()` after recording a general, non-entity error.
    fn add_general_error(&mut self, test_name: &str, error: &str) {
        self.add_error_to_entity(
            test_name,
            dcgmGroupEntityPair_t {
                entityGroupId: dcgm_field_entity_group_t_DCGM_FE_NONE,
                entityId: 0,
            },
            dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED,
            dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC,
            dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE,
            error,
        );
        self.set_fail_to_all_entities(test_name);
    }

    /// Serializes and stores the SDC binary version as auxiliary JSON data.
    ///
    /// Serialization failures are logged and leave any existing auxiliary data unchanged.
    ///
    /// Inputs: `test_name` and the SDC `version`. Output: `()` after storing serialized auxiliary data or logging failure.
    fn add_sdc_binary_version(&mut self, test_name: &str, version: &str) {
        let aux_data = serde_json::json!({ "version": version });
        match serde_json::to_vec(&aux_data) {
            Ok(aux_data) => {
                self.aux_data.insert(test_name.to_string(), aux_data);
                debug!("Adding SDC binary version {version} to the result.");
            }
            Err(e) => {
                error!("Failed to serialize SDC binary version aux data: {e}");
            }
        }
    }

    /// Handle the pass result.
    ///
    /// * It will set the result to pass for the passed entity.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `test_result` - The test result.
    ///
    /// # Output
    ///
    /// Returns `()` after applying pass states or recording malformed device identifiers.
    fn handle_pass_result(&mut self, test_name: &str, test_result: &TestResult) {
        if !test_result.device_type.eq_ignore_ascii_case(SDC_DEVICE_TYPE_CPU) {
            debug!(
                "Test result device type is not CPU, skipping: {}",
                test_result.device_type
            );
            return;
        }
        for device_id in &test_result.device_ids {
            // The SDC binary will do its best to give result to a specific entity.
            // If it cannot tell which entity it is (e.g., for system memory tests), it will set the device id to the undefined sentinel.
            // In this case, we will set the result to pass for all skipped entities.
            if device_id == SDC_DEVICE_ID_UNDEFINED {
                self.set_pass_to_all_skipped_entities(test_name);
                continue;
            }
            match device_id.parse::<u32>() {
                Ok(device_id) => {
                    self.set_pass_to_skipped_entity(
                        test_name,
                        dcgmGroupEntityPair_t {
                            entityGroupId: dcgm_field_entity_group_t_DCGM_FE_CPU,
                            entityId: device_id,
                        },
                    );
                }
                Err(e) => {
                    self.add_general_error(
                        test_name,
                        &format!("Failed to parse device id: {device_id}, error: {e}"),
                    );
                }
            }
        }
    }

    /// Handle the fail result.
    ///
    /// * It will set the result to fail for the failed entity.
    /// * It will add an error to the entity.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `test_result` - The test result.
    ///
    /// # Output
    ///
    /// Returns `()` after applying failure states and diagnostic errors.
    fn handle_fail_result(&mut self, test_name: &str, test_result: &TestResult) {
        if !test_result.device_type.eq_ignore_ascii_case(SDC_DEVICE_TYPE_CPU) {
            debug!(
                "Test result device type is not CPU, skipping: {}",
                test_result.device_type
            );
            return;
        }
        for device_id in &test_result.device_ids {
            let error_details = format!("{SDC_ERROR_DETAILS_PREFIX}{}", test_result.test_details);
            if let Ok(device_id) = device_id.parse::<u32>() {
                let entity = dcgmGroupEntityPair_t {
                    entityGroupId: dcgm_field_entity_group_t_DCGM_FE_CPU,
                    entityId: device_id,
                };
                self.set_fail_to_entity(test_name, entity);
                self.add_error_to_entity(
                    test_name,
                    entity,
                    dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED,
                    dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC,
                    dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE,
                    &error_details,
                );
            } else {
                self.set_fail_to_all_entities(test_name);
                self.add_error_to_entity(
                    test_name,
                    dcgmGroupEntityPair_t {
                        entityGroupId: dcgm_field_entity_group_t_DCGM_FE_NONE,
                        entityId: 0,
                    },
                    dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED,
                    dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC,
                    dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE,
                    &error_details,
                );
            }
        }
    }

    /// Handle the payload.
    ///
    /// * It will deserialize the payload.
    /// * It will handle the test result.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `payload` - The payload to handle.
    ///
    /// # Returns
    ///
    /// * `bool` - True if the payload is handled successfully.
    ///
    fn handle_payload(&mut self, test_name: &str, payload: &str) -> bool {
        // deserialize the payload
        let payload: ProtocolMessage = match serde_json::from_str(payload) {
            Ok(payload) => payload,
            Err(e) => {
                error!("Failed to deserialize payload, error: {}", e);
                return false;
            }
        };
        match payload {
            ProtocolMessage::TestResult(test_result) => {
                let test_status = &test_result.test_result.test_status;
                if test_status.eq_ignore_ascii_case(SDC_TEST_STATUS_PASS) {
                    self.handle_pass_result(test_name, &test_result.test_result);
                } else if test_status.eq_ignore_ascii_case(SDC_TEST_STATUS_FAIL) {
                    self.handle_fail_result(test_name, &test_result.test_result);
                } else {
                    error!("Unknown test status: {}", test_result.test_result.test_status);
                }
            }
            ProtocolMessage::Version(version) => {
                debug!("Received version: {}", version.sdc.version);
                self.add_sdc_binary_version(test_name, &version.sdc.version);
            }
        }
        true
    }

    /// Builds the SDC command-line arguments for an allowed test.
    ///
    /// Returns `None` when the test is disabled. A missing configuration also returns `None`
    /// after recording a general test error.
    ///
    /// Input: `test_name`, used to select per-test configuration. Output: SDC arguments for an allowed test, or `None`.
    fn build_sdc_args(&mut self, test_name: &str) -> Option<Vec<String>> {
        let Some(config) = self.configs.get(test_name) else {
            error!("Failed to get config for test {}", test_name);
            self.add_general_error(test_name, &format!("Failed to get config for test {test_name}"));
            return None;
        };
        if !config.is_allowed {
            info!("Test {} is not allowed, skipping", test_name);
            return None;
        }

        let mut sdc_args = vec![String::from(SDC_ARG_ENABLE_HEARTBEAT)];
        if !config.config_path.is_empty() {
            sdc_args.push(String::from(SDC_ARG_CONFIG_PATH));
            sdc_args.push(config.config_path.clone());
        }
        Some(sdc_args)
    }

    /// Creates append-mode paths for the SDC standard-output and standard-error logs.
    ///
    /// Input: `log_folder`, the parent log directory. Output: [`LogPaths`] for the CPU SDC stdout and stderr files.
    fn create_log_paths(log_folder: &Path) -> LogPaths {
        LogPaths::append(
            log_folder.join(SDC_STDOUT_LOG_FILENAME),
            log_folder.join(SDC_STDERR_LOG_FILENAME),
        )
    }

    /// Run the test.
    ///
    /// Invokes the SDC binary with the following arguments:
    /// * --enable-heartbeat: Enable the heartbeat.
    /// * --config-path: The path to the SDC binary config file.
    ///
    /// The shared child-process utility supervises the data and heartbeat channels.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    /// * `entity_info` - The entity info. This is used to initialize the result.
    /// * `test_parameters` - The test parameters. This is used to initialize the configuration.
    ///
    /// # Output
    ///
    /// Returns `()` after recording all available results or the error that stopped execution.
    pub fn run_test(
        &mut self,
        test_name: &str,
        entity_info: &dcgmDiagPluginEntityList_v1,
        test_parameters: &[dcgmDiagPluginTestParameter_t],
    ) {
        info!("Running test: {}", test_name);
        if self.init_result(test_name, entity_info).is_err() {
            return;
        }
        if self.init_config(test_name, test_parameters).is_err() {
            // error is already added by init_config
            return;
        }

        let Some(sdc_args) = self.build_sdc_args(test_name) else {
            return;
        };

        let sdc_binary_path = get_sdc_binary_path();
        let sdc_binary = match open_checked_sdc_binary(&sdc_binary_path) {
            Ok(sdc_binary) => sdc_binary,
            Err(e) => {
                self.add_general_error(test_name, &e);
                return;
            }
        };

        let dcgm_log_folder = get_default_dcgm_log_folder();
        let log_folder = Path::new(&dcgm_log_folder);
        let log_paths = Self::create_log_paths(log_folder);
        let stdout_log_path = log_paths.stdout.display().to_string();
        let stderr_log_path = log_paths.stderr.display().to_string();
        let mut child_config =
            ChildProcessConfig::new_from_fd(Path::new(&sdc_binary_path), sdc_binary.into(), sdc_args, log_paths);
        configure_heartbeat(&mut child_config);
        child_config.stop_requested = Some(Box::new(should_stop_main_thread));
        pass_cheating_env_to_child(&mut child_config);

        let Some(child_runner) = self.child_runner.take() else {
            self.add_general_error(test_name, "Child process runner is unavailable");
            return;
        };
        let mut payload_error = None;
        let child_output = {
            let mut handle_fd3_message = |payload| {
                if payload_error.is_some() {
                    return;
                }
                let payload = match String::from_utf8(payload) {
                    Ok(payload) => payload,
                    Err(err) => {
                        payload_error = Some(format!("Child process returned invalid UTF-8: {err}"));
                        return;
                    }
                };
                debug!("Received payload: {}", payload);
                if payload.is_empty() {
                    return;
                }
                if !self.handle_payload(test_name, &payload) {
                    payload_error = Some(format!("Failed to handle payload {payload}"));
                }
            };
            child_runner.run_child_blocking(child_config, &mut handle_fd3_message)
        };
        self.child_runner = Some(child_runner);

        let child_output = match child_output {
            Ok(output) => output,
            Err(e) => {
                self.add_general_error(test_name, &e.to_string());
                return;
            }
        };

        if let Some(protocol_error) = child_output.protocol_error {
            self.add_general_error(test_name, &protocol_error);
            return;
        }

        if child_output.timed_out {
            self.add_general_error(test_name, "Child process heartbeat timed out");
            return;
        }

        if child_output.stop_requested {
            self.add_general_error(test_name, "Child process stopped because NVVS requested stop");
            return;
        }

        let Some(code) = child_output.exit_code else {
            self.add_general_error(test_name, "Failed to wait for child process");
            return;
        };
        if code == 0 {
            info!("Child process exited successfully");
        } else {
            self.add_general_error(
                test_name,
                &format!(
                    "Child process exited with non-zero code, {code}. Please check the logs {stdout_log_path} and {stderr_log_path} for more details."
                ),
            );
            return;
        }

        if let Some(payload_error) = payload_error {
            self.add_general_error(test_name, &payload_error);
        }
    }

    /// Retrieve the results of the test.
    ///
    /// # Arguments
    ///
    /// * `test_name` - The name of the test.
    ///
    /// # Returns
    ///
    /// * `dcgmDiagEntityResults_v2` - The results of the test.
    ///
    /// # Safety
    ///
    /// If the returned result has a non-null `auxData.data`, `auxData.data` is a borrowed raw pointer into the
    /// plugin's internal state. Callers must not read through `auxData.data` after this `CpuSdcPlugin` is dropped.
    ///
    pub fn retrieve_results(&self, test_name: &str) -> dcgmDiagEntityResults_v2 {
        let mut result = self.results.get(test_name).map_or_else(
            || {
                error!("Failed to get result for test {}", test_name);
                dcgmDiagEntityResults_v2::default()
            },
            |result| *result,
        );

        if let Some(aux_data) = self.aux_data.get(test_name) {
            result.auxData.version = dcgmDiagAuxData_version1;
            result.auxData.type_ = dcgmDiagAuxDataType_JSON_VALUE_AUX_DATA_TYPE;
            result.auxData.size = aux_data.len();
            result.auxData.data = aux_data.as_ptr().cast_mut().cast::<c_void>();
        }

        result
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS};
    #[cfg(not(miri))]
    use child_process::{ChildExitStatus, ChildProcessBinary, MockChildProcessRunner};
    use dcgm_utils::{c_char_buf_to_string, write_c_char_buf};
    #[cfg(not(miri))]
    use std::ffi::{OsStr, OsString};
    #[cfg(not(miri))]
    use std::path::{Path, PathBuf};
    #[cfg(not(miri))]
    use std::sync::Mutex;

    #[cfg(not(miri))]
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const TEST_BOOLEAN_FALSE: &str = "false";
    const TEST_CONFIG_PATH: &str = "/etc/dcgm/sdc.json";
    const TEST_RESULT_DETAILS_OK: &str = "OK";
    const TEST_RESULT_DETAILS_MEMORY_MISMATCH: &str = "memory mismatch detected";
    const TEST_CPU_ID_0: &str = "0";
    const TEST_CPU_ID_1: &str = "1";
    const TEST_SDC_VERSION: &str = "4.5.6";
    const TEST_UNKNOWN_NAME: &str = "missing_cpu_sdc";

    #[cfg(not(miri))]
    struct EnvVarGuard {
        /// Name of the process environment variable managed by this guard.
        key: &'static str,
        /// Value that was present before the guard changed the environment.
        previous: Option<OsString>,
    }

    #[cfg(not(miri))]
    impl EnvVarGuard {
        /// Sets an environment variable and returns a guard that restores its previous value.
        ///
        /// Inputs: variable `key` and replacement `value`. Output: a guard containing the previous value.
        fn set(key: &'static str, value: impl AsRef<OsStr>) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: Tests that mutate CPU SDC environment variables hold `ENV_LOCK`,
            // preventing concurrent mutations of the same process-global state in this module.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }

        /// Removes an environment variable and returns a guard that restores its previous value.
        ///
        /// Input: variable `key`. Output: a guard containing the previous value.
        fn remove(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: Tests that mutate CPU SDC environment variables hold `ENV_LOCK`,
            // preventing concurrent mutations of the same process-global state in this module.
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    #[cfg(not(miri))]
    impl Drop for EnvVarGuard {
        /// Restores the environment variable to the state captured by the guard.
        ///
        /// Input: this guard's captured key and value. Output: `()` after the restoration attempt.
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => {
                    // SAFETY: See `EnvVarGuard::set`; this restores the value while the test still
                    // holds `ENV_LOCK`.
                    unsafe { std::env::set_var(self.key, value) };
                }
                None => {
                    // SAFETY: See `EnvVarGuard::set`; this removes the test value while the test
                    // still holds `ENV_LOCK`.
                    unsafe { std::env::remove_var(self.key) };
                }
            }
        }
    }

    #[cfg(not(miri))]
    struct TempFile {
        /// Path of the temporary file removed when the guard is dropped.
        path: PathBuf,
    }

    #[cfg(not(miri))]
    impl TempFile {
        /// Creates an empty, process-specific temporary file for a test binary.
        ///
        /// Input: `name`, used in the temporary filename. Output: a guard owning the created path.
        fn new(name: &str) -> Self {
            let process_id = std::process::id();
            let path = std::env::temp_dir().join(format!("{name}_{process_id}"));
            std::fs::write(&path, b"").expect("temporary binary should be created");
            Self { path }
        }

        /// Returns the filesystem path of the temporary file.
        ///
        /// Input: this file guard. Output: a borrowed [`Path`] valid for the guard's lifetime.
        fn path(&self) -> &Path {
            &self.path
        }
    }

    #[cfg(not(miri))]
    impl Drop for TempFile {
        /// Removes the temporary file, ignoring cleanup errors.
        ///
        /// Input: this file guard's path. Output: `()` after the removal attempt.
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Creates a CPU entity pair for the supplied identifier.
    ///
    /// Input: `entity_id`, the CPU identifier. Output: the corresponding CPU entity pair.
    fn cpu_entity(entity_id: u32) -> dcgmGroupEntityPair_t {
        dcgmGroupEntityPair_t {
            entityGroupId: dcgm_field_entity_group_t_DCGM_FE_CPU,
            entityId: entity_id,
        }
    }

    /// Build a minimal entity list for tests without placing a full
    /// [`dcgmDiagPluginEntityList_v1`] on the stack: it contains `entities[1024]`, and each slot
    /// includes GPU-sized aux data, which overflows typical test thread stacks when default-built.
    ///
    /// Input: `entity_ids`, the CPUs to include. Output: a heap-allocated entity list containing those CPUs.
    fn entity_list(entity_ids: &[u32]) -> Box<dcgmDiagPluginEntityList_v1> {
        let mut entity_info = Box::<dcgmDiagPluginEntityList_v1>::new_uninit();
        // SAFETY: The boxed allocation is valid for one `dcgmDiagPluginEntityList_v1`.
        unsafe { entity_info.as_mut_ptr().write_bytes(0, 1) };
        // SAFETY: All-zero is a valid C layout for this plugin input in tests; we then set
        // `numEntities` and only the entity pairs we use.
        let mut entity_info = unsafe { entity_info.assume_init() };
        entity_info.numEntities = entity_ids
            .len()
            .try_into()
            .expect("test entity count should fit in u32");
        for (index, entity_id) in entity_ids.iter().copied().enumerate() {
            entity_info.entities[index].entity = cpu_entity(entity_id);
        }
        entity_info
    }

    /// Creates a plugin test parameter with the supplied name and value.
    ///
    /// Inputs: parameter `name` and `value`. Output: an initialized C ABI parameter structure.
    fn parameter(name: &str, value: &str) -> dcgmDiagPluginTestParameter_t {
        let mut parameter = dcgmDiagPluginTestParameter_t::default();
        write_c_char_buf(&mut parameter.parameterName, name);
        write_c_char_buf(&mut parameter.parameterValue, value);
        parameter
    }

    /// Creates an SDC test result for the supplied status, details, and devices.
    ///
    /// Inputs: result `status`, diagnostic `details`, and `device_ids`. Output: the corresponding [`TestResult`].
    fn test_result(status: &str, details: &str, device_ids: &[&str]) -> TestResult {
        TestResult {
            return_code: 0,
            test_id: String::from("cpu_sdc_test"),
            test_num: 1,
            test_suite_len: 1,
            test_status: String::from(status),
            test_details: String::from(details),
            device_type: String::from(SDC_DEVICE_TYPE_CPU),
            device_ids: device_ids.iter().map(|device_id| String::from(*device_id)).collect(),
        }
    }

    #[cfg(not(miri))]
    /// Serializes an SDC test-result protocol message for a mocked child runner.
    ///
    /// Inputs: result `status`, diagnostic `details`, and `device_ids`. Output: its serialized JSON bytes.
    fn test_result_payload(status: &str, details: &str, device_ids: &[&str]) -> Vec<u8> {
        let payload = ProtocolMessage::TestResult(crate::payload::TestResultPayload {
            test_result: test_result(status, details, device_ids),
        });
        serde_json::to_vec(&payload).expect("test result payload should serialize")
    }

    #[cfg(not(miri))]
    /// Serializes an SDC version protocol message for a mocked child runner.
    ///
    /// Input: the SDC `version`. Output: its serialized JSON protocol bytes.
    fn sdc_version_payload(version: &str) -> Vec<u8> {
        let payload = ProtocolMessage::Version(crate::payload::VersionPayload {
            sdc: crate::payload::CpuSdcVersion {
                version: String::from(version),
            },
        });
        serde_json::to_vec(&payload).expect("version payload should serialize")
    }

    /// Asserts that result auxiliary data contains the expected SDC version.
    ///
    /// Inputs: diagnostic `result` and `expected_version`. Output: `()` when assertions pass; panics on mismatch.
    fn assert_sdc_binary_version_aux_data(result: &dcgmDiagEntityResults_v2, expected_version: &str) {
        assert_eq!(result.auxData.version, dcgmDiagAuxData_version1);
        assert_eq!(result.auxData.type_, dcgmDiagAuxDataType_JSON_VALUE_AUX_DATA_TYPE);
        assert!(result.auxData.size > 0);
        assert!(!result.auxData.data.is_null());

        // SAFETY: `retrieve_results` returned a pointer and size into plugin-owned aux data,
        // and the plugin remains alive for this assertion.
        let aux_data = unsafe { std::slice::from_raw_parts(result.auxData.data.cast::<u8>(), result.auxData.size) };
        let aux_json: serde_json::Value = serde_json::from_slice(aux_data).expect("aux data should contain valid json");
        assert_eq!(aux_json["version"], expected_version);
    }

    #[cfg(not(miri))]
    /// Creates a mocked child runner that emits one payload and exits successfully.
    ///
    /// Inputs: message `payload` and expected child configuration. Output: the configured mocked runner.
    fn mocked_child_runner_for_payload(
        expected_binary_path: &Path,
        expected_stdout: &Path,
        expected_stderr: &Path,
        expected_args: &[String],
        payload: Vec<u8>,
    ) -> MockChildProcessRunner {
        mocked_child_runner_for_payload_and_exit_code(
            expected_binary_path,
            expected_stdout,
            expected_stderr,
            expected_args,
            payload,
            Some(0),
        )
    }

    #[cfg(not(miri))]
    /// Creates a mocked child runner with configurable payload and exit status.
    ///
    /// Inputs: message `payload`, expected configuration, and optional `exit_code`. Output: the configured mocked runner.
    fn mocked_child_runner_for_payload_and_exit_code(
        expected_binary_path: &Path,
        expected_stdout: &Path,
        expected_stderr: &Path,
        expected_args: &[String],
        payload: Vec<u8>,
        exit_code: Option<i32>,
    ) -> MockChildProcessRunner {
        let expected_binary_path = expected_binary_path.to_path_buf();
        let expected_stdout = expected_stdout.to_path_buf();
        let expected_stderr = expected_stderr.to_path_buf();
        let expected_args = expected_args.to_vec();
        let expected_mock_return_path = std::env::var_os(ENV_MOCKED_SDC_RETURN_PATH);
        let expected_missed_heartbeat_interval = std::env::var_os(ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC);
        let expected_heartbeat_disabled = heartbeat_disabled();
        let expected_heartbeat_timeout = heartbeat_timeout();

        let mut runner = MockChildProcessRunner::new();
        runner
            .expect_run_child_blocking()
            .times(1)
            .withf(move |config, _| {
                matches!(
                    &config.binary,
                    ChildProcessBinary::Fd { binary_path, .. }
                        if binary_path.as_path() == expected_binary_path.as_path()
                ) && config.args.as_slice() == expected_args.as_slice()
                    && config.log_paths.append
                    && config.log_paths.stdout.as_path() == expected_stdout.as_path()
                    && config.log_paths.stderr.as_path() == expected_stderr.as_path()
                    && config.heartbeat_disabled == expected_heartbeat_disabled
                    && config.heartbeat_timeout == expected_heartbeat_timeout
                    && config.stop_requested.is_some()
                    && expected_mock_return_path.as_ref().is_none_or(|expected| {
                        config.environment.iter().any(|(name, value)| {
                            name.as_os_str() == OsStr::new(ENV_MOCKED_SDC_RETURN_PATH) && value == expected
                        })
                    })
                    && expected_missed_heartbeat_interval.as_ref().is_none_or(|expected| {
                        config.environment.iter().any(|(name, value)| {
                            name.as_os_str() == OsStr::new(ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC)
                                && value == expected
                        })
                    })
            })
            .returning(move |_, fd3_message_handler| {
                fd3_message_handler.handle_fd3_message(payload.clone());
                Ok(ChildExitStatus {
                    exit_code,
                    ..ChildExitStatus::default()
                })
            });
        runner
    }

    #[test]
    #[cfg(not(miri))]
    /// Verifies heartbeat defaults, valid overrides, and invalid-value fallback behavior.
    /// Input: none. Output: `()` when configuration assertions pass.
    fn configure_heartbeat_uses_environment_overrides_and_defaults() {
        let _env_lock = ENV_LOCK.lock().expect("environment lock should not be poisoned");
        let mut config = ChildProcessConfig::new(
            Path::new("/bin/true"),
            Vec::<String>::new(),
            LogPaths::create(PathBuf::from("stdout"), PathBuf::from("stderr")),
        );

        let _disabled_guard = EnvVarGuard::remove(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED);
        let _seconds_guard = EnvVarGuard::remove(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS);
        configure_heartbeat(&mut config);
        assert!(!config.heartbeat_disabled);
        assert_eq!(
            config.heartbeat_timeout,
            Duration::from_secs(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS)
        );

        let _disabled_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, SDC_BOOLEAN_TRUE_NUMERIC);
        let _seconds_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS, "17");
        configure_heartbeat(&mut config);
        assert!(config.heartbeat_disabled);
        assert_eq!(config.heartbeat_timeout, Duration::from_secs(17));

        let _disabled_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, "FALSE");
        configure_heartbeat(&mut config);
        assert!(!config.heartbeat_disabled);

        let _disabled_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, "invalid");
        let _seconds_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS, "invalid");
        configure_heartbeat(&mut config);
        assert!(!config.heartbeat_disabled);
        assert_eq!(
            config.heartbeat_timeout,
            Duration::from_secs(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS)
        );

        let _seconds_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS, "0");
        configure_heartbeat(&mut config);
        assert_eq!(
            config.heartbeat_timeout,
            Duration::from_secs(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS)
        );

        let _disabled_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, "yes");
        let _seconds_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS, "0.5");
        configure_heartbeat(&mut config);
        assert!(!config.heartbeat_disabled);
        assert_eq!(
            config.heartbeat_timeout,
            Duration::from_secs(DEFAULT_HEARTBEAT_TIMEOUT_SECONDS)
        );

        let _disabled_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, "on");
        configure_heartbeat(&mut config);
        assert!(!config.heartbeat_disabled);
    }

    #[test]
    /// Verifies parsing of all supported plugin configuration parameters.
    /// Input: none. Output: `()` when parsed configuration assertions pass.
    fn init_config_parses_supported_parameters() {
        let mut plugin = CpuSdcPlugin::new();
        let parameters = [
            parameter(crate::CPU_SDC_PARAMETER_IS_ALLOWED, TEST_BOOLEAN_FALSE),
            parameter(crate::CPU_SDC_PARAMETER_CONFIG_PATH, TEST_CONFIG_PATH),
        ];

        assert!(plugin.init_config(crate::CPU_SDC_PLUGIN_NAME, &parameters).is_ok());

        let config = plugin
            .configs
            .get(crate::CPU_SDC_PLUGIN_NAME)
            .expect("config should be inserted");
        assert!(!config.is_allowed);
        assert_eq!(config.config_path, TEST_CONFIG_PATH);
    }

    #[test]
    /// Verifies that a passing result marks the named CPU as passed.
    /// Input: none. Output: `()` when result assertions pass.
    fn handle_pass_result_marks_named_cpu_as_pass() {
        let mut plugin = CpuSdcPlugin::new();
        let entities = entity_list(&[0, 1]);
        assert!(
            plugin
                .init_result(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref())
                .is_ok()
        );

        plugin.handle_pass_result(
            crate::CPU_SDC_PLUGIN_NAME,
            &test_result(SDC_TEST_STATUS_PASS, TEST_RESULT_DETAILS_OK, &[TEST_CPU_ID_0]),
        );

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP);
    }

    #[test]
    /// Verifies that an undefined passing device marks every skipped CPU as passed.
    /// Input: none. Output: `()` when result assertions pass.
    fn handle_pass_result_with_undefined_device_marks_all_skipped_as_pass() {
        let mut plugin = CpuSdcPlugin::new();
        let entities = entity_list(&[0, 1]);
        assert!(
            plugin
                .init_result(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref())
                .is_ok()
        );

        plugin.handle_pass_result(
            crate::CPU_SDC_PLUGIN_NAME,
            &test_result(SDC_TEST_STATUS_PASS, TEST_RESULT_DETAILS_OK, &[SDC_DEVICE_ID_UNDEFINED]),
        );

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
    }

    #[test]
    /// Verifies that a failed result records both entity status and error details.
    /// Input: none. Output: `()` when failure assertions pass.
    fn handle_fail_result_records_entity_failure_and_error() {
        let mut plugin = CpuSdcPlugin::new();
        let entities = entity_list(&[0, 1]);
        assert!(
            plugin
                .init_result(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref())
                .is_ok()
        );

        plugin.handle_fail_result(
            crate::CPU_SDC_PLUGIN_NAME,
            &test_result(
                SDC_TEST_STATUS_FAIL,
                TEST_RESULT_DETAILS_MEMORY_MISMATCH,
                &[TEST_CPU_ID_1],
            ),
        );

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.numErrors, 1);
        assert_eq!(
            result.errors[0].entity.entityGroupId,
            dcgm_field_entity_group_t_DCGM_FE_CPU
        );
        assert_eq!(result.errors[0].entity.entityId, 1);
        assert_eq!(result.errors[0].code, dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED);
        assert_eq!(
            result.errors[0].category,
            dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC
        );
        assert_eq!(
            c_char_buf_to_string(&result.errors[0].msg).expect("error message should decode"),
            format!("{SDC_ERROR_DETAILS_PREFIX}{TEST_RESULT_DETAILS_MEMORY_MISMATCH}")
        );
    }

    #[test]
    /// Verifies that version messages are retained as diagnostic auxiliary data.
    /// Input: none. Output: `()` when auxiliary-data assertions pass.
    fn handle_version_payload_records_sdc_binary_version_aux_data() {
        let mut plugin = CpuSdcPlugin::new();
        let entities = entity_list(&[0]);
        assert!(plugin.init_result("cpu_sdc", entities.as_ref()).is_ok());

        assert!(plugin.handle_payload("cpu_sdc", r#"{"sdc":{"version":"1.2.3"}}"#));

        let result = plugin.retrieve_results("cpu_sdc");
        assert_sdc_binary_version_aux_data(&result, "1.2.3");
    }

    #[test]
    #[cfg(not(miri))]
    /// Verifies that SDC availability requires the configured path to be a file.
    /// Input: none. Output: `()` when availability assertions pass.
    fn sdc_binary_availability_requires_existing_file() {
        let _env_lock = ENV_LOCK.lock().expect("environment lock should not be poisoned");
        let binary = TempFile::new("cpu_sdc_available_binary");
        let _binary_guard = EnvVarGuard::set(ENV_CPU_SDC_BIN_PATH, binary.path().as_os_str());

        assert!(is_sdc_binary_available());

        let missing_path = std::env::temp_dir().join(format!("missing_cpu_sdc_binary_{}", std::process::id()));
        let _missing_binary_guard = EnvVarGuard::set(ENV_CPU_SDC_BIN_PATH, missing_path.as_os_str());

        assert!(!is_sdc_binary_available());
    }

    #[test]
    #[cfg(not(miri))]
    /// Verifies successful test execution and result capture through a mocked runner.
    /// Input: none. Output: `()` when execution assertions pass.
    fn run_test_uses_mocked_child_runner_and_records_results() {
        let _env_lock = ENV_LOCK.lock().expect("environment lock should not be poisoned");
        let binary = TempFile::new("cpu_sdc_run_test_binary");
        let log_root = std::env::temp_dir();
        let expected_log_paths = CpuSdcPlugin::create_log_paths(&log_root);
        let expected_stdout = expected_log_paths.stdout;
        let expected_stderr = expected_log_paths.stderr;
        let expected_args = vec![
            String::from(SDC_ARG_ENABLE_HEARTBEAT),
            String::from(SDC_ARG_CONFIG_PATH),
            String::from(TEST_CONFIG_PATH),
        ];

        let _skip_guard = EnvVarGuard::set(ENV_CPU_SDC_SKIP_BIN_PERMISSION_CHECK, SDC_BOOLEAN_TRUE_NUMERIC);
        let _binary_guard = EnvVarGuard::set(ENV_CPU_SDC_BIN_PATH, binary.path().as_os_str());
        let _heartbeat_disabled_guard =
            EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_DISABLED, SDC_BOOLEAN_TRUE_NUMERIC);
        let _heartbeat_seconds_guard = EnvVarGuard::set(ENV_CPU_SDC_HEARTBEAT_TIMEOUT_SECONDS, "17");
        let _home_guard = EnvVarGuard::set(ENV_DCGM_HOME_DIR, log_root.as_os_str());
        let mocked_return_path = log_root.join("mocked_sdc_returns.json");
        let _mocked_return_guard = EnvVarGuard::set(ENV_MOCKED_SDC_RETURN_PATH, mocked_return_path.as_os_str());
        let _missed_heartbeat_guard = EnvVarGuard::set(ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC, "2");

        let pass_runner = mocked_child_runner_for_payload(
            binary.path(),
            &expected_stdout,
            &expected_stderr,
            &expected_args,
            test_result_payload(SDC_TEST_STATUS_PASS, TEST_RESULT_DETAILS_OK, &[TEST_CPU_ID_0]),
        );
        let mut plugin = CpuSdcPlugin::with_child_runner(Box::new(pass_runner));
        let entities = entity_list(&[0, 1]);
        let parameters = [parameter(crate::CPU_SDC_PARAMETER_CONFIG_PATH, TEST_CONFIG_PATH)];

        plugin.run_test(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref(), &parameters);

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.numResults, 2);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP);
        assert_eq!(result.numErrors, 0);

        let fail_runner = mocked_child_runner_for_payload(
            binary.path(),
            &expected_stdout,
            &expected_stderr,
            &expected_args,
            test_result_payload(
                SDC_TEST_STATUS_FAIL,
                TEST_RESULT_DETAILS_MEMORY_MISMATCH,
                &[TEST_CPU_ID_1],
            ),
        );
        let mut plugin = CpuSdcPlugin::with_child_runner(Box::new(fail_runner));

        plugin.run_test(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref(), &parameters);

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.numResults, 2);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.numErrors, 1);
        assert_eq!(result.errors[0].entity.entityId, 1);
        assert_eq!(
            c_char_buf_to_string(&result.errors[0].msg).expect("error message should decode"),
            format!("{SDC_ERROR_DETAILS_PREFIX}{TEST_RESULT_DETAILS_MEMORY_MISMATCH}")
        );
    }

    #[test]
    #[cfg(not(miri))]
    /// Verifies that test execution retains the version emitted by the child process.
    /// Input: none. Output: `()` when version assertions pass.
    fn run_test_records_sdc_binary_version_from_mocked_child_runner() {
        let _env_lock = ENV_LOCK.lock().expect("environment lock should not be poisoned");
        let binary = TempFile::new("cpu_sdc_run_test_version_binary");
        let log_root = std::env::temp_dir();
        let expected_log_paths = CpuSdcPlugin::create_log_paths(&log_root);
        let expected_stdout = expected_log_paths.stdout;
        let expected_stderr = expected_log_paths.stderr;
        let expected_args = vec![
            String::from(SDC_ARG_ENABLE_HEARTBEAT),
            String::from(SDC_ARG_CONFIG_PATH),
            String::from(TEST_CONFIG_PATH),
        ];

        let _skip_guard = EnvVarGuard::set(ENV_CPU_SDC_SKIP_BIN_PERMISSION_CHECK, SDC_BOOLEAN_TRUE_NUMERIC);
        let _binary_guard = EnvVarGuard::set(ENV_CPU_SDC_BIN_PATH, binary.path().as_os_str());
        let _home_guard = EnvVarGuard::set(ENV_DCGM_HOME_DIR, log_root.as_os_str());

        let runner = mocked_child_runner_for_payload(
            binary.path(),
            &expected_stdout,
            &expected_stderr,
            &expected_args,
            sdc_version_payload(TEST_SDC_VERSION),
        );
        let mut plugin = CpuSdcPlugin::with_child_runner(Box::new(runner));
        let entities = entity_list(&[0]);
        let parameters = [parameter(crate::CPU_SDC_PARAMETER_CONFIG_PATH, TEST_CONFIG_PATH)];

        plugin.run_test(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref(), &parameters);

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.numResults, 1);
        assert_eq!(result.numErrors, 0);
        assert_sdc_binary_version_aux_data(&result, TEST_SDC_VERSION);
    }

    #[test]
    #[cfg(not(miri))]
    /// Verifies that a nonzero child exit status produces a diagnostic failure.
    /// Input: none. Output: `()` when failure assertions pass.
    fn run_test_reports_failure_when_child_exits_nonzero() {
        let _env_lock = ENV_LOCK.lock().expect("environment lock should not be poisoned");
        let binary = TempFile::new("cpu_sdc_run_test_nonzero_exit_binary");
        let log_root = std::env::temp_dir();
        let expected_log_paths = CpuSdcPlugin::create_log_paths(&log_root);
        let expected_stdout = expected_log_paths.stdout;
        let expected_stderr = expected_log_paths.stderr;
        let expected_args = vec![
            String::from(SDC_ARG_ENABLE_HEARTBEAT),
            String::from(SDC_ARG_CONFIG_PATH),
            String::from(TEST_CONFIG_PATH),
        ];

        let _skip_guard = EnvVarGuard::set(ENV_CPU_SDC_SKIP_BIN_PERMISSION_CHECK, SDC_BOOLEAN_TRUE_NUMERIC);
        let _binary_guard = EnvVarGuard::set(ENV_CPU_SDC_BIN_PATH, binary.path().as_os_str());
        let _home_guard = EnvVarGuard::set(ENV_DCGM_HOME_DIR, log_root.as_os_str());
        let mocked_return_path = log_root.join("mocked_sdc_returns.json");
        let _mocked_return_guard = EnvVarGuard::set(ENV_MOCKED_SDC_RETURN_PATH, mocked_return_path.as_os_str());

        let runner = mocked_child_runner_for_payload_and_exit_code(
            binary.path(),
            &expected_stdout,
            &expected_stderr,
            &expected_args,
            test_result_payload(SDC_TEST_STATUS_PASS, TEST_RESULT_DETAILS_OK, &[TEST_CPU_ID_0]),
            Some(7),
        );
        let mut plugin = CpuSdcPlugin::with_child_runner(Box::new(runner));
        let entities = entity_list(&[0, 1]);
        let parameters = [parameter(crate::CPU_SDC_PARAMETER_CONFIG_PATH, TEST_CONFIG_PATH)];

        plugin.run_test(crate::CPU_SDC_PLUGIN_NAME, entities.as_ref(), &parameters);

        let result = plugin.retrieve_results(crate::CPU_SDC_PLUGIN_NAME);
        assert_eq!(result.numResults, 2);
        assert_eq!(result.results[0].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.results[1].result, dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL);
        assert_eq!(result.numErrors, 1);
        assert_eq!(
            result.errors[0].entity.entityGroupId,
            dcgm_field_entity_group_t_DCGM_FE_NONE
        );
        assert_eq!(result.errors[0].entity.entityId, 0);
        assert_eq!(
            c_char_buf_to_string(&result.errors[0].msg).expect("error message should decode"),
            format!(
                "Child process exited with non-zero code, 7. Please check the logs {} and {} for more details.",
                expected_stdout.display(),
                expected_stderr.display()
            )
        );
    }

    #[test]
    /// Verifies that retrieving an unknown test returns empty default results.
    /// Input: none. Output: `()` when default-result assertions pass.
    fn retrieve_results_returns_default_for_unknown_test() {
        let plugin = CpuSdcPlugin::new();

        let result = plugin.retrieve_results(TEST_UNKNOWN_NAME);

        assert_eq!(result.numResults, 0);
        assert_eq!(result.numErrors, 0);
    }
}
