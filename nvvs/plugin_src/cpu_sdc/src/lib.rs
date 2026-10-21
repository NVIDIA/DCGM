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

mod common_plugin;
mod payload;
mod plugin;

use crate::plugin::{CpuSdcPlugin, is_sdc_binary_available};
use core::ffi::{c_char, c_void};
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use dcgm_logging::sinks::{HostEngineAppenderCallbackFp, NvvsPluginSink, Record as DcgmLoggingRecord};
use dcgm_logging::{LevelFilter, Log, add_sink, clear_sinks, error, set_max_level};
use dcgm_utils::write_c_char_buf;
use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

pub(crate) const CPU_SDC_PLUGIN_NAME: &str = "cpu_sdc";
const CPU_SDC_PLUGIN_DESCRIPTION: &str = "Silent Data Corruption tests for CPU";
const CPU_SDC_PLUGIN_CATEGORY: &str = "Hardware";
pub(crate) const CPU_SDC_PARAMETER_IS_ALLOWED: &str = "is_allowed";
pub(crate) const CPU_SDC_PARAMETER_CONFIG_PATH: &str = "config_path";

#[allow(
    dead_code,
    improper_ctypes,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_imports,
    unused_qualifications,
    clippy::all,
    clippy::cargo,
    clippy::nursery,
    clippy::pedantic,
    clippy::restriction
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[allow(non_camel_case_types, non_upper_case_globals)]
mod compat {
    pub use crate::bindings::root::{
        DCGM_DIAG_PLUGIN_INTERFACE_VERSION, DCGM_DIAG_RESPONSE_ERRORS_MAX, DCGM_DIAG_RESPONSE_INFO_MAX_V2,
        DCGM_DIAG_TEST_RUN_RESULTS_MAX, DCGM_ERR_MSG_LENGTH, DCGM_INT32_BLANK, HangDetectMonitorPtr, NvvsCommonPtr,
        dcgmDiagAuxData_t, dcgmDiagCustomStats_t, dcgmDiagEntityResult_v1, dcgmDiagEntityResults_v2, dcgmDiagError_v1,
        dcgmDiagInfo_v1, dcgmDiagPluginAttr_v1, dcgmDiagPluginEntityList_v1, dcgmDiagPluginInfo_t,
        dcgmDiagPluginStatFieldIds_t, dcgmDiagPluginTestParameter_t, dcgmGroupEntityPair_t, dcgmHandle_t, dcgmReturn_t,
        hostEngineAppenderCallbackFp_t,
    };

    pub type DcgmLoggingSeverity_t = crate::bindings::root::DcgmLoggingSeverity_t::Type;
    pub type dcgmErrorCategory_t = crate::bindings::root::dcgmErrorCategory_enum::Type;
    pub type dcgmErrorSeverity_t = crate::bindings::root::dcgmErrorSeverity_enum::Type;
    pub type dcgmError_t = crate::bindings::root::dcgmError_enum::Type;
    pub type dcgmDiagAuxDataType_t = crate::bindings::root::dcgmDiagAuxDataType::Type;
    pub type dcgmPluginValue_t = crate::bindings::root::dcgmPluginValue_t::Type;

    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityInfo: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityInfo;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityFatal: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityFatal;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityWarning: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityWarning;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityError: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityError;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityDebug: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityDebug;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityVerbose: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityVerbose;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityNone: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityNone;
    pub const DcgmLoggingSeverity_t_DcgmLoggingSeverityUnspecified: DcgmLoggingSeverity_t =
        crate::bindings::root::DcgmLoggingSeverity_t::DcgmLoggingSeverityUnspecified;

    pub const dcgmDiagResult_enum_DCGM_DIAG_RESULT_PASS: crate::bindings::root::dcgmDiagResult_t =
        crate::bindings::root::dcgmDiagResult_enum::DCGM_DIAG_RESULT_PASS;
    pub const dcgmDiagResult_enum_DCGM_DIAG_RESULT_SKIP: crate::bindings::root::dcgmDiagResult_t =
        crate::bindings::root::dcgmDiagResult_enum::DCGM_DIAG_RESULT_SKIP;
    pub const dcgmDiagResult_enum_DCGM_DIAG_RESULT_FAIL: crate::bindings::root::dcgmDiagResult_t =
        crate::bindings::root::dcgmDiagResult_enum::DCGM_DIAG_RESULT_FAIL;

    pub const dcgmError_enum_DCGM_FR_CPU_SDC_TEST_FAILED: dcgmError_t =
        crate::bindings::root::dcgmError_enum::DCGM_FR_CPU_SDC_TEST_FAILED;
    pub const dcgmErrorCategory_enum_DCGM_FR_EC_SOFTWARE_CPU_SDC: dcgmErrorCategory_t =
        crate::bindings::root::dcgmErrorCategory_enum::DCGM_FR_EC_SOFTWARE_CPU_SDC;
    pub const dcgmErrorSeverity_enum_DCGM_ERROR_TRIAGE: dcgmErrorSeverity_t =
        crate::bindings::root::dcgmErrorSeverity_enum::DCGM_ERROR_TRIAGE;

    pub const dcgmPluginValue_t_DcgmPluginParamBool: dcgmPluginValue_t =
        crate::bindings::root::dcgmPluginValue_t::DcgmPluginParamBool;
    pub const dcgmPluginValue_t_DcgmPluginParamString: dcgmPluginValue_t =
        crate::bindings::root::dcgmPluginValue_t::DcgmPluginParamString;

    pub const dcgmDiagAuxDataType_JSON_VALUE_AUX_DATA_TYPE: dcgmDiagAuxDataType_t =
        crate::bindings::root::dcgmDiagAuxDataType::JSON_VALUE_AUX_DATA_TYPE;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "This mirrors MAKE_DCGM_VERSION(dcgmDiagAuxData_t, 1); the C ABI struct size is bounded."
    )]
    pub const dcgmDiagAuxData_version1: u32 = std::mem::size_of::<dcgmDiagAuxData_t>() as u32 | (1_u32 << 24);

    pub const dcgmReturn_enum_DCGM_ST_BADPARAM: dcgmReturn_t = crate::bindings::root::dcgmReturn_enum::DCGM_ST_BADPARAM;
    pub const dcgmReturn_enum_DCGM_ST_GENERIC_ERROR: dcgmReturn_t =
        crate::bindings::root::dcgmReturn_enum::DCGM_ST_GENERIC_ERROR;
    pub const dcgmReturn_enum_DCGM_ST_OK: dcgmReturn_t = crate::bindings::root::dcgmReturn_enum::DCGM_ST_OK;
    pub const dcgmReturn_enum_DCGM_ST_PLUGIN_EXCEPTION: dcgmReturn_t =
        crate::bindings::root::dcgmReturn_enum::DCGM_ST_PLUGIN_EXCEPTION;
    pub const dcgmReturn_enum_DCGM_ST_VER_MISMATCH: dcgmReturn_t =
        crate::bindings::root::dcgmReturn_enum::DCGM_ST_VER_MISMATCH;

    pub const dcgm_field_entity_group_t_DCGM_FE_CPU: crate::bindings::root::dcgm_field_entity_group_t::Type =
        crate::bindings::root::dcgm_field_entity_group_t::DCGM_FE_CPU;
    pub const dcgm_field_entity_group_t_DCGM_FE_NONE: crate::bindings::root::dcgm_field_entity_group_t::Type =
        crate::bindings::root::dcgm_field_entity_group_t::DCGM_FE_NONE;
}

pub use compat::*;

type BoundLoggerCallbackFp = unsafe extern "C" fn(*const bindings::root::DcgmNs::Logging::Record);
type DcgmLoggingCallbackFp = unsafe extern "C" fn(*const DcgmLoggingRecord);

/// Converts a DCGM logging severity into the corresponding Rust log filter.
///
/// Input: `logging_severity`, the DCGM severity. Output: its [`LevelFilter`] equivalent.
const fn logging_level_filter(logging_severity: DcgmLoggingSeverity_t) -> LevelFilter {
    if logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityVerbose {
        LevelFilter::Trace
    } else if logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityDebug {
        LevelFilter::Debug
    } else if logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityInfo {
        LevelFilter::Info
    } else if logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityWarning {
        LevelFilter::Warn
    } else if logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityError
        || logging_severity == DcgmLoggingSeverity_t_DcgmLoggingSeverityFatal
    {
        LevelFilter::Error
    } else {
        LevelFilter::Off
    }
}

/// Configures the plugin log sink and maximum level for the host callback.
///
/// Inputs: the DCGM `logging_severity` and optional `logging_callback`. Output: `()` after installing or clearing sinks.
fn configure_logging(logging_severity: DcgmLoggingSeverity_t, logging_callback: hostEngineAppenderCallbackFp_t) {
    let max_level = logging_level_filter(logging_severity);
    let Some(callback) = logging_callback else {
        clear_sinks();
        set_max_level(LevelFilter::Off);
        return;
    };
    // SAFETY: bindgen and `dcgm-logging` model the same C++ `DcgmNs::Logging::Record`
    // layout for the same host-engine callback ABI.
    let callback: HostEngineAppenderCallbackFp =
        Some(unsafe { std::mem::transmute::<BoundLoggerCallbackFp, DcgmLoggingCallbackFp>(callback) });

    let sink = Arc::new(NvvsPluginSink::new(callback, max_level)) as Arc<dyn Log>;
    add_sink(&sink);
    set_max_level(max_level);
}

/// Recovers plugin state from an opaque pointer supplied through the C ABI.
///
/// Input: `user_data`, an opaque plugin-state pointer. Output: mutable plugin state, or `None` for a null pointer.
fn plugin_from_user_data(user_data: *mut c_void) -> Option<&'static mut CpuSdcPlugin> {
    let raw = NonNull::new(user_data.cast::<CpuSdcPlugin>())?;
    let ptr = raw.as_ptr();

    // SAFETY: `InitializePlugin` stores a `Box<CpuSdcPlugin>` in `userData`, and callers pass it
    // back unchanged until `ShutdownPlugin` consumes it.
    Some(unsafe { &mut *ptr })
}

/// Copies a nullable, NUL-terminated C string into an owned Rust string.
///
/// Input: `value`, a nullable C string pointer. Output: its lossy owned representation, or `None` when null.
fn c_string_from_ptr(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }

    // SAFETY: The plugin ABI passes valid, NUL-terminated strings for the duration of each call.
    let c_str = unsafe { CStr::from_ptr(value) };
    Some(c_str.to_string_lossy().into_owned())
}

/// Builds plugin metadata and advertises the test only when SDC is available.
///
/// Input: `sdc_binary_available`, whether the binary can be used. Output: populated plugin metadata.
fn build_plugin_info_with_availability(sdc_binary_available: bool) -> dcgmDiagPluginInfo_t {
    let mut info = dcgmDiagPluginInfo_t::default();
    write_c_char_buf(&mut info.pluginName, CPU_SDC_PLUGIN_NAME);
    write_c_char_buf(&mut info.description, CPU_SDC_PLUGIN_DESCRIPTION);

    if !sdc_binary_available {
        return info;
    }

    info.numTests = 1;
    write_c_char_buf(&mut info.tests[0].testName, CPU_SDC_PLUGIN_NAME);
    write_c_char_buf(&mut info.tests[0].description, CPU_SDC_PLUGIN_DESCRIPTION);
    write_c_char_buf(&mut info.tests[0].testCategory, CPU_SDC_PLUGIN_CATEGORY);
    info.tests[0].targetEntityGroup = dcgm_field_entity_group_t_DCGM_FE_CPU;

    let supported_params = [
        (CPU_SDC_PARAMETER_IS_ALLOWED, dcgmPluginValue_t_DcgmPluginParamBool),
        (CPU_SDC_PARAMETER_CONFIG_PATH, dcgmPluginValue_t_DcgmPluginParamString),
    ];
    info.tests[0].numValidParameters = u32::try_from(supported_params.len()).unwrap_or(2);
    for (index, (name, param_type)) in supported_params.into_iter().enumerate() {
        write_c_char_buf(&mut info.tests[0].validParameters[index].parameterName, name);
        info.tests[0].validParameters[index].parameterType = param_type;
    }

    info
}

/// Builds plugin metadata using the current SDC binary availability.
///
/// Input: none. Output: plugin metadata reflecting the configured SDC path.
fn build_plugin_info() -> dcgmDiagPluginInfo_t {
    build_plugin_info_with_availability(is_sdc_binary_available())
}

/// Validates the requested interface version and writes plugin metadata to ABI storage.
///
/// Inputs: `plugin_interface_version` and writable `info` storage. Output: a DCGM status describing success or rejection.
fn get_plugin_info_impl(plugin_interface_version: u32, info: *mut MaybeUninit<dcgmDiagPluginInfo_t>) -> dcgmReturn_t {
    let Some(info_ptr) = NonNull::new(info) else {
        return dcgmReturn_enum_DCGM_ST_BADPARAM;
    };

    if plugin_interface_version != DCGM_DIAG_PLUGIN_INTERFACE_VERSION {
        return dcgmReturn_enum_DCGM_ST_VER_MISMATCH;
    }

    // SAFETY: `info_ptr` is non-null and points to caller-owned storage for one
    // `dcgmDiagPluginInfo_t`. The storage may be uninitialized, so write through
    // `MaybeUninit` to avoid reading or dropping the previous contents.
    unsafe { info_ptr.as_ptr().write(MaybeUninit::new(build_plugin_info())) };

    dcgmReturn_enum_DCGM_ST_OK
}

/// Initializes logging and plugin state, returning both through C ABI output pointers.
///
/// Inputs: output storage plus logging severity and callback. Output: a DCGM status; successful calls initialize both outputs.
fn initialize_plugin_impl(
    stat_field_ids: *mut MaybeUninit<dcgmDiagPluginStatFieldIds_t>,
    user_data: *mut *mut c_void,
    logging_severity: DcgmLoggingSeverity_t,
    logging_callback: hostEngineAppenderCallbackFp_t,
) -> dcgmReturn_t {
    configure_logging(logging_severity, logging_callback);

    let Some(stat_field_ids_ptr) = NonNull::new(stat_field_ids) else {
        error!("cpu_sdc: InitializePlugin called with null stat_field_ids");
        return dcgmReturn_enum_DCGM_ST_BADPARAM;
    };
    let Some(user_data_ptr) = NonNull::new(user_data) else {
        error!("cpu_sdc: InitializePlugin called with null user_data");
        return dcgmReturn_enum_DCGM_ST_BADPARAM;
    };

    let plugin = Box::new(CpuSdcPlugin::new());
    let raw_plugin = Box::into_raw(plugin).cast::<c_void>();

    // SAFETY: `stat_field_ids_ptr` is non-null and points to caller-owned storage for one
    // `dcgmDiagPluginStatFieldIds_t`. The storage may be uninitialized, so write through
    // `MaybeUninit` to avoid reading or dropping the previous contents.
    unsafe {
        stat_field_ids_ptr
            .as_ptr()
            .write(MaybeUninit::new(dcgmDiagPluginStatFieldIds_t::default()));
    };

    let user_data_raw = user_data_ptr.as_ptr();
    // SAFETY: `user_data_ptr` is non-null and points to writable caller-owned storage.
    unsafe { user_data_raw.write(raw_plugin) };

    dcgmReturn_enum_DCGM_ST_OK
}

/// Validates C ABI test inputs and delegates test execution to the plugin state.
///
/// Inputs: a test name, parameter array and count, entity list, and plugin state. Output: `()` after execution or validation failure.
fn run_test_impl(
    test_name: *const c_char,
    num_parameters: u32,
    test_parameters: *const dcgmDiagPluginTestParameter_t,
    entity_info: *const dcgmDiagPluginEntityList_v1,
    user_data: *mut c_void,
) {
    let Some(plugin) = plugin_from_user_data(user_data) else {
        error!("cpu_sdc: RunTest called without plugin state");
        return;
    };
    let Some(entity_info_ptr) = NonNull::new(entity_info.cast_mut()) else {
        error!("cpu_sdc: RunTest called without entity info");
        return;
    };

    let name = c_string_from_ptr(test_name).unwrap_or_default();
    if test_parameters.is_null() && num_parameters != 0 {
        error!("cpu_sdc: RunTest called with null parameters and non-zero parameter count");
        return;
    }

    let parameters = if num_parameters == 0 {
        &[]
    } else {
        // SAFETY: `test_parameters` is non-null in this branch and points to `num_parameters`
        // elements per the plugin ABI contract.
        unsafe { std::slice::from_raw_parts(test_parameters, num_parameters as usize) }
    };

    let entity_info_ref = entity_info_ptr.as_ptr();
    // SAFETY: `entity_info_ptr` is non-null and borrowed only for this call.
    plugin.run_test(&name, unsafe { &*entity_info_ref }, parameters);
}

/// Retrieves results from plugin state and writes them to C ABI output storage.
///
/// Inputs: a test name, writable result storage, and plugin state. Output: `()` after writing results or rejecting invalid storage.
fn retrieve_results_impl(
    test_name: *const c_char,
    results: *mut MaybeUninit<dcgmDiagEntityResults_v2>,
    user_data: *mut c_void,
) {
    let Some(plugin) = plugin_from_user_data(user_data) else {
        error!("cpu_sdc: RetrieveResults called without plugin state");
        return;
    };
    let Some(results_ptr) = NonNull::new(results) else {
        error!("cpu_sdc: RetrieveResults called without result storage");
        return;
    };

    let name = c_string_from_ptr(test_name).unwrap_or_default();
    // SAFETY: `results_ptr` is non-null and points to caller-owned storage for one
    // `dcgmDiagEntityResults_v2`. The storage may be uninitialized, so write through
    // `MaybeUninit` to avoid reading or dropping the previous contents.
    unsafe {
        results_ptr
            .as_ptr()
            .write(MaybeUninit::new(plugin.retrieve_results(&name)));
    };
}

/// Releases plugin state previously returned through the C ABI and disables logging.
///
/// Input: `user_data`, the owned plugin-state pointer. Output: a DCGM success or bad-parameter status.
fn shutdown_plugin_impl(user_data: *mut c_void) -> dcgmReturn_t {
    let Some(plugin_ptr) = NonNull::new(user_data.cast::<CpuSdcPlugin>()) else {
        return dcgmReturn_enum_DCGM_ST_BADPARAM;
    };

    let ptr = plugin_ptr.as_ptr();
    // SAFETY: `ptr` was produced by `Box::into_raw` in `InitializePlugin` and is consumed exactly once here.
    drop(unsafe { Box::from_raw(ptr) });
    clear_sinks();
    set_max_level(LevelFilter::Off);

    dcgmReturn_enum_DCGM_ST_OK
}

#[unsafe(no_mangle)]
/// Returns the NVVS plugin interface version implemented by this plugin.
///
/// Input: none. Output: [`DCGM_DIAG_PLUGIN_INTERFACE_VERSION`], or zero if panic containment is triggered.
///
/// # Safety
///
/// This function is exported for the NVVS C plugin ABI. It does not dereference any caller-owned pointers.
pub unsafe extern "C" fn GetPluginInterfaceVersion() -> u32 {
    catch_unwind(AssertUnwindSafe(|| DCGM_DIAG_PLUGIN_INTERFACE_VERSION)).unwrap_or_default()
}

#[unsafe(no_mangle)]
/// Writes static plugin metadata into caller-owned storage.
///
/// Inputs: the requested interface version and `info` output storage. Output: a DCGM status describing the result.
///
/// # Safety
///
/// When `info` is non-null, it must point to writable storage for one `dcgmDiagPluginInfo_t`. The storage may be
/// uninitialized.
pub unsafe extern "C" fn GetPluginInfo(plugin_interface_version: u32, info: *mut dcgmDiagPluginInfo_t) -> dcgmReturn_t {
    catch_unwind(AssertUnwindSafe(|| {
        get_plugin_info_impl(
            plugin_interface_version,
            info.cast::<MaybeUninit<dcgmDiagPluginInfo_t>>(),
        )
    }))
    .unwrap_or(dcgmReturn_enum_DCGM_ST_PLUGIN_EXCEPTION)
}

#[unsafe(no_mangle)]
/// Initializes the CPU SDC plugin instance and returns plugin-owned state through `user_data`.
///
/// Inputs: ABI context, output storage, logging configuration, plugin attributes, and NVVS services. Output: a DCGM status;
/// successful calls initialize `stat_field_ids` and `user_data`.
///
/// # Safety
///
/// When non-null, `stat_field_ids` must point to writable storage for one `dcgmDiagPluginStatFieldIds_t`, and
/// `user_data` must point to writable storage for one opaque pointer. If `logging_callback` is present, it must remain
/// valid for the lifetime of this plugin instance.
pub unsafe extern "C" fn InitializePlugin(
    _handle: dcgmHandle_t,
    stat_field_ids: *mut dcgmDiagPluginStatFieldIds_t,
    user_data: *mut *mut c_void,
    logging_severity: DcgmLoggingSeverity_t,
    logging_callback: hostEngineAppenderCallbackFp_t,
    _plugin_attr: *const dcgmDiagPluginAttr_v1,
    _nvvs_common: NvvsCommonPtr,
    _monitor: HangDetectMonitorPtr,
) -> dcgmReturn_t {
    catch_unwind(AssertUnwindSafe(|| {
        initialize_plugin_impl(
            stat_field_ids.cast::<MaybeUninit<dcgmDiagPluginStatFieldIds_t>>(),
            user_data,
            logging_severity,
            logging_callback,
        )
    }))
    .unwrap_or(dcgmReturn_enum_DCGM_ST_PLUGIN_EXCEPTION)
}

#[unsafe(no_mangle)]
/// Runs the named CPU SDC test for the supplied entities and parameters.
///
/// Inputs: test name, timeout, parameter array and count, entity list, and plugin state. Output: `()` after running the test;
/// panics are contained at the ABI boundary.
///
/// # Safety
///
/// `user_data` must be null or a pointer previously returned by `InitializePlugin`. When non-null, `test_name` must be
/// a valid NUL-terminated C string, and `entity_info` must point to a valid `dcgmDiagPluginEntityList_v1`. When
/// `num_parameters` is nonzero, `test_parameters` must point to `num_parameters` valid entries.
pub unsafe extern "C" fn RunTest(
    test_name: *const c_char,
    _timeout: u32,
    num_parameters: u32,
    test_parameters: *const dcgmDiagPluginTestParameter_t,
    entity_info: *const dcgmDiagPluginEntityList_v1,
    user_data: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        run_test_impl(test_name, num_parameters, test_parameters, entity_info, user_data);
    }));
}

#[unsafe(no_mangle)]
/// Writes custom statistics for a completed test.
///
/// Inputs: test name, writable statistics storage, and plugin state. Output: `()` after reporting that no custom statistics exist.
///
/// # Safety
///
/// When `custom_stats` is non-null, it must point to writable storage for one `dcgmDiagCustomStats_t`.
pub unsafe extern "C" fn RetrieveCustomStats(
    _test_name: *const c_char,
    custom_stats: *mut dcgmDiagCustomStats_t,
    _user_data: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(custom_stats_ptr) = NonNull::new(custom_stats) {
            let ptr = custom_stats_ptr.as_ptr();
            // SAFETY: `ptr` is non-null and points to writable caller-owned storage.
            let more_stats = unsafe { &raw mut (*ptr).moreStats };
            // SAFETY: `ptr` is non-null and points to writable caller-owned storage.
            let num_stats = unsafe { &raw mut (*ptr).numStats };
            // SAFETY: `more_stats` points to the writable `moreStats` field of `custom_stats`.
            unsafe { more_stats.write(0) };
            // SAFETY: `num_stats` points to the writable `numStats` field of `custom_stats`.
            unsafe { num_stats.write(0) };
        }
    }));
}

#[unsafe(no_mangle)]
/// Retrieves the latest result for the named CPU SDC test.
///
/// Inputs: test name, writable result storage, and plugin state. Output: `()` after populating results when inputs are valid.
///
/// # Safety
///
/// `user_data` must be null or a pointer previously returned by `InitializePlugin`. When non-null, `test_name` must be
/// a valid NUL-terminated C string. When `results` is non-null, it must point to writable storage for one
/// `dcgmDiagEntityResults_v2`; the storage may be uninitialized.
pub unsafe extern "C" fn RetrieveResults(
    test_name: *const c_char,
    results: *mut dcgmDiagEntityResults_v2,
    user_data: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        retrieve_results_impl(
            test_name,
            results.cast::<MaybeUninit<dcgmDiagEntityResults_v2>>(),
            user_data,
        );
    }));
}

#[unsafe(no_mangle)]
/// Destroys plugin-owned state returned by `InitializePlugin`.
///
/// Input: `user_data`, the plugin-state pointer. Output: a DCGM status describing whether shutdown succeeded.
///
/// # Safety
///
/// `user_data` must be null or a pointer returned by a successful `InitializePlugin` call. A non-null pointer must be
/// passed at most once.
pub unsafe extern "C" fn ShutdownPlugin(user_data: *mut c_void) -> dcgmReturn_t {
    catch_unwind(AssertUnwindSafe(|| shutdown_plugin_impl(user_data)))
        .unwrap_or(dcgmReturn_enum_DCGM_ST_PLUGIN_EXCEPTION)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use dcgm_utils::c_char_buf_to_string;

    #[test]
    /// Verifies that available SDC support is fully described in plugin metadata.
    /// Input: none. Output: `()` when metadata assertions pass.
    fn build_plugin_info_advertises_cpu_sdc_test_and_parameters() {
        let info = build_plugin_info_with_availability(true);

        assert_eq!(info.numTests, 1);
        assert_eq!(
            c_char_buf_to_string(&info.pluginName).expect("plugin name should decode"),
            CPU_SDC_PLUGIN_NAME
        );
        assert_eq!(
            c_char_buf_to_string(&info.description).expect("description should decode"),
            CPU_SDC_PLUGIN_DESCRIPTION
        );
        assert_eq!(
            c_char_buf_to_string(&info.tests[0].testName).expect("test name should decode"),
            CPU_SDC_PLUGIN_NAME
        );
        assert_eq!(
            c_char_buf_to_string(&info.tests[0].testCategory).expect("test category should decode"),
            CPU_SDC_PLUGIN_CATEGORY
        );
        assert_eq!(info.tests[0].targetEntityGroup, dcgm_field_entity_group_t_DCGM_FE_CPU);
        assert_eq!(info.tests[0].numValidParameters, 2);
        assert_eq!(
            c_char_buf_to_string(&info.tests[0].validParameters[0].parameterName)
                .expect("first parameter should decode"),
            CPU_SDC_PARAMETER_IS_ALLOWED
        );
        assert_eq!(
            c_char_buf_to_string(&info.tests[0].validParameters[1].parameterName)
                .expect("second parameter should decode"),
            CPU_SDC_PARAMETER_CONFIG_PATH
        );
    }

    #[test]
    /// Verifies that an unavailable SDC binary suppresses test advertisement.
    /// Input: none. Output: `()` when metadata assertions pass.
    fn build_plugin_info_does_not_advertise_cpu_sdc_test_when_binary_is_missing() {
        let info = build_plugin_info_with_availability(false);

        assert_eq!(info.numTests, 0);
        assert_eq!(
            c_char_buf_to_string(&info.pluginName).expect("plugin name should decode"),
            CPU_SDC_PLUGIN_NAME
        );
        assert_eq!(
            c_char_buf_to_string(&info.description).expect("description should decode"),
            CPU_SDC_PLUGIN_DESCRIPTION
        );
    }

    #[test]
    /// Verifies that plugin metadata can be written into uninitialized ABI storage.
    /// Input: none. Output: `()` when ABI storage assertions pass.
    fn get_plugin_info_initializes_uninitialized_output_storage() {
        let mut info = MaybeUninit::<dcgmDiagPluginInfo_t>::uninit();

        assert_eq!(
            get_plugin_info_impl(
                DCGM_DIAG_PLUGIN_INTERFACE_VERSION,
                info.as_mut_ptr().cast::<MaybeUninit<dcgmDiagPluginInfo_t>>()
            ),
            dcgmReturn_enum_DCGM_ST_OK
        );

        // SAFETY: `get_plugin_info_impl` returned success after writing a complete
        // `dcgmDiagPluginInfo_t` into `info`.
        let info = unsafe { info.assume_init() };
        assert_eq!(
            c_char_buf_to_string(&info.pluginName).expect("plugin name should decode"),
            CPU_SDC_PLUGIN_NAME
        );
    }

    #[test]
    /// Verifies that initialization writes the stat-field output before returning.
    /// Input: none. Output: `()` when initialization assertions pass.
    fn initialize_plugin_initializes_uninitialized_stat_field_id_storage() {
        let mut stat_field_ids = MaybeUninit::<dcgmDiagPluginStatFieldIds_t>::uninit();
        let mut user_data = core::ptr::null_mut::<c_void>();

        assert_eq!(
            initialize_plugin_impl(
                &raw mut stat_field_ids,
                &raw mut user_data,
                DcgmLoggingSeverity_t_DcgmLoggingSeverityNone,
                None,
            ),
            dcgmReturn_enum_DCGM_ST_OK
        );

        // SAFETY: `initialize_plugin_impl` returned success after writing a complete
        // `dcgmDiagPluginStatFieldIds_t` into `stat_field_ids`.
        let _stat_field_ids = unsafe { stat_field_ids.assume_init() };
        assert!(!user_data.is_null());
        assert_eq!(shutdown_plugin_impl(user_data), dcgmReturn_enum_DCGM_ST_OK);
    }

    #[test]
    /// Verifies that result retrieval initializes caller-provided output storage.
    /// Input: none. Output: `()` when result-storage assertions pass.
    fn retrieve_results_initializes_uninitialized_output_storage() {
        let plugin = Box::new(CpuSdcPlugin::new());
        let user_data = Box::into_raw(plugin).cast::<c_void>();
        let mut results = MaybeUninit::<dcgmDiagEntityResults_v2>::uninit();

        retrieve_results_impl(
            core::ptr::null(),
            results.as_mut_ptr().cast::<MaybeUninit<dcgmDiagEntityResults_v2>>(),
            user_data,
        );
        assert_eq!(shutdown_plugin_impl(user_data), dcgmReturn_enum_DCGM_ST_OK);

        // SAFETY: `retrieve_results_impl` writes a complete `dcgmDiagEntityResults_v2`
        // when passed a valid plugin and result pointer.
        let results = unsafe { results.assume_init() };
        assert_eq!(results.numResults, 0);
    }

    #[test]
    /// Verifies that plugin metadata retrieval rejects a null output pointer.
    /// Input: none. Output: `()` when status assertions pass.
    fn get_plugin_info_rejects_null_output_pointer() {
        assert_eq!(
            get_plugin_info_impl(DCGM_DIAG_PLUGIN_INTERFACE_VERSION, core::ptr::null_mut()),
            dcgmReturn_enum_DCGM_ST_BADPARAM
        );
    }
}
