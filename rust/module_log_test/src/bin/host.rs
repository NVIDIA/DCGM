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

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::panic,
    reason = "integration test harness"
)]
#[path = "../shared_types.rs"]
mod shared_types;

use clap::Parser;
use clap::builder::TypedValueParser;
use libc::{LM_ID_NEWLM, RTLD_LAZY, RTLD_LOCAL, dlclose, dlerror, dlmopen, dlsym};
use log::debug;
use shared_types::Record;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

type InitFn = unsafe extern "C" fn(logging_callback: shared_types::hostEngineAppenderCallbackFp_t) -> i32;
type EmitFn = unsafe extern "C" fn();

const EXPECTED_RECORDS_PER_PLUGIN: usize = 3;

#[derive(Parser)]
#[command(
    name = "module_log_test_host",
    about = "Verifies per-cdylib module logging isolation"
)]
struct Args {
    /// Path to the plugin shared library (`libplugin_log_test.so`).
    #[arg(
        long = "plugin-path",
        required = true,
        value_name = "PATH",
        value_parser = clap::builder::PathBufValueParser::new().try_map(existing_plugin_path),
    )]
    plugin_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_path_argument_is_required() {
        let Err(err) = Args::try_parse_from(["module_log_test_host"]) else {
            panic!("plugin path argument should be required");
        };

        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }
}

fn existing_plugin_path(path: PathBuf) -> Result<PathBuf, String> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!("{} does not exist or is not a file", path.display()))
    }
}

fn plugin_path_to_cstring(path: &Path) -> Result<CString, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| "Plugin path is not valid UTF-8".to_owned())?;
    CString::new(path_str).map_err(|_| "Plugin path contains an interior nul byte".to_owned())
}

/// `log::Level` discriminants used by the test plugin sink (`level as usize`).
const SEVERITY_ERROR: u32 = 1;
const SEVERITY_INFO: u32 = 3;
const SEVERITY_TRACE: u32 = 5;

#[derive(Clone)]
struct CapturedRecord {
    message: String,
    severity: u32,
}

struct CallbackState {
    count: AtomicUsize,
    records: Mutex<Vec<CapturedRecord>>,
}

impl CallbackState {
    const fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            records: Mutex::new(Vec::new()),
        }
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Release);
        self.records.lock().unwrap().clear();
    }

    fn push_record(&self, record: &Record) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let message = unsafe { CStr::from_ptr(record.message) }
            .to_str()
            .unwrap_or("<invalid utf-8>")
            .to_owned();
        self.records.lock().unwrap().push(CapturedRecord {
            message,
            severity: record.severity,
        });
    }

    fn snapshot(&self) -> (usize, Vec<CapturedRecord>) {
        let records = self.records.lock().unwrap().clone();
        (self.count.load(Ordering::Acquire), records)
    }
}

static CALLBACK_1: CallbackState = CallbackState::new();
static CALLBACK_2: CallbackState = CallbackState::new();

unsafe extern "C" fn callback_1(record: *const Record) {
    use std::panic::catch_unwind;
    catch_unwind(|| {
        if record.is_null() {
            eprintln!("callback_1: null record");
        } else {
            CALLBACK_1.push_record(unsafe { &*record });
        }
    })
    .unwrap_or_else(|e| {
        eprintln!("callback_1: panic: {e:?}");
    });
}

unsafe extern "C" fn callback_2(record: *const Record) {
    use std::panic::catch_unwind;
    catch_unwind(|| {
        if record.is_null() {
            eprintln!("callback_2: null record");
        } else {
            CALLBACK_2.push_record(unsafe { &*record });
        }
    })
    .unwrap_or_else(|e| {
        eprintln!("callback_2: panic: {e:?}");
    });
}

struct Plugin {
    lib: *mut c_void,
    callback: shared_types::hostEngineAppenderCallbackFp_t,
}

impl Drop for Plugin {
    fn drop(&mut self) {
        if !self.lib.is_null() {
            unsafe { dlclose(self.lib) };
            self.lib = ptr::null_mut();
        }
    }
}

fn dlerror_message() -> String {
    // SAFETY: `dlerror()` returns a pointer to a static error string or null.
    let err = unsafe { dlerror() };
    if err.is_null() {
        "unknown".to_owned()
    } else {
        // SAFETY: non-null `dlerror()` pointers refer to a nul-terminated C string.
        unsafe { CStr::from_ptr(err) }
            .to_str()
            .unwrap_or("<invalid dlerror utf-8>")
            .to_owned()
    }
}

fn load_plugin(lib_path: &CStr) -> Result<*mut c_void, String> {
    // SAFETY: `dlmopen` loads a shared object in a new link-map namespace.
    let ptr = unsafe { dlmopen(LM_ID_NEWLM, lib_path.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
    if ptr.is_null() {
        return Err(format!(
            "Unable to load {}: {}",
            lib_path.to_str().unwrap_or("<non-utf8 path>"),
            dlerror_message()
        ));
    }
    Ok(ptr)
}

fn resolve_symbol<T>(lib: *mut c_void, name: &CStr) -> Result<T, String> {
    // SAFETY: `dlsym` looks up an exported symbol in a loaded shared object.
    let sym = unsafe { dlsym(lib, name.as_ptr()) };
    if sym.is_null() {
        return Err(format!(
            "Unable to find symbol {}: {}",
            name.to_str().unwrap_or("<non-utf8 name>"),
            dlerror_message()
        ));
    }
    // SAFETY: caller verifies the symbol type matches the exported function.
    Ok(unsafe { std::mem::transmute_copy(&sym) })
}

fn extract_instance_token(message: &str) -> Option<String> {
    let (token, _rest) = message.split_once(':')?;
    Some(token.trim().trim_matches('"').to_owned())
}

fn verify_callback_records(callback_name: &str, count: usize, records: &[CapturedRecord]) -> Result<String, String> {
    if count != EXPECTED_RECORDS_PER_PLUGIN {
        return Err(format!(
            "{callback_name}: expected {EXPECTED_RECORDS_PER_PLUGIN} callbacks, got {count}"
        ));
    }
    if records.len() != EXPECTED_RECORDS_PER_PLUGIN {
        return Err(format!(
            "{callback_name}: expected {EXPECTED_RECORDS_PER_PLUGIN} captured records, got {}",
            records.len()
        ));
    }

    let mut severities: Vec<u32> = records.iter().map(|record| record.severity).collect();
    severities.sort_unstable();
    let mut expected = [SEVERITY_ERROR, SEVERITY_INFO, SEVERITY_TRACE];
    expected.sort_unstable();
    if severities != expected {
        return Err(format!(
            "{callback_name}: unexpected severities {severities:?}, expected {expected:?}"
        ));
    }

    let tokens: HashSet<String> = records
        .iter()
        .filter_map(|record| extract_instance_token(&record.message))
        .collect();
    if tokens.len() != 1 {
        return Err(format!(
            "{callback_name}: expected one instance id across records, got {tokens:?}"
        ));
    }

    Ok(tokens.into_iter().next().unwrap())
}

fn verify_cross_callback_isolation(
    id_a: &str,
    id_b: &str,
    records_a: &[CapturedRecord],
    records_b: &[CapturedRecord],
) -> Result<(), String> {
    if id_a == id_b {
        return Err(format!(
            "Plugin instances must have distinct ids, both reported {id_a:?}"
        ));
    }

    for record in records_a {
        if record.message.contains(id_b) {
            return Err(format!(
                "Callback 1 received a record belonging to instance {id_b:?}: {}",
                record.message
            ));
        }
    }

    for record in records_b {
        if record.message.contains(id_a) {
            return Err(format!(
                "Callback 2 received a record belonging to instance {id_a:?}: {}",
                record.message
            ));
        }
    }

    Ok(())
}

fn run() -> Result<(), String> {
    static LOGGER: OnceLock<()> = OnceLock::new();
    LOGGER.get_or_init(|| {
        simple_logger::init_with_level(log::Level::Trace).unwrap();
    });

    let args = Args::parse();
    let lib_path = plugin_path_to_cstring(&args.plugin_path)?;
    debug!("Using plugin path: {}", args.plugin_path.display());

    let plugins = [
        Plugin {
            lib: load_plugin(&lib_path)?,
            callback: Some(callback_1),
        },
        Plugin {
            lib: load_plugin(&lib_path)?,
            callback: Some(callback_2),
        },
        Plugin {
            lib: load_plugin(&lib_path)?,
            callback: None,
        },
    ];

    CALLBACK_1.reset();
    CALLBACK_2.reset();

    for (idx, plugin) in plugins.iter().enumerate() {
        let plugin_init_fp: InitFn = resolve_symbol(plugin.lib, c"InitializePlugin")?;
        let emit_fp: EmitFn = resolve_symbol(plugin.lib, c"emit_log_events")?;

        let (count_1_before_init, _) = CALLBACK_1.snapshot();
        let (count_2_before_init, _) = CALLBACK_2.snapshot();

        let init_status = unsafe { plugin_init_fp(plugin.callback) };
        if init_status != 0 {
            return Err(format!("InitializePlugin for plugin {idx} returned {init_status}"));
        }

        let (count_1_after_init, _) = CALLBACK_1.snapshot();
        let (count_2_after_init, _) = CALLBACK_2.snapshot();
        if count_1_after_init != count_1_before_init || count_2_after_init != count_2_before_init {
            return Err(format!(
                "InitializePlugin for plugin {idx} must not emit records \
                 (CALL1 {count_1_before_init}->{count_1_after_init}, \
                 CALL2 {count_2_before_init}->{count_2_after_init})"
            ));
        }

        unsafe {
            emit_fp();
        }

        debug!(
            "Plugin {idx} emit complete. CALL1={}, CALL2={}",
            CALLBACK_1.count.load(Ordering::Acquire),
            CALLBACK_2.count.load(Ordering::Acquire)
        );
    }

    let (count_1, records_1) = CALLBACK_1.snapshot();
    let (count_2, records_2) = CALLBACK_2.snapshot();
    let id_1 = verify_callback_records("callback_1", count_1, &records_1)?;
    let id_2 = verify_callback_records("callback_2", count_2, &records_2)?;
    verify_cross_callback_isolation(&id_1, &id_2, &records_1, &records_2)?;

    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {
            println!("PASS: module log isolation");
        }
        Err(err) => {
            eprintln!("FAIL: {err}");
            std::process::exit(1);
        }
    }
}
