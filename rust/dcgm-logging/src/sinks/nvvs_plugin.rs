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

//! NVVS plugin sink that forwards [`Record`]s to the host-engine appender callback.

use std::ffi::{CString, c_char, c_long, c_void};
use std::time::UNIX_EPOCH;

use crate::{Level, LevelFilter, Log, Metadata, Record as LogRecord};

/// `DcgmLoggingSeverity_t` values from `dcgmlib/dcgm_structs.h`.
#[allow(dead_code, reason = "public FFI constants for NVVS plugin consumers")]
pub mod severity {
    /// No logging.
    pub const NONE: i32 = 0;
    /// Fatal errors.
    pub const FATAL: i32 = 1;
    /// Errors.
    pub const ERROR: i32 = 2;
    /// Warnings.
    pub const WARNING: i32 = 3;
    /// Informative.
    pub const INFO: i32 = 4;
    /// Debug information.
    pub const DEBUG: i32 = 5;
    /// Verbose debugging information.
    pub const VERBOSE: i32 = 6;
}

/// Timestamp portion of a host-engine log record (`DcgmNs::Logging::RecordTime`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecordTime {
    /// Seconds since the Unix epoch.
    pub time: i64,
    /// Milliseconds within the second.
    pub millitm: u16,
    /// Padding; not used by the host engine.
    _pad: [u8; 6],
}

/// Host-engine log record layout (`DcgmNs::Logging::Record`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Record {
    /// Logging message.
    pub message: *const c_char,
    /// Function or module name.
    pub func: *const c_char,
    /// Source file name.
    pub file: *const c_char,
    /// Associated object; usually null for NVVS plugins.
    pub object: *const c_void,
    /// Source line number.
    pub line: usize,
    /// Thread id.
    pub tid: u32,
    /// Record severity (`DcgmLoggingSeverity_t`).
    pub severity: i32,
    /// Record timestamp.
    pub time: RecordTime,
}

/// Host-engine appender callback type (`hostEngineAppenderCallbackFp_t`).
pub type LoggerCallbackFp = unsafe extern "C" fn(*const Record);

/// Optional host-engine appender callback supplied by NVVS at plugin init.
pub type HostEngineAppenderCallbackFp = Option<LoggerCallbackFp>;

/// Forwards [`LogRecord`]s to NVVS via the host-supplied callback.
///
/// The host (verified in `common/DcgmLogging.cpp` `HostengineAppender::write`
/// and `appendRecordToLogger`) consumes the [`Record`] synchronously, so the
/// borrowed [`CString`] buffers used to populate the C record only need to live
/// for the duration of the callback invocation.
pub struct NvvsPluginSink {
    callback: HostEngineAppenderCallbackFp,
    max_level: LevelFilter,
}

impl NvvsPluginSink {
    /// Builds a sink that forwards records to `callback` (when `Some`) and
    /// drops records strictly more verbose than `max_level`.
    #[must_use]
    pub const fn new(callback: HostEngineAppenderCallbackFp, max_level: LevelFilter) -> Self {
        Self { callback, max_level }
    }
}

const fn level_to_severity(level: Level) -> i32 {
    match level {
        Level::Error => severity::ERROR,
        Level::Warn => severity::WARNING,
        Level::Info => severity::INFO,
        Level::Debug => severity::DEBUG,
        Level::Trace => severity::VERBOSE,
    }
}

fn cstring_from_record_field(value: &str) -> CString {
    let prefix = value.split_once('\0').map_or(value, |(prefix, _)| prefix);
    CString::new(prefix).unwrap_or_default()
}

impl Log for NvvsPluginSink {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &LogRecord) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let Some(callback) = self.callback else {
            return;
        };

        let file = cstring_from_record_field(record.file().unwrap_or("<unknown>"));
        let message = cstring_from_record_field(&format!("{}", record.args()));
        let module = CString::new(record.target()).unwrap_or_default();
        let line = record.line().unwrap_or(0) as usize;

        // SAFETY: `syscall(SYS_gettid)` takes no inputs and returns a small integer.
        let tid: c_long = unsafe { nix::libc::syscall(nix::libc::SYS_gettid) };
        let tid = u32::try_from(tid.cast_unsigned()).unwrap_or_default();

        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = i64::try_from(now.as_secs()).unwrap_or(i64::MAX);
        let millis = u16::try_from(now.subsec_millis()).unwrap_or_default();

        let ffi_record = Record {
            message: message.as_ptr(),
            func: module.as_ptr(),
            file: file.as_ptr(),
            object: core::ptr::null(),
            line,
            tid,
            severity: level_to_severity(record.level()),
            time: RecordTime {
                time: secs,
                millitm: millis,
                _pad: [0; 6],
            },
        };

        // SAFETY: NVVS' `HostengineAppender::write` / `appendRecordToLogger` consume the record
        // synchronously, so the borrowed `CString` buffers remain valid until the callback returns.
        unsafe {
            callback(&raw const ffi_record);
        }
    }

    fn flush(&self) {}
}

#[cfg(all(test, not(miri)))]
#[allow(clippy::unwrap_used)]
mod tests {
    use core::ffi::c_uint;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use serial_test::serial;
    use std::ffi::CStr;
    use std::sync::Mutex;

    use super::{HostEngineAppenderCallbackFp, NvvsPluginSink, Record, severity};
    use crate::{Level, LevelFilter, Log, Record as LogRecord};

    static CAPTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static LAST_SEVERITY: AtomicUsize = AtomicUsize::new(usize::MAX);

    extern "C" fn capture_callback(record_ptr: *const Record) {
        // SAFETY: tests always pass a valid stack-allocated `Record`.
        let record = unsafe { &*record_ptr };
        CAPTURE_COUNT.fetch_add(1, Ordering::Release);
        #[allow(clippy::cast_sign_loss)]
        LAST_SEVERITY.store(record.severity as usize, Ordering::Release);
    }

    #[test]
    #[serial(capture_count)]
    fn nvvs_plugin_sink_forwards_record() {
        let cb: HostEngineAppenderCallbackFp = Some(capture_callback);
        let sink = NvvsPluginSink::new(cb, LevelFilter::Trace);
        CAPTURE_COUNT.store(0, Ordering::Release);

        sink.log(
            &LogRecord::builder()
                .args(format_args!("hello dcgm"))
                .level(Level::Info)
                .target("nic_eud::test")
                .module_path(Some("nic_eud::test"))
                .file(Some("test.rs"))
                .line(Some(42))
                .build(),
        );

        assert_eq!(CAPTURE_COUNT.load(Ordering::Acquire), 1);
        assert_eq!(LAST_SEVERITY.load(Ordering::Acquire), severity::INFO as c_uint as usize);
    }

    #[test]
    #[serial(capture_count)]
    fn nvvs_plugin_sink_below_filter_is_dropped() {
        let cb: HostEngineAppenderCallbackFp = Some(capture_callback);
        let sink = NvvsPluginSink::new(cb, LevelFilter::Error);
        CAPTURE_COUNT.store(0, Ordering::Release);

        sink.log(
            &LogRecord::builder()
                .args(format_args!("noisy debug"))
                .level(Level::Debug)
                .target("nic_eud::test")
                .build(),
        );

        assert_eq!(CAPTURE_COUNT.load(Ordering::Acquire), 0);
    }

    static LAST_MESSAGE: Mutex<Option<Vec<u8>>> = Mutex::new(None);

    extern "C" fn capture_message_callback(record_ptr: *const Record) {
        // SAFETY: tests always pass a valid stack-allocated `Record`.
        let record = unsafe { &*record_ptr };
        // SAFETY: `NvvsPluginSink::log` writes a NUL-terminated C string into `message` and the
        // pointer remains valid for the synchronous callback call.
        let bytes = unsafe { CStr::from_ptr(record.message) }.to_bytes().to_vec();
        let _ = LAST_MESSAGE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .replace(bytes);
    }

    #[test]
    fn nvvs_plugin_sink_handles_interior_nul() {
        let cb: HostEngineAppenderCallbackFp = Some(capture_message_callback);
        let sink = NvvsPluginSink::new(cb, LevelFilter::Trace);
        let _ = LAST_MESSAGE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();

        sink.log(
            &LogRecord::builder()
                .args(format_args!("nul-\0-inside"))
                .level(Level::Info)
                .target("nic_eud::test")
                .module_path(Some("nic_eud::test"))
                .file(Some("test.rs"))
                .line(Some(7))
                .build(),
        );

        let captured = LAST_MESSAGE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap();
        assert_eq!(captured, b"nul-");
    }
}
