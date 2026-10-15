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

#![allow(clippy::unwrap_used, reason = "testing code")]

mod shared_types;

use dcgm_logging::{Level, LevelFilter, Log, Metadata, Record, add_sink, error, info, set_max_level, trace};
use libc::{SYS_gettid, syscall};
use shared_types::Record as FfiRecord;
use shared_types::hostEngineAppenderCallbackFp_t;
use std::ffi::CString;
use std::ptr;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

struct CallbackSink {
    callback: hostEngineAppenderCallbackFp_t,
    log_level: Level,
}

impl Log for CallbackSink {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.log_level
    }

    fn log(&self, record: &Record) {
        if let Some(log_callback) = self.callback {
            let msg = CString::new(format!("{}", record.args())).unwrap();
            let func = CString::new(record.module_path().unwrap_or_else(|| record.target())).unwrap();
            let file = CString::new(record.file().unwrap_or("<unknown>")).unwrap();
            // SAFETY: no safety concerns in calling gettid
            let thread_id = u32::try_from(unsafe { syscall(SYS_gettid) }.cast_unsigned()).unwrap_or_default();
            let rec = FfiRecord {
                message: msg.as_ptr(),
                func: func.as_ptr(),
                file: file.as_ptr(),
                object: ptr::null(),
                line: record.line().unwrap_or_default() as _,
                tid: thread_id,
                severity: u32::try_from(record.metadata().level() as usize).unwrap(),
                time: usize::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()).unwrap(),
            };
            // SAFETY: the function pointer is not null. The record is not required to live beyond the function call,
            // so we can safely drop it at the exit, and no local stack pointers will be preserved.
            unsafe {
                log_callback(&raw const rec);
            }
        }
    }

    fn flush(&self) {}
}

static UNIQ: OnceLock<String> = OnceLock::new();

fn generate_id() -> String {
    use rand::distr::{Alphanumeric, SampleString};
    Alphanumeric.sample_string(&mut rand::rng(), 5)
}

#[unsafe(no_mangle)]
pub extern "C" fn emit_log_events() {
    info!("{:?}: emit_log_events", UNIQ.get());
    error!("{:?}: some critical stuff", UNIQ.get());
    trace!("{:?}: tracing is also interesting", UNIQ.get());

    log::error!("{:?}: Make sure this does not do anything", UNIQ.get());
}

#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub extern "C" fn InitializePlugin(logging_callback: hostEngineAppenderCallbackFp_t) -> i32 {
    // We use add_sink here instead of set_sinks to verify two modules with different provided callbacks will not affect
    // each other's loggers. The set_sinks overwrites previously registered sinks and may hide the problem.
    add_sink(
        &(Arc::new(CallbackSink {
            callback: logging_callback,
            log_level: Level::Trace,
        }) as Arc<dyn Log>),
    );
    set_max_level(LevelFilter::Trace);

    UNIQ.get_or_init(generate_id);

    0
}
