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

//! Regression test for the central premise of [`dcgm_logging`]:
//! it must not route records through the `::log` crate's global logger,
//! so registering a third-party `::log` global logger does not steal records
//! emitted via `dcgm_logging::info!` (or its siblings).
#![allow(clippy::unwrap_used, reason = "this is an integration test, not a production code")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use dcgm_logging as module_log;

/// Serializes the tests in this binary; both touch the per-cdylib
/// `module_log` static singleton, and `cargo test` runs tests in parallel by
/// default.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lock_test_globals() -> MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Captures every record routed through the `::log` crate's global logger
/// into [`SEEN`] so the test can assert isolation.
struct GlobalCapturingLogger;

static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
static GLOBAL_CAPTURE: GlobalCapturingLogger = GlobalCapturingLogger;
static SIDE_EFFECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl log::Log for GlobalCapturingLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

/// Module-local capture sink.
struct ModuleCapture {
    count: AtomicUsize,
    last: Mutex<Option<String>>,
}

impl module_log::Log for ModuleCapture {
    fn enabled(&self, _: &module_log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &module_log::Record) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .replace(format!("{}", record.args()));
    }
    fn flush(&self) {}
}

fn evaluate_side_effect() -> &'static str {
    SIDE_EFFECT_COUNTER.fetch_add(1, Ordering::Relaxed);
    "evaluated"
}

#[test]
fn module_log_does_not_route_through_global_log_logger() {
    let _guard = lock_test_globals();

    // Step 1: install a real `::log` global logger and bump its max level so
    // it would happily accept any record routed through the `log` crate's
    // global state. set_logger may have been called by another test in this
    // binary; tolerate the Err and proceed.
    let _ = log::set_logger(&GLOBAL_CAPTURE);
    log::set_max_level(log::LevelFilter::Trace);

    // Step 2: configure module_log with its own capturing sink.
    let module_sink = std::sync::Arc::new(ModuleCapture {
        count: AtomicUsize::new(0),
        last: Mutex::new(None),
    });
    module_log::set_max_level(module_log::LevelFilter::Trace);
    module_log::set_sinks(vec![module_sink.clone() as std::sync::Arc<dyn module_log::Log>]);
    SEEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();

    // Step 3: emit through module_log::info!.
    module_log::info!("module-log-only");

    // Assertions:
    assert_eq!(
        module_sink.count.load(Ordering::Relaxed),
        1,
        "module_log sink should have received the record"
    );
    let module_last = module_sink
        .last
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(module_last, Some("module-log-only".to_owned()));

    let global_seen = SEEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    assert!(
        global_seen.is_empty(),
        "::log global logger MUST NOT see records emitted via module_log; saw: {global_seen:?}"
    );
}

#[test]
fn disabled_module_log_arm_does_not_evaluate_args() {
    let _guard = lock_test_globals();

    // set_logger may already have been called; tolerate the `Err`.
    let _ = log::set_logger(&GLOBAL_CAPTURE);

    SIDE_EFFECT_COUNTER.store(0, Ordering::Relaxed);
    module_log::set_max_level(module_log::LevelFilter::Off);
    module_log::clear_sinks();

    module_log::info!("{}", evaluate_side_effect());

    assert_eq!(
        SIDE_EFFECT_COUNTER.load(Ordering::Relaxed),
        0,
        "format args must not be evaluated when the module_log macro arm is disabled"
    );
}
