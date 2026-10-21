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

//! Per-shared-library logger that fans out into multiple [`log::Log`] sinks without using [`log::set_logger`].
//!
//! Use this crate when a Rust shared library needs its own logging infrastructure that
//! does not collide with the global `log` crate state.
//!
//! The standard `log` crate types ([`Level`], [`LevelFilter`], [`Log`], [`Metadata`], [`Record`])
//! are reused throughout, and re-exported from this crate so consumers can write
//! `pub use dcgm_logging as log;` and get familiar logging API.
//!
//! # Overview
//!
//! - [`SinkingLog`] holds an [`arc_swap::ArcSwapOption`] of an immutable
//!   `Arc<Vec<Arc<dyn log::Log>>>` plus a single [`AtomicUsize`] storing the
//!   early-out [`LevelFilter`] used by the [`info!`], [`warn!`], etc. macros'
//!   fast path.
//! - Sinks implement `log::Log` trait that allows using 3rd party loggers unchanged.
//! - [`SinkingLog`] itself implements [`Log`], so it can be used anywhere a
//!   `Log` is expected.
//! - A `static LOGGER: SinkingLog` is the per-cdylib singleton driven by the
//!   macros. The free functions [`set_sinks`], [`add_sink`], [`clear_sinks`],
//!   [`set_max_level`], and [`max_level`] are convenience wrappers that
//!   delegate to that static.
//!
//! # Threading and reentrancy
//!
//! - The hot-disabled path is a single relaxed atomic load + numeric compare
//!   in the macro arm; format arguments are not evaluated.
//! - The hot enabled path performs one wait-free [`ArcSwapOption::load`] to
//!   obtain a snapshot of the sink list, then iterates. There is no per-sink
//!   atomic increment and no heap allocation per record; writers swap in a
//!   fresh `Arc<Vec<…>>` via copy-on-write rather than mutating the live one.
//!   Sinks may safely re-enter the logger or call [`set_sinks`] /
//!   [`clear_sinks`] from inside their own [`Log::log`]; the held snapshot is
//!   immutable and writers are lock-free, so re-entry is race-free and
//!   deadlock-free.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use ::log::{Level, LevelFilter, Log, Metadata, MetadataBuilder, Record, RecordBuilder};
use arc_swap::ArcSwapOption;

pub mod sinks;

const fn usize_to_level_filter(value: usize) -> LevelFilter {
    match value {
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        5 => LevelFilter::Trace,
        _ => LevelFilter::Off,
    }
}

/// Multi-sink logger that augments [`log::Log`] with sink management.
///
/// `SinkingLog` is the building block of this crate. Its [`Log`]
/// impl fans out incoming records to every registered sink whose own
/// [`Log::enabled`] accepts the record. The early-out [`LevelFilter`] is
/// stored in an atomic so the macro fast path does not need to acquire any
/// lock when logging is disabled.
///
/// The sink list lives behind an [`ArcSwapOption`] so the hot read path is
/// wait-free and does *not* clone individual sink `Arc`s per record. Writers
/// swap in a fresh `Arc<Vec<…>>` via copy-on-write.
///
/// All public methods are safe to call from multiple threads. Construction
/// is `const`, so a `SinkingLog` may live in a `static`.
pub struct SinkingLog {
    sinks: ArcSwapOption<Vec<Arc<dyn Log>>>,
    max_level: AtomicUsize,
}

impl SinkingLog {
    /// Constructs an empty logger with [`LevelFilter::Off`]
    ///
    /// No sinks are installed.
    /// Calling [`Log::log`] before [`set_sinks`] / [`add_sink`] is a no-op.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sinks: ArcSwapOption::const_empty(),
            max_level: AtomicUsize::new(LevelFilter::Off as _),
        }
    }

    /// Replaces the registered sinks with the supplied vector.
    pub fn set_sinks(&self, sinks: Vec<Arc<dyn Log>>) {
        self.sinks.store(Some(Arc::new(sinks)));
    }

    /// Appends `sink` to the registered set without disturbing existing sinks.
    pub fn add_sink(&self, sink: &Arc<dyn Log>) {
        // `rcu` may run its closure more than once under writer contention, so we must not
        // consume `sink` inside the closure; we clone the outer Arc on each retry instead.
        self.sinks.rcu(|current| {
            let mut next = current.as_deref().cloned().unwrap_or_default();
            next.push(sink.clone());
            Some(Arc::new(next))
        });
    }

    /// Removes all registered sinks.
    pub fn clear_sinks(&self) {
        self.sinks.store(None);
    }

    /// Sets the early-out [`LevelFilter`] used by the macro fast path and by [`Log::enabled`].
    ///
    /// Records whose level is more verbose than `level` are dropped before allocation or formatting work happens.
    /// Per-sink filters are still applied via each sink's own [`Log::enabled`] after this check.
    pub fn set_max_level(&self, level: LevelFilter) {
        self.max_level.store(level as _, Ordering::Relaxed);
    }

    /// Returns the current early-out [`LevelFilter`].
    #[must_use]
    pub fn max_level(&self) -> LevelFilter {
        usize_to_level_filter(self.max_level.load(Ordering::Relaxed))
    }
}

impl Default for SinkingLog {
    fn default() -> Self {
        Self::new()
    }
}

impl Log for SinkingLog {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level()
    }

    fn log(&self, record: &Record) {
        // The `log` crate contract allows callers to invoke `log` without a
        // prior `enabled` call, so re-check here for correctness.
        if !self.enabled(record.metadata()) {
            return;
        }

        // `load()` is wait-free and the returned guard pins an immutable
        // snapshot of the sink list. Writers (`set_sinks` / `add_sink` /
        // `clear_sinks`) operate by swapping in a fresh `Arc<Vec<…>>`, so a
        // sink that reconfigures the logger from inside `log` does not race
        // with this iteration.
        let snapshot = self.sinks.load();
        if let Some(sinks) = &*snapshot {
            for sink in sinks.iter() {
                if sink.enabled(record.metadata()) {
                    sink.log(record);
                }
            }
        }
    }

    fn flush(&self) {
        let snapshot = self.sinks.load();
        if let Some(sinks) = &*snapshot {
            for sink in sinks.iter() {
                sink.flush();
            }
        }
    }
}

/// Per-cdylib singleton driven by the [`info!`], [`warn!`], [`error!`], [`debug!`], [`trace!`] macros.
///
/// Because rlibs are statically linked into each `cdylib`, this static is
/// private to the loaded shared library; two shared libraries that both link
/// `dcgm_logging` get independent loggers.
static LOGGER: SinkingLog = SinkingLog::new();

/// Returns a reference to the per-cdylib [`SinkingLog`].
///
/// Useful when a consumer wants to install [`SinkingLog`] itself as the
/// `::log` global via `log::set_logger(dcgm_logging::logger())`. The default
/// configuration (no sinks installed, [`LevelFilter::Off`]) silently drops
/// every record until [`set_sinks`] and [`set_max_level`] have run.
#[must_use]
pub fn logger() -> &'static SinkingLog {
    &LOGGER
}

/// Replaces the static logger's sinks. See [`SinkingLog::set_sinks`].
pub fn set_sinks(sinks: Vec<Arc<dyn Log>>) {
    LOGGER.set_sinks(sinks);
}

/// Appends a sink to the static logger. See [`SinkingLog::add_sink`].
pub fn add_sink(sink: &Arc<dyn Log>) {
    LOGGER.add_sink(sink);
}

/// Removes all sinks from the static logger. See [`SinkingLog::clear_sinks`].
pub fn clear_sinks() {
    LOGGER.clear_sinks();
}

/// Sets the static logger's early-out level. See [`SinkingLog::set_max_level`].
pub fn set_max_level(level: LevelFilter) {
    LOGGER.set_max_level(level);
}

/// Returns the static logger's early-out level. See [`SinkingLog::max_level`].
#[must_use]
pub fn max_level() -> LevelFilter {
    LOGGER.max_level()
}

/// Internal entry point used by the `__module_log_*` macros.
///
/// Builds a [`Record`] from the supplied metadata and dispatches it through
/// the static [`SinkingLog`]. This is `pub` only because the macros need to
/// reach it from any consumer crate; user code should call the macros
/// instead.
#[doc(hidden)]
pub fn __log(
    level: Level,
    target: &str,
    module: &'static str,
    file: &'static str,
    line: u32,
    args: core::fmt::Arguments<'_>,
) {
    let record = Record::builder()
        .args(args)
        .level(level)
        .target(target)
        .module_path(Some(module))
        .file(Some(file))
        .line(Some(line))
        .build();
    LOGGER.log(&record);
}

#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_impl {
    (@internal target: $target:expr, lvl: $lvl:expr, $($arg:tt)+) => {{
        if $lvl <= $crate::max_level() {
            $crate::__log(
                $lvl,
                $target,
                ::core::module_path!(),
                ::core::file!(),
                ::core::line!(),
                ::core::format_args!($($arg)+),
            );
        }
    }};
}

/// `error!`-level macro for this crate.
///
/// See module-level docs for the supported grammar (basic format and
/// `target:` arms). Format arguments are not evaluated when
/// [`max_level`] is below [`Level::Error`].
#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_error {
    (target: $target:expr, $($arg:tt)+) => {
        $crate::__module_log_impl!(@internal target: $target, lvl: $crate::Level::Error, $($arg)+);
    };
    ($($arg:tt)+) => {
        $crate::__module_log_impl!(@internal target: ::core::module_path!(), lvl: $crate::Level::Error, $($arg)+);
    };
}

/// `warn!`-level macro for this crate.
#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_warn {
    (target: $target:expr, $($arg:tt)+) => {
        $crate::__module_log_impl!(@internal target: $target, lvl: $crate::Level::Warn, $($arg)+);
    };
    ($($arg:tt)+) => {
        $crate::__module_log_impl!(@internal target: ::core::module_path!(), lvl: $crate::Level::Warn, $($arg)+);
    };
}

/// `info!`-level macro for this crate.
#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_info {
    (target: $target:expr, $($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: $target,
            lvl: $crate::Level::Info,
            $($arg)+);
    };
    ($($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: ::core::module_path!(),
            lvl: $crate::Level::Info,
            $($arg)+);
    };
}

/// `debug!`-level macro for this crate.
#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_debug {
    (target: $target:expr, $($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: $target,
            lvl: $crate::Level::Debug,
            $($arg)+);
    };
    ($($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: ::core::module_path!(),
            lvl: $crate::Level::Debug,
            $($arg)+);
    };
}

/// `trace!`-level macro for this crate.
#[doc(hidden)]
#[macro_export]
macro_rules! __module_log_trace {
    (target: $target:expr, $($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: $target,
            lvl: $crate::Level::Trace,
            $($arg)+);
    };
    ($($arg:tt)+) => {
        $crate::__module_log_impl!(
            @internal
            target: ::core::module_path!(),
            lvl: $crate::Level::Trace,
            $($arg)+);
    };
}

/// Logs a record at [`Level::Error`].
///
/// Mirrors the basic-format and `target:` arms of `log::error!`. Format
/// arguments are not evaluated when [`max_level`] is below [`Level::Error`].
pub use crate::__module_log_error as error;

/// Logs a record at [`Level::Warn`]. See [`error!`] for grammar.
pub use crate::__module_log_warn as warn;

/// Logs a record at [`Level::Info`]. See [`error!`] for grammar.
pub use crate::__module_log_info as info;

/// Logs a record at [`Level::Debug`]. See [`error!`] for grammar.
pub use crate::__module_log_debug as debug;

/// Logs a record at [`Level::Trace`]. See [`error!`] for grammar.
pub use crate::__module_log_trace as trace;

#[cfg(all(test, not(miri)))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{
        Level, LevelFilter, Log, Metadata, Record, SinkingLog, clear_sinks, max_level, set_max_level, set_sinks,
    };
    use serial_test::serial;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard, Weak};

    /// Builds a [`Record`] and dispatches it through `$sinking.log(&record)` in a single expression.
    /// We must not stash the `Record` in a `let` binding because [`format_args!`] returns a temporary whose backing
    /// storage is dropped at the end of the let statement, leaving the `Arguments<'_>` inside the `Record` dangling.
    macro_rules! emit {
        ($sinking:expr, $level:expr, $target:expr, $($arg:tt)+) => {
            $sinking.log(
                &Record::builder()
                    .args(format_args!($($arg)+))
                    .level($level)
                    .target($target)
                    .build(),
            )
        };
    }

    /// Counts every record forwarded to it, captures the most recent message
    /// and target. Per-sink filtering is configurable.
    pub struct CapturingSink {
        pub count: AtomicUsize,
        pub last_message: Mutex<Option<String>>,
        pub last_target: Mutex<Option<String>>,
        pub max_level: LevelFilter,
    }

    impl CapturingSink {
        pub const fn new(max_level: LevelFilter) -> Self {
            Self {
                count: AtomicUsize::new(0),
                last_message: Mutex::new(None),
                last_target: Mutex::new(None),
                max_level,
            }
        }
    }

    impl Log for CapturingSink {
        fn enabled(&self, metadata: &Metadata) -> bool {
            metadata.level() <= self.max_level
        }
        fn log(&self, record: &Record) {
            if !self.enabled(record.metadata()) {
                return;
            }
            self.count.fetch_add(1, Ordering::Relaxed);
            let _ = self
                .last_message
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .replace(format!("{}", record.args()));
            let _ = self
                .last_target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .replace(record.target().to_owned());
        }
        fn flush(&self) {}
    }

    // ----- Tests using local SinkingLog instances (no globals) -----

    #[test]
    #[serial(set_sinks)]
    fn disabled_path_drops_record() {
        let sinking = SinkingLog::new();
        sinking.set_max_level(LevelFilter::Off);
        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        sinking.set_sinks(vec![cap.clone() as Arc<dyn Log>]);

        emit!(&sinking, Level::Error, "test", "should not arrive");

        assert_eq!(cap.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[serial(set_sinks)]
    fn multi_sink_fanout_respects_per_sink_filter() {
        let sinking = SinkingLog::new();
        sinking.set_max_level(LevelFilter::Trace);
        let permissive = Arc::new(CapturingSink::new(LevelFilter::Trace));
        let restrictive = Arc::new(CapturingSink::new(LevelFilter::Warn));
        sinking.set_sinks(vec![
            permissive.clone() as Arc<dyn Log>,
            restrictive.clone() as Arc<dyn Log>,
        ]);

        emit!(&sinking, Level::Info, "test", "info-message");

        assert_eq!(permissive.count.load(Ordering::Relaxed), 1);
        assert_eq!(restrictive.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[serial(set_sinks)]
    fn set_sinks_replaces_existing() {
        let sinking = SinkingLog::new();
        sinking.set_max_level(LevelFilter::Trace);
        let first = Arc::new(CapturingSink::new(LevelFilter::Trace));
        let second = Arc::new(CapturingSink::new(LevelFilter::Trace));
        sinking.set_sinks(vec![first.clone() as Arc<dyn Log>]);
        sinking.set_sinks(vec![second.clone() as Arc<dyn Log>]);

        emit!(&sinking, Level::Info, "test", "msg");

        assert_eq!(first.count.load(Ordering::Relaxed), 0);
        assert_eq!(second.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial(set_sinks)]
    fn add_sink_appends() {
        let sinking = SinkingLog::new();
        sinking.set_max_level(LevelFilter::Trace);
        let first = Arc::new(CapturingSink::new(LevelFilter::Trace));
        let second = Arc::new(CapturingSink::new(LevelFilter::Trace));
        sinking.set_sinks(vec![first.clone() as Arc<dyn Log>]);
        sinking.add_sink(&(second.clone() as Arc<dyn Log>));

        emit!(&sinking, Level::Info, "test", "msg");

        assert_eq!(first.count.load(Ordering::Relaxed), 1);
        assert_eq!(second.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial(set_sinks)]
    fn sinking_log_self_filters_when_called_via_dyn_log() {
        let sinking = SinkingLog::new();
        sinking.set_max_level(LevelFilter::Off);
        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        sinking.set_sinks(vec![cap.clone() as Arc<dyn Log>]);

        // Drive the impl through `&dyn Log` to skip the macro's pre-check;
        // SinkingLog::log must enforce filtering itself per the log crate
        // contract.
        let log_ref: &dyn Log = &sinking;
        let record = Record::builder()
            .args(format_args!("nope"))
            .level(Level::Error)
            .target("test")
            .build();
        log_ref.log(&record);

        assert_eq!(cap.count.load(Ordering::Relaxed), 0);
    }

    /// Sink that, on every `log` call, re-enters the parent [`SinkingLog`] AND
    /// calls `clear_sinks` on it. Verifies that the snapshot pattern in
    /// `SinkingLog::log` releases the read lock before sinks run.
    struct ReentrantSink {
        target: Weak<SinkingLog>,
        nested_count: Arc<AtomicUsize>,
    }

    impl Log for ReentrantSink {
        fn enabled(&self, _: &Metadata) -> bool {
            true
        }
        fn log(&self, _: &Record) {
            self.nested_count.fetch_add(1, Ordering::Relaxed);
            if let Some(target) = self.target.upgrade() {
                // Clear first so the recursive log() snapshot is empty and
                // we don't recurse into ourselves indefinitely.
                target.clear_sinks();
                let nested = Record::builder()
                    .args(format_args!("nested"))
                    .level(Level::Info)
                    .target("test")
                    .build();
                target.log(&nested);
            }
        }
        fn flush(&self) {}
    }

    #[test]
    #[serial(set_sinks)]
    fn sink_can_reenter_logger_without_deadlock() {
        let sinking = Arc::new(SinkingLog::new());
        sinking.set_max_level(LevelFilter::Trace);
        let nested_count = Arc::new(AtomicUsize::new(0));
        let reentrant = Arc::new(ReentrantSink {
            target: Arc::downgrade(&sinking),
            nested_count: nested_count.clone(),
        });
        sinking.set_sinks(vec![reentrant as Arc<dyn Log>]);

        let outer = Record::builder()
            .args(format_args!("outer"))
            .level(Level::Info)
            .target("test")
            .build();
        sinking.log(&outer);

        // Reentrant log impl ran exactly once on the outer call's snapshot.
        // Subsequent attempts find sinks already cleared.
        assert_eq!(nested_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[serial(set_sinks)]
    fn multi_thread_concurrent_logging() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 200;

        let sinking = Arc::new(SinkingLog::new());
        sinking.set_max_level(LevelFilter::Trace);
        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        sinking.set_sinks(vec![cap.clone() as Arc<dyn Log>]);

        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let sinking = sinking.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..PER_THREAD {
                    let record = Record::builder()
                        .args(format_args!("msg"))
                        .level(Level::Info)
                        .target("t")
                        .build();
                    sinking.log(&record);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cap.count.load(Ordering::Relaxed), THREADS.saturating_mul(PER_THREAD));
    }

    // ----- Tests that touch the static LOGGER singleton (serialized) -----

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// Acquires the global test mutex (recovering from poison) and resets the
    /// static logger to a known-clean state both on construction and on drop.
    /// This makes tests that touch the per-cdylib singleton parallel-safe and
    /// panic-safe.
    struct GlobalTestEnv {
        _guard: MutexGuard<'static, ()>,
    }

    impl GlobalTestEnv {
        fn new() -> Self {
            let guard = TEST_MUTEX.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            clear_sinks();
            set_max_level(LevelFilter::Off);
            Self { _guard: guard }
        }
    }

    impl Drop for GlobalTestEnv {
        fn drop(&mut self) {
            clear_sinks();
            set_max_level(LevelFilter::Off);
        }
    }

    #[test]
    #[serial(set_sinks)]
    fn macro_to_static_pipeline() {
        let _env = GlobalTestEnv::new();

        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        set_sinks(vec![cap.clone() as Arc<dyn Log>]);
        set_max_level(LevelFilter::Trace);

        super::info!("greeting from macro");

        assert_eq!(cap.count.load(Ordering::Relaxed), 1);
        assert_eq!(max_level(), LevelFilter::Trace);
        assert_eq!(
            cap.last_message
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            Some("greeting from macro".to_owned())
        );
    }

    #[test]
    #[serial(set_sinks)]
    fn target_arm_passes_explicit_target() {
        let _env = GlobalTestEnv::new();

        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        set_sinks(vec![cap.clone() as Arc<dyn Log>]);
        set_max_level(LevelFilter::Trace);

        super::info!(target: "custom-tag", "{}", "msg");

        assert_eq!(cap.count.load(Ordering::Relaxed), 1);
        assert_eq!(
            cap.last_target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            Some("custom-tag".to_owned())
        );
    }

    static SIDE_EFFECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn evaluate_side_effect() -> &'static str {
        SIDE_EFFECT_COUNTER.fetch_add(1, Ordering::Relaxed);
        "evaluated"
    }

    #[test]
    #[serial(set_sinks)]
    fn disabled_macro_arm_does_not_evaluate_args() {
        let _env = GlobalTestEnv::new();

        SIDE_EFFECT_COUNTER.store(0, Ordering::Relaxed);
        // Set_max_level to Off so info! short-circuits before format_args!.
        set_max_level(LevelFilter::Off);

        super::info!("{}", evaluate_side_effect());

        assert_eq!(SIDE_EFFECT_COUNTER.load(Ordering::Relaxed), 0);

        // Now bump the level and verify the side effect runs.
        set_max_level(LevelFilter::Info);
        let cap = Arc::new(CapturingSink::new(LevelFilter::Trace));
        set_sinks(vec![cap.clone() as Arc<dyn Log>]);

        super::info!("{}", evaluate_side_effect());

        assert_eq!(SIDE_EFFECT_COUNTER.load(Ordering::Relaxed), 1);
        assert_eq!(cap.count.load(Ordering::Relaxed), 1);
    }
}
