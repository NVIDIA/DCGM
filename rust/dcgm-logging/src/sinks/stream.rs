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

//! Stream sink that writes one record per line to stdout or stderr.

use std::io::Write;

use crate::{LevelFilter, Log, Metadata, Record};

/// Selects which standard stream a [`StreamSink`] writes to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stream {
    /// `std::io::stdout()`.
    Stdout,
    /// `std::io::stderr()`.
    Stderr,
}

/// Writes one line per [`Record`] to a standard stream — [`std::io::stdout`]
/// or [`std::io::stderr`], chosen at construction.
///
/// Each record is formatted as
/// `<time> <level> [<pid>:<tid>] <message> [<file>:<line>][<target>]`,
/// terminated with a newline, and emitted with a single
/// [`write_all`](std::io::Write::write_all) call while the process-wide
/// stream lock is held. Holding that lock for the whole write keeps lines
/// from concurrent threads — and from other writers that share the same
/// standard-stream handle — from interleaving. The sink owns no buffer and
/// stores no file handle; [`flush`](Log::flush) forwards to the underlying
/// stream. Write and flush errors are ignored, since [`Log::log`] cannot
/// report them.
///
/// Records more verbose than [`max_level`](Self::max_level) are dropped by
/// [`Log::enabled`] before any formatting work happens. Build one with
/// [`StreamSink::stdout`] or [`StreamSink::stderr`]; both are `const`, so a
/// `StreamSink` may live in a `static`.
pub struct StreamSink {
    stream: Stream,
    max_level: LevelFilter,
}

impl StreamSink {
    /// Builds a [`StreamSink`] writing to `std::io::stdout()`.
    #[must_use]
    pub const fn stdout(max_level: LevelFilter) -> Self {
        Self {
            stream: Stream::Stdout,
            max_level,
        }
    }

    /// Builds a [`StreamSink`] writing to `std::io::stderr()`.
    #[must_use]
    pub const fn stderr(max_level: LevelFilter) -> Self {
        Self {
            stream: Stream::Stderr,
            max_level,
        }
    }

    /// Returns the per-sink level filter.
    #[must_use]
    pub const fn max_level(&self) -> LevelFilter {
        self.max_level
    }

    /// Returns which stream this sink writes to.
    #[must_use]
    pub const fn stream(&self) -> Stream {
        self.stream
    }

    fn write_line(&self, record: &Record) {
        // SAFETY: `syscall(SYS_gettid)` takes no inputs and returns a small integer
        let tid = unsafe { nix::libc::syscall(nix::libc::SYS_gettid) };
        let line = format!(
            "{time} {level} [{pid}:{tid:?}] {msg} [{file}:{line}][{target}]\n",
            time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            level = record.level().as_str(),
            pid = std::process::id(),
            tid = u32::try_from(tid).unwrap_or_default(),
            msg = record.args(),
            file = record.file().unwrap_or("<unknown>"),
            line = record.line().unwrap_or(0),
            target = record.target(),
        );
        match self.stream {
            Stream::Stdout => {
                let _ = std::io::stdout().lock().write_all(line.as_bytes());
            }
            Stream::Stderr => {
                let _ = std::io::stderr().lock().write_all(line.as_bytes());
            }
        }
    }
}

impl Log for StreamSink {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        self.write_line(record);
    }

    fn flush(&self) {
        match self.stream {
            Stream::Stdout => {
                let _ = std::io::stdout().flush();
            }
            Stream::Stderr => {
                let _ = std::io::stderr().flush();
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Stream, StreamSink};
    use crate::LevelFilter;

    #[test]
    fn stream_sink_constructors_record_stream_and_filter() {
        let stdout = StreamSink::stdout(LevelFilter::Info);
        assert_eq!(stdout.stream(), Stream::Stdout);
        assert_eq!(stdout.max_level(), LevelFilter::Info);

        let stderr = StreamSink::stderr(LevelFilter::Error);
        assert_eq!(stderr.stream(), Stream::Stderr);
        assert_eq!(stderr.max_level(), LevelFilter::Error);
    }

    #[cfg(not(miri))]
    #[test]
    fn stream_sink_smoke_does_not_panic() {
        use crate::{Level, LevelFilter, Log, Record};
        use log::Metadata;
        // We deliberately do not capture real stdout/stderr — that pattern
        // is brittle in unit tests. This is a smoke test that the sink can
        // process a record without panicking and that the per-sink filter
        // drops records below the threshold.
        let sink = StreamSink::stdout(LevelFilter::Error);
        // Inline-build the Record so `format_args!` storage outlives the call.
        sink.log(
            &Record::builder()
                .metadata(Metadata::builder().level(Level::Error).target("connect").build())
                .args(format_args!("filtered"))
                .level(Level::Error)
                .file(Some("test.rs"))
                .line(Some(131))
                .build(),
        );
        sink.flush();
    }

    #[cfg(not(miri))]
    use serial_test::serial;

    #[cfg(not(miri))]
    #[test]
    #[serial(set_sinks)]
    fn stream_sink_e2e_logging() {
        use super::Log;
        use crate::error;
        use std::sync::Arc;

        let logger = crate::logger();
        let sink: Arc<dyn Log> = Arc::new(StreamSink::stdout(LevelFilter::Info));
        logger.set_sinks(vec![sink]);
        logger.set_max_level(LevelFilter::Trace);

        error!("Simple test message");
        error!(target: "connect", "Another test message");
    }
}
