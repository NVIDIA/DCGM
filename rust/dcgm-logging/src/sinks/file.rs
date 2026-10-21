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

//! File sink that line-buffers records to disk.

use std::fs::{File, OpenOptions};
use std::io::{LineWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use crate::{LevelFilter, Log, Metadata, Record};

/// Logs one line per [`Record`] to an append-mode file.
///
/// The file handle is wrapped in `Mutex<LineWriter<File>>`. [`LineWriter`]
/// flushes to the underlying file every time a `'\n'` is written, so each
/// record reaches disk before the next one is processed without paying the
/// cost of a `flush` system call inside the lock.
pub struct FileSink {
    inner: Mutex<LineWriter<File>>,
    max_level: LevelFilter,
}

impl FileSink {
    /// Opens `path` in append mode (creating it if missing) and returns a
    /// [`FileSink`] writing into it.
    ///
    /// # Errors
    ///
    /// Forwards any [`std::io::Error`] from [`OpenOptions`].
    pub fn append<P: AsRef<Path>>(path: P, max_level: LevelFilter) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            inner: Mutex::new(LineWriter::new(file)),
            max_level,
        })
    }

    /// Returns the per-sink level filter.
    #[must_use]
    pub const fn max_level(&self) -> LevelFilter {
        self.max_level
    }

    fn write_line(&self, record: &Record) {
        // SAFETY: `syscall(SYS_gettid)` takes no inputs and returns a small integer
        let tid = unsafe { nix::libc::syscall(nix::libc::SYS_gettid) };

        // Format the line first so the lock window only covers the actual
        // writing (LineWriter flushes on `'\n'`, so a single write_all is the
        // whole record). Drop the formatted string after the lock releases.
        let line = format!(
            "{time} {level} [{pid}:{tid:?}] {msg} [{file}:{line}][{target}]\n",
            time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            level = record.level().as_str(),
            pid = std::process::id(),
            tid = tid,
            msg = record.args(),
            file = record.file().unwrap_or("<unknown>"),
            line = record.line().unwrap_or(0),
            target = record.target(),
        );

        let _ = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write_all(line.as_bytes());
    }
}

impl Log for FileSink {
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
        let _ = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush();
    }
}

#[cfg(not(miri))]
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::FileSink;
    use crate::{Level, LevelFilter, Log, Record};
    use serial_test::serial;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Tiny test-only RAII wrapper around a unique path under `temp_dir()`
    struct TempPath(PathBuf);

    impl TempPath {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("dcgm_module_log_{tag}_{pid}_{nanos}_{n}.log"));
            Self(path)
        }
        fn as_path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn file_sink_writes_record_line() {
        let temp = TempPath::new("write");
        let sink = FileSink::append(temp.as_path(), LevelFilter::Trace).unwrap();

        // Inline the Record build + log call so the `format_args!` temporary
        // lives long enough; binding `Record` to a `let` would drop the
        // backing storage too early.
        sink.log(
            &Record::builder()
                .args(format_args!("hello-{}", 42))
                .level(Level::Info)
                .target("test")
                .module_path(Some("dcgm_logging::sinks::file::tests"))
                .build(),
        );
        sink.flush();

        let contents = std::fs::read_to_string(temp.as_path()).unwrap();
        assert!(contents.contains("INFO"), "expected level label in line: {contents:?}");
        assert!(contents.contains("hello-42"), "expected message in line: {contents:?}");
        assert!(
            contents.ends_with('\n'),
            "expected trailing newline so LineWriter flushes"
        );
    }

    #[test]
    fn file_sink_below_filter_drops() {
        let temp = TempPath::new("filter");
        let sink = FileSink::append(temp.as_path(), LevelFilter::Warn).unwrap();

        sink.log(
            &Record::builder()
                .args(format_args!("debug-line"))
                .level(Level::Debug)
                .target("test")
                .build(),
        );
        sink.flush();

        let contents = std::fs::read_to_string(temp.as_path()).unwrap_or_default();
        assert!(
            !contents.contains("debug-line"),
            "expected debug line to be filtered out, got: {contents:?}"
        );
    }

    #[test]
    #[serial(set_sinks)]
    fn file_sink_e2e() {
        use super::Log;
        use crate::{error, warn};
        use std::sync::Arc;
        let logger = crate::logger();
        let tmp_file = TempPath::new("file_sink_e2e");
        logger.set_sinks(vec![
            Arc::new(FileSink::append(tmp_file.as_path(), LevelFilter::Info).unwrap()) as Arc<dyn Log>,
        ]);
        logger.set_max_level(LevelFilter::Trace);

        error!("Diagnostic");
        warn!(target: "connect", "Success");

        let contents = std::fs::read_to_string(tmp_file.as_path()).unwrap_or_default();
        let contents = contents.lines().collect::<Vec<_>>();
        assert_eq!(
            contents.len(),
            2,
            "expected 2 lines in the file {}, got: {}. Content:\n{}",
            tmp_file.as_path().display(),
            contents.len(),
            contents.join("\n")
        );
        assert!(
            contents[0].contains("Diagnostic") && contents[0].contains("ERROR"),
            "expected error line contains 'Diagnostic', got: {:?}",
            contents[0]
        );
        assert!(
            contents[1].contains("connect") && contents[1].contains("WARN"),
            "expected warn line contains 'connect' target, got: {:?}",
            contents[1]
        );
    }
}
