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

//! # Mocked CPU SDC
//!
//! This file is used to create a mocked SDC binary for testing the CPU SDC plugin.

use libc::{F_GETFD, fcntl};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::os::fd::FromRawFd;
use std::os::unix::io::RawFd;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENV_MOCKED_SDC_EXIT_CODE: &str = "DCGM_MOCKED_SDC_EXIT_CODE";
const ENV_MOCKED_SDC_RETURN_PATH: &str = "DCGM_MOCKED_SDC_RETURN_PATH";
const ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC: &str = "DCGM_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC";
const ARG_STDOUT: &str = "--stdout";
const ARG_ENABLE_HEARTBEAT: &str = "--enable-heartbeat";

/// The arguments for the mocked SDC binary.
#[derive(Debug, Default)]
struct Args {
    /// Print the result to stdout instead of the fd channel for debugging purpose.
    stdout: bool,

    /// Enable the heartbeat thread to send heartbeat messages to the heartbeat channel.
    enable_heartbeat: bool,
}

impl Args {
    /// Parses supported command-line flags into the mocked binary's arguments.
    ///
    /// Input: process command-line arguments. Output: parsed [`Args`], ignoring unsupported flags.
    fn parse() -> Self {
        let mut args = Self::default();

        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                ARG_STDOUT => args.stdout = true,
                ARG_ENABLE_HEARTBEAT => args.enable_heartbeat = true,
                _ => {}
            }
        }

        args
    }
}

/// Parses a mocked child-process exit code in the range `0..=255`.
///
/// Input: `value`, the textual exit code. Output: its `i32` value or a descriptive parse/range error.
fn parse_mocked_exit_code(value: &str) -> Result<i32, String> {
    value
        .parse::<u8>()
        .map(i32::from)
        .map_err(|e| format!("Failed to parse {ENV_MOCKED_SDC_EXIT_CODE} as an exit code in 0..=255: {e}"))
}

/// Reads the configured mocked exit code, defaulting to success.
///
/// Input: the mocked-exit-code environment variable. Output: the configured code, zero, or an environment/parse error.
fn mocked_exit_code() -> Result<i32, String> {
    match std::env::var(ENV_MOCKED_SDC_EXIT_CODE) {
        Ok(value) => parse_mocked_exit_code(&value),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(e) => Err(format!("Failed to get {ENV_MOCKED_SDC_EXIT_CODE}: {e}")),
    }
}

/// Parses a missed-heartbeat interval expressed in seconds.
///
/// Input: `value`, a textual second count. Output: the corresponding [`Duration`] or a parse error.
fn parse_mocked_missed_heartbeat_interval(value: &str) -> Result<Duration, String> {
    value.parse::<u64>().map(Duration::from_secs).map_err(|e| {
        format!("Failed to parse {ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC} as a number of seconds: {e}")
    })
}

/// Reads the configured missed-heartbeat interval, defaulting to zero.
///
/// Input: the missed-heartbeat environment variable. Output: its duration, zero, or an environment/parse error.
fn mocked_missed_heartbeat_interval() -> Result<Duration, String> {
    match std::env::var(ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC) {
        Ok(value) => parse_mocked_missed_heartbeat_interval(&value),
        Err(std::env::VarError::NotPresent) => Ok(Duration::ZERO),
        Err(e) => Err(format!(
            "Failed to get {ENV_MOCKED_SDC_MISSED_HEARTBEAT_INTERVAL_SEC}: {e}"
        )),
    }
}

/// Terminates the mocked SDC process with the supplied exit code.
///
/// Input: `exit_code`, returned to the parent process. Output: none; this function never returns.
fn exit_process(exit_code: i32) -> ! {
    std::process::exit(exit_code);
}

/// Print the result to stdout.
///
/// Input: `sdc_returns`, containing newline-separated payloads. Output: `()` after printing every line.
fn print_result_to_stdout(sdc_returns: &str) {
    for line in sdc_returns.lines() {
        println!("{line}");
    }
}

/// Get the file from the raw fd.
///
/// # Arguments
///
/// * `fd` - The raw fd.
///
/// # Returns
///
/// * `File` - The file.
///
/// Returns an I/O error when `fd` is not open; otherwise ownership of `fd` is transferred to the output [`File`].
fn checked_file_from_raw_fd(fd: RawFd) -> Result<File, std::io::Error> {
    // SAFETY: `fcntl` only inspects the supplied file descriptor here.
    let flags = unsafe { fcntl(fd, F_GETFD) };
    if flags == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: The descriptor was verified as open above, and ownership is transferred to
        // `File` exactly once by the caller for the mocked child process channel.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

/// Print the result to the fd channel.
///
/// # Arguments
///
/// * `sdc_returns` - The result.
///
/// # Output
///
/// Returns `()` after writing framed payload lines or reporting the first channel error.
fn print_result_to_fd_channel(sdc_returns: &str) {
    let channel_fd = 3;

    let mut file = match checked_file_from_raw_fd(channel_fd) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to get file from channel fd: {e}");
            return;
        }
    };
    for line in sdc_returns.lines() {
        let len: u32 = match line.len().try_into() {
            Ok(len) => len,
            Err(e) => {
                eprintln!("Failed to convert line length to u32: {e}");
                return;
            }
        };
        let length_le_bytes = len.to_le_bytes();
        if let Err(e) = file.write_all(&length_le_bytes) {
            eprintln!("Failed to write to file: {e}");
            return;
        }
        if let Err(e) = file.write_all(line.as_bytes()) {
            eprintln!("Failed to write to file: {e}");
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// Enable the heartbeat thread to send heartbeat messages to the heartbeat channel.
///
/// # Returns
///
/// * `Box<dyn FnOnce()>` - The function to stop the heartbeat thread.
///
/// Input: none; the heartbeat descriptor is obtained from the child-process ABI.
fn enable_heartbeat() -> Box<dyn FnOnce()> {
    let (tx, rx) = mpsc::channel::<()>();

    let channel_fd = 4;
    let mut file = match checked_file_from_raw_fd(channel_fd) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to get file from channel fd: {e}");
            return Box::new(|| {});
        }
    };

    let heartbeat_thread = thread::spawn(move || {
        loop {
            match rx.try_recv() {
                Ok(()) | Err(mpsc::TryRecvError::Disconnected) => {
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_or(0, |duration| duration.as_secs());
                    let heartbeat_str = format!("{{\"timestamp\": {timestamp}}}");
                    let len: u32 = match heartbeat_str.len().try_into() {
                        Ok(len) => len,
                        Err(e) => {
                            eprintln!("Failed to convert heartbeat string length to u32: {e}");
                            break;
                        }
                    };
                    let length_le_bytes = len.to_le_bytes();
                    if let Err(e) = file.write_all(&length_le_bytes) {
                        eprintln!("Failed to write to file: {e}");
                        break;
                    }
                    if let Err(e) = file.write_all(heartbeat_str.as_bytes()) {
                        eprintln!("Failed to write to file: {e}");
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
        }
    });
    Box::new(move || {
        let _ = tx.send(());
        if let Err(e) = heartbeat_thread.join() {
            eprintln!("Failed to join heartbeat thread: {e:?}");
        }
    })
}

/// Runs the mocked SDC binary using its configured payload and channel behavior.
///
/// Inputs: command-line arguments and mocked-SDC environment variables. Output: process output and the configured exit status.
fn main() {
    let args = Args::parse();
    let mut heartbeat_thread_stop_fn: Option<Box<dyn FnOnce()>> = None;
    let exit_code = match mocked_exit_code() {
        Ok(exit_code) => exit_code,
        Err(e) => {
            eprintln!("{e}");
            exit_process(1);
        }
    };
    let missed_heartbeat_interval = match mocked_missed_heartbeat_interval() {
        Ok(duration) => duration,
        Err(e) => {
            eprintln!("{e}");
            exit_process(1);
        }
    };

    let sdc_path = match std::env::var(ENV_MOCKED_SDC_RETURN_PATH) {
        Ok(sdc_path) => sdc_path,
        Err(e) => {
            eprintln!("Failed to get SDC return path: {e}");
            exit_process(exit_code);
        }
    };
    let sdc_returns = match fs::read_to_string(sdc_path) {
        Ok(sdc_returns) => sdc_returns,
        Err(e) => {
            eprintln!("Failed to read SDC return file: {e}");
            exit_process(exit_code);
        }
    };

    if args.enable_heartbeat {
        heartbeat_thread_stop_fn = Some(enable_heartbeat());
    }

    if args.stdout {
        print_result_to_stdout(&sdc_returns);
    } else {
        print_result_to_fd_channel(&sdc_returns);
    }

    if let Some(stop_fn) = heartbeat_thread_stop_fn {
        stop_fn();
    }
    if !args.stdout {
        thread::sleep(missed_heartbeat_interval);
    }

    exit_process(exit_code);
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{parse_mocked_exit_code, parse_mocked_missed_heartbeat_interval};
    use std::time::Duration;

    #[test]
    /// Verifies that valid process exit codes are accepted.
    /// Input: none. Output: `()` when parse assertions pass.
    fn parse_mocked_exit_code_accepts_valid_exit_codes() {
        assert_eq!(parse_mocked_exit_code("0"), Ok(0));
        assert_eq!(parse_mocked_exit_code("7"), Ok(7));
        assert_eq!(parse_mocked_exit_code("255"), Ok(255));
    }

    #[test]
    /// Verifies that out-of-range and nonnumeric exit codes are rejected.
    /// Input: none. Output: `()` when rejection assertions pass.
    fn parse_mocked_exit_code_rejects_invalid_exit_codes() {
        assert!(parse_mocked_exit_code("-1").is_err());
        assert!(parse_mocked_exit_code("256").is_err());
        assert!(parse_mocked_exit_code("not-an-int").is_err());
    }

    #[test]
    /// Verifies that heartbeat intervals are parsed as seconds.
    /// Input: none. Output: `()` when duration assertions pass.
    fn parse_mocked_missed_heartbeat_interval_accepts_seconds() {
        assert_eq!(parse_mocked_missed_heartbeat_interval("0"), Ok(Duration::ZERO));
        assert_eq!(parse_mocked_missed_heartbeat_interval("2"), Ok(Duration::from_secs(2)));
    }

    #[test]
    /// Verifies that invalid heartbeat intervals are rejected.
    /// Input: none. Output: `()` when rejection assertions pass.
    fn parse_mocked_missed_heartbeat_interval_rejects_invalid_values() {
        assert!(parse_mocked_missed_heartbeat_interval("-1").is_err());
        assert!(parse_mocked_missed_heartbeat_interval("not-an-int").is_err());
    }
}
