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

use std::collections::BTreeMap;
use std::ffi::{CString, NulError, OsString};
use std::fmt;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::time::Duration;

use tokio::io::AsyncRead;
use tokio::net::unix::pipe::Receiver as PipeReceiver;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

use crate::child_waiter::ChildWaiter;
use crate::framed_stream;

const CHILD_DATA_FD: i32 = 3;
const CHILD_HEARTBEAT_FD: i32 = 4;
const DEFAULT_PATH_ENV: &str = "/usr/bin:/bin:/usr/sbin:/sbin";
const DEFAULT_FD3_MESSAGE_QUEUE_CAPACITY: usize = 64;
const DEFAULT_STOP_CHECK_INTERVAL: Duration = Duration::from_millis(100);
// Matches C++ ChildProcess::Kill(sigTermTimeoutSec=10) and the post-SIGKILL Wait(1s)
// in common/ChildProcess/ChildProcess.cpp.
const DEFAULT_SIGTERM_GRACE: Duration = Duration::from_secs(10);
const DEFAULT_SIGKILL_GRACE: Duration = Duration::from_secs(1);

#[derive(Debug)]
pub enum ChildProcessError {
    RuntimeCreationFailed(std::io::Error),
    SpawnFailed(std::io::Error),
    WaiterSpawnFailed(std::io::Error),
    PipeCreationFailed(std::io::Error),
    PipeRegistrationFailed(std::io::Error),
    BinaryFdDuplicationFailed(std::io::Error),
    InvalidCString {
        which: &'static str,
        source: NulError,
    },
    LogFileOpenFailed {
        which: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for ChildProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeCreationFailed(err) => {
                write!(f, "failed to create child process runtime: {err}")
            }
            Self::SpawnFailed(err) => write!(f, "failed to spawn process: {err}"),
            Self::WaiterSpawnFailed(err) => write!(f, "failed to spawn child wait thread: {err}"),
            Self::PipeCreationFailed(err) => write!(f, "failed to create pipe: {err}"),
            Self::PipeRegistrationFailed(err) => write!(f, "failed to register pipe with runtime: {err}"),
            Self::BinaryFdDuplicationFailed(err) => {
                write!(f, "failed to duplicate binary file descriptor: {err}")
            }
            Self::InvalidCString { which, source } => write!(f, "invalid {which}: {source}"),
            Self::LogFileOpenFailed { which, path, source } => {
                write!(f, "failed to open {which} log file {}: {source}", path.display())
            }
        }
    }
}

/// Destination files for the child's stdout and stderr.
#[derive(Debug, Clone)]
pub struct LogPaths {
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub append: bool,
}

impl LogPaths {
    #[must_use]
    pub const fn create(stdout: PathBuf, stderr: PathBuf) -> Self {
        Self {
            stdout,
            stderr,
            append: false,
        }
    }

    #[must_use]
    pub const fn append(stdout: PathBuf, stderr: PathBuf) -> Self {
        Self {
            stdout,
            stderr,
            append: true,
        }
    }
}

/// Configuration for spawning and supervising a child process.
pub struct ChildProcessConfig {
    pub binary: ChildProcessBinary,
    pub args: Vec<String>,
    pub log_paths: LogPaths,
    pub clear_environment: bool,
    pub environment: Vec<(OsString, OsString)>,
    pub heartbeat_timeout: Duration,
    pub heartbeat_disabled: bool,
    pub fd3_message_queue_capacity: usize,
    pub stop_requested: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    pub stop_check_interval: Duration,
    /// Grace period between SIGTERM and SIGKILL when terminating the child.
    /// `None` skips SIGTERM entirely and goes straight to SIGKILL.
    pub sigterm_grace: Option<Duration>,
    /// Time to wait for the child to exit after SIGKILL before giving up.
    pub sigkill_grace: Duration,
}

/// Binary source for spawning a child process.
#[derive(Debug)]
pub enum ChildProcessBinary {
    Path(PathBuf),
    Fd { fd: OwnedFd, binary_path: PathBuf },
}

impl ChildProcessConfig {
    #[must_use]
    pub fn new(binary_path: impl Into<PathBuf>, args: impl Into<Vec<String>>, log_paths: LogPaths) -> Self {
        Self {
            binary: ChildProcessBinary::Path(binary_path.into()),
            args: args.into(),
            log_paths,
            clear_environment: true,
            environment: Vec::new(),
            heartbeat_timeout: Duration::from_mins(1),
            heartbeat_disabled: false,
            fd3_message_queue_capacity: DEFAULT_FD3_MESSAGE_QUEUE_CAPACITY,
            stop_requested: None,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
            sigterm_grace: Some(DEFAULT_SIGTERM_GRACE),
            sigkill_grace: DEFAULT_SIGKILL_GRACE,
        }
    }

    #[must_use]
    pub fn new_from_fd(
        binary_path: impl Into<PathBuf>,
        binary_fd: OwnedFd,
        args: impl Into<Vec<String>>,
        log_paths: LogPaths,
    ) -> Self {
        Self {
            binary: ChildProcessBinary::Fd {
                fd: binary_fd,
                binary_path: binary_path.into(),
            },
            args: args.into(),
            log_paths,
            clear_environment: true,
            environment: Vec::new(),
            heartbeat_timeout: Duration::from_mins(1),
            heartbeat_disabled: false,
            fd3_message_queue_capacity: DEFAULT_FD3_MESSAGE_QUEUE_CAPACITY,
            stop_requested: None,
            stop_check_interval: DEFAULT_STOP_CHECK_INTERVAL,
            sigterm_grace: Some(DEFAULT_SIGTERM_GRACE),
            sigkill_grace: DEFAULT_SIGKILL_GRACE,
        }
    }
}

/// Receives FD3 messages while a child process is running.
pub trait Fd3MessageHandler {
    fn handle_fd3_message(&mut self, message: Vec<u8>);
}

impl<F> Fd3MessageHandler for F
where
    F: FnMut(Vec<u8>),
{
    fn handle_fd3_message(&mut self, message: Vec<u8>) {
        self(message);
    }
}

/// Interface for code that launches and supervises child processes.
#[cfg_attr(feature = "mock", mockall::automock)]
pub trait ChildProcessRunner {
    /// Runs the child process described by `config` to completion.
    ///
    /// FD3 messages are delivered to `fd3_message_handler` as they arrive.
    ///
    /// # Errors
    ///
    /// Returns any error produced while setting up, spawning, or supervising the
    /// child process.
    ///
    /// # Panics
    ///
    /// The production implementation creates and blocks on a Tokio runtime, so
    /// it panics if called from within an existing Tokio runtime. Async callers
    /// should use [`run_child`] directly.
    fn run_child_blocking(
        &self,
        config: ChildProcessConfig,
        fd3_message_handler: &mut dyn Fd3MessageHandler,
    ) -> Result<ChildExitStatus, ChildProcessError>;
}

/// Production child process runner.
#[derive(Debug, Default, Clone, Copy)]
pub struct RealChildProcessRunner;

impl ChildProcessRunner for RealChildProcessRunner {
    fn run_child_blocking(
        &self,
        config: ChildProcessConfig,
        fd3_message_handler: &mut dyn Fd3MessageHandler,
    ) -> Result<ChildExitStatus, ChildProcessError> {
        run_child_blocking(config, fd3_message_handler)
    }
}

/// Running child process output.
#[derive(Debug)]
pub struct ChildOutput {
    fd3_messages: mpsc::Receiver<Vec<u8>>,
    completion: JoinHandle<ChildExitStatus>,
}

impl ChildOutput {
    /// Receives the next FD3 message, or `None` after the process supervisor has finished.
    pub async fn next_fd3_message(&mut self) -> Option<Vec<u8>> {
        self.fd3_messages.recv().await
    }

    /// Waits for the child process to finish.
    ///
    /// Any unread FD3 messages are drained and discarded so the bounded queue
    /// cannot block process completion. If process completion is already
    /// available, unread FD3 messages are left for the receiver drop path.
    pub async fn wait(self) -> ChildExitStatus {
        let Self {
            mut fd3_messages,
            mut completion,
        } = self;

        if let Some(status) = Self::drain_fd3_messages(&mut fd3_messages, &mut completion).await {
            return status;
        }

        Self::join_completion(completion).await
    }

    async fn drain_fd3_messages(
        fd3_messages: &mut mpsc::Receiver<Vec<u8>>,
        completion: &mut JoinHandle<ChildExitStatus>,
    ) -> Option<ChildExitStatus> {
        loop {
            tokio::select! {
                biased;

                status = &mut *completion => return Some(Self::status_from_completion(status)),

                message = fd3_messages.recv() => {
                    message?;
                }
            }
        }
    }

    async fn join_completion(completion: JoinHandle<ChildExitStatus>) -> ChildExitStatus {
        Self::status_from_completion(completion.await)
    }

    fn status_from_completion(result: Result<ChildExitStatus, tokio::task::JoinError>) -> ChildExitStatus {
        match result {
            Ok(status) => status,
            Err(err) => ChildExitStatus {
                protocol_error: Some(format!("child supervisor task failed: {err}")),
                ..ChildExitStatus::default()
            },
        }
    }
}

/// Final status from a completed supervised process.
#[derive(Debug, Default)]
pub struct ChildExitStatus {
    pub exit_code: Option<i32>,
    pub heartbeat_count: u32,
    pub timed_out: bool,
    pub stop_requested: bool,
    pub protocol_error: Option<String>,
}

/// Spawns a child binary with FD3/FD4 pipes and starts supervising it.
///
/// stdout and stderr are sent directly to the files in [`LogPaths`]. FD3 is
/// interpreted as a length-delimited data stream delivered through a bounded
/// queue. FD4 is interpreted as a length-delimited heartbeat stream.
///
/// This function must be called from within a Tokio runtime.
///
/// # Errors
///
/// Returns [`ChildProcessError::LogFileOpenFailed`] if either log file cannot
/// be opened, [`ChildProcessError::PipeCreationFailed`] if pipe creation fails,
/// [`ChildProcessError::PipeRegistrationFailed`] if the pipe cannot be registered
/// with Tokio, or [`ChildProcessError::SpawnFailed`] if the binary cannot be spawned.
pub fn run_child(config: ChildProcessConfig) -> Result<ChildOutput, ChildProcessError> {
    // Best-effort: re-assert subreaper status before each spawn. The result is
    // intentionally dropped — the spawn itself proceeds either way.
    #[cfg(feature = "subreaper")]
    let _ = crate::subreaper::ensure_child_subreaper();

    let stdout_file = open_log_file(&config.log_paths.stdout, config.log_paths.append).map_err(|source| {
        ChildProcessError::LogFileOpenFailed {
            which: "stdout",
            path: config.log_paths.stdout.clone(),
            source,
        }
    })?;
    let stderr_file = open_log_file(&config.log_paths.stderr, config.log_paths.append).map_err(|source| {
        ChildProcessError::LogFileOpenFailed {
            which: "stderr",
            path: config.log_paths.stderr.clone(),
            source,
        }
    })?;

    let (fd3_read, fd3_write) = create_pipe().map_err(ChildProcessError::PipeCreationFailed)?;
    let (fd4_read, fd4_write) = create_pipe().map_err(ChildProcessError::PipeCreationFailed)?;

    let fd3_write_raw = fd3_write.as_raw_fd();
    let fd4_write_raw = fd4_write.as_raw_fd();
    let fd3_async = PipeReceiver::from_owned_fd(fd3_read).map_err(ChildProcessError::PipeRegistrationFailed)?;
    let fd4_async = PipeReceiver::from_owned_fd(fd4_read).map_err(ChildProcessError::PipeRegistrationFailed)?;

    let mut cmd = match &config.binary {
        ChildProcessBinary::Path(path) => Command::new(path.as_path()),
        ChildProcessBinary::Fd { binary_path, .. } => Command::new(binary_path.as_path()),
    };
    cmd.args(&config.args);
    if config.clear_environment {
        cmd.env_clear();
        cmd.env("PATH", DEFAULT_PATH_ENV);
    }
    for (name, value) in &config.environment {
        cmd.env(name, value);
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::from(stdout_file));
    cmd.stderr(std::process::Stdio::from(stderr_file));

    let child = spawn_command(cmd, &config, fd3_write_raw, fd4_write_raw)?;

    drop(fd3_write);
    drop(fd4_write);

    // Observe the child's exit off tokio's process reaper so that supervision
    // works even when another runtime in this process owns SIGCHLD (see
    // `child_waiter`).
    let mut waiter = ChildWaiter::spawn(child).map_err(ChildProcessError::WaiterSpawnFailed)?;

    let (fd3_messages_sender, fd3_messages) = mpsc::channel(config.fd3_message_queue_capacity.max(10));
    let completion = tokio::spawn(async move {
        let mut fd3_framed = framed_stream::framed_reader(fd3_async);
        let mut fd4_framed = framed_stream::framed_reader(fd4_async);
        let mut status = ChildExitStatus::default();

        supervise_loop(
            &mut waiter,
            &mut fd3_framed,
            &mut fd4_framed,
            fd3_messages_sender,
            &config,
            &mut status,
        )
        .await;

        status
    });

    Ok(ChildOutput {
        fd3_messages,
        completion,
    })
}

/// Runs [`run_child`] on a current-thread Tokio runtime.
///
/// FD3 messages are delivered to `fd3_message_handler` as they arrive.
///
/// # Errors
///
/// Returns [`ChildProcessError::RuntimeCreationFailed`] if the runtime cannot be
/// created, or any error returned by [`run_child`].
///
/// # Panics
///
/// Panics if called from within an existing Tokio runtime. Async callers should
/// use [`run_child`] directly.
pub fn run_child_blocking(
    config: ChildProcessConfig,
    fd3_message_handler: &mut dyn Fd3MessageHandler,
) -> Result<ChildExitStatus, ChildProcessError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(ChildProcessError::RuntimeCreationFailed)?;
    runtime.block_on(async {
        let mut child_output = run_child(config)?;
        while let Some(message) = child_output.next_fd3_message().await {
            fd3_message_handler.handle_fd3_message(message);
        }
        Ok(child_output.wait().await)
    })
}

fn open_log_file(path: &Path, append: bool) -> Result<std::fs::File, std::io::Error> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    options.open(path)
}

fn create_pipe() -> Result<(OwnedFd, OwnedFd), std::io::Error> {
    // Create pipes with O_CLOEXEC so unrelated children spawned concurrently
    // by the same parent do not inherit these FDs. The intended child of this
    // spawn picks them up via the `dup2` in `setup_child_fds` (which clears
    // FD_CLOEXEC on FD3/FD4), so the receiving binary still sees them open.
    nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC).map_err(std::io::Error::from)
}

fn spawn_command(
    cmd: Command,
    config: &ChildProcessConfig,
    fd3_write_raw: i32,
    fd4_write_raw: i32,
) -> Result<std::process::Child, ChildProcessError> {
    match &config.binary {
        ChildProcessBinary::Path(_) => spawn_with_pre_exec(cmd, move || setup_child_fds(fd3_write_raw, fd4_write_raw)),
        ChildProcessBinary::Fd { fd, binary_path } => {
            let fexecve = FexecveCommand::new(fd.as_fd(), binary_path, config)?;
            spawn_with_pre_exec(cmd, move || {
                setup_child_fds(fd3_write_raw, fd4_write_raw)?;
                fexecve.exec()
            })
        }
    }
}

fn spawn_with_pre_exec<F>(mut cmd: Command, pre_exec: F) -> Result<std::process::Child, ChildProcessError>
where
    F: FnMut() -> std::io::Result<()> + Send + Sync + 'static,
{
    // SAFETY: `pre_exec` runs after fork and before exec in the child process.
    // The installed closure calls only async-signal-safe fd operations and, for
    // fd-backed binaries, `fexecve`, which replaces the process on success.
    unsafe { cmd.pre_exec(pre_exec) };
    cmd.spawn().map_err(ChildProcessError::SpawnFailed)
}

struct FexecveCommand {
    fd: OwnedFd,
    _argv: Vec<CString>,
    argv_ptrs: Vec<*const libc::c_char>,
    _env: Vec<CString>,
    env_ptrs: Vec<*const libc::c_char>,
}

// SAFETY: Pointer fields refer to immutable buffers owned by `_argv` and `_env`.
// The buffers are built before fork and are not mutated while the pre_exec
// closure uses them.
unsafe impl Send for FexecveCommand {}
// SAFETY: See the `Send` impl; all pointed-to data is immutable for the
// lifetime of the command.
unsafe impl Sync for FexecveCommand {}

impl FexecveCommand {
    fn new(
        binary_fd: BorrowedFd<'_>,
        binary_path: &Path,
        config: &ChildProcessConfig,
    ) -> Result<Self, ChildProcessError> {
        let fd = duplicate_binary_fd_for_exec(binary_fd)?;
        let argv = build_fexecve_argv(binary_path, &config.args)?;
        let argv_ptrs = cstring_ptrs(&argv);
        let env = build_fexecve_env(config)?;
        let env_ptrs = cstring_ptrs(&env);

        Ok(Self {
            fd,
            _argv: argv,
            argv_ptrs,
            _env: env,
            env_ptrs,
        })
    }

    fn exec(&self) -> std::io::Result<()> {
        // SAFETY: `fd` is a valid executable fd, and argv/env pointer arrays
        // are null-terminated and point to live `CString` buffers owned by self.
        let result = unsafe { libc::fexecve(self.fd.as_raw_fd(), self.argv_ptrs.as_ptr(), self.env_ptrs.as_ptr()) };
        if result == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

fn build_fexecve_argv(binary_path: &Path, args: &[String]) -> Result<Vec<CString>, ChildProcessError> {
    let mut arguments = Vec::with_capacity(args.len() + 1);
    arguments.push(cstring_from_bytes(binary_path.as_os_str().as_bytes(), "argv[0]")?);
    for arg in args {
        arguments.push(cstring_from_bytes(arg.as_bytes(), "argument")?);
    }
    Ok(arguments)
}

fn build_fexecve_env(config: &ChildProcessConfig) -> Result<Vec<CString>, ChildProcessError> {
    let mut env = BTreeMap::new();
    if config.clear_environment {
        env.insert(OsString::from("PATH"), OsString::from(DEFAULT_PATH_ENV));
    } else {
        env.extend(std::env::vars_os());
    }
    for (name, value) in &config.environment {
        env.insert(name.clone(), value.clone());
    }

    env.into_iter()
        .map(|(name, value)| {
            let mut item = Vec::new();
            item.extend_from_slice(name.as_os_str().as_bytes());
            item.push(b'=');
            item.extend_from_slice(value.as_os_str().as_bytes());
            cstring_from_bytes(&item, "environment variable")
        })
        .collect()
}

fn cstring_from_bytes(bytes: &[u8], which: &'static str) -> Result<CString, ChildProcessError> {
    CString::new(bytes).map_err(|source| ChildProcessError::InvalidCString { which, source })
}

fn cstring_ptrs(values: &[CString]) -> Vec<*const libc::c_char> {
    let mut ptrs: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
    ptrs.push(ptr::null());
    ptrs
}

fn duplicate_binary_fd_for_exec(fd: BorrowedFd<'_>) -> Result<OwnedFd, ChildProcessError> {
    // Keep the executable fd away from FD3/FD4, which are reserved for child
    // protocol streams and are reassigned in `pre_exec`.
    // SAFETY: `fcntl(F_DUPFD_CLOEXEC)` only reads the valid borrowed fd and
    // returns a new descriptor on success.
    let duplicated_fd = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, CHILD_HEARTBEAT_FD + 1) };
    if duplicated_fd == -1 {
        return Err(ChildProcessError::BinaryFdDuplicationFailed(
            std::io::Error::last_os_error(),
        ));
    }

    // SAFETY: `fcntl(F_DUPFD_CLOEXEC)` returned a new fd owned by this process.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicated_fd) })
}

fn setup_child_fds(fd3_write_raw: i32, fd4_write_raw: i32) -> std::io::Result<()> {
    // Move source fds out of the way if they collide with target slots.
    let fd3_write_raw = move_if_reserved_child_fd(fd3_write_raw)?;
    let fd4_write_raw = move_if_reserved_child_fd(fd4_write_raw)?;

    // SAFETY: `dup2` is async-signal-safe and `fd3_write_raw` is a valid pipe fd.
    if unsafe { libc::dup2(fd3_write_raw, CHILD_DATA_FD) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `dup2` is async-signal-safe and `fd4_write_raw` is a valid pipe fd.
    if unsafe { libc::dup2(fd4_write_raw, CHILD_HEARTBEAT_FD) } == -1 {
        return Err(std::io::Error::last_os_error());
    }

    close_original_fd(fd3_write_raw);
    close_original_fd(fd4_write_raw);

    Ok(())
}

fn move_if_reserved_child_fd(fd: i32) -> std::io::Result<i32> {
    if fd != CHILD_DATA_FD && fd != CHILD_HEARTBEAT_FD {
        return Ok(fd);
    }

    // SAFETY: `fcntl(F_DUPFD)` is async-signal-safe. The returned descriptor is
    // above the reserved child fd slots.
    let moved_fd = unsafe { libc::fcntl(fd, libc::F_DUPFD, CHILD_HEARTBEAT_FD + 1) };
    if moved_fd == -1 {
        return Err(std::io::Error::last_os_error());
    }

    // SAFETY: `close` is async-signal-safe and `moved_fd` now owns the stream
    // that previously used `fd`.
    unsafe { libc::close(fd) };
    Ok(moved_fd)
}

fn close_original_fd(original_fd: i32) {
    if original_fd != CHILD_DATA_FD && original_fd != CHILD_HEARTBEAT_FD {
        // SAFETY: `close` is async-signal-safe and the duplicated fd owns the
        // child-side stream.
        unsafe { libc::close(original_fd) };
    }
}

async fn next_frame<R: AsyncRead + Unpin>(
    framed: &mut FramedRead<R, LengthDelimitedCodec>,
) -> Option<Result<Vec<u8>, std::io::Error>> {
    use futures::StreamExt;
    StreamExt::next(framed)
        .await
        .map(|result| result.map(|frame| frame.to_vec()))
}

/// Terminate the supervised child: SIGTERM with a grace window, then SIGKILL
/// with a second grace window. Mirrors `ChildProcess::Kill` in
/// `common/ChildProcess/ChildProcess.cpp`. The SIGTERM phase is skipped
/// when [`ChildProcessConfig::sigterm_grace`] is `None`.
async fn graceful_kill(waiter: &mut ChildWaiter, config: &ChildProcessConfig, output: &mut ChildExitStatus) {
    let pid = waiter.pid();

    if let Some(grace) = config.sigterm_grace {
        // Skip signalling if the child already exited: its PID may have been
        // reaped (and possibly recycled) by the wait thread.
        if let Some(exit) = waiter.try_exited() {
            output.exit_code = exit.code;
            return;
        }
        send_signal(pid, libc::SIGTERM);
        if let Ok(exit) = tokio::time::timeout(grace, waiter.wait()).await {
            output.exit_code = exit.code;
            return;
        }
    }

    if let Some(exit) = waiter.try_exited() {
        output.exit_code = exit.code;
        return;
    }
    send_signal(pid, libc::SIGKILL);
    if let Ok(exit) = tokio::time::timeout(config.sigkill_grace, waiter.wait()).await {
        output.exit_code = exit.code;
        return;
    }

    let msg = "child did not exit after SIGKILL";
    output.protocol_error = Some(
        output
            .protocol_error
            .take()
            .map_or_else(|| msg.to_string(), |prev| format!("{prev}; {msg}")),
    );
}

fn send_signal(pid: libc::pid_t, sig: libc::c_int) {
    // SAFETY: `kill(2)` reads the two scalar arguments; no memory access.
    // ESRCH (child already exited) is benign and ignored; the wait thread
    // reaps whatever status the kernel has.
    unsafe { libc::kill(pid, sig) };
}

async fn supervise_loop<R1, R2>(
    waiter: &mut ChildWaiter,
    fd3_framed: &mut FramedRead<R1, LengthDelimitedCodec>,
    fd4_framed: &mut FramedRead<R2, LengthDelimitedCodec>,
    fd3_messages: mpsc::Sender<Vec<u8>>,
    config: &ChildProcessConfig,
    output: &mut ChildExitStatus,
) where
    R1: AsyncRead + Send + Unpin,
    R2: AsyncRead + Send + Unpin,
{
    use tokio::time::{Instant, sleep_until};

    let mut fd3_done = false;
    let mut fd4_done = false;
    let mut pending_fd3_message = None;

    let mut heartbeat_deadline = Instant::now() + config.heartbeat_timeout;
    let mut stop_check_deadline = Instant::now() + config.stop_check_interval;

    loop {
        let all_streams_done = fd3_done && fd4_done && pending_fd3_message.is_none();

        tokio::select! {
            biased;

            exit = waiter.wait(), if all_streams_done => {
                output.exit_code = exit.code;
                break;
            }

            () = sleep_until(stop_check_deadline), if config.stop_requested.is_some() => {
                if config.stop_requested.as_ref().is_some_and(|check| check()) {
                    output.stop_requested = true;
                    graceful_kill(waiter, config, output).await;
                    break;
                }
                stop_check_deadline = Instant::now() + config.stop_check_interval;
            }

            permit = fd3_messages.reserve(), if pending_fd3_message.is_some() => {
                if let Ok(permit) = permit {
                    if let Some(message) = pending_fd3_message.take() {
                        permit.send(message);
                    }
                } else {
                    graceful_kill(waiter, config, output).await;
                    break;
                }
            }

            frame = next_frame(fd3_framed), if !fd3_done && pending_fd3_message.is_none() => {
                match frame {
                    Some(Ok(data)) => {
                        pending_fd3_message = Some(data);
                        heartbeat_deadline = Instant::now() + config.heartbeat_timeout;
                    },
                    Some(Err(err)) => {
                        output.protocol_error = Some(format!("FD3 framing error: {err}"));
                        graceful_kill(waiter, config, output).await;
                        break;
                    }
                    None => fd3_done = true,
                }
            }

            frame = next_frame(fd4_framed), if !fd4_done => {
                match frame {
                    Some(Ok(_)) => {
                        heartbeat_deadline = Instant::now() + config.heartbeat_timeout;
                        output.heartbeat_count = output.heartbeat_count.saturating_add(1);
                    }
                    Some(Err(err)) => {
                        output.protocol_error = Some(format!("FD4 framing error: {err}"));
                        graceful_kill(waiter, config, output).await;
                        break;
                    }
                    None => fd4_done = true,
                }
            }

            () = sleep_until(heartbeat_deadline), if !config.heartbeat_disabled => {
                output.timed_out = true;
                graceful_kill(waiter, config, output).await;
                break;
            }
        }
    }
}

#[cfg(all(test, not(miri)))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::cast_possible_wrap)]
mod tests {
    use super::*;

    #[test]
    fn child_exit_status_default_has_sensible_values() {
        let output = ChildExitStatus::default();
        assert!(output.exit_code.is_none());
        assert_eq!(output.heartbeat_count, 0);
        assert!(!output.timed_out);
        assert!(!output.stop_requested);
        assert!(output.protocol_error.is_none());
    }

    #[test]
    fn process_error_display() {
        let err = ChildProcessError::PipeCreationFailed(std::io::Error::from_raw_os_error(libc::EMFILE));
        let msg = format!("{err}");
        assert!(msg.contains("pipe"));

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err2 = ChildProcessError::SpawnFailed(io_err);
        let msg2 = format!("{err2}");
        assert!(msg2.contains("spawn"));
    }

    #[test]
    fn log_file_open_failed_display_mentions_stream_and_path() {
        let err = ChildProcessError::LogFileOpenFailed {
            which: "stdout",
            path: PathBuf::from("/no/such/dir/out.txt"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        let msg = format!("{err}");
        assert!(msg.contains("stdout"), "got {msg}");
        assert!(msg.contains("/no/such/dir/out.txt"), "got {msg}");
    }

    #[test]
    fn log_file_open_failure_is_reported() {
        let stdout_directory = std::env::temp_dir();
        let bad = LogPaths::create(
            stdout_directory.clone(),
            stdout_directory.join("child_process_unused_stderr.txt"),
        );
        let config = ChildProcessConfig::new(Path::new("/bin/true"), Vec::<String>::new(), bad);
        let mut ignore_fd3_message = |_| {};
        let err = run_child_blocking(config, &mut ignore_fd3_message).unwrap_err();
        assert!(matches!(
            err,
            ChildProcessError::LogFileOpenFailed { which: "stdout", .. }
        ));
    }

    #[cfg(feature = "mock")]
    #[test]
    fn child_process_runner_can_be_mocked() {
        let args = Vec::new();
        let log_paths = LogPaths::create(PathBuf::from("/tmp/stdout.log"), PathBuf::from("/tmp/stderr.log"));
        let config = ChildProcessConfig::new(Path::new("/bin/true"), args, log_paths);

        let mut runner = MockChildProcessRunner::new();
        runner
            .expect_run_child_blocking()
            .withf(|config, _| {
                matches!(&config.binary, ChildProcessBinary::Path(path) if path.as_path() == Path::new("/bin/true"))
            })
            .returning(|_, _| {
                Ok(ChildExitStatus {
                    exit_code: Some(7),
                    ..ChildExitStatus::default()
                })
            });

        let mut ignore_fd3_message = |_| {};
        let output = runner.run_child_blocking(config, &mut ignore_fd3_message).unwrap();
        assert_eq!(output.exit_code, Some(7));
    }

    #[test]
    fn fexecve_argv_uses_binary_path_for_argv0() {
        let args = vec![String::from("--flag")];
        let cstring_arguments = build_fexecve_argv(Path::new("/custom/binary"), &args).unwrap();

        assert_eq!(cstring_arguments[0].as_bytes(), b"/custom/binary");
        assert_eq!(cstring_arguments[1].as_bytes(), b"--flag");
    }

    #[test]
    fn run_child_can_execute_binary_from_fd() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let log_root = std::env::temp_dir();
        let stdout = log_root.join(format!("child_process_fd_test_{unique}.stdout"));
        let stderr = log_root.join(format!("child_process_fd_test_{unique}.stderr"));
        let log_paths = LogPaths::create(stdout.clone(), stderr.clone());

        let binary = std::fs::File::open("/bin/sh").unwrap();
        let args = vec![
            String::from("-c"),
            String::from(
                "test \"$0\" = /bin/sh && test \"$CHILD_PROCESS_FEXECVE_TEST\" = expected && exit 7 || exit 3",
            ),
        ];
        let environment = vec![(OsString::from("CHILD_PROCESS_FEXECVE_TEST"), OsString::from("expected"))];
        let mut config = ChildProcessConfig::new_from_fd(Path::new("/bin/sh"), binary.into(), args, log_paths);
        config.clear_environment = true;
        config.environment = environment;
        config.heartbeat_disabled = true;

        let mut ignore_fd3_message = |_| {};
        let output = run_child_blocking(config, &mut ignore_fd3_message).unwrap();

        let _ = std::fs::remove_file(stdout);
        let _ = std::fs::remove_file(stderr);

        assert_eq!(output.exit_code, Some(7));
    }

    #[tokio::test]
    async fn run_child_streams_fd3_messages_before_exit() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let log_root = std::env::temp_dir();
        let stdout = log_root.join(format!("child_process_stream_test_{unique}.stdout"));
        let stderr = log_root.join(format!("child_process_stream_test_{unique}.stderr"));
        let log_paths = LogPaths::create(stdout.clone(), stderr.clone());

        let args = vec![
            String::from("-c"),
            String::from("printf '\\005\\000\\000\\000hello' >&3; sleep 1; exit 7"),
        ];
        let mut config = ChildProcessConfig::new(Path::new("/bin/sh"), args, log_paths);
        config.heartbeat_disabled = true;

        let mut output = run_child(config).unwrap();
        let message = tokio::time::timeout(Duration::from_millis(500), output.next_fd3_message())
            .await
            .expect("fd3 message should arrive before child exits")
            .expect("fd3 message stream should be open");
        assert_eq!(message, b"hello");

        let status = output.wait().await;

        let _ = std::fs::remove_file(stdout);
        let _ = std::fs::remove_file(stderr);

        assert_eq!(status.exit_code, Some(7));
    }

    #[tokio::test]
    async fn wait_returns_completion_without_draining_fd3_when_completion_is_ready() {
        let (sender, fd3_messages) = mpsc::channel(1);
        sender.send(b"unread".to_vec()).await.unwrap();

        let completion = tokio::spawn(async {
            ChildExitStatus {
                exit_code: Some(17),
                ..ChildExitStatus::default()
            }
        });

        while !completion.is_finished() {
            tokio::task::yield_now().await;
        }

        let output = ChildOutput {
            fd3_messages,
            completion,
        };
        let status = tokio::time::timeout(Duration::from_millis(500), output.wait())
            .await
            .expect("wait should use the ready completion instead of waiting for fd3 closure");

        drop(sender);

        assert_eq!(status.exit_code, Some(17));
    }

    #[tokio::test]
    async fn wait_drains_fd3_messages_to_unblock_completion() {
        let (sender, fd3_messages) = mpsc::channel(1);
        sender.send(b"queued".to_vec()).await.unwrap();

        let completion = tokio::spawn(async move {
            sender.send(b"blocked until wait drains".to_vec()).await.unwrap();
            ChildExitStatus {
                exit_code: Some(23),
                ..ChildExitStatus::default()
            }
        });

        let output = ChildOutput {
            fd3_messages,
            completion,
        };
        let status = tokio::time::timeout(Duration::from_millis(500), output.wait())
            .await
            .expect("wait should drain fd3 messages while completion is blocked");

        assert_eq!(status.exit_code, Some(23));
    }

    fn unique_log_paths(tag: &str) -> (LogPaths, PathBuf, PathBuf) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir();
        let stdout = root.join(format!("child_process_{tag}_{unique}.stdout"));
        let stderr = root.join(format!("child_process_{tag}_{unique}.stderr"));
        let paths = LogPaths::create(stdout.clone(), stderr.clone());
        (paths, stdout, stderr)
    }

    fn cleanup_log_files(stdout: &Path, stderr: &Path) {
        let _ = std::fs::remove_file(stdout);
        let _ = std::fs::remove_file(stderr);
    }

    // The graceful-kill tests below use python3 instead of /bin/sh so the
    // child is a single process. Shell-based children that background a long
    // sleep leak the FD3/FD4 write ends to that grandchild, which keeps the
    // parent's pipe read end open after the child exits. tokio's blocking
    // reads on those pipes then hold the runtime open until the grandchild
    // exits naturally — wallclock-confusing for tests, even though
    // `graceful_kill` itself behaves correctly.
    const PY_SIGTERM_TRAP: &str = "import signal,sys,time\n\
         def h(*a): sys.exit(42)\n\
         signal.signal(signal.SIGTERM, h)\n\
         time.sleep(30)";
    const PY_SIGTERM_IGN: &str = "import signal,time\n\
         signal.signal(signal.SIGTERM, signal.SIG_IGN)\n\
         time.sleep(30)";

    #[test]
    fn graceful_kill_lets_sigterm_handler_exit_cleanly() {
        let (log_paths, stdout, stderr) = unique_log_paths("sigterm_trap");
        let args = vec![String::from("-c"), String::from(PY_SIGTERM_TRAP)];
        let mut config = ChildProcessConfig::new(Path::new("/usr/bin/python3"), &*args, log_paths);
        config.heartbeat_disabled = true;
        let stop: Box<dyn Fn() -> bool + Send + Sync> = Box::new(|| true);
        config.stop_requested = Some(stop);
        config.stop_check_interval = Duration::from_millis(50);

        let start = std::time::Instant::now();
        let mut ignore_fd3_message = |_| {};
        let output = run_child_blocking(config, &mut ignore_fd3_message).unwrap();
        let elapsed = start.elapsed();
        cleanup_log_files(&stdout, &stderr);

        assert!(output.stop_requested);
        assert_eq!(
            output.exit_code,
            Some(42),
            "expected SIGTERM handler to run; output: {output:?}"
        );
        assert!(
            output.protocol_error.is_none(),
            "unexpected: {:?}",
            output.protocol_error
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "SIGTERM-handler exit took too long: {elapsed:?}"
        );
    }

    #[test]
    fn graceful_kill_falls_back_to_sigkill_when_sigterm_ignored() {
        let (log_paths, stdout, stderr) = unique_log_paths("sigkill_fallback");
        let args = vec![String::from("-c"), String::from(PY_SIGTERM_IGN)];
        let mut config = ChildProcessConfig::new(Path::new("/usr/bin/python3"), &*args, log_paths);
        config.heartbeat_disabled = true;
        let stop: Box<dyn Fn() -> bool + Send + Sync> = Box::new(|| true);
        config.stop_requested = Some(stop);
        config.stop_check_interval = Duration::from_millis(50);
        config.sigterm_grace = Some(Duration::from_millis(200));
        config.sigkill_grace = Duration::from_secs(2);

        let start = std::time::Instant::now();
        let mut ignore_fd3_message = |_| {};
        let output = run_child_blocking(config, &mut ignore_fd3_message).unwrap();
        let elapsed = start.elapsed();
        cleanup_log_files(&stdout, &stderr);

        assert!(output.stop_requested);
        assert!(
            output.exit_code.is_none(),
            "expected signal kill, got {:?}",
            output.exit_code
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "graceful kill took too long: {elapsed:?}; protocol_error={:?}",
            output.protocol_error
        );
        assert!(
            elapsed >= Duration::from_millis(200),
            "expected SIGTERM grace to elapse before SIGKILL, got {elapsed:?}"
        );
    }

    // Subreaper tests. The python script forks a grandchild, writes the
    // grandchild's PID to FD3, then both the immediate child and grandchild
    // close the inherited FD3/FD4 so the parent supervise_loop sees EOF and
    // doesn't hang on the now-orphaned grandchild's still-open pipe FDs.
    #[cfg(feature = "subreaper")]
    const PY_FORK_AND_REPORT_GPID: &str = "import os, struct, signal, sys, time\n\
         gpid = os.fork()\n\
         if gpid == 0:\n\
         \x20   os.close(3); os.close(4)\n\
         \x20   signal.signal(signal.SIGTERM, signal.SIG_DFL)\n\
         \x20   time.sleep(30)\n\
         \x20   sys.exit(0)\n\
         msg = str(gpid).encode()\n\
         os.write(3, struct.pack('<I', len(msg)) + msg)\n\
         os.close(3); os.close(4)\n\
         sys.exit(0)";

    #[cfg(feature = "subreaper")]
    fn read_ppid(pid: libc::pid_t) -> Option<libc::pid_t> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("PPid:") {
                return rest.trim().parse().ok();
            }
        }
        None
    }

    #[cfg(feature = "subreaper")]
    fn run_fork_and_report_gpid() -> (libc::pid_t, ChildExitStatus) {
        let (log_paths, stdout, stderr) = unique_log_paths("subreaper");
        let args = vec![String::from("-c"), String::from(PY_FORK_AND_REPORT_GPID)];
        let mut config = ChildProcessConfig::new(Path::new("/usr/bin/python3"), &*args, log_paths);
        config.heartbeat_disabled = true;
        let mut fd3_messages: Vec<Vec<u8>> = Vec::new();
        let mut collect_fd3_message = |message| fd3_messages.push(message);
        let output = run_child_blocking(config, &mut collect_fd3_message).unwrap();
        cleanup_log_files(&stdout, &stderr);
        assert_eq!(fd3_messages.len(), 1, "expected one FD3 frame: {output:?}");
        let gpid_str = std::str::from_utf8(&fd3_messages[0]).unwrap();
        let gpid: libc::pid_t = gpid_str.parse().unwrap_or_else(|_| panic!("bad pid: {gpid_str:?}"));
        (gpid, output)
    }

    #[cfg(feature = "subreaper")]
    fn our_pid() -> libc::pid_t {
        // SAFETY: getpid(2) has no preconditions and cannot fail.
        unsafe { libc::getpid() }
    }

    #[cfg(feature = "subreaper")]
    fn kill_and_reap_if_ours(gpid: libc::pid_t, ppid: libc::pid_t, our_pid: libc::pid_t) {
        // SAFETY: kill(2) reads two scalar args. ESRCH is benign.
        unsafe { libc::kill(gpid, libc::SIGKILL) };
        if ppid == our_pid {
            let mut status: libc::c_int = 0;
            let status_ptr = std::ptr::from_mut(&mut status);
            // SAFETY: waitpid(2) writes through a valid pointer to a c_int we own.
            unsafe { libc::waitpid(gpid, status_ptr, 0) };
        }
    }

    #[cfg(feature = "subreaper")]
    #[test]
    fn subreaper_reparents_orphan_grandchild_to_us() {
        let (gpid, _output) = run_fork_and_report_gpid();
        let our_pid = our_pid();
        let ppid = read_ppid(gpid).unwrap_or_else(|| panic!("could not read PPid for grandchild {gpid}"));

        // Cleanup before asserting, so a failure doesn't leak the grandchild.
        kill_and_reap_if_ours(gpid, ppid, our_pid);

        assert_eq!(
            ppid, our_pid,
            "grandchild {gpid} PPid was {ppid}, expected our pid {our_pid} (subreaper not in effect?)"
        );
    }

    #[cfg(feature = "subreaper")]
    #[test]
    fn subreaper_bit_reasserts_after_external_clear() {
        crate::subreaper::ensure_child_subreaper().unwrap();
        // SAFETY: prctl(PR_SET_CHILD_SUBREAPER, 0) toggles a process-wide flag.
        let rc = unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 0u64, 0u64, 0u64, 0u64) };
        assert_eq!(rc, 0, "prctl clear failed: {}", std::io::Error::last_os_error());

        // run_child should re-assert the bit and the grandchild should still
        // reparent to us.
        let (gpid, _) = run_fork_and_report_gpid();
        let our_pid = our_pid();
        let ppid = read_ppid(gpid).unwrap_or_else(|| panic!("could not read PPid for grandchild {gpid}"));

        kill_and_reap_if_ours(gpid, ppid, our_pid);

        assert_eq!(
            ppid, our_pid,
            "grandchild {gpid} PPid was {ppid}, expected our pid {our_pid} after external clear"
        );
    }

    #[test]
    fn graceful_kill_with_no_sigterm_grace_skips_straight_to_sigkill() {
        let (log_paths, stdout, stderr) = unique_log_paths("sigkill_only");
        // Use the SIGTERM-trap script: if SIGTERM were sent (it shouldn't be),
        // the process would exit 42. With sigterm_grace = None we expect
        // SIGKILL only and a signal-terminated exit_code (None).
        let args = vec![String::from("-c"), String::from(PY_SIGTERM_TRAP)];
        let mut config = ChildProcessConfig::new(Path::new("/usr/bin/python3"), &*args, log_paths);
        config.heartbeat_disabled = true;
        let stop: Box<dyn Fn() -> bool + Send + Sync> = Box::new(|| true);
        config.stop_requested = Some(stop);
        config.stop_check_interval = Duration::from_millis(50);
        config.sigterm_grace = None;
        config.sigkill_grace = Duration::from_secs(2);

        let start = std::time::Instant::now();
        let mut ignore_fd3_message = |_| {};
        let output = run_child_blocking(config, &mut ignore_fd3_message).unwrap();
        let elapsed = start.elapsed();
        cleanup_log_files(&stdout, &stderr);

        assert!(output.stop_requested);
        assert!(
            output.exit_code.is_none(),
            "expected signal kill, got {:?}",
            output.exit_code
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "skip-SIGTERM kill took too long: {elapsed:?}"
        );
    }
}
