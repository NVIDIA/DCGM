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

//! Child-process waiter that does not depend on tokio's process reaper.
//!
//! `tokio::process::Child::wait` observes child exit through tokio's global
//! reaper: a `pidfd` integration on new kernels, falling back to a process-wide
//! `SIGCHLD` handler on older ones. Neither is usable when this crate runs in
//! the same process as another child-reaping runtime that owns `SIGCHLD` (e.g.
//! the Boost.Process `sigchld_service` used by `common/ChildProcess`): Boost
//! installs its `SIGCHLD` disposition with `sigaction(..., NULL)` and resets it
//! to `SIG_DFL` on teardown, silently clobbering tokio's handler so tokio's
//! waits hang. Our minimum target (Ubuntu 18.04, kernel 4.15/5.4) also predates
//! the `pidfd` support tokio needs, so the `SIGCHLD` fallback is what runs.
//!
//! [`ChildWaiter`] sidesteps the whole problem: a dedicated OS thread blocks in
//! a per-PID `waitid(P_PID, …, WEXITED | WNOWAIT)` (never `waitpid(-1)`), so exit
//! is observed independently of any `SIGCHLD` disposition. The result is
//! published through a `watch` channel, making [`ChildWaiter::wait`] cancel-safe
//! and re-awaitable as `tokio::select!` and the graceful-kill escalation require.
//!
//! `WNOWAIT` means the observation does *not* reap the child: it stays a zombie,
//! so its PID remains reserved (cannot be recycled) for as long as the
//! [`ChildWaiter`] lives. That is what makes the supervisor's later
//! `kill(pid, …)` calls safe — they can never land on an unrelated process that
//! reused the PID.
//!
//! The actual reap is deferred until the [`ChildWaiter`] is dropped — i.e. once
//! supervision has finished and no further signals will be sent — and is then
//! performed *on the wait thread*, not in `drop`. Dropping the waiter only
//! signals the thread (by dropping an mpsc sender); the thread reaps after the
//! child has exited. Doing it this way means an early teardown (the supervisor
//! giving up before the child exits) still reaps the child once it finally
//! exits — no zombie leak — and `drop` never blocks.

use std::process::Child;
use std::sync::mpsc;

use dcgm_logging::debug;
use tokio::sync::watch;

/// Outcome of a finished child process.
///
/// `code` is the process exit code, or `None` if the process was terminated by
/// a signal (mirroring [`std::process::ExitStatus::code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildExit {
    pub code: Option<i32>,
}

/// Waits for a child process to exit without relying on tokio's process reaper.
///
/// See the module documentation for the rationale. A dedicated OS thread
/// observes exit with `waitid(WNOWAIT)` and publishes the outcome through a
/// `watch` channel so [`ChildWaiter::wait`] can be awaited repeatedly and
/// cancelled safely. The child stays an unreaped zombie (PID reserved, so
/// signalling stays safe) until this waiter is dropped, at which point the wait
/// thread reaps it.
pub struct ChildWaiter {
    pid: libc::pid_t,
    exit: watch::Receiver<Option<ChildExit>>,
    /// Dropping this sender releases the wait thread to reap the child once it
    /// has exited. Held only for that drop side effect; never read directly.
    #[allow(dead_code)]
    reap_release: mpsc::Sender<()>,
}

impl ChildWaiter {
    /// Takes ownership of `child` and starts waiting for its exit on a dedicated
    /// thread.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`std::io::Error`] if the waiter thread cannot be
    /// spawned.
    pub fn spawn(mut child: Child) -> std::io::Result<Self> {
        let pid = child.id().cast_signed();
        let (tx, rx) = watch::channel(None);
        // Dropping `reap_release` (held by the returned waiter) makes
        // `reap_signal.recv()` on the thread return `Err`, releasing it to reap.
        let (reap_release, reap_signal) = mpsc::channel::<()>();

        // The thread captures only `pid` (a `Copy`), the channels — never
        // `child` — so on a spawn failure we still own `child` to clean it up.
        let spawned = std::thread::Builder::new()
            .name(format!("child-wait-{pid}"))
            .spawn(move || {
                // Observe exit without reaping. Only a successful observation
                // leaves a zombie that *we* own and must reap; a failed one
                // (e.g. `ECHILD` because the child was already auto-reaped under
                // `SIGCHLD = SIG_IGN`) leaves nothing for us — and the PID may
                // have been reused, so reaping it would block on / steal an
                // unrelated child.
                let (exit, owns_zombie) = match observe_exit_without_reaping(pid) {
                    Observed::Zombie(code) => (ChildExit { code }, true),
                    Observed::Gone => (ChildExit { code: None }, false),
                };
                // The receiver may already be gone if supervision ended early;
                // dropping the send result is intentional.
                let _ = tx.send(Some(exit));
                // Block until the waiter is dropped (supervision over, no more
                // signals) before reaping, so the PID stays reserved for the
                // whole signalling window. The child has already exited here, so
                // the reap is immediate.
                let _ = reap_signal.recv();
                if owns_zombie {
                    reap(pid);
                }
            });

        match spawned {
            Ok(_handle) => Ok(Self {
                pid,
                exit: rx,
                reap_release,
            }),
            Err(err) => {
                // No waiter exists, so nothing else will reap this child.
                // Terminate and reap it here rather than leaking a running
                // process / zombie. (`child` was never moved into the thread.)
                let _ = child.kill();
                let _ = child.wait();
                Err(err)
            }
        }
    }

    /// Process identifier of the supervised child, for signal delivery.
    pub const fn pid(&self) -> libc::pid_t {
        self.pid
    }

    /// Returns the exit outcome if the child has already exited, without waiting.
    ///
    /// Used to avoid signalling a PID that may have been reaped (and possibly
    /// recycled) between supervision steps.
    pub fn try_exited(&self) -> Option<ChildExit> {
        *self.exit.borrow()
    }

    /// Waits for the child to exit.
    ///
    /// Cancel-safe and re-awaitable: the outcome is cached, so awaiting again
    /// after completion returns the same value immediately.
    pub async fn wait(&mut self) -> ChildExit {
        // `wait_for` resolves as soon as the value is `Some` and is cancel-safe.
        if self.exit.wait_for(Option::is_some).await.is_err() {
            // Sender dropped before publishing (waiter thread vanished); treat
            // as a signal-terminated child so supervision can complete.
            return ChildExit { code: None };
        }
        let exit = *self.exit.borrow();
        exit.unwrap_or(ChildExit { code: None })
    }
}

/// Outcome of observing the child with `waitid(WEXITED | WNOWAIT)`.
enum Observed {
    /// The child exited and `waitid` left it as a reapable zombie, so its PID
    /// is reserved and the caller must [`reap`] it once signalling is done. The
    /// field is the exit code, or `None` if the child was terminated by a signal.
    Zombie(Option<i32>),
    /// `waitid` could not observe the child (e.g. `ECHILD`: it was already
    /// auto-reaped, as happens when `SIGCHLD` is `SIG_IGN`). No zombie is owned
    /// by this waiter, so the caller must **not** reap — the PID may have been
    /// reused by an unrelated child.
    Gone,
}

/// Blocks until the child `pid` exits, observing its status **without reaping**.
///
/// Uses `waitid` with `WNOWAIT` so a successfully observed child stays a reapable
/// zombie (PID reserved); [`reap`] consumes it later, once signalling is done.
/// Returns [`Observed::Gone`] when `waitid` cannot observe the child, signalling
/// that this waiter owns no zombie and must not reap.
fn observe_exit_without_reaping(pid: libc::pid_t) -> Observed {
    loop {
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();

        // SAFETY: `waitid` initialises `info` for the target child. `P_PID` with
        // `pid` selects exactly our child; `WEXITED | WNOWAIT` blocks until it
        // exits while leaving it reapable (the PID stays reserved).
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid.cast_unsigned(),
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOWAIT,
            )
        };

        if rc == -1 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            // ECHILD (e.g. SIGCHLD set to SIG_IGN auto-reaped the child) or
            // another error: no zombie exists for this PID.
            return Observed::Gone;
        }

        // SAFETY: `waitid` returned success, so it initialised `info`.
        let info = unsafe { info.assume_init() };
        // SAFETY: `WEXITED` guarantees the SIGCHLD fields of `info` are valid.
        let si_status = unsafe { info.si_status() };

        return Observed::Zombie(if info.si_code == libc::CLD_EXITED {
            Some(si_status)
        } else {
            // CLD_KILLED / CLD_DUMPED: terminated by a signal.
            None
        });
    }
}

/// Reaps the zombie that [`observe_exit_without_reaping`] left behind
/// ([`Observed::Zombie`]).
///
/// Only called when this waiter owns the zombie, so `waitpid` finds it and
/// returns immediately. The status is discarded (the exit code was already
/// captured via `waitid`).
fn reap(pid: libc::pid_t) {
    loop {
        // SAFETY: `waitpid` reaps the zombie `pid`; a null status pointer is
        // permitted when the caller does not need the status.
        let rc = unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) };
        if rc == -1 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }

            debug!("waitpid({}) failed in reap: rc=-1, error={:?}", pid, err);
        }
        break;
    }
}

// Gated out of Miri: every test here spawns a real process via
// `std::process::Command`, which Miri does not support (mirrors the
// `not(miri)` guard on the `child_process` test module).
#[cfg(all(test, not(miri)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }

    #[test]
    fn reports_exit_code() {
        let child = std::process::Command::new("/bin/sh")
            .args(["-c", "exit 7"])
            .spawn()
            .unwrap();
        let mut waiter = ChildWaiter::spawn(child).unwrap();

        let exit = block_on(async {
            tokio::time::timeout(Duration::from_secs(5), waiter.wait())
                .await
                .expect("waiter should observe child exit before timeout")
        });

        assert_eq!(exit.code, Some(7));
    }

    #[test]
    fn reports_signal_termination_and_exposes_pid() {
        let child = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .unwrap();
        let pid = child.id().cast_signed();
        let mut waiter = ChildWaiter::spawn(child).unwrap();
        assert_eq!(waiter.pid(), pid, "waiter must expose the child pid for signalling");

        // SAFETY: kill(2) reads two scalar arguments; `pid` is our live child.
        unsafe { libc::kill(waiter.pid(), libc::SIGKILL) };

        let exit = block_on(async {
            tokio::time::timeout(Duration::from_secs(5), waiter.wait())
                .await
                .expect("waiter should observe signal death before timeout")
        });

        assert_eq!(exit.code, None);
    }

    #[test]
    fn wait_is_reawaitable_and_cached() {
        let child = std::process::Command::new("/bin/sh")
            .args(["-c", "exit 3"])
            .spawn()
            .unwrap();
        let mut waiter = ChildWaiter::spawn(child).unwrap();

        block_on(async {
            let first = tokio::time::timeout(Duration::from_secs(5), waiter.wait())
                .await
                .expect("first wait observes exit");
            assert_eq!(first.code, Some(3));

            // Re-awaiting must return the same cached result, because the
            // supervise loop's `select!` and graceful kill await it repeatedly.
            let second = waiter.wait().await;
            assert_eq!(second.code, Some(3));

            assert_eq!(waiter.try_exited(), Some(ChildExit { code: Some(3) }));
        });
    }

    #[test]
    fn exited_child_pid_stays_reserved_until_waiter_dropped() {
        let child = std::process::Command::new("/bin/sh")
            .args(["-c", "exit 0"])
            .spawn()
            .unwrap();
        let pid = child.id().cast_signed();
        let mut waiter = ChildWaiter::spawn(child).unwrap();

        // The wait thread observes exit with WNOWAIT, so the child is not reaped
        // here: it stays a zombie we own.
        block_on(async {
            tokio::time::timeout(Duration::from_secs(5), waiter.wait())
                .await
                .expect("waiter should observe exit before timeout");
        });

        // Because the PID is still reserved (zombie), a later `kill(pid, …)` can
        // never land on an unrelated process that reused it. `kill(pid, 0)` does
        // nothing but check existence.
        // SAFETY: kill(2) with signal 0 only performs an existence/permission
        // check on its two scalar arguments.
        let exists = unsafe { libc::kill(pid, 0) };
        assert_eq!(
            exists, 0,
            "exited child must remain a reserved zombie until the waiter is dropped"
        );

        // Dropping the waiter reaps the zombie.
        drop(waiter);
    }

    /// Reads the single-letter process state from `/proc/<pid>/stat`, or `None`
    /// if the process no longer exists. The state follows the parenthesised
    /// `comm` field (which may contain spaces and `)`), so parse after the last
    /// `") "`.
    fn proc_state(pid: libc::pid_t) -> Option<char> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        stat.rsplit_once(") ").and_then(|(_, rest)| rest.chars().next())
    }

    #[test]
    fn early_drop_still_reaps_child_when_it_exits_later() {
        // Simulate a supervisor that gives up before the child exits: build the
        // waiter for a still-running child, then drop it immediately. The
        // detached wait thread must still reap the child once it exits, rather
        // than leaving a zombie behind.
        let child = std::process::Command::new("/bin/sh")
            .args(["-c", "sleep 0.3"])
            .spawn()
            .unwrap();
        let pid = child.id().cast_signed();
        let waiter = ChildWaiter::spawn(child).unwrap();
        drop(waiter); // early teardown while the child is still alive

        // The child exits ~0.3s later and must then be reaped, so its /proc
        // entry disappears. A leak would leave it as a 'Z' (zombie) indefinitely.
        let mut reaped = false;
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(100));
            if proc_state(pid).is_none() {
                reaped = true;
                break;
            }
        }
        assert!(
            reaped,
            "child must be reaped by the wait thread after an early drop (no zombie leak)"
        );
    }

    #[test]
    fn observe_reports_gone_when_pid_is_not_our_zombie() {
        // `waitid` on a PID that is not our child fails with ECHILD. That must
        // map to `Observed::Gone` (not a signal-style `Zombie`), so the wait
        // thread skips `reap` and never `waitpid`s a PID it does not own. Our
        // own PID is conveniently never our child, and yields ECHILD without
        // blocking.
        // SAFETY: getpid(2) has no preconditions and cannot fail.
        let not_our_child = unsafe { libc::getpid() };
        assert!(matches!(observe_exit_without_reaping(not_our_child), Observed::Gone));
    }
}
