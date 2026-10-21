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

//! Process subreaper helper (crate-internal).
//!
//! Enabling `PR_SET_CHILD_SUBREAPER` causes orphan descendants of spawned
//! children to reparent to the calling process instead of init. This matches
//! `SubreaperGuard` in `common/ChildProcess/`.
//!
//! This module is gated by `cfg(feature = "subreaper")`: in the
//! default crate configuration (`subreaper` disabled) it does not compile
//! and the crate touches no process-global state.
//!
//! Tokio + subreaper coexistence: the subreaper bit only changes how orphaned
//! descendants are *reparented*; it does not change how any `waitpid` is
//! issued. Tokio reaps only the PIDs it owns (its spawned children and its own
//! orphan queue) with per-PID `waitpid(pid, WNOHANG)`, never `waitpid(-1)`.
//! That holds on both of tokio's reaping paths:
//!
//! - the `pidfd_open` reaper (`tokio/src/process/unix/pidfd_reaper.rs`), used
//!   when the running kernel supports it; and
//! - the `SIGCHLD`-driven fallback (`tokio/src/process/unix/reap.rs` and
//!   `orphan.rs`), used otherwise.
//!
//! The fallback is what we actually run on: our minimum supported OS is Ubuntu
//! 18.04, whose kernels (4.15 GA, up to 5.4 HWE) predate the `pidfd_open`
//! support tokio needs (it calls the raw `SYS_pidfd_open` with `PIDFD_NONBLOCK`,
//! which needs ~5.10), so `Pidfd::open` fails and tokio uses the `SIGCHLD`
//! reaper. Either way, becoming a subreaper does not interfere with tokio's
//! per-PID child waits. What *would* interfere is adding our own
//! `waitpid(-1, WNOHANG)` reaper: it could race with tokio and steal the exit
//! status of a child tokio is waiting on. That is why this crate does not add
//! one — at the cost that reparented orphans are not reaped here.

/// Become a subreaper for descendants (see `prctl(2)` `PR_SET_CHILD_SUBREAPER`).
///
/// Idempotent and cheap (one syscall). Called per spawn from
/// [`crate::run_child`] so the bit is re-asserted if some other code path
/// (e.g. C++ `~SubreaperGuard`) cleared it. Never panics.
pub fn ensure_child_subreaper() -> std::io::Result<()> {
    // SAFETY: `prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)` reads only the
    // first argument; the trailing args are ignored. No memory aliasing.
    let result = unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1u64, 0u64, 0u64, 0u64) };
    if result == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
