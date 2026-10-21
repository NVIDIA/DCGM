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

//! Shared helpers for spawning and supervising privileged child processes.
//!
//! # Feature flags
//!
//! - `subreaper` (**disabled by default**) — the crate calls
//!   `prctl(2) PR_SET_CHILD_SUBREAPER`.
//!
//!   ```toml
//!   child_process = { workspace = true, default-features = false }
//!   ```
//!
//! - `mock` — exposes `MockChildProcessRunner` for use in tests.

// The `subreaper` feature should only be enabled if C++ side does not set the subreaper itself.
#![allow(clippy::negative_feature_names)]

pub mod child_process;
mod child_waiter;
pub mod framed_stream;
pub mod permission_check;
#[cfg(feature = "subreaper")]
mod subreaper;

#[cfg(feature = "mock")]
pub use child_process::MockChildProcessRunner;
pub use child_process::{
    ChildExitStatus, ChildOutput, ChildProcessBinary, ChildProcessConfig, ChildProcessError, ChildProcessRunner,
    Fd3MessageHandler, LogPaths, RealChildProcessRunner, run_child, run_child_blocking,
};
pub use permission_check::{PermissionError, check_binary_fd};
