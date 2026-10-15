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

use std::fmt;
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum PermissionError {
    NotRegularFile(PathBuf),
    NotOwnedByRootUserAndGroup(PathBuf),
    WritableByNonRoot(PathBuf),
    NotExecutable(PathBuf),
    IoError(std::io::Error),
}

impl fmt::Display for PermissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRegularFile(path) => write!(f, "not a regular file: {}", path.display()),
            Self::NotOwnedByRootUserAndGroup(path) => write!(f, "not owned by root: {}", path.display()),
            Self::WritableByNonRoot(path) => {
                write!(f, "writable by group or other: {}", path.display())
            }
            Self::NotExecutable(path) => write!(f, "not executable: {}", path.display()),
            Self::IoError(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl From<std::io::Error> for PermissionError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

/// Validates an already-open binary file descriptor for privileged execution.
///
/// This performs permission checks with `fstat`, so the checked file is the
/// same opened file that can later be passed to the child process runner for
/// execution.
///
/// # Errors
///
/// Returns a [`PermissionError`] variant describing the first failed check.
pub fn check_binary_fd(
    binary_fd: BorrowedFd<'_>,
    path_for_error: &Path,
    skip_permission_check: bool,
) -> Result<(), PermissionError> {
    check_metadata(
        path_for_error.into(),
        PermissionMetadata::from_fd(binary_fd)?,
        skip_permission_check,
    )
}

#[derive(Clone, Copy)]
struct PermissionMetadata {
    is_regular_file: bool,
    uid: libc::uid_t,
    gid: libc::gid_t,
    mode: libc::mode_t,
}

impl PermissionMetadata {
    fn from_fd(fd: BorrowedFd<'_>) -> Result<Self, PermissionError> {
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `stat` points to valid writable memory and `fd` is a live borrowed descriptor.
        let result = unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) };
        if result == -1 {
            return Err(PermissionError::IoError(std::io::Error::last_os_error()));
        }
        // SAFETY: `fstat` succeeded, so it initialized `stat`.
        let stat = unsafe { stat.assume_init() };

        Ok(Self {
            is_regular_file: stat.st_mode & libc::S_IFMT == libc::S_IFREG,
            uid: stat.st_uid,
            gid: stat.st_gid,
            mode: stat.st_mode,
        })
    }
}

struct ProcessIdentity {
    euid: libc::uid_t,
    egid: libc::gid_t,
    groups: Vec<libc::gid_t>,
}

impl ProcessIdentity {
    fn current() -> Result<Self, PermissionError> {
        let groups = nix::unistd::getgroups()
            .map_err(std::io::Error::from)?
            .into_iter()
            .map(nix::unistd::Gid::as_raw)
            .collect();

        Ok(Self {
            euid: nix::unistd::Uid::effective().as_raw(),
            egid: nix::unistd::Gid::effective().as_raw(),
            groups,
        })
    }

    fn can_execute(&self, metadata: &PermissionMetadata) -> bool {
        let any_execute = libc::S_IXUSR | libc::S_IXGRP | libc::S_IXOTH;
        if self.euid == 0 {
            return metadata.mode & any_execute != 0;
        }

        if self.euid == metadata.uid {
            return metadata.mode & libc::S_IXUSR != 0;
        }

        if self.egid == metadata.gid || self.groups.contains(&metadata.gid) {
            return metadata.mode & libc::S_IXGRP != 0;
        }

        metadata.mode & libc::S_IXOTH != 0
    }
}

fn check_metadata(
    path: PathBuf,
    metadata: PermissionMetadata,
    skip_permission_check: bool,
) -> Result<(), PermissionError> {
    if !metadata.is_regular_file {
        return Err(PermissionError::NotRegularFile(path));
    }

    if skip_permission_check {
        return Ok(());
    }

    if metadata.uid != 0 || metadata.gid != 0 {
        return Err(PermissionError::NotOwnedByRootUserAndGroup(path));
    }

    let group_other_write = libc::S_IWGRP | libc::S_IWOTH;
    if metadata.mode & group_other_write != 0 {
        return Err(PermissionError::WritableByNonRoot(path));
    }

    if !ProcessIdentity::current()?.can_execute(&metadata) {
        return Err(PermissionError::NotExecutable(path));
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn permission_error_display() {
        let err = PermissionError::NotExecutable(PathBuf::from("/foo/bar"));
        let msg = format!("{err}");
        assert!(msg.contains("/foo/bar"));
        assert!(msg.contains("not executable"));
    }

    #[cfg(not(miri))]
    #[test]
    fn executable_permission_accepts_when_current_process_can_execute() {
        let identity = ProcessIdentity::current().unwrap();
        let mut metadata = PermissionMetadata {
            is_regular_file: true,
            uid: 0,
            gid: 0,
            mode: 0,
        };
        metadata.mode = executable_mode_for_identity(&identity, &metadata);

        let result = check_metadata(PathBuf::from("/tmp/test-binary"), metadata, false);

        assert!(result.is_ok(), "unexpected: {result:?}");
    }

    #[cfg(not(miri))]
    #[test]
    fn executable_permission_rejects_missing_execute_bits() {
        let result = check_metadata(
            PathBuf::from("/tmp/test-binary"),
            PermissionMetadata {
                is_regular_file: true,
                uid: 0,
                gid: 0,
                mode: 0,
            },
            false,
        );

        assert!(
            matches!(result, Err(PermissionError::NotExecutable(_))),
            "expected NotExecutable, got {result:?}"
        );
    }

    #[test]
    fn current_process_execute_permission_uses_owner_bit_for_owner() {
        let identity = ProcessIdentity {
            euid: 1000,
            egid: 2000,
            groups: vec![3000],
        };
        let mut metadata = PermissionMetadata {
            is_regular_file: true,
            uid: 1000,
            gid: 3000,
            mode: libc::S_IXUSR,
        };

        assert!(identity.can_execute(&metadata));

        metadata.mode = libc::S_IXGRP | libc::S_IXOTH;
        assert!(!identity.can_execute(&metadata));
    }

    #[test]
    fn current_process_execute_permission_uses_matching_group_bit() {
        let metadata = PermissionMetadata {
            is_regular_file: true,
            uid: 0,
            gid: 3000,
            mode: libc::S_IXGRP,
        };

        assert!(
            ProcessIdentity {
                euid: 1000,
                egid: 3000,
                groups: vec![],
            }
            .can_execute(&metadata)
        );

        assert!(
            ProcessIdentity {
                euid: 1000,
                egid: 2000,
                groups: vec![3000],
            }
            .can_execute(&metadata)
        );
    }

    #[test]
    fn current_process_execute_permission_uses_other_bit_without_owner_or_group_match() {
        let identity = ProcessIdentity {
            euid: 1000,
            egid: 2000,
            groups: vec![],
        };
        let mut metadata = PermissionMetadata {
            is_regular_file: true,
            uid: 0,
            gid: 3000,
            mode: libc::S_IXOTH,
        };

        assert!(identity.can_execute(&metadata));

        metadata.mode = libc::S_IXGRP;
        assert!(!identity.can_execute(&metadata));
    }

    #[test]
    fn root_execute_permission_requires_at_least_one_execute_bit() {
        let identity = ProcessIdentity {
            euid: 0,
            egid: 0,
            groups: vec![],
        };
        let mut metadata = PermissionMetadata {
            is_regular_file: true,
            uid: 0,
            gid: 0,
            mode: libc::S_IXOTH,
        };

        assert!(identity.can_execute(&metadata));

        metadata.mode = 0;
        assert!(!identity.can_execute(&metadata));
    }

    #[cfg(not(miri))]
    fn executable_mode_for_identity(identity: &ProcessIdentity, metadata: &PermissionMetadata) -> libc::mode_t {
        if identity.euid == 0 || identity.euid == metadata.uid {
            libc::S_IXUSR
        } else if identity.egid == metadata.gid || identity.groups.contains(&metadata.gid) {
            libc::S_IXGRP
        } else {
            libc::S_IXOTH
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn check_binary_fd_uses_open_file_even_after_path_is_removed() {
        use std::os::fd::AsFd;

        let tmp = std::env::temp_dir().join("child_process_fd_check_test");
        std::fs::write(&tmp, b"").unwrap();
        let file = std::fs::File::open(&tmp).unwrap();
        std::fs::remove_file(&tmp).unwrap();

        let result = check_binary_fd(file.as_fd(), &tmp, true);

        assert!(result.is_ok(), "unexpected: {result:?}");
    }
}
