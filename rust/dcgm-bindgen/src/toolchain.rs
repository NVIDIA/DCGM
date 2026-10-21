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

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use serde::Deserialize;

/// Subset of a `CMake` toolchain file consumed by this build helper.
///
/// The file at `/opt/cross/<arch>/toolchain-gcc.cmake` is the source of truth
/// for the cross-compile sysroot, target triple, and the standard system
/// include directories used by the rest of the `CMake` build. Mirroring those
/// values into bindgen keeps it in sync with the GCC cross toolchain so the
/// same headers as a regular C++ compilation are used.
///
/// Two construction paths are supported. The fast path -- [`from_env`] --
/// reads the values from environment variables that the outer `CMake`
/// build (via Corrosion in `cmake/Rust.cmake`) injects into Cargo. The
/// fallback -- [`from_cmake_extract`] -- spawns `cmake -P` against the
/// toolchain file and parses a JSON payload back out, so direct
/// `cargo build`/IDE workflows still succeed without going through the
/// outer `CMake`.
///
/// [`from_env`]: Self::from_env
/// [`from_cmake_extract`]: Self::from_cmake_extract
#[derive(Debug, Clone, Deserialize)]
pub struct CmakeToolchain {
    /// `CMAKE_SYSROOT` -- passed to clang as `--sysroot=...`.
    pub sysroot: PathBuf,
    /// `CMAKE_LIBRARY_ARCHITECTURE` -- passed to clang as `--target=...`.
    pub target_triple: String,
    /// `CMAKE_C_STANDARD_INCLUDE_DIRECTORIES` -- libc and gcc include paths,
    /// passed to clang as `-isystem<dir>`.
    pub c_include_dirs: Vec<PathBuf>,
    /// `CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES` -- the GCC C++ stdlib search
    /// path, passed to clang as `-isystem<dir>`. Superset of the C list.
    pub cxx_include_dirs: Vec<PathBuf>,
}

/// Environment variable holding `CMAKE_SYSROOT` from the outer `CMake` build.
pub const ENV_SYSROOT: &str = "DCGM_BINDGEN_SYSROOT";
/// Environment variable holding `CMAKE_LIBRARY_ARCHITECTURE`.
pub const ENV_TARGET_TRIPLE: &str = "DCGM_BINDGEN_TARGET_TRIPLE";
/// Environment variable holding `CMAKE_C_STANDARD_INCLUDE_DIRECTORIES`,
/// joined with `;` (the native `CMake` list separator).
pub const ENV_C_INCLUDE_DIRS: &str = "DCGM_BINDGEN_C_INCLUDE_DIRS";
/// Environment variable holding `CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES`,
/// joined with `;`.
pub const ENV_CXX_INCLUDE_DIRS: &str = "DCGM_BINDGEN_CXX_INCLUDE_DIRS";

/// All four `DCGM_BINDGEN_*` env-var names, in display order.
pub const ENV_VARS: [&str; 4] = [ENV_SYSROOT, ENV_TARGET_TRIPLE, ENV_C_INCLUDE_DIRS, ENV_CXX_INCLUDE_DIRS];

const CMAKE_SCRIPT_TEMPLATE: &str = include_str!("extract_toolchain.cmake");
const TOOLCHAIN_FILE_PLACEHOLDER: &str = "@TOOLCHAIN_FILE@";
const JSON_BEGIN_MARKER: &str = "___DCGM_TOOLCHAIN_JSON_BEGIN___";
const JSON_END_MARKER: &str = "___DCGM_TOOLCHAIN_JSON_END___";

impl CmakeToolchain {
    /// Reads the toolchain values from environment variables set by the
    /// outer `CMake` build (see `cmake/Rust.cmake`).
    ///
    /// Returns `Ok(None)` when none of the four `DCGM_BINDGEN_*` variables
    /// are set, signalling that the caller should fall back to running
    /// `cmake -P` directly. Returns `Err` only on a *partial* configuration
    /// (some set, others not): that combination is always a bug in the
    /// outer build glue and silently masking it would lead to a confusing
    /// half-correct bindgen invocation.
    ///
    /// # Errors
    ///
    /// - [`ToolchainError::PartialEnv`] when only some of the four required
    ///   environment variables are set.
    pub fn from_env() -> Result<Option<Self>, ToolchainError> {
        Self::from_env_lookup(|name| std::env::var(name).ok())
    }

    /// Pure helper split out of [`from_env`] so tests can drive every branch
    /// without mutating process-global environment state.
    ///
    /// [`from_env`]: Self::from_env
    fn from_env_lookup<F>(mut lookup: F) -> Result<Option<Self>, ToolchainError>
    where
        F: FnMut(&'static str) -> Option<String>,
    {
        let sysroot = lookup(ENV_SYSROOT);
        let target_triple = lookup(ENV_TARGET_TRIPLE);
        let c_dirs = lookup(ENV_C_INCLUDE_DIRS);
        let cxx_dirs = lookup(ENV_CXX_INCLUDE_DIRS);

        match (sysroot, target_triple, c_dirs, cxx_dirs) {
            (None, None, None, None) => Ok(None),
            (Some(sysroot), Some(target_triple), Some(c_dirs), Some(cxx_dirs)) => Ok(Some(Self {
                sysroot: PathBuf::from(sysroot),
                target_triple,
                c_include_dirs: split_path_list(&c_dirs),
                cxx_include_dirs: split_path_list(&cxx_dirs),
            })),
            (sysroot, target_triple, c_dirs, cxx_dirs) => {
                let mut missing = Vec::new();
                if sysroot.is_none() {
                    missing.push(ENV_SYSROOT);
                }
                if target_triple.is_none() {
                    missing.push(ENV_TARGET_TRIPLE);
                }
                if c_dirs.is_none() {
                    missing.push(ENV_C_INCLUDE_DIRS);
                }
                if cxx_dirs.is_none() {
                    missing.push(ENV_CXX_INCLUDE_DIRS);
                }
                Err(ToolchainError::PartialEnv { missing })
            }
        }
    }

    /// Spawns `cmake -P` against the given toolchain file and parses the
    /// JSON object the extractor script prints between sentinel markers.
    ///
    /// The extractor script is shipped with the crate and templated at
    /// runtime to embed the toolchain path; it is written into `out_dir`
    /// (typically Cargo's `OUT_DIR`) before `cmake` is invoked.
    ///
    /// # Errors
    ///
    /// - [`ToolchainError::Io`] if writing the templated script fails.
    /// - [`ToolchainError::CmakeNotFound`] if the `cmake` binary cannot be
    ///   spawned (missing from `PATH` or not executable).
    /// - [`ToolchainError::CmakeFailed`] if `cmake -P` exits non-zero.
    /// - [`ToolchainError::CmakeOutputMissingMarkers`] if the expected
    ///   begin/end sentinel lines are absent from the captured output.
    /// - [`ToolchainError::Json`] if the payload between markers is not
    ///   well-formed JSON or does not match the expected shape.
    pub fn from_cmake_extract(toolchain_path: &Path, out_dir: &Path) -> Result<Self, ToolchainError> {
        let script_contents = render_script(toolchain_path);
        let script_path = out_dir.join("dcgm_bindgen_extract_toolchain.cmake");
        fs::write(&script_path, script_contents)?;

        let cmake_bin = std::env::var_os("CMAKE").unwrap_or_else(|| OsString::from("cmake"));

        let output = Command::new(&cmake_bin)
            .arg("-P")
            .arg(&script_path)
            .env("LC_ALL", "C")
            .env("LANG", "C")
            .output()
            .map_err(|source| {
                if source.kind() == std::io::ErrorKind::NotFound {
                    ToolchainError::CmakeNotFound {
                        binary: cmake_bin.clone(),
                        source,
                    }
                } else {
                    ToolchainError::Io(source)
                }
            })?;

        if !output.status.success() {
            return Err(ToolchainError::CmakeFailed {
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        // CMake `message()` writes to stderr by default. Parse stderr first;
        // if the markers are not there, try stdout as a defensive fallback
        // for unusual CMake configurations.
        let stderr_str = String::from_utf8_lossy(&output.stderr).into_owned();
        let stdout_str = String::from_utf8_lossy(&output.stdout).into_owned();
        let json_payload = extract_between_markers(&stderr_str)
            .or_else(|| extract_between_markers(&stdout_str))
            .map(str::to_owned);

        let json = json_payload.ok_or(ToolchainError::CmakeOutputMissingMarkers {
            stderr: stderr_str,
            stdout: stdout_str,
        })?;
        let toolchain: Self = serde_json::from_str(&json)?;
        Ok(toolchain)
    }
}

/// Splits a `;`-separated path list into `PathBuf` entries. Empty entries
/// (e.g. from a trailing `;`) are dropped; the four `DCGM_BINDGEN_*` lists
/// are never expected to contain empty paths.
fn split_path_list(value: &str) -> Vec<PathBuf> {
    value.split(';').filter(|s| !s.is_empty()).map(PathBuf::from).collect()
}

fn render_script(toolchain_path: &Path) -> String {
    CMAKE_SCRIPT_TEMPLATE.replace(TOOLCHAIN_FILE_PLACEHOLDER, &toolchain_path.display().to_string())
}

fn extract_between_markers(haystack: &str) -> Option<&str> {
    let begin_idx = haystack.find(JSON_BEGIN_MARKER)?;
    let after_begin = &haystack[begin_idx + JSON_BEGIN_MARKER.len()..];
    let end_idx = after_begin.find(JSON_END_MARKER)?;
    Some(after_begin[..end_idx].trim())
}

/// Errors produced when loading or interpreting a `CMake` toolchain file.
#[derive(Debug, thiserror::Error)]
pub enum ToolchainError {
    /// I/O failure unrelated to spawning `cmake` (e.g., writing the
    /// templated extractor script into `OUT_DIR`).
    #[error("toolchain I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The `cmake` binary could not be executed. Typically means `cmake`
    /// is not on `PATH` or the override in the `CMAKE` env var points at a
    /// non-existent file.
    #[error("toolchain extractor: failed to spawn `{}`: {source}", binary.to_string_lossy())]
    CmakeNotFound {
        /// The path that was attempted (either `cmake` or the value of
        /// the `CMAKE` env var).
        binary: OsString,
        /// The underlying spawn failure.
        #[source]
        source: std::io::Error,
    },

    /// `cmake -P` returned a non-zero exit status. The captured stderr is
    /// included for diagnostics.
    #[error("toolchain extractor: cmake -P exited with {status}; stderr: {stderr}")]
    CmakeFailed {
        /// The exit status from `cmake`.
        status: ExitStatus,
        /// Captured stderr from the invocation.
        stderr: String,
    },

    /// `cmake -P` succeeded but its output did not include the expected
    /// sentinel markers around the JSON payload. Both streams are reported
    /// because in unusual configurations cmake may route `message()`
    /// differently.
    #[error(
        "toolchain extractor: output did not include the expected JSON markers; stderr: {stderr}; stdout: {stdout}"
    )]
    CmakeOutputMissingMarkers {
        /// Captured stderr from the invocation.
        stderr: String,
        /// Captured stdout from the invocation.
        stdout: String,
    },

    /// The JSON between the markers could not be parsed into a
    /// [`CmakeToolchain`]. Wraps the underlying [`serde_json::Error`].
    #[error("toolchain extractor: JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Some, but not all, of the `DCGM_BINDGEN_*` environment variables
    /// were set. A partial configuration is treated as a hard error
    /// because it almost certainly indicates a bug in the outer `CMake`
    /// glue rather than a deliberate fallback request.
    #[error(
        "toolchain extractor: partial DCGM_BINDGEN_* environment configuration; \
         missing: {missing:?}"
    )]
    PartialEnv {
        /// The subset of [`ENV_VARS`] that was not set.
        missing: Vec<&'static str>,
    },

    /// The target architecture is not a triple this crate knows how to
    /// map to a cross toolchain location.
    #[error("toolchain extractor: {0}")]
    UnsupportedTarget(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn fixture_lookup<'a>(
        map: &'a HashMap<&'static str, &'static str>,
    ) -> impl Fn(&'static str) -> Option<String> + 'a {
        move |name| map.get(name).map(|s| (*s).to_string())
    }

    fn full_env_map() -> HashMap<&'static str, &'static str> {
        HashMap::from([
            (ENV_SYSROOT, "/opt/cross/x86_64-linux-gnu/sysroot"),
            (ENV_TARGET_TRIPLE, "x86_64-linux-gnu"),
            (
                ENV_C_INCLUDE_DIRS,
                "/opt/cross/include;/opt/cross/include-fixed;/opt/cross/sysroot/usr/include",
            ),
            (
                ENV_CXX_INCLUDE_DIRS,
                "/opt/cross/include/c++/14.2.0;/opt/cross/include;/opt/cross/sysroot/usr/include",
            ),
        ])
    }

    // -----------------------------------------------------------------
    // from_env behavior
    // -----------------------------------------------------------------

    #[test]
    fn from_env_returns_none_when_no_vars_set() {
        let map: HashMap<&'static str, &'static str> = HashMap::new();
        let result = CmakeToolchain::from_env_lookup(fixture_lookup(&map)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn from_env_returns_some_when_all_vars_set() {
        let map = full_env_map();
        let toolchain = CmakeToolchain::from_env_lookup(fixture_lookup(&map))
            .unwrap()
            .expect("env should yield Some");

        assert_eq!(toolchain.sysroot, Path::new("/opt/cross/x86_64-linux-gnu/sysroot"));
        assert_eq!(toolchain.target_triple, "x86_64-linux-gnu");
        assert_eq!(toolchain.c_include_dirs.len(), 3);
        assert_eq!(toolchain.cxx_include_dirs.len(), 3);
        assert_eq!(toolchain.c_include_dirs[0], Path::new("/opt/cross/include"));
        assert_eq!(
            toolchain.cxx_include_dirs[0],
            Path::new("/opt/cross/include/c++/14.2.0")
        );
    }

    #[test]
    fn from_env_errors_on_partial_config() {
        let mut map = full_env_map();
        map.remove(ENV_TARGET_TRIPLE);
        let err = CmakeToolchain::from_env_lookup(fixture_lookup(&map)).unwrap_err();
        match err {
            ToolchainError::PartialEnv { missing } => {
                assert_eq!(missing, vec![ENV_TARGET_TRIPLE]);
            }
            other => panic!("expected PartialEnv {{ [TARGET_TRIPLE] }}, got {other:?}"),
        }
    }

    #[test]
    fn from_env_errors_lists_all_missing_in_order() {
        let map = HashMap::from([(ENV_SYSROOT, "/sysroot")]);
        let err = CmakeToolchain::from_env_lookup(fixture_lookup(&map)).unwrap_err();
        match err {
            ToolchainError::PartialEnv { missing } => {
                assert_eq!(
                    missing,
                    vec![ENV_TARGET_TRIPLE, ENV_C_INCLUDE_DIRS, ENV_CXX_INCLUDE_DIRS]
                );
            }
            other => panic!("expected PartialEnv with three missing, got {other:?}"),
        }
    }

    #[test]
    fn split_path_list_drops_empty_entries() {
        assert_eq!(
            split_path_list("/a;;/b;;"),
            vec![PathBuf::from("/a"), PathBuf::from("/b")]
        );
    }

    #[test]
    fn split_path_list_handles_single_path() {
        assert_eq!(split_path_list("/a"), vec![PathBuf::from("/a")]);
    }

    // -----------------------------------------------------------------
    // JSON parser behavior
    // -----------------------------------------------------------------

    #[test]
    fn parses_well_formed_json_payload() {
        let json = r#"{
            "sysroot": "/opt/cross/x86_64-linux-gnu/sysroot",
            "target_triple": "x86_64-linux-gnu",
            "c_include_dirs": ["/a", "/b"],
            "cxx_include_dirs": ["/x", "/y", "/z"]
        }"#;
        let toolchain: CmakeToolchain = serde_json::from_str(json).unwrap();
        assert_eq!(toolchain.sysroot, Path::new("/opt/cross/x86_64-linux-gnu/sysroot"));
        assert_eq!(toolchain.target_triple, "x86_64-linux-gnu");
        assert_eq!(toolchain.c_include_dirs.len(), 2);
        assert_eq!(toolchain.cxx_include_dirs.len(), 3);
    }

    #[test]
    fn json_with_missing_field_fails_to_parse() {
        let json = r#"{
            "sysroot": "/x",
            "target_triple": "t",
            "c_include_dirs": []
        }"#;
        let result: Result<CmakeToolchain, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn json_with_wrong_types_fails_to_parse() {
        let json = r#"{
            "sysroot": "/x",
            "target_triple": 42,
            "c_include_dirs": [],
            "cxx_include_dirs": []
        }"#;
        let result: Result<CmakeToolchain, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // Marker extraction
    // -----------------------------------------------------------------

    #[test]
    fn extract_markers_finds_payload_in_middle_of_output() {
        let output =
            format!("-- some cmake noise\n{JSON_BEGIN_MARKER}\n{{\"sysroot\":\"/x\"}}\n{JSON_END_MARKER}\n-- more\n");
        let extracted = extract_between_markers(&output).unwrap();
        assert_eq!(extracted, "{\"sysroot\":\"/x\"}");
    }

    #[test]
    fn extract_markers_returns_none_when_begin_marker_absent() {
        let output = format!("nothing here\n{JSON_END_MARKER}\n");
        assert!(extract_between_markers(&output).is_none());
    }

    #[test]
    fn extract_markers_returns_none_when_end_marker_absent() {
        let output = format!("{JSON_BEGIN_MARKER}\nrun-on payload\n");
        assert!(extract_between_markers(&output).is_none());
    }

    #[test]
    fn extract_markers_trims_payload_whitespace() {
        let output = format!("{JSON_BEGIN_MARKER}\n  {{\"a\":1}}  \n{JSON_END_MARKER}");
        assert_eq!(extract_between_markers(&output).unwrap(), "{\"a\":1}");
    }

    // -----------------------------------------------------------------
    // Script template rendering
    // -----------------------------------------------------------------

    #[test]
    fn render_script_substitutes_toolchain_path() {
        let rendered = render_script(Path::new("/opt/cross/x86_64-linux-gnu/toolchain-gcc.cmake"));
        assert!(rendered.contains("include(\"/opt/cross/x86_64-linux-gnu/toolchain-gcc.cmake\")"));
        assert!(!rendered.contains(TOOLCHAIN_FILE_PLACEHOLDER));
    }

    // -----------------------------------------------------------------
    // End-to-end against the build container's real toolchain file.
    // Skipped automatically outside the build container.
    // -----------------------------------------------------------------

    #[cfg(not(miri))]
    #[test]
    fn end_to_end_against_real_toolchain_when_available() {
        let real_toolchain = Path::new("/opt/cross/x86_64-linux-gnu/toolchain-gcc.cmake");
        if !real_toolchain.is_file() {
            return;
        }
        let tmp = std::env::temp_dir();
        let toolchain = CmakeToolchain::from_cmake_extract(real_toolchain, &tmp).unwrap();
        assert_eq!(toolchain.target_triple, "x86_64-linux-gnu");
        assert_eq!(toolchain.sysroot, Path::new("/opt/cross/x86_64-linux-gnu/sysroot"));
        assert!(!toolchain.c_include_dirs.is_empty());
        assert!(!toolchain.cxx_include_dirs.is_empty());
    }
}
