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

//! Build-helper crate for DCGM Rust crates that consume `bindgen`.
//!
//! [`cpp_builder`] returns a fully preconfigured [`bindgen::Builder`] for
//! parsing DCGM C++ headers. It picks up the cross-compile sysroot, target
//! triple, and standard system include directories from
//! `/opt/cross/<arch>/toolchain-gcc.cmake` when present, layers in the project's
//! common include roots, and applies all bindgen options that should be uniform
//! across DCGM Rust bindings. Consumers only need to add `.header(...)`,
//! their own `.allowlist_*(...)` rules, and call `.generate()`.
//!
//! Toolchain values are obtained either from environment variables that the
//! outer `CMake` build (via Corrosion in `cmake/Rust.cmake`) injects into
//! Cargo, or -- when those are absent -- by spawning `cmake -P` against the
//! cross toolchain file. See [`CmakeToolchain`] for the exact contract.

mod toolchain;

use std::path::PathBuf;

pub use toolchain::{CmakeToolchain, ENV_VARS, ToolchainError};

/// Returns a fully preconfigured [`bindgen::Builder`] for parsing DCGM C++
/// headers.
///
/// As a side effect, emits `cargo:rerun-if-changed=<toolchain-gcc.cmake>` and
/// `cargo:rerun-if-env-changed=...` for both `CARGO_CFG_TARGET_ARCH` and the
/// four `DCGM_BINDGEN_*` env vars, so consumers automatically rebuild when
/// the cross toolchain or the outer `CMake` configuration changes.
///
/// # Note
/// This function is meant to work inside the DCGM build container, and it
/// hardcodes the toolchain location paths inside the build container.
///
/// # Errors
///
/// Returns an error if the target architecture is unsupported, the
/// environment configuration is partial, the toolchain file cannot be
/// located/read, or `cmake -P` fails.
pub fn cpp_builder() -> Result<bindgen::Builder, ToolchainError> {
    for env_var in ENV_VARS {
        println!("cargo:rerun-if-env-changed={env_var}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CMAKE");

    let triple = cargo_target_triple()?;
    let toolchain_path = PathBuf::from(format!("/opt/cross/{triple}/toolchain-gcc.cmake"));
    println!("cargo:rerun-if-changed={}", toolchain_path.display());

    let toolchain = if let Some(toolchain) = CmakeToolchain::from_env()? {
        toolchain
    } else {
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").ok_or_else(|| {
            ToolchainError::UnsupportedTarget(
                "OUT_DIR is not set; dcgm-bindgen must be invoked from a Cargo build script".into(),
            )
        })?);
        CmakeToolchain::from_cmake_extract(&toolchain_path, &out_dir)?
    };

    let repo_root = repo_root();
    let mut clang_args: Vec<String> = Vec::new();

    for dir in [
        repo_root.join("dcgmlib"),
        repo_root.join("nvvs/include"),
        repo_root.join("common"),
        repo_root.join("sdk/nvidia/nvml"),
    ] {
        clang_args.push(format!("-I{}", dir.display()));
    }
    // Vendored headers (e.g. libfmt) live under the cross sysroot's
    // /usr/local prefix, which is not on the toolchain's standard list.
    clang_args.push(format!("-I{}/usr/local/include", toolchain.sysroot.display()));
    clang_args.push(format!("--sysroot={}", toolchain.sysroot.display()));
    clang_args.push(format!("--target={}", toolchain.target_triple));
    for dir in toolchain.cxx_include_dirs.iter().chain(toolchain.c_include_dirs.iter()) {
        clang_args.push(format!("-isystem{}", dir.display()));
    }

    Ok(bindgen::Builder::default()
        .clang_arg("-xc++")
        .clang_arg("-std=c++23")
        .clang_args(clang_args)
        .clang_macro_fallback()
        .derive_default(true)
        .derive_debug(true)
        .derive_hash(true)
        .generate_comments(false)
        .layout_tests(false)
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .enable_cxx_namespaces()
        .enable_function_attribute_detection()
        .formatter(bindgen::Formatter::Rustfmt)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new())))
}

/// Returns the GCC-style triple (`x86_64-linux-gnu`/`aarch64-linux-gnu`) for
/// the Cargo build target, used to locate the matching cross toolchain.
///
/// `cfg!(target_arch = ...)` would resolve to the host arch because build
/// scripts are compiled for the host, so it cannot distinguish a native build
/// from an `--target=aarch64-unknown-linux-gnu` cross-compile. Cargo exposes
/// the actual build target to build scripts via `CARGO_CFG_TARGET_ARCH` (the
/// arch alone, e.g. `x86_64`) and `TARGET` (the full triple), which both honour
/// `--target` overrides set by CMake/Corrosion.
fn cargo_target_triple() -> Result<&'static str, ToolchainError> {
    target_triple_from_arch_var(std::env::var("CARGO_CFG_TARGET_ARCH"))
}

/// Pure helper split out of [`cargo_target_triple`] so tests can drive every
/// branch without mutating process-global environment state.
fn target_triple_from_arch_var(arch_var: Result<String, std::env::VarError>) -> Result<&'static str, ToolchainError> {
    let arch = arch_var.map_err(|_| {
        ToolchainError::UnsupportedTarget(
            "CARGO_CFG_TARGET_ARCH is not set; dcgm-bindgen must be invoked from a Cargo build script".into(),
        )
    })?;
    match arch.as_str() {
        "x86_64" => Ok("x86_64-linux-gnu"),
        "aarch64" => Ok("aarch64-linux-gnu"),
        other => Err(ToolchainError::UnsupportedTarget(format!(
            "unsupported target architecture for dcgm-bindgen: {other}"
        ))),
    }
}

/// Returns the DCGM repo root, derived from this crate's location at
/// compile time (`<repo>/rust/dcgm-bindgen`).
#[allow(clippy::unwrap_used)]
fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../..").canonicalize().unwrap()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn x86_64_maps_to_x86_64_linux_gnu() {
        assert_eq!(
            target_triple_from_arch_var(Ok("x86_64".into())).unwrap(),
            "x86_64-linux-gnu"
        );
    }

    #[test]
    fn aarch64_maps_to_aarch64_linux_gnu() {
        assert_eq!(
            target_triple_from_arch_var(Ok("aarch64".into())).unwrap(),
            "aarch64-linux-gnu"
        );
    }

    #[test]
    fn missing_env_var_yields_unsupported_target_with_explanatory_message() {
        let err = target_triple_from_arch_var(Err(std::env::VarError::NotPresent)).unwrap_err();
        match err {
            ToolchainError::UnsupportedTarget(msg) => {
                assert!(msg.contains("CARGO_CFG_TARGET_ARCH"), "got: {msg}");
                assert!(msg.contains("not set"), "got: {msg}");
            }
            other => panic!("expected UnsupportedTarget(CARGO_CFG_TARGET_ARCH not set ...), got {other:?}"),
        }
    }

    #[test]
    fn unsupported_arch_yields_unsupported_target_naming_the_arch() {
        let err = target_triple_from_arch_var(Ok("riscv64".into())).unwrap_err();
        match err {
            ToolchainError::UnsupportedTarget(msg) => {
                assert!(msg.contains("unsupported"), "got: {msg}");
                assert!(msg.contains("riscv64"), "got: {msg}");
            }
            other => panic!("expected UnsupportedTarget(unsupported ... riscv64), got {other:?}"),
        }
    }
}
