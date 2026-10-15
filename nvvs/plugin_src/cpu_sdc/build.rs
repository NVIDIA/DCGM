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

use std::env;
use std::error::Error;
use std::path::PathBuf;

/// Resolves and validates the C++ headers used to generate CPU SDC bindings.
///
/// # Inputs
///
/// Reads `CARGO_WORKSPACE_DIR` from the process environment.
///
/// # Output
///
/// Returns the validated header paths, or an error when the environment variable is unavailable or a header is missing.
fn include_cpp_headers() -> Result<Vec<String>, Box<dyn Error>> {
    // CARGO_WORKSPACE_DIR is defined in .cargo/config.toml.
    let project_root = PathBuf::from(env::var("CARGO_WORKSPACE_DIR")?);

    let bindgen_includes = [
        project_root.join("nvvs/include/PluginInterface.h"),
        project_root.join("nvvs/plugin_src/cpu_sdc/wrapper.h"),
    ]
    .into_iter()
    .map(|path| {
        if path.try_exists()? {
            println!("cargo::rerun-if-changed={}", path.display());
            Ok(path.display().to_string())
        } else {
            Err(format!("File not found: {}", path.display()).into())
        }
    })
    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(bindgen_includes)
}

/// Generates Rust bindings for the DCGM plugin interface during the Cargo build.
///
/// # Inputs
///
/// Reads Cargo build variables and the headers resolved by [`include_cpp_headers`].
///
/// # Output
///
/// Returns `Ok(())` after writing `bindings.rs`, or propagates configuration, generation, and file-writing errors.
fn main() -> Result<(), Box<dyn Error>> {
    let bindgen_includes = include_cpp_headers()?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let bindings = dcgm_bindgen::cpp_builder()?
        .headers(&bindgen_includes)
        .allowlist_type("dcgm.*")
        .allowlist_type("DcgmLoggingSeverity_t")
        .allowlist_type("HangDetectMonitorPtr")
        .allowlist_type("NvvsCommonPtr")
        .allowlist_type("hostEngineAppenderCallbackFp_t")
        .allowlist_var("DCGM_.*")
        .opaque_type("std::.*")
        .opaque_type("__gnu_cxx::.*")
        .opaque_type("__pstl::.*")
        .generate()?;

    bindings.write_to_file(out_dir.join("bindings.rs"))?;

    Ok(())
}
