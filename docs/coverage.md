# Code coverage

DCGM collects line coverage for both its C/C++ and Rust code. The standard
workflow is driven by a single flag:

```sh
./build.sh --coverage
```

This enables `gcov` instrumentation for C/C++, runs the C++ tests through
CTest, and runs the Rust workspace tests under
[`cargo-llvm-cov`][cargo-llvm-cov]. Reports are written under the build tree
(`_out/build/Linux-*-<type>/coverage/`):

| Path | Tool | Contents |
| --- | --- | --- |
| `coverage/coverage.xml` | gcovr | Merged C/C++ and Rust Cobertura report |
| `coverage/html/index.html` | gcovr | Merged C/C++ and Rust detailed HTML report |
| `coverage/gcovr/coverage.xml` | gcovr | C/C++ Cobertura report (line, branch, function) |
| `coverage/cargo-llvm-cov/coverage.xml` | cargo-llvm-cov | Rust Cobertura report |

gcovr combines the two intermediate Cobertura files with
`--cobertura-add-tracefile`. Both merged reports therefore use the same input
data and cover both language trees. The HTML output is a single browsable
report rather than separate C/C++ and Rust reports.

`coverage/coverage.xml` can be consumed by tools that support Cobertura. The
merged HTML target also prints a combined `lines: X% (Y out of Z)` summary.

## CMake targets

`build.sh --coverage` requests the two top-level report targets after the C++
tests complete:

- `coverage-cobertura` merges the C/C++ and Rust Cobertura inputs into
  `coverage/coverage.xml`.
- `coverage-html` merges the same inputs into the detailed report under
  `coverage/html/` and prints the combined line-coverage summary.

Those targets pull in the language-specific producers through their
dependencies:

- `gcovr-cobertura` scans the C/C++ coverage data and writes
  `coverage/gcovr/coverage.xml`.
- `cargo-llvm-cov-profile` removes stale Rust coverage profiles, runs the Rust
  workspace tests with `--no-report`, and keeps its instrumented Cargo build
  under `coverage/cargo-llvm-cov/target/`.
- `cargo-llvm-cov-cobertura` exports the collected Rust profile to
  `coverage/cargo-llvm-cov/coverage.xml`.

The `gcovr-html` and `cargo-llvm-cov-html` targets remain available when a
language-specific HTML report is useful, but the standard build workflow
requests only the merged HTML report.

## Cleaning coverage data

Coverage output and instrumentation data are part of CMake's clean graph. For
a build configured with `DCGM_BUILD_COVERAGE=ON`, this command removes the
generated coverage state along with the normal build outputs, without deleting
the configured build tree:

```sh
cmake --build <binary-dir> --target clean
```

The clean target removes the complete `coverage/` directory, including the
dedicated cargo-llvm-cov target directory, and registers C/C++ `.gcno`/`.gcda`
and LLVM `.profraw`/`.profdata` files associated with compiled targets. It also
removes default LLVM profile files from each binary directory. Registration is
deferred until the top-level directory has finished processing so targets from
subdirectories are included.

Corrosion's regular Cargo build directory is also registered with CMake's clean
target. Cargo's downloaded source cache under `CARGO_HOME` is intentionally
preserved. `build.sh --clean` remains the full reset: it deletes the build and
install trees before configuring them again.

LLVM profile names containing runtime substitutions such as `%p` or `%m`
cannot be enumerated individually by CMake. Code that sets `LLVM_PROFILE_FILE`
that way must place the profiles in a dedicated directory and register that
directory for cleanup.

## Requirements

C/C++ reports require `gcov` and `gcovr`. Rust reports require
`cargo-llvm-cov` and the `llvm-tools-preview` rustup component. These tools are
preinstalled in the dcgmbuild container.

Tool discovery is intentionally non-required so configuring DCGM does not fail
in environments that are not set up to generate coverage. The C/C++ targets
are created only when `gcov` and `gcovr` are available, the Rust targets only
when `cargo-llvm-cov` is available, and the two merged targets only when both
toolchains are available. Consequently, `build.sh --coverage` fails when it
requests the merged targets in an incomplete environment instead of silently
generating single-language coverage.

## Limitations

Rust `cdylib`s loaded by C++ test binaries are not yet captured by the Rust
coverage run. When that support is added, set `LLVM_PROFILE_FILE` in
`main.py.in` before the tests run and fold the emitted `.profraw` files into the
cargo-llvm-cov data. Place those dynamically named profiles in a directory
registered with CMake's clean target.

[cargo-llvm-cov]: https://github.com/taiki-e/cargo-llvm-cov
