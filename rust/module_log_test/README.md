# module_log_test

Integration harness for [`dcgm-logging`](../dcgm-logging): verifies that two dynamically loaded copies of the same plugin shared object keep **separate** logging state and deliver records only to the callback each instance was given.

## What it verifies

DCGM loads multiple NVVS-style plugins as separate shared libraries. Each plugin should register its own host-engine log callback without clobbering another module's sinks. This harness models that with Linux `dlmopen(3)`:

1. Load `libplugin_log_test.so` twice, each time in a **new link-map namespace** (`LM_ID_NEWLM`).
2. Call `InitializePlugin` on each copy with a different C callback.
3. Call `emit_log_events` on each copy.
4. Assert each callback received exactly **three** records (one per `info!`, `error!`, and `trace!` from `dcgm-logging`).

The plugin uses [`add_sink`](../dcgm-logging/src/lib.rs) rather than `set_sinks` on purpose: `set_sinks` replaces the entire sink list and can mask bugs where one module overwrites another's registration.

## Layout

| Artifact | Role |
|----------|------|
| `plugin_log_test` (`rlib` + `cdylib`) | Fake plugin: registers a `dcgm-logging` sink that forwards records to a host-supplied C callback. |
| `module_log_test_host` (binary) | Loads two plugin instances, drives init/emit, validates isolation, exits 0 on success. |

Cargo package name: `module_log_test`. Library crate name: `plugin_log_test` (produces `libplugin_log_test.so`).

## Python integration test

After a normal DCGM build with install, the harness is installed under:

`share/dcgm_tests/apps/amd64/` or `share/dcgm_tests/apps/aarch64/` (per build target)

Run from the installed test tree:

```bash
cd _out/Linux-amd64-debug/share/dcgm_tests
python3 main.py --eris --no-lint --test test_module_log_test.test_module_log_isolation
```

The Python test executes `module_log_test_host` with `cwd` set to the install directory, passes the installed
`libplugin_log_test.so` with `--plugin-path`, and asserts exit code 0.

## Manual run

```bash
cd _out/Linux-amd64-debug/share/dcgm_tests/apps/amd64
./module_log_test_host --plugin-path ./libplugin_log_test.so
```

On success the process prints `PASS: module log isolation` and exits 0. On failure it prints `FAIL: ...` to stderr and exits 1.

Optional: enable `RUST_LOG=debug` before running for host diagnostics.
