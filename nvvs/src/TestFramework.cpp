/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <NvvsExitCode.h>
#include <TestFramework.inl>

#include <cstdint>

/* Global boolean to say whether we should be exiting or not. This is set by the signal handler if we receive a CTRL-C
 * or other terminating signal */
std::atomic_int32_t main_should_stop __attribute__((visibility("default"))) = 0;

int mainReturnCode = NVVS_ST_SUCCESS;

/**
 * Loads the current NVVS global stop-request flag for plugins loaded from shared libraries.
 *
 * This C ABI wrapper lets plugins query whether NVVS has received a termination request without linking directly
 * against the C++ `main_should_stop` atomic. This is required for Rust plugins because `std::atomic_int32_t` is a C++
 * library type, not a stable C ABI type that Rust can declare and load directly. Keeping the atomic load in C++ exposes
 * a plain `int32_t` return value across the FFI boundary.
 *
 * @return Non-zero when NVVS has requested plugin work to stop; zero otherwise.
 * @note This function performs a relaxed atomic load and is intended for polling only.
 */
extern "C" int32_t nvvs_main_should_stop_load() __attribute__((visibility("default")));
extern "C" int32_t nvvs_main_should_stop_load()
{
    return main_should_stop.load(std::memory_order_relaxed);
}

namespace impl
{

template class TestFramework<PluginLib, DynamicLibraryLoader, SoftwarePluginFramework, FileSystemOperator>;

} // namespace impl
