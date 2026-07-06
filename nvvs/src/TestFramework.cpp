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

#include <TestFramework.inl>

/* Global boolean to say whether we should be exiting or not. This is set by the signal handler if we receive a CTRL-C
 * or other terminating signal */
std::atomic_int32_t main_should_stop __attribute__((visibility("default"))) = 0;

namespace impl
{

template class TestFramework<PluginLib, DynamicLibraryLoader, SoftwarePluginFramework, FileSystemOperator>;

} // namespace impl
