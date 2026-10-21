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
#pragma once

#include "DistributedCudaContextRuntimeAnyUniquePtr.hpp"

namespace DcgmNs::ProfTester
{
/** Creates owned operating-system runtime adapters for distributed CUDA contexts. */
class DistributedCudaContextRuntimeFactory
{
public:
    /**
     * Create a runtime adapter.
     *
     * The returned handle uniquely owns the runtime and may be transferred across threads. Each runtime instance must
     * outlive every borrowed AnyPtr view created from it. The factory itself has no shared state.
     *
     * @return A unique ownership handle to the runtime.
     * @throws std::bad_alloc If runtime allocation fails.
     */
    [[nodiscard]] DistributedCudaContextRuntimeAnyUniquePtr Create() const;
};
} // namespace DcgmNs::ProfTester
