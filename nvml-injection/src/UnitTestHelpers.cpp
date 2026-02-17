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

#include "UnitTestHelpers.h"
#include <WithInjectionSku.h>

#include <filesystem>

[[nodiscard]] NvmlInjectionGuard WithNvmlInjectionSkuFile(std::string const &skuFileName)
{
    std::string skuFilePath = std::string(PROJECT_SRC_DIR) + "/dcgm_private/testing/SKUs/" + skuFileName;
    if (!std::filesystem::exists(skuFilePath))
    {
        return nullptr;
    }

    setenv("NVML_INJECTION_MODE", "True", 1);
    setenv("NVML_YAML_FILE", skuFilePath.c_str(), 1);
    return std::make_unique<DcgmNs::Defer<std::function<void()>>>([]() {
        unsetenv("NVML_YAML_FILE");
        unsetenv("NVML_INJECTION_MODE");
    });
}
