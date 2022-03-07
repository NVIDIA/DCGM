/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmMigTypes.hpp"

std::ostream &DcgmNs::Mig::operator<<(std::ostream &os, DcgmNs::Mig::GpuInstanceId const &val)
{
    static_assert(std::is_trivially_copyable<DcgmNs::Mig::GpuInstanceId>::value,
                  "DcgmNs::Mig::GpuInstanceId should be trivially copyable");
    os << "GPU_I(" << val.id << ")";
    return os;
}
std::ostream &DcgmNs::Mig::operator<<(std::ostream &os, DcgmNs::Mig::ComputeInstanceId const &val)
{
    static_assert(std::is_trivially_copyable<DcgmNs::Mig::ComputeInstanceId>::value,
                  "DcgmNs::Mig::ComputeInstanceId should be trivially copyable");
    os << "GPU_CI(" << val.id << ")";
    return os;
}
std::ostream &DcgmNs::Mig::Nvml::operator<<(std::ostream &os, DcgmNs::Mig::Nvml::GpuInstanceId const &val)
{
    static_assert(std::is_trivially_copyable<DcgmNs::Mig::Nvml::GpuInstanceId>::value,
                  "DcgmNs::Mig::Nvml::GpuInstanceId should be trivially copyable");
    os << "NVML_GI(" << val.id << ")";
    return os;
}
std::ostream &DcgmNs::Mig::Nvml::operator<<(std::ostream &os, DcgmNs::Mig::Nvml::ComputeInstanceId const &val)
{
    static_assert(std::is_trivially_copyable<DcgmNs::Mig::Nvml::ComputeInstanceId>::value,
                  "DcgmNs::Mig::Nvml::ComputeInstanceId should be trivially copyable");
    os << "NVML_CI(" << val.id << ")";
    return os;
}
