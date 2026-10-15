# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

declare_nvvs_plugin(cudaCommon .)

macro(cuda_common_plugin CUDA_VER)
    nvvs_plugin_srcs(pluginCudaCommon_${CUDA_VER})

    target_link_libraries(pluginCudaCommon_${CUDA_VER}
        PRIVATE
            cuda_ops${CUDA_VER}
        PUBLIC
            fmt::fmt)
endmacro()

foreach(CUDA_VER IN LISTS SUPPORTED_CUDA_VERSIONS)
    add_subdirectory(Cuda${CUDA_VER})
endforeach()

add_subdirectory(mock)
add_subdirectory(tests)
