# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
set(
  NSCQ_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR})

add_library(sdk_nscq_interface INTERFACE)
target_link_libraries(sdk_nscq_interface INTERFACE sdk_nvidia sdk_nscq_loader)
target_include_directories(sdk_nscq_interface INTERFACE ${NSCQ_INCLUDE_DIRS})

 add_subdirectory(dlwrap)
