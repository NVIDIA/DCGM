# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

target_include_directories(childprocess_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(childprocess PRIVATE
    FramedChannel.cpp
    FramedChannel.hpp
    ChildProcess.cpp
    ChildProcess.hpp
    SubreaperGuard.cpp
    SubreaperGuard.hpp
    SigChldGuard.cpp
    SigChldGuard.hpp
    ChildProcessBuilder.cpp
    ChildProcessBuilder.hpp
    StdLines.cpp
    StdLines.hpp
    Pipe.cpp
    Pipe.hpp
)

target_link_libraries(childprocess PUBLIC
    childprocess_interface
    dcgm_common
    dcgm
    dcgm_logging
    fmt::fmt
    Boost::system
    Boost::filesystem
)

add_subdirectory(tests)
