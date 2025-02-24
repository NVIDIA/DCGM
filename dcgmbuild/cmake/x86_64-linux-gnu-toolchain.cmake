#
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
#
# CMake toolchain definition for x86_64-linux-gnu target

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_SYSROOT /opt/cross/x86_64-linux-gnu/sysroot)
set(CMAKE_FIND_ROOT_PATH /opt/cross/x86_64-linux-gnu)

set(CMAKE_C_COMPILER /opt/cross/bin/x86_64-linux-gnu-gcc)
set(CMAKE_C_COMPILER_TARGET x86_64-linux-gnu)

set(CMAKE_CXX_COMPILER /opt/cross/bin/x86_64-linux-gnu-g++)
set(CMAKE_CXX_COMPILER_TARGET x86_64-linux-gnu)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_SYSTEM_PREFIX_PATH /opt/cross;/opt/cross/x86_64-linux-gnu)

set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/../../../../x86_64-linux-gnu/include/c++/14.2.0
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/../../../../x86_64-linux-gnu/include/c++/14.2.0/x86_64-linux-gnu
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/../../../../x86_64-linux-gnu/include/c++/14.2.0/backward
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/include
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/include-fixed
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/../../../../x86_64-linux-gnu/include
    /opt/cross/bin/../x86_64-linux-gnu/sysroot/usr/include
)

set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/include
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/include-fixed
    /opt/cross/bin/../lib/gcc/x86_64-linux-gnu/14.2.0/../../../../x86_64-linux-gnu/include
    /opt/cross/bin/../x86_64-linux-gnu/sysroot/usr/include
)
