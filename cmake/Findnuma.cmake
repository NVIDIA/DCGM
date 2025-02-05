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
# - Find Numa (a NUMA utility library)
# This module defines
# LIBNUMA_INCLUDE_DIR, where to find Libnuma headers
# Libnuma_FOUND, If false, do not try to use libnuma

set(Libnuma_prefixes /usr/lib/x86_64-linux-gnu /usr / /usr/local /opt/local "$ENV{HOME}" "${Libnuma_ROOT}")

foreach(prefix ${Libnuma_prefixes})
    list(APPEND Libnuma_INCLUDE_PATHS "${prefix}/include" "${prefix}")
endforeach()

message(STATUS "Searching ${Libnuma_INCLUDE_PATHS} for numa.h")

file(GLOB potential_headers FILES /usr/include/*numa*)
message(STATUS "potential matches for numa.h: ${potential_headers}")

find_path(LIBNUMA_INCLUDE_DIR numa.h PATHS ${Libnuma_INCLUDE_PATHS})

if (LIBNUMA_INCLUDE_DIR)
    message(STATUS "Found the NUMA include directory: ${LIBNUMA_INCLUDE_DIR}")
else()
    message(STATUS "Didn't find the NUMA include directory")
endif()

if (LIBNUMA_INCLUDE_DIR)
    set(Libnuma_FOUND TRUE)
else ()
    set(Libnuma_FOUND FALSE)
endif ()

if (Libnuma_FOUND)
    if (NOT Libnuma_FIND_QUIETLY)
        message(STATUS "Found numa.h: ${LIBNUMA_INCLUDE_DIR}")
    endif ()
else ()
    if (Libnuma_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find numa.h")
    endif ()
    message(STATUS "numa.h NOT found.")
endif ()

mark_as_advanced(LIBNUMA_INCLUDE_DIR)
unset(Libnuma_INCLUDE_PATHS)
