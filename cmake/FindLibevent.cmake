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
# - Find Libevent (a cross event library)
# This module defines
# LIBEVENT_INCLUDE_DIR, where to find Libevent headers
# LIBEVENT_STATIC_LIBS, Libevent static libraries
# Libevent_FOUND, If false, do not try to use libevent

find_path(Libevent_INCLUDE_DIR "event2/event.h")
mark_as_advanced(Libevent_INCLUDE_DIR)

if (Libevent_INCLUDE_DIR)
    cmake_path(GET Libevent_INCLUDE_DIR PARENT_PATH _Libevent_INSTALL_PREFIX)

    # TODO: Add dispatching based on ``Libevent_USE_STATIC_LIBS`` variable

    set(_Libevent_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(_Libevent_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})

    find_library(Libevent_core_LIBRARY
        NAMES ${_Libevent_LIBRARY_PREFIX}event_core${_Libevent_LIBRARY_SUFFIX}
        HINTS "${_Libevent_INSTALL_PREFIX}")

    find_library(Libevent_extra_LIBRARY
        NAMES ${_Libevent_LIBRARY_PREFIX}event_extra${_Libevent_LIBRARY_SUFFIX}
        HINTS "${_Libevent_INSTALL_PREFIX}")

    find_library(Libevent_pthreads_LIBRARY
        NAMES ${_Libevent_LIBRARY_PREFIX}event_pthreads${_Libevent_LIBRARY_SUFFIX}
        HINTS "${_Libevent_INSTALL_PREFIX}")

    mark_as_advanced(
        Libevent_core_LIBRARY
        Libevent_extra_LIBRARY
        Libevent_pthreads_LIBRARY)

    unset(_Libevent_INSTALL_PREFIX)
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Libevent
    REQUIRED_VARS
        Libevent_INCLUDE_DIR
        Libevent_core_LIBRARY
        Libevent_extra_LIBRARY
        Libevent_pthreads_LIBRARY)

if (Libevent_FOUND)
    set(Libevent_INCLUDE_DIRS "${Libevent_INCLUDE_DIR}")
    set(Libevent_LIBRARIES
        "${Libevent_core_LIBRARY}"
        "${Libevent_extra_LIBRARY}"
        "${Libevent_pthreads_LIBRARY}")

    if (NOT TARGET Libevent::core)
        add_library(Libevent::core UNKNOWN IMPORTED)
        set_target_properties(Libevent::core PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Libevent_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${Libevent_core_LIBRARY}")
    endif()

    if (NOT TARGET Libevent::extra)
        add_library(Libevent::extra UNKNOWN IMPORTED)
        set_target_properties(Libevent::extra PROPERTIES
            IMPORTED_LOCATION "${Libevent_extra_LIBRARY}"
            INTERFACE_LINK_LIBRARIES Libevent::core)
    endif()

    if (NOT TARGET Libevent::pthreads)
        add_library(Libevent::pthreads UNKNOWN IMPORTED)
        set_target_properties(Libevent::pthreads PROPERTIES
            IMPORTED_LOCATION "${Libevent_pthreads_LIBRARY}"
            INTERFACE_LINK_LIBRARIES Libevent::core)
    endif()

    # TODO: Remove legacy targets and variables. Update call sites

    # Legacy variables
    set(LIBEVENT_INCLUDE_DIR "${Libevent_INCLUDE_DIRS}")
    set(LIBEVENT_STATIC_LIB
        "${Libevent_core_LIBRARY}"
        "${Libevent_extra_LIBRARY}")
    set(LIBEVENT_PTHREAD_STATIC_LIB "${Libevent_pthreads_LIBRARY}")
    set(LIBEVENT_STATIC_LIBS libevent_event_static libevent_event_pthread)

    # Legacy Targets
    if (NOT TARGET libevent_event_static)
        add_library(libevent_event_static UNKNOWN IMPORTED)
        set_target_properties(libevent_event_static PROPERTIES
            IMPORTED_LOCATION "${Libevent_core_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "${Libevent_extra_LIBRARY}")
    endif()

    if (NOT TARGET libevent_event_pthread)
        add_library(libevent_event_pthread UNKNOWN IMPORTED)
        set_target_properties(libevent_event_pthread PROPERTIES IMPORTED_LOCATION ${LIBEVENT_PTHREAD_STATIC_LIB})
    endif()
endif()
