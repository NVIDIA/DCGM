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

if (NOT TARGET JsonCpp::JsonCpp)
    find_package(jsoncpp REQUIRED CONFIG)
    set(JSONCPP_STATIC_LIBS jsoncpp_static)
    set(JSONCPP_INCLUDE_PATH $<TARGET_PROPERTY:jsoncpp_static,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
# set(Jsoncpp_PATH_PREFIXES /usr/local "${Jsoncpp_ROOT}" "$ENV{HOME}")
# foreach(prefix ${Jsoncpp_PATH_PREFIXES})
#     list(APPEND Jsoncpp_INCLUDE_PATHS ${prefix}/include)
#     list(APPEND Jsoncpp_LIB_PATHS ${prefix}/lib ${prefix}/lib64)
# endforeach()

# find_path(JSONCPP_INCLUDE_PATH json/json.h PATHS ${Jsoncpp_INCLUD_PATHS})
# find_file(JSONCPP_STATIC_LIB NAMES libjsoncpp.a PATHS ${Jsoncpp_LIB_PATHS})

# if(JSONCPP_INCLUDE_PATH AND JSONCPP_STATIC_LIB)
#     set(Jsoncpp_FOUND TRUE)
#     add_library(libjsoncpp STATIC IMPORTED)
#     set_target_properties(libjsoncpp PROPERTIES IMPORTED_LOCATION ${JSONCPP_STATIC_LIB})
#     target_include_directories(libjsoncpp INTERFACE ${JSONCPP_INCLUDE_PATH})
#     set(JSONCPP_STATIC_LIBS libjsoncpp)
# else()
#     set(Jsoncpp_FOUND FALSE)
# endif()

# if (Jsoncpp_FOUND)
#     if (NOT Jsoncpp_FIND_QUIETLY)
#         message(STATUS "Found libjsoncpp: ${JSONCPP_STATIC_LIBS}")
#     endif ()
# else ()
#     if (Jsoncpp_FIND_REQUIRED)
#         message(FATAL_ERROR "Could NOT find libjsoncpp")
#     endif ()
#     message(STATUS "libjsoncpp NOT found")
# endif ()

# unset(Jsoncpp_PATH_PREFIXES)
# unset(Jsoncpp_INCLUDE_PATHS)
# unset(Jsoncpp_LIB_PATHS)
# unset(JSONCPP_STATIC_LIB)

mark_as_advanced(JSONCPP_INCLUDE_PATH JSONCPP_STATIC_LIBS)
