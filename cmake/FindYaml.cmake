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

find_package(yaml-cpp REQUIRED CONFIG)
set(YAML_STATIC_LIBS ${YAML_CPP_LIBRARIES})
set(YAML_INCLUDE_PATH ${YAML_CPP_INCLUDE_DIR})
# set(Yaml_PATH_PREFIXES /usr/local "${Yaml_ROOT}" "$ENV{HOME}")
# foreach(prefix ${Yaml_PATH_PREFIXES})
#     list(APPEND Yaml_INCLUDE_PATHS ${prefix}/include)
#     list(APPEND Yaml_LIB_PATHS ${prefix}/lib ${prefix}/lib64)
# endforeach()

# find_path(YAML_INCLUDE_PATH yaml-cpp/yaml.h PATHS ${Yaml_INCLUD_PATHS})
# find_file(YAML_STATIC_LIB NAMES libyaml-cpp.a PATHS ${Yaml_LIB_PATHS})

# if(YAML_INCLUDE_PATH AND YAML_STATIC_LIB)
#     set(Yaml_FOUND TRUE)
#     add_library(libyamlcpp STATIC IMPORTED)
#     set_target_properties(libyamlcpp PROPERTIES IMPORTED_LOCATION ${YAML_STATIC_LIB})
#     set(YAML_STATIC_LIBS libyamlcpp)
# else()
#     set(Yaml_FOUND FALSE)
# endif()

# if (Yaml_FOUND)
#     if (NOT Yaml_FIND_QUIETLY)
#         message(STATUS "Found libyaml-cpp: ${YAML_STATIC_LIBS}")
#     endif ()
# else ()
#     if (Yaml_FIND_REQUIRED)
#         message(FATAL_ERROR "Could NOT find libyaml-cpp")
#     endif ()
#     message(STATUS "libyaml-cpp NOT found.")
# endif ()

# unset(Yaml_PATH_PREFIXES)
# unset(Yaml_INCLUDE_PATHS)
# unset(Yaml_LIB_PATHS)
# unset(YAML_STATIC_LIB)

mark_as_advanced(YAML_INCLUDE_PATH YAML_STATIC_LIBS)
