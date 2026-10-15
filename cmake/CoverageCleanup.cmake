# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#[[
Register gcov-compatible and LLVM source-based coverage files with CMake's
clean target.

The registration is deferred until the end of top-level directory processing
so targets declared in child directories are included.

LLVM_PROFILE_FILE may contain runtime substitutions such as %p and %m. CMake's
ADDITIONAL_CLEAN_FILES property cannot enumerate those dynamic names, so such
profiles should be placed in a dedicated directory that is itself registered
for cleanup. This module registers LLVM's default profile names and common
object- and target-derived names.
]]

include_guard(GLOBAL)

function(_coverage_cleanup_register_directory directory)
  get_property(binary_directory DIRECTORY "${directory}" PROPERTY BINARY_DIR)
  set_property(DIRECTORY "${directory}" APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    "${binary_directory}/default.profraw"
    "${binary_directory}/default.profdata")

  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  set(compiled_target_types
    EXECUTABLE
    STATIC_LIBRARY
    SHARED_LIBRARY
    MODULE_LIBRARY
    OBJECT_LIBRARY)

  foreach(target IN LISTS targets)
    get_target_property(target_type "${target}" TYPE)
    if(target_type IN_LIST compiled_target_types)
      set_property(TARGET "${target}" APPEND PROPERTY ADDITIONAL_CLEAN_FILES
        "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_OBJECTS:${target}>,.gcno>"
        "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_OBJECTS:${target}>,.gcda>"
        "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_OBJECTS:${target}>,.profraw>"
        "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_OBJECTS:${target}>,.profdata>")

      if(NOT target_type STREQUAL "OBJECT_LIBRARY")
        set_property(TARGET "${target}" APPEND PROPERTY ADDITIONAL_CLEAN_FILES
          "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_FILE:${target}>,.profraw>"
          "$<PATH:REPLACE_EXTENSION,LAST_ONLY,$<TARGET_FILE:${target}>,.profdata>")
      endif()
    endif()
  endforeach()

  get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    _coverage_cleanup_register_directory("${subdirectory}")
  endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL _coverage_cleanup_register_directory "${CMAKE_SOURCE_DIR}")
