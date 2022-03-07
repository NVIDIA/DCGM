#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

function (get_absolute_path pathVal resultAbsPath)
        execute_process(COMMAND readlink -enq "${pathVal}" OUTPUT_VARIABLE LOCAL_RESULT)
        set(${resultAbsPath} ${LOCAL_RESULT} PARENT_SCOPE)
endfunction()

function (find_library_ex nameVal valPrefix)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${nameVal}
        OUTPUT_VARIABLE LIBNAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    #file(REAL_PATH "${LIBNAME}" REALLIBNAME)
    get_absolute_path(${LIBNAME} REALLIBNAME)
    get_filename_component(LIBNAME "${REALLIBNAME}" NAME)
    message(DEBUG "${nameVal} file is ${REALLIBNAME}")

    execute_process(
        COMMAND bash -c "objdump -p '${REALLIBNAME}' | grep SONAME | sed 's/^.*SONAME[ ]*//'"
        OUTPUT_VARIABLE LIB_SONAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(DEBUG "${nameVal} SONAME is ${LIB_SONAME}")

    set ("${valPrefix}_SONAME" ${LIB_SONAME} PARENT_SCOPE)
    set ("${valPrefix}_ABSPATH" ${REALLIBNAME} PARENT_SCOPE)
    set ("${valPrefix}_LIBNAME" ${LIBNAME} PARENT_SCOPE)
endfunction()