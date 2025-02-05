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

include(utils)

function (add_3rd_party_library condition libName installDir)
    if (${${condition}})
        find_library_ex(${libName} LIB)
        message(STATUS "${libName} LIB_SONAME  = ${LIB_SONAME}")
        message(STATUS "${libName} LIB_ABSPATH = ${LIB_ABSPATH}")
        message(STATUS "${libName} LIB_LIBNAME = ${LIB_LIBNAME}")


        add_custom_target(make${libName}_symlink ALL
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIB_SONAME} ${libName}
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIB_LIBNAME} ${LIB_SONAME}
            BYPRODUCTS ${libName} ${LIB_SONAME} ${LIB_LIBNAME}
        )

        install(
            PROGRAMS
                "${LIB_ABSPATH}"
                "${CMAKE_CURRENT_BINARY_DIR}/${LIB_SONAME}"
                "${CMAKE_CURRENT_BINARY_DIR}/${libName}"
            DESTINATION ${installDir}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            COMPONENT Tests
        )
    endif()
endfunction()


add_3rd_party_library(ANY_SANITIZER "libstdc++.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(ADDRESS_SANITIZER "libasan.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(THREAD_SANITIZER "libtsan.so" ${DCGM_TESTS_APP_DIR})
add_3rd_party_library(LEAK_SANITIZER "liblsan.so" ${DCGM_TESTS_APP_DIR})