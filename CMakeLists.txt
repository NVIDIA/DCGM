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
cmake_minimum_required(VERSION 3.21.0)
project(dcgm VERSION 2.3.4)
set(DCGM_CONFIG_REVISION 0)
set(DCGM_ROOT_DIR ${PROJECT_SOURCE_DIR})

set(DCGM_PKG "datacenter-gpu-manager" CACHE STRING "DCGM package names base")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build Position Independent Code")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

set(DCGM_CXX_FLAGS "-Werror -Wpedantic -Wno-sign-compare -Wno-unused-parameter -Wall -Wextra")

include(CheckCCompilerFlag)
check_c_compiler_flag("-Wno-cast-function-type" CC_HAS_WNO_CAST_FUNCTION_TYPE)
if (CC_HAS_WNO_CAST_FUNCTION_TYPE)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -Wno-cast-function-type")
endif()

option(ENABLE_LTO "Enable LTO during the build" OFF)
if (ENABLE_LTO)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -flto")
endif()

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)
if (ADDRESS_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=address")
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize-address-use-after-scope")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize=address")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize-address-use-after-scope")
    set(ANY_SANITIZER TRUE)
endif()

option(THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
if (THREAD_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=thread")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize=thread")
    set(ANY_SANITIZER TRUE)
endif()

option(UB_SANITIZER "Enable UndefinedSanitizer" OFF)
if (UB_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=undefined")
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize-undefined-trap-on-error")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize=undefined")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize-undefined-trap-on-error")
    set(ANY_SANITIZER TRUE)
endif()

option(LEAK_SANITIZER "Enable LeakSanitizer" OFF)
if (LEAK_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=leak")
    set(DCGM_C_FLAGS "${DCGM_C_FLAGS} -fsanitize=leak")
    set(ANY_SANITIZER TRUE)
endif()

if (ANY_SANITIZER)
    add_compile_definitions(SANITIZERS=1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -g -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DCGM_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Werror -g -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DCGM_CXX_FLAGS} -g -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -s -O3 -g -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DCGM_CXX_FLAGS} -s -O3 -g -fno-omit-frame-pointer")
endif()

if(DCGM_BUILD_COVERAGE STREQUAL "1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(NV_UNIX=1 NV_LINUX=1 _UNIX=1)
add_compile_definitions(DRIVER_MAJOR_VERSION="418")
add_compile_definitions(DRIVER_VERSION="418")
add_compile_definitions(DCGM_BUILD_VGPU_MODULE)
if(${CMAKE_POSITION_INDEPENDENT_CODE})
    add_compile_options(-fPIC)
endif()

if (NOT ANY_SANITIZER)
    add_link_options(-static-libstdc++)
endif()

add_link_options(-Wl,--exclude-libs,libstdc++ -Wl,--exclude-libs,libprotobuf)

include(GNUInstallDirs)

if (NOT DEFINED DCGM_LIB_INSTALL_PREFIX)
    set(DCGM_LIB_INSTALL_PREFIX "${CMAKE_INSTALL_LIBDIR}")
endif()

if (NOT DEFINED DCGM_INCLUDE_INSTALL_PREFIX)
    set(DCGM_INCLUDE_INSTALL_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

mark_as_advanced(DCGM_LIB_INSTALL_PREFIX)
message("DCGM_LIB_INSTALL_PREFIX=${DCGM_LIB_INSTALL_PREFIX}")

set(DCGM_RPATH "$ORIGIN/:$ORIGIN/../${DCGM_LIB_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH ${DCGM_RPATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${DCGM_RPATH}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${DCGM_RPATH}")

set(DCGM_NVVS_CONFIG_DIR "${CMAKE_INSTALL_SYSCONFDIR}/nvidia-validation-suite")
set(DCGM_NVVS_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/nvidia-validation-suite")
set(DCGM_NVVS_PLUGINS_INSTALL_DIR "${DCGM_NVVS_INSTALL_DIR}/plugins")
set(DCGM_TESTS_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/dcgm_tests")
set(DCGM_BINDINGS_INSTALL_DIR "local/dcgm/bindings")
set(DCGM_DOCS_INSTALL_DIR "local/dcgm")
set(DCGM_SAMPLES_INSTALL_DIR "local/dcgm/sdk_samples")

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(DCGM_TESTS_ARCH "amd64")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
    set(DCGM_TESTS_ARCH "ppc64le")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(DCGM_TESTS_ARCH "aarch64")
else()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Unsupported architecture")
endif()

message(DEBUG "DCGM_TESTS_ARCH=${DCGM_TESTS_ARCH}")

set(DCGM_TESTS_APP_DIR "${DCGM_TESTS_INSTALL_DIR}/apps/${DCGM_TESTS_ARCH}")
set(PYTHON_VER python3)

find_package(Git QUIET REQUIRED)

# Let CMake rebuild cache every time git commit is changed
set_property(GLOBAL APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/index")

# Git Commit -> ${COMMIT_ID}
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                OUTPUT_VARIABLE COMMIT_ID
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Git branch -> ${BUILD_BRANCH}
# Jenkins does some weird checkouts so it ends up in a detached HEAD state.
# The command below will
# - grab all refs that are related to the current commit
# - filter out detached HEAD
# - use rel_dcgm_* if found
# - use main or master if found
# - use the first branch we find otherwise
execute_process(COMMAND "./scripts/get_build_branch.py" ${COMMIT_ID}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                OUTPUT_VARIABLE BUILD_BRANCH)
# Git has uncommitted changes -> ${HAS_UNCOMMITTED_CHANGES}
execute_process(COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE HAS_UNCOMMITTED_CHANGES)
# Build Platform -> ${BUILDPLATFORM}
execute_process(COMMAND uname -srvm
                OUTPUT_VARIABLE BUILDPLATFORM
                OUTPUT_STRIP_TRAILING_WHITESPACE)
# Build Date in UTC -> ${BUILD_DATE}
execute_process(COMMAND date -u -Idate
                OUTPUT_VARIABLE BUILD_DATE
                OUTPUT_STRIP_TRAILING_WHITESPACE)

add_compile_definitions(BUILD_BRANCH="${BUILD_BRANCH}")

set(BUILDBRANCH ${BUILD_BRANCH})
set(DRIVERVERSION 418)
set(BUILDDATE ${BUILD_DATE})

if (NOT HAS_UNCOMMITTED_CHANGES EQUAL 0)
    set(COMMIT_ID "${COMMIT_ID}(dirty)")

    if ("$ENV{PRINT_UNCOMMITTED_CHANGES}" STREQUAL "1")
        execute_process(COMMAND "${GIT_EXECUTABLE}" diff-index -p HEAD --
                        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                        OUTPUT_VARIABLE INDEX_DIFF
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        message("Git index diff:\n${INDEX_DIFF}")
    endif()
endif()

list(APPEND DCGM_BUILD_INFO "version:${CMAKE_PROJECT_VERSION}")
list(APPEND DCGM_BUILD_INFO "arch:${CMAKE_SYSTEM_PROCESSOR}")
list(APPEND DCGM_BUILD_INFO "buildtype:${CMAKE_BUILD_TYPE}")
list(APPEND DCGM_BUILD_INFO "buildid:$ENV{BUILD_NUMBER}")
list(APPEND DCGM_BUILD_INFO "builddate:${BUILD_DATE}")
list(APPEND DCGM_BUILD_INFO "commit:${COMMIT_ID}")
list(APPEND DCGM_BUILD_INFO "branch:${BUILD_BRANCH}")
list(APPEND DCGM_BUILD_INFO "buildplatform:${BUILDPLATFORM}")
list(APPEND DCGM_BUILD_INFO "$ENV{DCGM_BUILD_INFO}")

execute_process(COMMAND bash -c "echo '${DCGM_BUILD_INFO}' | md5sum - | awk '{print $1}'" OUTPUT_VARIABLE BUILD_INFO_CRC OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND DCGM_BUILD_INFO "crc:${BUILD_INFO_CRC}")

include(artifactory)
prepare_rt_props("dcgm")
prepare_rt_props("config" REVISION "${DCGM_CONFIG_REVISION}")

message("${DCGM_BUILD_INFO}")

mark_as_advanced(DCGM_NVVS_CONFIG_DIR, DCGM_NVVS_INSTALL_DIR, DCGM_NVVS_PLUGINS_INSTALL_DIR)

string(TIMESTAMP BUILD_DATE "%Y%m%d")
add_compile_definitions(BUILD_DATE="${BUILD_DATE}")

macro(update_lib_ver libname)
    set_target_properties(
        ${libname}
        PROPERTIES
            VERSION
            ${PROJECT_VERSION}
            SOVERSION
            ${PROJECT_VERSION_MAJOR}
    )
endmacro()

define_property(TARGET PROPERTY INSTALL_TO_DIR BRIEF_DOCS "Installation dir" FULL_DOCS "Installation dir")

enable_testing()

add_subdirectory(docs)
add_subdirectory(sdk)
add_subdirectory(cublas_proxy)
add_subdirectory(dcgmlib)
add_subdirectory(common)
add_subdirectory(nvvs)
add_subdirectory(modules)
add_subdirectory(dcgmlib/src)

add_subdirectory(modules/common)
add_subdirectory(modules/introspect)
add_subdirectory(modules/health)
add_subdirectory(modules/policy)
add_subdirectory(modules/profiling)
add_subdirectory(modules/config)
add_subdirectory(modules/diag)
add_subdirectory(modules/nvswitch)
add_subdirectory(modules/vgpu)

add_subdirectory(hostengine)
# add_subdirectory(cuda_loader)
add_subdirectory(nvvs/plugin_src)
add_subdirectory(nvvs/src)
add_subdirectory(dcgmi)
add_subdirectory(testing)
add_subdirectory(dcgmproftester)
add_subdirectory(dcgm_stub)

add_subdirectory(testing/python3/apps/cuda_ctx_create)
add_subdirectory(sdk_samples)

add_subdirectory(config-files)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt
    COMMAND bash -c "find ${CMAKE_SOURCE_DIR} -iname '*.cpp' -or -iname '*.c' -not -ipath '*_out*' -exec grep -n -H -B1 -A4 -e '^\\sPRINT_' -e 'DEBUG_' {} \; > ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt"
    VERBATIM
)
add_custom_target(dcgm_decode_db ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt)

set(DCGM_PRIVATE_DIR "${CMAKE_SOURCE_DIR}/dcgm_private")
if (EXISTS ${DCGM_PRIVATE_DIR})
    add_subdirectory(${DCGM_PRIVATE_DIR})
endif()


install(
    TARGETS
        dcgm
        dcgmi
        nv-hostengine
        dcgmmoduleconfig
        dcgmmodulediag
        dcgmmodulehealth
        dcgmmoduleintrospect
        dcgmmodulenvswitch
        dcgmmodulepolicy
        ${DCGM_CUBLAS_PROXY}
        ${DCGMPROFTESTER}
    ARCHIVE DESTINATION ${DCGM_TESTS_APP_DIR} COMPONENT Tests
    LIBRARY DESTINATION ${DCGM_TESTS_APP_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT Tests
    RUNTIME DESTINATION ${DCGM_TESTS_APP_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT Tests
)

include(sanitizers)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt DESTINATION ${DCGM_TESTS_INSTALL_DIR}/data COMPONENT Tests)

foreach(plugin ${NVVS_PLUGINS})
    get_target_property(INSTALL_TO_DIR ${plugin} INSTALL_TO_DIR)
    string(
        REPLACE
            "${DCGM_NVVS_PLUGINS_INSTALL_DIR}"
            "${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins"
            INSTALL_TO
            ${INSTALL_TO_DIR}
)
    install(TARGETS ${plugin}
        LIBRARY DESTINATION ${INSTALL_TO}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT Tests)
endforeach()

set_target_properties(nvvs PROPERTIES INSTALL_RPATH "${DCGM_RPATH}:$ORIGIN/../../lib:$ORIGIN/../${DCGM_TESTS_ARCH}")
install(TARGETS nvvs RUNTIME DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/ COMPONENT Tests)

install(
    TARGETS
        dcgm
        dcgmi
        nv-hostengine
        dcgmmoduleconfig
        dcgmmodulediag
        dcgmmodulehealth
        dcgmmoduleintrospect
        dcgmmodulenvswitch
        dcgmmodulepolicy
        ${DCGM_CUBLAS_PROXY}
        ${DCGMPROFTESTER}
    EXPORT dcgm-export
    ARCHIVE DESTINATION ${DCGM_LIB_INSTALL_PREFIX} COMPONENT DCGM
    LIBRARY DESTINATION ${DCGM_LIB_INSTALL_PREFIX}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT DCGM
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT DCGM
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT DCGM
)

install(
    EXPORT dcgm-export
    FILE dcgmConfig.cmake
    NAMESPACE Dcgm::
    DESTINATION ${DCGM_LIB_INSTALL_PREFIX}/cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    COMPONENT DCGM
)

install(TARGETS nvvs RUNTIME DESTINATION ${DCGM_NVVS_INSTALL_DIR} COMPONENT DCGM)

install(
    FILES
        LICENSE
    DESTINATION ${DCGM_DOCS_INSTALL_DIR}
    COMPONENT DCGM
)

if (ANY_SANITIZER)
    set(CPACK_COMPONENTS_ALL Tests)
else()
set(CPACK_COMPONENTS_ALL DCGM Tests)
endif()

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    set(CPACK_TGZ_PACKAGE_ARCHITECTURE "amd64")

elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")

    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "ppc64el")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "ppc64le")
    set(CPACK_TGZ_PACKAGE_ARCHITECTURE "ppc64le")

elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")

    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
    set(CPACK_TGZ_PACKAGE_ARCHITECTURE "aarch64")

endif()

if (DEFINED DCGM_PACKAGING_ENGINE)
    if(${DCGM_PACKAGING_ENGINE} STREQUAL "RPM")
        install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/${DCGM_PKG}-${CMAKE_PROJECT_VERSION}/ COMPONENT DCGM)
    elseif(${DCGM_PACKAGING_ENGINE} STREQUAL "DEB")
        install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${DCGM_PKG} RENAME copyright COMPONENT DCGM)
    endif()
endif()

set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_SOURCE_DIR}/cmake/packaging.cmake")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/dcgm_config/CMakeLists.txt.in
    ${CMAKE_CURRENT_BINARY_DIR}/dcgm_config/CMakeLists.txt
    ESCAPE_QUOTES
    @ONLY)

include(CPack)
