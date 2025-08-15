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

cmake_minimum_required(VERSION 3.21.0)

project(datacenter-gpu-manager-4 VERSION 4.3.1)

set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF CACHE BOOL
    "Specify whether to link the target in the build tree with the INSTALL_RPATH")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CTEST_USE_LAUNCHERS ON CACHE BOOL [[
When enabled, the generated build system wraps each invocation of the compiler,
linker, or custom command line with a "launcher" that communicates with CTest
via environment variables and files to report granular build warning and error
information. Otherwise, CTest must "scrape" the build output log for
diagnostics.
]])

find_package(Boost REQUIRED COMPONENTS filesystem stacktrace_basic system)
find_package(Catch2 REQUIRED)
find_package(Cuda REQUIRED)
find_package(fmt REQUIRED)
find_package(Git REQUIRED)
find_package(Jsoncpp REQUIRED)
find_package(Libevent REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
find_package(Yaml REQUIRED)

include(Catch)
include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

set(DCGM_CONFIG_REVISION 0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build Position Independent Code")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

set(BUILD_NUMBER "$ENV{BUILD_NUMBER}" CACHE STRING
    "The build number associated with the generated build system")

option(ENABLE_LTO "Enable LTO during the build" OFF)
if (ENABLE_LTO)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -flto")
endif ()

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)
if (ADDRESS_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope")
    set(ANY_SANITIZER TRUE)
endif ()

option(THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
if (THREAD_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=thread")
    set(ANY_SANITIZER TRUE)
endif ()

option(UB_SANITIZER "Enable UndefinedSanitizer" OFF)
if (UB_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=undefined -fsanitize-undefined-trap-on-error")
    set(ANY_SANITIZER TRUE)
endif ()

option(LEAK_SANITIZER "Enable LeakSanitizer" OFF)
if (LEAK_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=leak")
    set(ANY_SANITIZER TRUE)
endif ()

if (ANY_SANITIZER)
    add_compile_definitions(SANITIZERS=1)
endif ()

if (DCGM_BUILD_COVERAGE)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} --coverage")
endif ()

set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fmax-errors=1 -Werror -fno-omit-frame-pointer")

# Set common CXXFLAGS as a base for CFLAGS
set(DCGM_C_FLAGS ${DCGM_CXX_FLAGS})

string(JOIN " " DCGM_CXX_FLAGS
    "${DCGM_CXX_FLAGS}"
    -Wdangling-reference
    -Winvalid-constexpr
    -Wsign-promo
    -Wplacement-new=2
    -Wcatch-value=2
    -Wextra-semi
    -Wsuggest-override
    -Wpedantic
    -Wall
    -Wextra)

option(GCC_ANALYZER "Enable GCC static analyzer" OFF)
if (GCC_ANALYZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fanalyzer")
endif ()

include(CheckCCompilerFlag)

# Set base flags but allow to override from the environment
set(CMAKE_C_FLAGS "${DCGM_C_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${DCGM_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

# Customize flags per CMake build type
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s -g")

option(VMWARE "Build for VMWare environment" OFF)
if (VMWARE)
    add_compile_definitions(NV_VMWARE=1)
endif ()

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

add_compile_definitions(
    DCGM_BUILD_VGPU_MODULE
    DRIVER_MAJOR_VERSION="418"
    DRIVER_VERSION="418"
    NV_LINUX=1
    NV_UNIX=1
    PLOG_LOCAL
    _UNIX=1)

if (NOT ANY_SANITIZER)
    add_link_options(-static-libstdc++)
endif ()

add_link_options(-Wl,--exclude-libs,libstdc++)

set(DCGM_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_INSTALL_RPATH ${DCGM_RPATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(DCGM_TESTS_ARCH "amd64")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(DCGM_TESTS_ARCH "aarch64")
else ()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Unsupported architecture")
endif ()

message(DEBUG "DCGM_TESTS_ARCH=${DCGM_TESTS_ARCH}")

set(DCGM_NVVS_PLUGINS_INSTALL_DIR "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins")
set(DCGM_TESTS_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/dcgm_tests")
set(DCGM_TESTS_APP_DIR "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}")

# Let CMake rebuild cache every time git commit is changed
set_property(GLOBAL APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/index")

# Git Commit -> ${COMMIT_ID}
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE COMMIT_ID
    RESULT_VARIABLE GIT_EXIT_CODE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (GIT_EXIT_CODE EQUAL 0)
    # Git branch -> ${BUILD_BRANCH}
    # Jenkins does some weird checkouts so it ends up in a detached HEAD state.
    # The command below will
    # - grab all refs that are related to the current commit
    # - filter out detached HEAD
    # - use rel_dcgm_* if found
    # - use main or master if found (RELINGO_IGNORE until DCGM-1855 is done)
    # - use the first branch we find otherwise
    execute_process(COMMAND "./scripts/get_build_branch.py" ${COMMIT_ID}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE BUILD_BRANCH)
    # Git has uncommitted changes -> ${HAS_UNCOMMITTED_CHANGES}
    execute_process(COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE HAS_UNCOMMITTED_CHANGES)
else ()
    message(STATUS "The working directory is not a Git repository")
    set(BUILD_BRANCH "non-branched")
    set(HAS_UNCOMMITED_CHANGES 1)
endif ()

# Build Platform -> ${BUILDPLATFORM}
execute_process(COMMAND uname -srvm
    OUTPUT_VARIABLE BUILDPLATFORM
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Build Date in UTC -> ${BUILD_DATE}
execute_process(COMMAND date -u -Idate
    OUTPUT_VARIABLE BUILD_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

add_compile_definitions(BUILD_BRANCH="${BUILD_BRANCH}")

if (NOT HAS_UNCOMMITTED_CHANGES EQUAL 0)
    string(APPEND COMMIT_ID "(dirty)")

    if ("$ENV{PRINT_UNCOMMITTED_CHANGES}" STREQUAL "1")
        execute_process(COMMAND "${GIT_EXECUTABLE}" diff-index -p HEAD --
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE INDEX_DIFF
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        message("Git index diff:\n${INDEX_DIFF}")
    endif ()
endif ()

set(DCGM_BUILD_INFO
    "version:${CMAKE_PROJECT_VERSION}"
    "arch:${CMAKE_SYSTEM_PROCESSOR}"
    "buildtype:${CMAKE_BUILD_TYPE}"
    "buildid:${BUILD_NUMBER}"
    "builddate:${BUILD_DATE}"
    "commit:${COMMIT_ID}"
    "branch:${BUILD_BRANCH}"
    "buildplatform:${BUILDPLATFORM}"
    "$ENV{DCGM_BUILD_INFO}")

execute_process(
    COMMAND bash -c "echo '${DCGM_BUILD_INFO}' | md5sum - | awk '{print $1}'"
    OUTPUT_VARIABLE BUILD_INFO_CRC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND DCGM_BUILD_INFO "crc:${BUILD_INFO_CRC}")

include(artifactory)
prepare_rt_props("dcgm")
prepare_rt_props("config" REVISION "${DCGM_CONFIG_REVISION}")

message("${DCGM_BUILD_INFO}")

string(TIMESTAMP BUILD_DATE "%Y%m%d")
add_compile_definitions(BUILD_DATE="${BUILD_DATE}")

add_compile_definitions(INJECTION_LIBRARY_AVAILABLE)

add_library(nvml_injection SHARED)
add_library(nvmli_interface INTERFACE)
add_library(nvmli_public OBJECT)

set(LINK_INJECTION_LIB "true")

if (BUILD_TESTING)
    add_library(testing_cuda_stubs STATIC)
    add_library(testing_cuda_stubs_interface INTERFACE)
    add_library(testing_dcgm_cublas_stubs STATIC)
endif()

add_executable(BwChecker_11)
add_executable(BwChecker_12)
add_executable(dcgmi)
add_executable(dcgmproftester11)
add_executable(dcgmproftester12)
add_executable(nv-hostengine)
add_executable(nvvs)
add_executable(stub_library_test)

add_library(config_interface INTERFACE)
add_library(diag_interface INTERFACE)
add_library(health_interface INTERFACE)
add_library(introspect_interface INTERFACE)
add_library(mndiag_interface INTERFACE)
add_library(module_common_interface INTERFACE)
add_library(modules_interface INTERFACE)
add_library(nvswitch_interface INTERFACE)
add_library(policy_interface INTERFACE)
add_library(public_profiling_interface INTERFACE)
add_library(sysmon_interface INTERFACE)

add_library(childprocess_interface INTERFACE)
add_library(common_interface INTERFACE)
add_library(mock_interface INTERFACE)
add_library(cuda_mock_interface INTERFACE)
add_library(cuda_lib_base_interface INTERFACE)
add_library(cuda_common_interface INTERFACE)
add_library(dcgm_interface INTERFACE)
add_library(dcgm_thread_interface INTERFACE)
add_library(dcgmi_interface INTERFACE)
add_library(dcgmtest_interface INTERFACE)
add_library(nvvs_interface INTERFACE)
add_library(nvvs_plugins INTERFACE)
add_library(nvvs_plugins_interface INTERFACE)
add_library(nvvs_plugins_cudaCommon_interface INTERFACE)
add_library(nvvs_plugins_targetedstress_interface INTERFACE)
add_library(nvvs_plugins_targetedpower_interface INTERFACE)
add_library(nvvs_plugins_gpuburn_interface INTERFACE)
add_library(nvvs_plugins_busgrind_interface INTERFACE)
add_library(nvvs_plugins_nvbandwidth_interface INTERFACE)
add_library(nvvs_plugins_memtest_interface INTERFACE)
add_library(nvvs_plugins_memory_interface INTERFACE)
add_library(nvvs_plugins_contextcreate_interface INTERFACE)
add_library(nvvs_plugins_common_interface INTERFACE)
add_library(public_vgpu_interface INTERFACE)
add_library(sdk_nscq_interface INTERFACE)
add_library(sdk_nscq_loader_interface INTERFACE)
add_library(sdk_nvidia INTERFACE)
add_library(sdk_nvml_interface INTERFACE)
add_library(sdk_nvsdm_interface INTERFACE)
add_library(sdk_nvsdm_loader_interface INTERFACE)
add_library(serialize_interface INTERFACE)
add_library(transport_interface INTERFACE)

add_library(buildinfo_objects STATIC)
add_library(childprocess STATIC)
add_library(common_watch_objects STATIC)
add_library(dcgm_common STATIC)
add_library(dcgm_cuda_worker11 STATIC)
add_library(dcgm_cuda_worker12 STATIC)
add_library(dcgm_cuda_lib11 STATIC)
add_library(dcgm_cuda_lib12 STATIC)
add_library(dcgm_entity_types STATIC)
add_library(dcgm_logging STATIC)
add_library(dcgm_mutex STATIC)
add_library(dcgm_static_private STATIC)
add_library(dcgm_stub STATIC)
add_library(dcgm_thread STATIC)
add_library(dcgmi_objects STATIC)
add_library(dcgmi_objects_without_main STATIC)
add_library(nvvs_without_main_objects OBJECT)
add_library(nvvs_main_objects OBJECT)
add_library(nvvs_plugins_common_objects OBJECT)
add_library(pluginCudaCommon_11 STATIC)
add_library(pluginCudaCommon_12 STATIC)
add_library(remoteconn STATIC)
add_library(sdk_nvml_essentials_objects STATIC)
add_library(sdk_nvml_loader STATIC)
add_library(serialize STATIC)
add_library(transport_objects STATIC)

add_library(dcgmmodulediag_common STATIC)
add_library(dcgmmodulediag_private_static STATIC)
add_library(dcgmmodulemndiag_common STATIC)
add_library(dcgmmodulemndiag_private_static STATIC)
add_library(module_common_objects STATIC)
add_library(modules_objects STATIC)
add_library(nvswitch_objects STATIC)
add_library(sysmon_objects STATIC)

add_library(dcgmmoduleconfig SHARED)
add_library(dcgmmodulediag SHARED)
add_library(dcgmmodulehealth SHARED)
add_library(dcgmmoduleintrospect SHARED)
add_library(dcgmmodulemndiag SHARED)
add_library(dcgmmodulenvswitch SHARED)
add_library(dcgmmodulepolicy SHARED)
add_library(dcgmmodulesysmon SHARED)

add_library(ContextCreate_11 SHARED)
add_library(ContextCreate_12 SHARED)
add_library(Diagnostic_11 SHARED)
add_library(Diagnostic_12 SHARED)
add_library(Memory_11 SHARED)
add_library(Memory_12 SHARED)
add_library(Memtest_11 SHARED)
add_library(Memtest_12 SHARED)
add_library(NVBandwidth_12 SHARED)
add_library(Pcie_11 SHARED)
add_library(Pcie_12 SHARED)
add_library(TargetedPower_11 SHARED)
add_library(TargetedPower_12 SHARED)
add_library(TargetedStress_11 SHARED)
add_library(TargetedStress_12 SHARED)

add_library(dcgm SHARED)
add_library(dcgm_cublas_proxy11 SHARED)
add_library(dcgm_cublas_proxy12 SHARED)
add_library(pluginCommon SHARED)

add_library(sdk_nscq_loader)
add_library(sdk_nvsdm_loader)

add_library(DCGM::dcgm ALIAS dcgm)

set_target_properties(
    ContextCreate_11
    ContextCreate_12
    Diagnostic_11
    Diagnostic_12
    Memory_11
    Memory_12
    Memtest_11
    Memtest_12
    NVBandwidth_12
    Pcie_11
    Pcie_12
    TargetedPower_11
    TargetedPower_12
    TargetedStress_11
    TargetedStress_12
    dcgm
    dcgmmoduleconfig
    dcgmmodulediag
    dcgmmodulehealth
    dcgmmoduleintrospect
    dcgmmodulemndiag
    dcgmmodulenvswitch
    dcgmmodulepolicy
    dcgmmodulesysmon
    dcgm_cublas_proxy11
    dcgm_cublas_proxy12
    pluginCommon
    PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

set_target_properties(nvml_injection PROPERTIES SOVERSION 1 VERSION 1.0.0)

set_target_properties(nvvs PROPERTIES
    INSTALL_RPATH "${DCGM_RPATH}:$ORIGIN/../../lib:$ORIGIN/../${DCGM_TESTS_ARCH}"
    RUNTIME_OUTPUT_DIRECTORY nvvs)

set_target_properties(
    BwChecker_11
    BwChecker_12
    PROPERTIES
        INSTALL_RPATH "${DCGM_RPATH}:$ORIGIN/../../../${DCGM_TESTS_ARCH}")

set_target_properties(dcgmi PROPERTIES RUNTIME_OUTPUT_DIRECTORY dcgmi)

set_target_properties(
    ContextCreate_11
    ContextCreate_12
    Diagnostic_11
    Diagnostic_12
    Memory_11
    Memory_12
    Memtest_11
    Memtest_12
    NVBandwidth_12
    Pcie_11
    Pcie_12
    TargetedPower_11
    TargetedPower_12
    TargetedStress_11
    TargetedStress_12
    nvml_injection
    nvmli_public
    nvvs_without_main_objects
    nvvs_plugins_common_objects
    pluginCommon
    pluginCudaCommon_11
    pluginCudaCommon_12
    PROPERTIES
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN FALSE)

add_subdirectory(common)
add_subdirectory(config-files)
add_subdirectory(cublas_proxy)
add_subdirectory(dcgm_stub)
add_subdirectory(dcgmi)
add_subdirectory(dcgmlib)
add_subdirectory(dcgmproftester)
add_subdirectory(docs)
add_subdirectory(hostengine)
add_subdirectory(modules)
add_subdirectory(multi-node)
add_subdirectory(nvml-injection)
add_subdirectory(nvvs)
add_subdirectory(scripts)
add_subdirectory(sdk)
add_subdirectory(sdk_samples)
add_subdirectory(testing)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt
    COMMAND bash -c "find ${CMAKE_SOURCE_DIR} -iname '*.cpp' -or -iname '*.c' -not -ipath '*_out*' -exec grep -n -H -B1 -A4 -e '^\\sPRINT_' -e 'DEBUG_' {} \; > ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt"
    VERBATIM
)
add_custom_target(dcgm_decode_db ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dcgm_decode_db.txt)

include(sanitizers)

install(
    FILES "${PROJECT_BINARY_DIR}/dcgm_decode_db.txt"
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/data
    COMPONENT Tests)

install(
    EXPORT dcgm-export
    FILE DCGMConfig.cmake
    NAMESPACE DCGM::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/DCGM" 
    COMPONENT Development)

write_basic_package_version_file(
    DCGMConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(
    FILES "${PROJECT_BINARY_DIR}/DCGMConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/DCGM"
    COMPONENT Development)

install(
    FILES LICENSE
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
    COMPONENT Core)

#[[
This install call serves to establish an empty install component. This empty
component is used in the context of packages for dependency management
systems, such as rpm or apt, to provide a "metapackage", i.e. a package that
expresses a dependency on two or more other packages, but which provide no
content themselves.
#]]
install(FILES DESTINATION . COMPONENT Cuda_All)

install(
    TARGETS
        dcgm
        dcgm_interface
    EXPORT dcgm-export
    ARCHIVE
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Development
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Core
        NAMELINK_COMPONENT Development
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Core)

install(
    TARGETS
        dcgm_stub
        nvml_injection
    ARCHIVE
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Development
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Development
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Development)

install(
    TARGETS
        dcgmi
        dcgmmoduleconfig
        dcgmmodulediag
        dcgmmodulehealth
        dcgmmoduleintrospect
        dcgmmodulenvswitch
        dcgmmodulepolicy
        dcgmmodulesysmon
        nv-hostengine
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Core
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Core)

install(
    TARGETS
        dcgmmodulemndiag
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Multinode
        NAMELINK_SKIP)

install(
    TARGETS nvvs
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT Core)

install(
    TARGETS
        dcgm_cublas_proxy11
        dcgmproftester11
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Cuda11
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Cuda11)

install(
    TARGETS pluginCommon
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cudaless"
        COMPONENT Core
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        NAMELINK_SKIP
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cudaless"
        COMPONENT Core)

install(
    TARGETS
        BwChecker_11
        ContextCreate_11
        Diagnostic_11
        Memory_11
        Memtest_11
        Pcie_11
        TargetedPower_11
        TargetedStress_11
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda11"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        COMPONENT Cuda11
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda11"
        COMPONENT Cuda11)

install(
    TARGETS
        dcgm_cublas_proxy12
        dcgmproftester12
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Cuda12
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Cuda12)

install(
    TARGETS
        BwChecker_12
        ContextCreate_12
        Diagnostic_12
        Memory_12
        Memtest_12
        NVBandwidth_12
        Pcie_12
        TargetedPower_12
        TargetedStress_12
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        COMPONENT Cuda12
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12"
        COMPONENT Cuda12)

# Tests component
install(
    TARGETS nvvs
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/"
        COMPONENT Tests)

install(
    TARGETS
        dcgm
        dcgm_cublas_proxy11
        dcgm_cublas_proxy12
        dcgmi
        dcgmmoduleconfig
        dcgmmodulediag
        dcgmmodulehealth
        dcgmmoduleintrospect
        dcgmmodulemndiag
        dcgmmodulenvswitch
        dcgmmodulepolicy
        dcgmmodulesysmon
        dcgmproftester11
        dcgmproftester12
        nv-hostengine
        nvml_injection
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)

install(
    TARGETS pluginCommon
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cudaless"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cudaless"
        COMPONENT Tests)

install(
    TARGETS
        BwChecker_11
        ContextCreate_11
        Diagnostic_11
        Memory_11
        Memtest_11
        Pcie_11
        TargetedPower_11
        TargetedStress_11
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda11"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda11"
        COMPONENT Tests)

install(
    TARGETS
        BwChecker_12
        ContextCreate_12
        Diagnostic_12
        Memory_12
        Memtest_12
        NVBandwidth_12
        Pcie_12
        TargetedPower_12
        TargetedStress_12
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12"
        COMPONENT Tests)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    set(CPACK_TGZ_PACKAGE_ARCHITECTURE "amd64")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
    set(CPACK_TGZ_PACKAGE_ARCHITECTURE "aarch64")
endif ()

if (VMWARE)
    set(CPACK_VMWARE ${VMWARE} CACHE BOOL "Indicate if the VMWare build was initiated" FORCE)
endif ()

set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_SOURCE_DIR}/cmake/packaging.cmake")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_STRIP "${CMAKE_STRIP}")
set(CPACK_OBJDUMP "${CMAKE_OBJDUMP}")

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_DEBUGINFO_PACKAGE ON)
set(CPACK_DEBIAN_PACKAGE_EPOCH 1)

set(CPACK_RPM_DEBUGINFO_PACKAGE ON)
set(CPACK_RPM_PACKAGE_EPOCH 1)

configure_file(
    dcgm_config/CMakeLists.txt.in
    dcgm_config/CMakeLists.txt
    ESCAPE_QUOTES
    @ONLY)

include(CPack)
