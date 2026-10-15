# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

cmake_minimum_required(VERSION 4.0.0)

project(datacenter-gpu-manager-4 VERSION 4.7.0)

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

find_package(Boost REQUIRED COMPONENTS filesystem regex stacktrace_basic system)
find_package(Catch2 REQUIRED)
find_package(fmt REQUIRED)
find_package(Git REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Libevent REQUIRED)
find_package(plog REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
find_package(TCLAP REQUIRED)
find_package(yaml-cpp REQUIRED CONFIG)

include(Catch)
include(CheckCCompilerFlag)
include(CheckPIESupported)
include(CheckIPOSupported)
include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

include(Coverity OPTIONAL)
include(Cuda)

check_pie_supported(OUTPUT_VARIABLE PIE_SUPPORT_OUTPUT)

set(DCGM_CONFIG_REVISION 0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build Position Independent Code")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Enable LTO
option(CMAKE_INTERPROCEDURAL_OPTIMIZATION "Enable interprocedural optimization for all targets" OFF)

# Build targets in parallel where possible
option(CMAKE_OPTIMIZE_DEPENDENCIES "Activates dependency optimization of static and object libraries" ON)

# Disable modules, as we don't use them and ccache does not currently support them
option(CMAKE_CXX_SCAN_FOR_MODULES "Whether CMake will scan C++ sources in the target for module dependencies" OFF)

if (CMAKE_POSITION_INDEPENDENT_CODE)
    if (NOT CMAKE_C_LINK_PIE_SUPPORTED OR NOT CMAKE_CXX_LINK_PIE_SUPPORTED)
        message(FATAL_ERROR "${PIE_SUPPORT_OUTPUT}")
    endif()
else()
    if (NOT CMAKE_C_LINK_NO_PIE_SUPPORTED OR NOT CMAKE_CXX_LINK_NO_PIE_SUPPORTED)
        message(WARNING "${PIE_SUPPORT_OUTPUT}")
    endif()
endif()

# Verify toolchain IPO support
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  check_ipo_supported()
endif()

set(BUILD_NUMBER "$ENV{BUILD_NUMBER}" CACHE STRING
    "The build number associated with the generated build system")

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)
if (ADDRESS_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope")
    set(ANY_SANITIZER TRUE)
endif ()

option(THREAD_SANITIZER "Enable ThreadSanitizer" OFF)
if (THREAD_SANITIZER)
    set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fsanitize=thread -Wno-tsan")
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
    add_compile_definitions(NDEBUG)
endif ()

set(DCGM_CXX_FLAGS "${DCGM_CXX_FLAGS} -fmax-errors=1 -Werror -fno-omit-frame-pointer")
# Necessary for ccache to correctly handle debug builds
string(JOIN " " DCGM_CXX_FLAGS
    "${DCGM_CXX_FLAGS}"
    "-fcanon-prefix-map"
    "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=/workspaces/dcgm")

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
    _UNIX=1
    __CUDA_API_VERSION_INTERNAL=1
)

if (NOT ANY_SANITIZER)
    add_link_options(-static-libstdc++)
endif ()

add_link_options(-Wl,--exclude-libs,libstdc++)

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
set(DCGM_TESTS_APP_DIR "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps")

set(DCGM_BIN_RPATH
    "$ORIGIN:$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
file(RELATIVE_PATH DCGM_LIB_TO_LIBEXEC_DIR
    "/${CMAKE_INSTALL_LIBDIR}"
    "/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}")
set(DCGM_MODULE_RPATH
    "$ORIGIN:$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}")
set(DCGM_NVVS_CUDA_PLUGIN_RPATH
    "$ORIGIN/../cudaless:$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}:$ORIGIN/../../../${DCGM_TESTS_ARCH}")
set(DCGM_NVVS_CUDALESS_PLUGIN_RPATH
    "$ORIGIN:$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}:$ORIGIN/../../../${DCGM_TESTS_ARCH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Git information
set(GIT_COMMIT_SHA "unknown" CACHE STRING "Current git commit SHA")
set(GIT_COMMIT_REF "unknown" CACHE STRING "Current git commit branch or tag")

# Build Platform -> ${BUILDPLATFORM}
execute_process(COMMAND uname -srvm
    OUTPUT_VARIABLE BUILDPLATFORM
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Build Date in UTC -> ${BUILD_DATE}
execute_process(COMMAND date -u -Idate
    OUTPUT_VARIABLE BUILD_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(DCGM_BUILD_INFO
    "version:${CMAKE_PROJECT_VERSION}"
    "arch:${CMAKE_SYSTEM_PROCESSOR}"
    "buildtype:${CMAKE_BUILD_TYPE}"
    "buildid:${BUILD_NUMBER}"
    "builddate:${BUILD_DATE}"
    "commit:${GIT_COMMIT_SHA}"
    "branch:${GIT_COMMIT_REF}"
    "buildplatform:${BUILDPLATFORM}"
    "$ENV{DCGM_BUILD_INFO}")

execute_process(
    COMMAND bash -c "echo '${DCGM_BUILD_INFO}' | md5sum - | awk '{print $1}'"
    OUTPUT_VARIABLE BUILD_INFO_CRC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND DCGM_BUILD_INFO "crc:${BUILD_INFO_CRC}")
message(STATUS "${DCGM_BUILD_INFO}")

add_compile_definitions(INJECTION_LIBRARY_AVAILABLE)

add_library(nvml_injection SHARED)
add_library(nvmli_interface INTERFACE)
add_library(nvmli_public OBJECT)

set(LINK_INJECTION_LIB "true")

add_executable(BwChecker_11)
add_executable(BwChecker_12)
add_executable(BwChecker_13)
add_executable(dcgmi)
add_executable(dcgmproftester11)
add_executable(dcgmproftester12)
add_executable(dcgmproftester13)
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
add_library(nvvs_plugins_nccl_tests_interface INTERFACE)
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
add_library(dcgm_cuda_worker13 STATIC)
add_library(dcgm_entity_types STATIC)
add_library(dcgm_logging STATIC)
add_library(dcgm_mutex STATIC)
add_library(dcgm_static_private STATIC)
add_library(dcgm_stub STATIC)
add_library(dcgm_thread STATIC)
add_library(dcgmproftester_lib11 STATIC)
add_library(dcgmproftester_lib12 STATIC)
add_library(dcgmproftester_lib13 STATIC)
add_library(dcgmi_objects_without_main STATIC)
add_library(file_handle STATIC)
add_library(nvvs_without_main_objects OBJECT)
add_library(nvvs_main_objects OBJECT)
add_library(nvvs_plugins_common_objects OBJECT)
add_library(pluginCudaCommon_11 STATIC)
add_library(pluginCudaCommon_12 STATIC)
add_library(pluginCudaCommon_13 STATIC)
add_library(remoteconn STATIC)
add_library(sdk_nvml_essentials_objects STATIC)
add_library(sdk_nvml_loader STATIC)
add_library(serialize STATIC)
add_library(transport_objects STATIC)
add_library(user_credentials STATIC)

add_library(dcgmmodulediag_common STATIC)
add_library(dcgmmodulediag_private_static STATIC)
add_library(dcgmmodulemndiag_common STATIC)
add_library(dcgmmodulemndiag_private_static STATIC)
add_library(dcgmmodulepolicy_common STATIC)
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
add_library(ContextCreate_13 SHARED)
add_library(Diagnostic_11 SHARED)
add_library(Diagnostic_12 SHARED)
add_library(Diagnostic_13 SHARED)
add_library(Memory_11 SHARED)
add_library(Memory_12 SHARED)
add_library(Memory_13 SHARED)
add_library(Memtest_11 SHARED)
add_library(Memtest_12 SHARED)
add_library(Memtest_13 SHARED)
add_library(NcclTests_12 SHARED)
add_library(NcclTests_13 SHARED)
add_library(NVBandwidth_12 SHARED)
add_library(NVBandwidth_13 SHARED)
add_library(Pcie_11 SHARED)
add_library(Pcie_12 SHARED)
add_library(Pcie_13 SHARED)
add_library(TargetedPower_11 SHARED)
add_library(TargetedPower_12 SHARED)
add_library(TargetedPower_13 SHARED)
add_library(TargetedStress_11 SHARED)
add_library(TargetedStress_12 SHARED)
add_library(TargetedStress_13 SHARED)

add_library(dcgm SHARED)
add_library(cublas_ops11 STATIC)
add_library(cublas_ops12 STATIC)
add_library(cublas_ops13 STATIC)
add_library(cublas_ops_interface11 INTERFACE)
add_library(cublas_ops_interface12 INTERFACE)
add_library(cublas_ops_interface13 INTERFACE)
add_library(cuda_ops11 STATIC)
add_library(cuda_ops12 STATIC)
add_library(cuda_ops13 STATIC)
add_library(cuda_ops_interface11 INTERFACE)
add_library(cuda_ops_interface12 INTERFACE)
add_library(cuda_ops_interface13 INTERFACE)
add_library(pluginCommon SHARED)

add_library(sdk_nscq_loader)
add_library(sdk_nvsdm_loader)

add_library(DCGM::dcgm ALIAS dcgm)

set_target_properties(
    ContextCreate_11
    ContextCreate_12
    ContextCreate_13
    Diagnostic_11
    Diagnostic_12
    Diagnostic_13
    Memory_11
    Memory_12
    Memory_13
    Memtest_11
    Memtest_12
    Memtest_13
    NcclTests_12
    NcclTests_13
    NVBandwidth_12
    NVBandwidth_13
    Pcie_11
    Pcie_12
    Pcie_13
    TargetedPower_11
    TargetedPower_12
    TargetedPower_13
    TargetedStress_11
    TargetedStress_12
    TargetedStress_13
    dcgm
    dcgmmoduleconfig
    dcgmmodulediag
    dcgmmodulehealth
    dcgmmoduleintrospect
    dcgmmodulemndiag
    dcgmmodulenvswitch
    dcgmmodulepolicy
    dcgmmodulesysmon
    pluginCommon
    PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

set_target_properties(nvml_injection PROPERTIES SOVERSION 1 VERSION 1.0.0)

set_target_properties(dcgmi PROPERTIES
    INSTALL_RPATH "${DCGM_BIN_RPATH}")

set_target_properties(nv-hostengine PROPERTIES
    INSTALL_RPATH "${DCGM_BIN_RPATH}")

set_target_properties(dcgm PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/${DCGM_LIB_TO_LIBEXEC_DIR}")

set_target_properties(
    dcgmmoduleconfig
    dcgmmodulediag
    dcgmmodulehealth
    dcgmmoduleintrospect
    dcgmmodulemndiag
    dcgmmodulenvswitch
    dcgmmodulepolicy
    dcgmmodulesysmon
    PROPERTIES
        INSTALL_RPATH "${DCGM_MODULE_RPATH}")

set_target_properties(nvvs PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}:$ORIGIN/../${DCGM_TESTS_ARCH}"
    RUNTIME_OUTPUT_DIRECTORY nvvs)

# These targets are installed in both the production and test layouts.
set_target_properties(
    ContextCreate_11
    ContextCreate_12
    ContextCreate_13
    Memory_11
    Memory_12
    Memory_13
    Memtest_11
    Memtest_12
    Memtest_13
    NVBandwidth_12
    NVBandwidth_13
    NcclTests_12
    NcclTests_13
    PROPERTIES
        INSTALL_RPATH "${DCGM_NVVS_CUDA_PLUGIN_RPATH}")

set_property(TARGET
    Diagnostic_11
    Pcie_11
    TargetedPower_11
    TargetedStress_11
    PROPERTY INSTALL_RPATH
        "$ORIGIN/../cudaless"
        "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
        "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
        "$ORIGIN/../../cuda11/lib")

set_property(TARGET BwChecker_11 PROPERTY INSTALL_RPATH
    "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
    "$ORIGIN/../../cuda11/lib")

set_property(TARGET
    Diagnostic_12
    Pcie_12
    TargetedPower_12
    TargetedStress_12
    PROPERTY INSTALL_RPATH
        "$ORIGIN/../cudaless"
        "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
        "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
        "$ORIGIN/../../cuda12/lib")

set_property(TARGET BwChecker_12 PROPERTY INSTALL_RPATH
    "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
    "$ORIGIN/../../cuda12/lib")

set_property(TARGET
    Diagnostic_13
    Pcie_13
    TargetedPower_13
    TargetedStress_13
    PROPERTY INSTALL_RPATH
        "$ORIGIN/../cudaless"
        "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
        "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
        "$ORIGIN/../../cuda13/lib")

set_property(TARGET BwChecker_13 PROPERTY INSTALL_RPATH
    "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../../../${DCGM_TESTS_ARCH}"
    "$ORIGIN/../../cuda13/lib")

set_property(TARGET pluginCommon PROPERTY INSTALL_RPATH
    "$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../../../${DCGM_TESTS_ARCH}")

set_property(TARGET dcgmproftester11 PROPERTY INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda11/lib")

set_property(TARGET dcgmproftester12 PROPERTY INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib")

set_property(TARGET dcgmproftester13 PROPERTY INSTALL_RPATH
    "$ORIGIN"
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/../${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib")

set_target_properties(dcgmi PROPERTIES RUNTIME_OUTPUT_DIRECTORY dcgmi)

set_target_properties(
    ContextCreate_11
    ContextCreate_12
    ContextCreate_13
    Diagnostic_11
    Diagnostic_12
    Diagnostic_13
    Memory_11
    Memory_12
    Memory_13
    Memtest_11
    Memtest_12
    Memtest_13
    NcclTests_12
    NcclTests_13
    NVBandwidth_12
    NVBandwidth_13
    Pcie_11
    Pcie_12
    Pcie_13
    TargetedPower_11
    TargetedPower_12
    TargetedPower_13
    TargetedStress_11
    TargetedStress_12
    TargetedStress_13
    nvml_injection
    nvmli_public
    nvvs_without_main_objects
    nvvs_plugins_common_objects
    pluginCommon
    pluginCudaCommon_11
    pluginCudaCommon_12
    pluginCudaCommon_13
    PROPERTIES
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN FALSE)

add_subdirectory(common)
add_subdirectory(config-files)
add_subdirectory(data)
add_subdirectory(dcgm_stub)
add_subdirectory(dcgmi)
add_subdirectory(dcgmlib)
add_subdirectory(dcgmproftester)
add_subdirectory(docs)
add_subdirectory(hostengine)
add_subdirectory(modules)
add_subdirectory(nvml-injection)
add_subdirectory(nvvs)
add_subdirectory(scripts)
add_subdirectory(sdk)
add_subdirectory(sdk_samples)
add_subdirectory(rust/module_log_test)
add_subdirectory(testing)

include(sanitizers)

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
    COMPONENT libdcgm)

#[[
These install calls serve to establish empty install components. These empty
components are used in the context of packages for dependency management
systems, such as rpm or apt, to provide a "metapackage", i.e. a package that
expresses a dependency on two or more other packages, but which provide no
content themselves.
#]]
install(FILES DESTINATION . COMPONENT Core)
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
        COMPONENT libdcgm
        NAMELINK_COMPONENT Development
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT libdcgm)

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
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT dcgmi)

install(
    TARGETS
        nv-hostengine
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT nv-hostengine)

install(
    TARGETS
        dcgmmoduleconfig
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmoduleconfig
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmoduleconfig)

install(
    TARGETS
        dcgmmodulediag
        nvvs
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulediag
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulediag)

install(
    TARGETS pluginCommon
    LIBRARY
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cudaless"
        COMPONENT libdcgmmodulediag
        NAMELINK_SKIP)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    install(CODE "
        file(INSTALL
            DESTINATION \"$<INSTALL_PREFIX>/${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cudaless\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES \"$<TARGET_PROPERTY:cpu_sdc-shared,RUST_SONAME_LINK>\")
    "
    COMPONENT libdcgmmodulediag)
endif()

install(
    TARGETS
        dcgmmodulehealth
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulehealth
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulehealth)

install(
    TARGETS
        dcgmmoduleintrospect
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmoduleintrospect
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmoduleintrospect)

install(
    TARGETS
        dcgmmodulenvswitch
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulenvswitch
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulenvswitch)

install(
    TARGETS
        dcgmmodulepolicy
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulepolicy
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulepolicy)

install(
    TARGETS
        dcgmmodulesysmon
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulesysmon
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT libdcgmmodulesysmon)

install(
    TARGETS
        dcgmmodulemndiag
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
        COMPONENT Multinode
        NAMELINK_SKIP)

install(
    TARGETS
        dcgmproftester11
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Cuda11
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Cuda11)

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
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda11"
        COMPONENT Cuda11
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda11"
        COMPONENT Cuda11)

install(
    TARGETS
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
        dcgmproftester13
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT Cuda13
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT Cuda13)

install(
    TARGETS
        BwChecker_12
        ContextCreate_12
        Diagnostic_12
        Memory_12
        Memtest_12
        NcclTests_12
        NVBandwidth_12
        Pcie_12
        TargetedPower_12
        TargetedStress_12
    LIBRARY
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda12"
        COMPONENT Cuda12
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda12"
        COMPONENT Cuda12)

 install(
    TARGETS
        BwChecker_13
        ContextCreate_13
        Diagnostic_13
        Memory_13
        Memtest_13
        NcclTests_13
        NVBandwidth_13
        Pcie_13
        TargetedPower_13
        TargetedStress_13
    LIBRARY
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda13"
        COMPONENT Cuda13
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda13"
        COMPONENT Cuda13)

# Start with the SONAME links so FOLLOW_SYMLINK_CHAIN excludes the unversioned linker links.
install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda11/lib\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas11>/$<TARGET_PROPERTY:Cuda::cublas11,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt11>/$<TARGET_PROPERTY:Cuda::cublasLt11,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda11/lib\"
        TYPE SHARED_LIBRARY
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas11>/libcublas.so\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt11>/libcublasLt.so\")
    "
    COMPONENT CUDA11_CUBLAS)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas12>/$<TARGET_PROPERTY:Cuda::cublas12,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt12>/$<TARGET_PROPERTY:Cuda::cublasLt12,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib\"
        TYPE SHARED_LIBRARY
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas12>/libcublas.so\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt12>/libcublasLt.so\")
    "
    COMPONENT CUDA12_CUBLAS)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas13>/$<TARGET_PROPERTY:Cuda::cublas13,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt13>/$<TARGET_PROPERTY:Cuda::cublasLt13,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib\"
        TYPE SHARED_LIBRARY
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas13>/libcublas.so\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt13>/libcublasLt.so\")
    "
    COMPONENT CUDA13_CUBLAS)

# Tests component
install(
    TARGETS nvvs
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/"
        COMPONENT Tests)

install(
    TARGETS
        dcgm
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
        dcgmproftester13
        nv-hostengine
        nvml_injection
    LIBRARY
        DESTINATION "${DCGM_TESTS_APP_DIR}/${DCGM_TESTS_ARCH}"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)

install(
    TARGETS pluginCommon
    LIBRARY
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cudaless"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cudaless"
        COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cudaless\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES \"$<TARGET_PROPERTY:cpu_sdc-shared,RUST_SONAME_LINK>\")
    "
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
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda11"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda11"
        COMPONENT Tests)

install(
    TARGETS
        BwChecker_12
        ContextCreate_12
        Diagnostic_12
        Memory_12
        Memtest_12
        NVBandwidth_12
        NcclTests_12
        Pcie_12
        TargetedPower_12
        TargetedStress_12
    LIBRARY
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12"
        COMPONENT Tests)

install(
    TARGETS
        BwChecker_13
        ContextCreate_13
        Diagnostic_13
        Memory_13
        Memtest_13
        NVBandwidth_13
        NcclTests_13
        Pcie_13
        TargetedPower_13
        TargetedStress_13
    LIBRARY
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13"
        COMPONENT Tests
        NAMELINK_SKIP
    RUNTIME
        DESTINATION "${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13"
        COMPONENT Tests)

# Keep the Tests component standalone. Preserve its shared runtime directory and place the
# CUDA-specific chains in their plugin directories so the unversioned linker links do not collide.
install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas11>/$<TARGET_PROPERTY:Cuda::cublas11,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt11>/$<TARGET_PROPERTY:Cuda::cublasLt11,IMPORTED_SONAME>\")
    "
    COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas12>/$<TARGET_PROPERTY:Cuda::cublas12,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt12>/$<TARGET_PROPERTY:Cuda::cublasLt12,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas12>/$<TARGET_PROPERTY:Cuda::cublas12,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt12>/$<TARGET_PROPERTY:Cuda::cublasLt12,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12\"
        TYPE SHARED_LIBRARY
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas12>/libcublas.so\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt12>/libcublasLt.so\")
    "
    COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas13>/$<TARGET_PROPERTY:Cuda::cublas13,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt13>/$<TARGET_PROPERTY:Cuda::cublasLt13,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas13>/$<TARGET_PROPERTY:Cuda::cublas13,IMPORTED_SONAME>\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt13>/$<TARGET_PROPERTY:Cuda::cublasLt13,IMPORTED_SONAME>\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13\"
        TYPE SHARED_LIBRARY
        FILES
            \"$<TARGET_FILE_DIR:Cuda::cublas13>/libcublas.so\"
            \"$<TARGET_FILE_DIR:Cuda::cublasLt13>/libcublasLt.so\")
    "
    COMPONENT Tests)

if (DCGM_BUILD_COVERAGE)
    if (CMAKE_INSTALL_PREFIX)
       string(REGEX REPLACE "(.)" "[\\1]"
           CMAKE_INSTALL_PREFIX_PATTERN
           "${CMAKE_INSTALL_PREFIX}")
    else()
       set(CMAKE_INSTALL_PREFIX_PATTERN "$^") # Intentionally unmatchable
    endif()

    install(
        DIRECTORY "${CMAKE_BINARY_DIR}/"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/coverage/gcov"
        COMPONENT Tests
        FILES_MATCHING
            PATTERN "*.gcno"
            REGEX ".*/_CPack_Packages/.*" EXCLUDE
            REGEX "${CMAKE_INSTALL_PREFIX_PATTERN}/.*" EXCLUDE)
endif()

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
