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
set(CPACK_PACKAGE_CONTACT "dcgm-support <dcgm-support@nvidia.com>")
set(CPACK_PACKAGE_VENDOR "NVIDIA Corp.")

set(CPACK_COMPONENT_CORE_DESCRIPTION
    "CUDA-version agnostic components of the NVIDIA® Datacenter GPU Management Tools")

set(CPACK_COMPONENT_CUDA11_DESCRIPTION
    "NVIDIA® Datacenter GPU Management binaries supporting CUDA11 environments")

set(CPACK_COMPONENT_CUDA12_DESCRIPTION
    "NVIDIA® Datacenter GPU Management binaries supporting CUDA12 environments")

set(CPACK_COMPONENT_CUDA_ALL_DESCRIPTION
    "NVIDIA® Datacenter GPU Management binaries for all supported CUDA environments")

set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION
    "Development files for the NVIDIA® Datacenter GPU Management Tools")

set(CPACK_COMPONENT_MULTINODE_DESCRIPTION
    "CUDA-version agnostic binaries used in NVIDIA® Datacenter GPU Management multi-node diagnostics")

if (CPACK_GENERATOR MATCHES "DEB")
    list(APPEND CPACK_COMPONENTS_ALL Cuda_All)
    list(REMOVE_ITEM CPACK_COMPONENTS_ALL Tests)

    if (CPACK_VMWARE)
        message(FATAL_ERROR "DEB packages are not provided for VMWare builds")
    endif()

    if (NOT DEFINED CPACK_DEBIAN_PACKAGE_VERSION)
        set(CPACK_DEBIAN_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")

        if (DEFINED ENV{BUILD_NUMBER})
            string(APPEND CPACK_DEBIAN_PACKAGE_VERSION "~$ENV{BUILD_NUMBER}")
        endif()
    endif()

    set(CPACK_COMPONENTS_GROUPING "IGNORE")
    set(CPACK_PRE_BUILD_SCRIPTS
        "${CMAKE_CURRENT_LIST_DIR}/cpack-deb-prebuild.cmake")
    set(CPACK_DEBIAN_PACKAGE_RELEASE 1)

    set(CPACK_DEBIAN_CORE_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-core")
    set(CPACK_DEBIAN_CUDA_ALL_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda-all")
    set(CPACK_DEBIAN_CUDA11_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda11")
    set(CPACK_DEBIAN_CUDA12_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda12")
    set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dev")
    set(CPACK_DEBIAN_MULTINODE_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-multinode")
    set(CPACK_DEBIAN_PYTHON_PACKAGE_NAME "python3-${CPACK_PACKAGE_NAME}")

    set(CPACK_DEBIAN_CORE_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_CORE_PACKAGE_CONFLICTS "datacenter-gpu-manager")
    set(CPACK_DEBIAN_CORE_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_LIST_DIR}/../scripts/deb/postinst")
    set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libc6 (>= 2.27), lshw")
    string(JOIN ", " CPACK_DEBIAN_CORE_PACKAGE_PREDEPENDS
        "passwd"
        "login")
    set(CPACK_DEBIAN_CUDA_ALL_DEBUGINFO_PACKAGE FALSE)
    string(JOIN ", " CPACK_DEBIAN_CUDA_ALL_PACKAGE_DEPENDS
        "${CPACK_DEBIAN_CUDA11_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})"
        "${CPACK_DEBIAN_CUDA12_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")

    set(CPACK_DEBIAN_CUDA11_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_CUDA11_PACKAGE_DEPENDS
        "${CPACK_DEBIAN_CORE_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")
    set(CPACK_DEBIAN_CUDA11_PACKAGE_PROVIDES
        "${CPACK_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")
    set(CPACK_DEBIAN_CUDA11_PACKAGE_RECOMMENDS "libnuma1")

    set(CPACK_DEBIAN_CUDA12_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_CUDA12_PACKAGE_DEPENDS
        "${CPACK_DEBIAN_CORE_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")
    set(CPACK_DEBIAN_CUDA12_PACKAGE_PROVIDES
        "${CPACK_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")
    set(CPACK_DEBIAN_CUDA12_PACKAGE_RECOMMENDS "libnuma1")

    set(CPACK_DEBIAN_DEVELOPMENT_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_DEPENDS
        "${CPACK_DEBIAN_CORE_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")

    set(CPACK_DEBIAN_MULTINODE_DEBUGINFO_PACKAGE TRUE)
    string(JOIN ", " CPACK_DEBIAN_MULTINODE_PACKAGE_DEPENDS
        "${CPACK_DEBIAN_CORE_PACKAGE_NAME} (= ${CPACK_DEBIAN_PACKAGE_EPOCH}:${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})"
        "openmpi | openmpi-bin (>= 4.1.1)") # Play nice with the clients using the NVIDIA DOCA package repository

    set(SUFFIX
        "_${CPACK_DEBIAN_PACKAGE_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")

    foreach(component IN LISTS CPACK_COMPONENTS_ALL)
        string(TOUPPER "${component}" component)
        string(CONCAT CPACK_DEBIAN_${component}_FILE_NAME
            "${CPACK_DEBIAN_${component}_PACKAGE_NAME}"
            "${SUFFIX}")
    endforeach()
elseif(CPACK_GENERATOR MATCHES "TGZ")
    if (CPACK_VMWARE)
        set(DCGM_TGZ_SUFFIX "${CPACK_TGZ_PACKAGE_ARCHITECTURE}-vmware")
    else()
        set(DCGM_TGZ_SUFFIX "${CPACK_TGZ_PACKAGE_ARCHITECTURE}")
    endif()

    set(CPACK_TGZ_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
    if (DEFINED ENV{BUILD_NUMBER})
        string(APPEND CPACK_TGZ_PACKAGE_VERSION "+$ENV{BUILD_NUMBER}")
    endif()

    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

    set(CPACK_COMPONENT_CORE_GROUP "Amalgam")
    set(CPACK_COMPONENT_CUDA11_GROUP "Amalgam")
    set(CPACK_COMPONENT_CUDA12_GROUP "Amalgam")
    set(CPACK_COMPONENT_DEVELOPMENT_GROUP "Amalgam")
    set(CPACK_COMPONENT_MULTINODE_GROUP "Amalgam")

    set(CPACK_ARCHIVE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_TGZ_PACKAGE_VERSION}-${DCGM_TGZ_SUFFIX}")
    set(CPACK_ARCHIVE_AMALGAM_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_TGZ_PACKAGE_VERSION}-${DCGM_TGZ_SUFFIX}")
    set(CPACK_ARCHIVE_TESTS_FILE_NAME "${CPACK_PACKAGE_NAME}-tests-${CPACK_TGZ_PACKAGE_VERSION}-${DCGM_TGZ_SUFFIX}")
elseif(CPACK_GENERATOR MATCHES "RPM")
    list(APPEND CPACK_COMPONENTS_ALL Cuda_All)
    list(REMOVE_ITEM CPACK_COMPONENTS_ALL Tests)

    if (CPACK_VMWARE)
        message(FATAL_ERROR "RPM packages are not provided for VMWare builds")
    endif()

    if (NOT DEFINED CPACK_RPM_PACKAGE_VERSION)
        set(CPACK_RPM_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")

        if (DEFINED ENV{BUILD_NUMBER})
            string(APPEND CPACK_RPM_PACKAGE_VERSION "~$ENV{BUILD_NUMBER}")
        endif()
    endif()

    set(CPACK_COMPONENTS_GROUPING "IGNORE")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_PRE_BUILD_SCRIPTS
        "${CMAKE_CURRENT_LIST_DIR}/cpack-rpm-prebuild.cmake")

    set(CPACK_RPM_BUILD_SOURCE_DIRS_PREFIX "/-")
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
        "/usr/lib/systemd"
        "/usr/lib/systemd/system"
        "/usr/lib/systemd/system-generators"
        "/usr/libexec"
        "/usr/sbin"
        "/usr/share/licenses"
        "/usr/src")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    set(CPACK_RPM_INSTALL_WITH_EXEC TRUE)
    set(CPACK_RPM_PACKAGE_AUTOREQ "no") # Removes dependency on libcuda.so.1
    set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
    set(CPACK_RPM_PACKAGE_LICENSE "NVIDIA Proprietary")
    set(CPACK_RPM_PACKAGE_RELEASE 1) # this is the package spec version, not the software version
    set(CPACK_RPM_SPEC_MORE_DEFINE "
%define __strip ${CPACK_STRIP}
%define __objdump ${CPACK_OBJDUMP}
%define __brp_elfperms /bin/true")

    set(CPACK_RPM_CORE_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-core")
    set(CPACK_RPM_CUDA_ALL_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda-all")
    set(CPACK_RPM_CUDA11_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda11")
    set(CPACK_RPM_CUDA12_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-cuda12")
    set(CPACK_RPM_DEVELOPMENT_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-devel")
    set(CPACK_RPM_MULTINODE_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-multinode")
    set(CPACK_RPM_PYTHON_PACKAGE_NAME "python3-${CPACK_PACKAGE_NAME}")

    set(CPACK_RPM_CORE_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_RPM_CORE_PACKAGE_CONFLICTS "datacenter-gpu-manager")

    # The CPack RPM generator doesn't support the Recommends spec file header.
    # As a work-around, we abuse the substitution of another header.
    #
    # Note that weak dependency spec file headers are only supported in rpm
    # version 4.12.0 or later. The Linux distribution shipping with a
    # sufficiently recent rpm version include:
    # - Red Hat Enterprise Linux >= 8,
    # - SUSE Linux Enterprise Server >= 10
    # - Fedora Linux >= 24
    #
    set(CPACK_RPM_CORE_PACKAGE_REQUIRES "glibc >= 2.27, lshw")

    # The RPM post-installation script configures an unprivileged user account,
    # `nvidia-dcgm`, to satisfy the principle of least privilege. In RPM
    # packages, this account is created via the `useradd` command. The name of
    # the package varies between SUSE Linux Enterprise Server (SLES) and Red
    # Hat Enterprise Linux (RHEL).
    #
    # Fortunately, the name 'shadow' refers to an appropriate package on SLES
    # and to an appropriate virtual package on RHEL 9. An alternative referring
    # to the concrete package has been added to support RHEL 8
    string(JOIN ", " CPACK_RPM_CORE_PACKAGE_REQUIRES_PRE
        "(shadow or shadow-utils)"
        "/sbin/nologin")

    set(CPACK_RPM_CORE_POST_INSTALL_SCRIPT_FILE
        "${CMAKE_CURRENT_LIST_DIR}/../scripts/rpm/postinstall")

    set(CPACK_RPM_CUDA_ALL_DEBUGINFO_PACKAGE FALSE)
    string(JOIN ", " CPACK_RPM_CUDA_ALL_PACKAGE_REQUIRES
        "${CPACK_RPM_CUDA11_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}"
        "${CPACK_RPM_CUDA12_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}")

    set(CPACK_RPM_CUDA11_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_RPM_CUDA11_PACKAGE_PROVIDES
        "${CPACK_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}")
    # The CPack RPM generator doesn't support the Recommends spec file header.
    # As a work-around, we abuse the substitution of another header
    set(CPACK_RPM_CUDA11_PACKAGE_REQUIRES
        "${CPACK_RPM_CORE_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}
Recommends: libnuma.so.1()(64bit)")

    set(CPACK_RPM_CUDA12_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_RPM_CUDA12_PACKAGE_PROVIDES
        "${CPACK_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}")
    # The CPack RPM generator doesn't support the Recommends spec file header.
    # As a work-around, we abuse the substitution of another header
    set(CPACK_RPM_CUDA12_PACKAGE_REQUIRES
        "${CPACK_RPM_CORE_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}
Recommends: libnuma.so.1()(64bit)")

    set(CPACK_RPM_DEVELOPMENT_DEBUGINFO_PACKAGE TRUE)
    set(CPACK_RPM_DEVELOPMENT_PACKAGE_DEPENDS
        "${CPACK_RPM_CORE_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}")

    set(CPACK_RPM_MULTINODE_DEBUGINFO_PACKAGE TRUE)
    string(JOIN ", " CPACK_RPM_MULTINODE_PACKAGE_DEPENDS
        "${CPACK_RPM_CORE_PACKAGE_NAME} = ${CPACK_RPM_PACKAGE_EPOCH}:${CPACK_RPM_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}"
        "openmpi >= 4.1.1")

    foreach(component IN LISTS CPACK_COMPONENTS_ALL)
        string(TOUPPER "${component}" component)
        set(CPACK_RPM_${component}_PACKAGE_SUMMARY
            "${CPACK_COMPONENT_${component}_DESCRIPTION}")
    endforeach()
endif()
