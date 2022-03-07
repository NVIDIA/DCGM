set(CPACK_SET_DESTDIR OFF)
set(DCGM_PKG "datacenter-gpu-manager")
set(DCGM_CONFIG_PKG "${DCGM_PKG}-config")

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NVIDIA® Datacenter GPU Management Tools
 The Datacenter GPU Manager package contains tools for managing NVIDIA® GPUs in
 high performance and cluster computing environments.
 .
 This package also contains the DCGM GPU Diagnostic. DCGM GPU Diagnostic is the system
 administrator and cluster manager's tool for detecting and troubleshooting
 common problems affecting NVIDIA® Tesla GPUs.")

set(CPACK_PACKAGE_CONTACT "dcgm-support <dcgm-support@nvidia.com>")
set(CPACK_PACKAGE_VENDOR "NVIDIA Corp.")

IF (CPACK_GENERATOR MATCHES "DEB")

    set(CPACK_DEBIAN_PACKAGE_EPOCH 1)

    set(
        CPACK_DEBIAN_DCGM_FILE_NAME
        "${DCGM_PKG}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
    )
    set(
        CPACK_DEBIAN_TESTS_FILE_NAME
        "${DCGM_PKG}-tests_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
    )
    set(
        CPACK_DEBIAN_CONFIG_FILE_NAME
        "${DCGM_CONFIG_PKG}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
    )

    set(CPACK_DEBIAN_DCGM_PACKAGE_DEPENDS "libc6 (>= 2.17), libgomp1 (>= 4.8)")
    set(CPACK_DEBIAN_TESTS_PACKAGE_DEPENDS "libc6 (>= 2.17), libgomp1 (>= 4.8)")
    # Not actually necessary, but CMake prints a warning if there are no dependencies
    set(CPACK_DEBIAN_CONFIG_PACKAGE_DEPENDS "libc6 (>= 2.17)")

    set(CPACK_DEBIAN_TESTS_PACKAGE_NAME "${DCGM_PKG}-tests")
    set(CPACK_DEBIAN_TESTS_PACKAGE_PROVIDES "${DCGM_PKG}-tests")
    set(CPACK_DEBIAN_TESTS_PACKAGE_REPLACES "${DCGM_PKG}-tests")
    set(CPACK_DEBIAN_TESTS_PACKAGE_CONFLICTS "${DCGM_PKG}-tests")

    set(CPACK_DEBIAN_CONFIG_PACKAGE_NAME "${DCGM_CONFIG_PKG}")

    set(CPACK_DEBIAN_DCGM_PACKAGE_NAME "${DCGM_PKG}")
    set(CPACK_DEBIAN_DCGM_PACKAGE_PROVIDES "${DCGM_PKG}")
    string(CONCAT CPACK_DEBIAN_DCGM_PACKAGE_REPLACES
        "${DCGM_PKG}"
        ", datacenter-gpu-manager-fabricmanager (<<2.0)"
        ", datacenter-gpu-manager-dcp-nda-only"
        ", datacenter-gpu-manager-collectd"
        ", datacenter-gpu-manager-wsgi"
        ", datacenter-gpu-manager-fabricmanager-internal-api-header")

    string(CONCAT CPACK_DEBIAN_DCGM_PACKAGE_CONFLICTS
        "${DCGM_PKG}"
        ", datacenter-gpu-manager-fabricmanager (<<2.0)"
        ", datacenter-gpu-manager-dcp-nda-only"
        ", datacenter-gpu-manager-collectd"
        ", datacenter-gpu-manager-wsgi"
        ", datacenter-gpu-manager-fabricmanager-internal-api-header")

ELSEIF(CPACK_GENERATOR MATCHES "TGZ")

    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
    set(CPACK_ARCHIVE_DCGM_FILE_NAME "${DCGM_PKG}-${CPACK_PACKAGE_VERSION}-${CPACK_TGZ_PACKAGE_ARCHITECTURE}")
    set(CPACK_ARCHIVE_CONFIG_FILE_NAME "${DCGM_CONFIG_PKG}-${CPACK_PACKAGE_VERSION}-${CPACK_TGZ_PACKAGE_ARCHITECTURE}")
    set(CPACK_ARCHIVE_TESTS_FILE_NAME "${DCGM_PKG}-tests-${CPACK_PACKAGE_VERSION}-${CPACK_TGZ_PACKAGE_ARCHITECTURE}")

ELSEIF(CPACK_GENERATOR MATCHES "RPM")
    set(CPACK_RPM_PACKAGE_RELEASE 1) # this is the package spec version, not the software version
    set(CPACK_RPM_PACKAGE_EPOCH 1)
    set(CPACK_RPM_PACKAGE_LICENSE "NVIDIA Proprietary")

    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_RPM_PACKAGE_SUMMARY "NVIDIA Datacenter GPU Manager")
    set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")

    set(CPACK_RPM_DCGM_FILE_NAME
        "${DCGM_PKG}-${CPACK_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}-${CPACK_RPM_PACKAGE_ARCHITECTURE}.rpm")
    set(CPACK_RPM_TESTS_FILE_NAME
        "${DCGM_PKG}-tests-${CPACK_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}-${CPACK_RPM_PACKAGE_ARCHITECTURE}.rpm")
    set(CPACK_RPM_CONFIG_FILE_NAME
        "${DCGM_CONFIG_PKG}-${CPACK_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}-${CPACK_RPM_PACKAGE_ARCHITECTURE}.rpm")

    set(CPACK_RPM_DCGM_PACKAGE_REQUIRES "glibc >= 2.17, libgomp")
    set(CPACK_RPM_TESTS_PACKAGE_REQUIRES "glibc >= 2.17, libgomp")

    set(CPACK_RPM_CONFIG_PACKAGE_DESCRIPTION "Auxiliary definitions for NVIDIA® Datacenter GPU Management Tools")

    set(CPACK_RPM_DCGM_PACKAGE_CONFLICTS "datacenter-gpu-manager-fabricmanager < 2.0")
    set(CPACK_RPM_DCGM_PACKAGE_OBSOLETES "datacenter-gpu-manager-fabricmanager < 2.0")

    set(CPACK_RPM_DCGM_PACKAGE_PROVIDES "${DCGM_PKG}")
    set(CPACK_RPM_TESTS_PACKAGE_PROVIDES "${DCGM_PKG}-tests")

    set(CPACK_RPM_DCGM_PACKAGE_NAME "${DCGM_PKG}")
    set(CPACK_RPM_CONFIG_PACKAGE_NAME "${DCGM_CONFIG_PKG}")
    set(CPACK_RPM_TESTS_PACKAGE_NAME "${DCGM_PKG}-tests")

    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/etc")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/etc")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/lib")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/lib/systemd")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/lib/systemd/system")
    list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/share/licenses")


    # Stop fpm from scanning binaries for their dependencies. That removes dependency on installed libcuda.so.1
    set(CPACK_RPM_PACKAGE_AUTOREQ " no")

    # Remove .build-id generation to comply with Cuda repo requirements.
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define _build_id_links none \n%define debug_package %{nil}")
    set(CPACK_RPM_DCGM_USER_FILELIST "%license LICENSE")

ENDIF()