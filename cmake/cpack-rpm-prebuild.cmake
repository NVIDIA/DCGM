#[[
Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

macro(install_license)
    file(MAKE_DIRECTORY "${PREFIX}/usr/share/licenses/${PACKAGE_NAME}-${PACKAGE_VERSION}/")

    file(COPY_FILE
        "${CPACK_RESOURCE_FILE_LICENSE}"
        "${PREFIX}/usr/share/licenses/${PACKAGE_NAME}-${PACKAGE_VERSION}/LICENSE")
endmacro()

if(DEFINED CPACK_RPM_PACKAGE_VERSION)
  set(PACKAGE_VERSION "${CPACK_RPM_PACKAGE_VERSION}")
else()
  set(PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
endif()

if (NOT CPACK_RPM_COMPONENT_INSTALL)
    set(PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
    set(PREFIX "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
    install_license()
else()
    foreach(Component IN LISTS CPACK_COMPONENTS_ALL)
        string(TOUPPER "${Component}" COMPONENT)

        if(CPACK_RPM_MAIN_COMPONENT STREQUAL Component)
            if(DEFINED CPACK_RPM_PACKAGE_NAME)
                set(PACKAGE_NAME "${CPACK_RPM_PACKAGE_NAME}")
            elseif(DEFINED CPACK_PACKAGE_NAME)
                set(PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
            endif()
        elseif(DEFINED CPACK_RPM_${Component}_PACKAGE_NAME)
            set(PACKAGE_NAME "${CPACK_RPM_${Component}_PACKAGE_NAME}")
        elseif(DEFINED CPACK_RPM_${COMPONENT}_PACKAGE_NAME)
            set(PACKAGE_NAME "${CPACK_RPM_${COMPONENT}_PACKAGE_NAME}")
        elseif(DEFINED CPACK_RPM_PACKAGE_NAME)
            set(PACKAGE_NAME "${CPACK_RPM_PACKAGE_NAME}-${Component}")
        else()
            set(PACKAGE_NAME "${CPACK_PACKAGE_NAME}-${Component}")
        endif()

        set(PREFIX "${CPACK_TEMPORARY_INSTALL_DIRECTORY}/${Component}")
        install_license()
    endforeach()
endif()
