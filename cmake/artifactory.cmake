function(prepare_rt_props COMPONENT)
set(options)
set(oneValueArgs REVISION)
set(multiValueArgs)
cmake_parse_arguments(PARSED "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

set(arch ${CMAKE_SYSTEM_PROCESSOR})
if (${arch} STREQUAL "aarch64")
    set(arch "sbsa")
endif()

# Default values for the "dcgm" component
set(COMPONENT_NAME ${COMPONENT})
set(TARGET_FILE "rt.props")

# Other components such as "config" have names like "dcgm_config" and print to "config.rt.props"
if (NOT ${COMPONENT} STREQUAL "dcgm")
    set(TARGET_FILE "${COMPONENT}.rt.props")
    set(COMPONENT_NAME "dcgm_${COMPONENT}")
endif()

set(RT_PROPS "")
list(APPEND RT_PROPS "arch=${arch}")
list(APPEND RT_PROPS "branch=${BUILD_BRANCH}")
list(APPEND RT_PROPS "changelist=${COMMIT_ID}")
list(APPEND RT_PROPS "os=linux")
list(APPEND RT_PROPS "platform=linux-${arch}")
if("${PARSED_REVISION}" STREQUAL "")
set(VER_VAL "${CMAKE_PROJECT_VERSION}")
else()
set(VER_VAL "${CMAKE_PROJECT_VERSION}.${PARSED_REVISION}")
endif()
if(DEFINED ENV{BUILD_NUMBER} AND NOT "$ENV{BUILD_NUMBER}" STREQUAL "")
set (VER_VAL "${VER_VAL}.$ENV{BUILD_NUMBER}")
endif()
list(APPEND RT_PROPS "version=${VER_VAL}")
list(APPEND RT_PROPS "component_name=${COMPONENT_NAME}")
#list(APPEND RT_PROPS "source=")

message("RT_PROPS: ${RT_PROPS}")
file(WRITE "${CMAKE_BINARY_DIR}/${TARGET_FILE}" "${RT_PROPS}")

endfunction()
