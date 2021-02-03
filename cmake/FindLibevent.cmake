# - Find Libevent (a cross event library)
# This module defines
# LIBEVENT_INCLUDE_DIR, where to find Libevent headers
# LIBEVENT_STATIC_LIBS, Libevent static libraries
# Libevent_FOUND, If false, do not try to use libevent

set(Libevent_EXTRA_PREFIXES / /lib /lib64 /usr/local /opt/local "$ENV{HOME}" "${Libevent_ROOT}")
foreach(prefix ${Libevent_EXTRA_PREFIXES})
  list(APPEND Libevent_INCLUDE_PATHS "${prefix}/include")
  list(APPEND Libevent_LIB_PATHS "${prefix}/lib" "${prefix}/lib64")
endforeach()

find_path(LIBEVENT_INCLUDE_DIR evhttp.h event.h PATHS ${Libevent_INCLUDE_PATHS})
find_library(LIBEVENT_STATIC_LIB NAMES libevent.a libevent_core.a libevent_extra.a PATHS ${Libevent_LIB_PATHS})
find_library(LIBEVENT_PTHREAD_STATIC_LIB NAMES libevent_pthreads.a PATHS ${Libevent_LIB_PATHS})

if (LIBEVENT_INCLUDE_DIR AND LIBEVENT_STATIC_LIB AND LIBEVENT_PTHREAD_STATIC_LIB)
    set(Libevent_FOUND TRUE)
    add_library(libevent_event_static STATIC IMPORTED)
    set_target_properties(libevent_event_static PROPERTIES IMPORTED_LOCATION ${LIBEVENT_STATIC_LIB})
    add_library(libevent_event_pthread STATIC IMPORTED)
    set_target_properties(libevent_event_pthread PROPERTIES IMPORTED_LOCATION ${LIBEVENT_PTHREAD_STATIC_LIB})
    set(LIBEVENT_STATIC_LIBS libevent_event_static libevent_event_pthread)
else ()
    set(Libevent_FOUND FALSE)
endif ()

if (Libevent_FOUND)
    if (NOT Libevent_FIND_QUIETLY)
        message(STATUS "Found libevent: ${LIBEVENT_LIB}")
    endif ()
else ()
    if (Libevent_FIND_REQUIRED)
        message(FATAL_ERROR "Could NOT find libevent and libevent_pthread.")
    endif ()
    message(STATUS "libevent and libevent_pthread NOT found.")
endif ()

unset(Libevent_EXTRA_PREFIXES)
unset(LIBEVENT_PTHREAD_STATIC_LIB)
unset(LIBEVENT_STATIC_LIB)

mark_as_advanced(
    LIBEVENT_STATIC_LIBS
    LIBEVENT_INCLUDE_DIR
    )
