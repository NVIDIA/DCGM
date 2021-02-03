include(utils)

set(Cuda9_prefix usr/local/cuda-9.2)
set(Cuda10_prefix usr/local/cuda-10.0)
set(Cuda11_prefix usr/local/cuda-11.0)

macro (load_cuda cuda_version)
    foreach (prefix ${Cuda${cuda_version}_prefix})
        list(APPEND Cuda${cuda_version}_INCLUDE_PATHS "${CMAKE_FIND_ROOT_PATH}/${prefix}/include")
        list(APPEND Cuda${cuda_version}_LIB_PATHS "${CMAKE_FIND_ROOT_PATH}/${prefix}/lib" "${CMAKE_FIND_ROOT_PATH}/${prefix}/lib64")
    endforeach ()

    find_path(CUDA${cuda_version}_INCLUDE_DIR cuda.h PATHS ${Cuda${cuda_version}_INCLUDE_PATHS} NO_DEFAULT_PATHS)
    if (CUDA${cuda_version}_INCLUDE_DIR)
        # Dereference symbolic links
        get_absolute_path(${CUDA${cuda_version}_INCLUDE_DIR} CUDA${cuda_version}_INCLUDE_DIR)
    endif()
    find_library(libcudart${cuda_version} libcudart_static.a PATHS ${Cuda${cuda_version}_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(libculibos${cuda_version} libculibos.a PATHS ${Cuda${cuda_version}_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(libcuda${cuda_version} libcuda.so PATHS ${Cuda${cuda_version}_LIB_PATHS} PATH_SUFFIXES stubs NO_DEFAULT_PATH)

    find_library(libcublas${cuda_version} libcublas_static.a PATHS ${Cuda${cuda_version}_LIB_PATHS} NO_DEFAULT_PATH)
    find_library(libcublaslt${cuda_version} libcublasLt_static.a PATHS ${Cuda${cuda_version}_LIB_PATHS} NO_DEFAULT_PATH)

    if (CUDA${cuda_version}_INCLUDE_DIR 
        AND libcublas${cuda_version}
        AND libcudart${cuda_version}
        AND libculibos${cuda_version}
        AND libcuda${cuda_version})
        set(Cuda${cuda_version}_FOUND TRUE)

        set(CUDA${cuda_version}_STATIC_LIBS
            ${libcudart${cuda_version}}
            ${libculibos${cuda_version}}
            CACHE STRING "Cuda${cuda_version} static libs")
        set(CUDA${cuda_version}_LIBS ${libcuda${cuda_version}} CACHE STRING "Cuda${cuda_version} shared libs")
        set(CUDA${cuda_version}_STATIC_CUBLAS_LIBS ${libcublas${cuda_version}} CACHE STRING "Cuda${cuda_version} static libs")
    else ()
        set(Cuda${cuda_version}_FOUND FALSE)
    endif ()

    if (libcublaslt${cuda_version})
        set(CUDA${cuda_version}_STATIC_CUBLAS_LIBS
            ${CUDA${cuda_version}_STATIC_CUBLAS_LIBS}
            ${libcublaslt${cuda_version}}
            CACHE STRING "Cuda${cuda_version} static libs"
            FORCE)
    endif ()

    if (Cuda${cuda_version}_FOUND)
        if (NOT Cuda_FIND_QUETLY)
            message(STATUS "Found CUDA ${cuda_version}. CUDA${cuda_version}_INCLUDE_DIR=${CUDA${cuda_version}_INCLUDE_DIR}")
            message(${libcublas${cuda_version}})
            message(${libcudart${cuda_version}})
            message(${libculibos${cuda_version}})
            message(${libcublaslt${cuda_version}})
            message(${libcuda${cuda_version}})
        endif ()
    else ()
        if (Cuda_FIND_REQUIRED)
            message(${CUDA${cuda_version}_INCLUDE_DIR})
            message(${libcublas${cuda_version}})
            message(${libcudart${cuda_version}})
            message(${libculibos${cuda_version}})
            message(${libcublaslt${cuda_version}})
            message(${libcuda${cuda_version}})
            message(FATAL_ERROR "Could NOT find Cuda ${cuda_version}")
        endif ()
        message(STATUS "Cuda ${cuda_version} NOT found")
    endif ()

    unset(libcublas${cuda_version})
    unset(libcublaslt${cuda_version})
    unset(libcudart${cuda_version})
    unset(libculibos${cuda_version})
    unset(libcuda${cuda_version})

    mark_as_advanced(CUDA${cuda_version}_INCLUDE_DIR CUDA${cuda_version}_STATIC_LIBS CUDA${cuda_version}_LIBS CUDA${cuda_version}_STATIC_CUBLAS_LIBS)

endmacro()

if (NOT DEFINED CUDA9_INCLUDE_DIR AND NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    load_cuda(9)
endif()

if (NOT DEFINED CUDA10_INCLUDE_DIR AND NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    load_cuda(10)
endif()

if (NOT DEFINED CUDA11_INCLUDE_DIR)
    load_cuda(11)
endif()

unset(Cuda9_prefix)
unset(Cuda10_prefix)
unset(Cuda11_prefix)
