include(utils)

set(CUDA9_TOOLKIT_PREFIX "${CMAKE_FIND_ROOT_PATH}/usr/local/cuda-9.2"
    CACHE PATH "Path to CUDA9 toolkit installation")
set(CUDA10_TOOLKIT_PREFIX "${CMAKE_FIND_ROOT_PATH}/usr/local/cuda-10.0"
    CACHE PATH "Path to CUDA9 toolkit installation")
set(CUDA11_TOOLKIT_PREFIX "${CMAKE_FIND_ROOT_PATH}/usr/local/cuda-11.0"
    CACHE PATH "Path to CUDA9 toolkit installation")

mark_as_advanced(CUDA9_TOOLKIT_PREFIX CUDA10_TOOLKIT_PREFIX CUDA11_TOOLKIT_PREFIX)

macro (load_cuda cuda_version)
    foreach (cudapath ${CUDA${cuda_version}_TOOLKIT_PREFIX})
        list(APPEND Cuda${cuda_version}_INCLUDE_PATHS "${cudapath}/include")
        list(APPEND Cuda${cuda_version}_LIB_PATHS "${cudapath}/lib" "${cudapath}/lib64")
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
        unset(CUDA${cuda_version}_INCLUDE_DIR CACHE)
    endif ()

    if (libcublaslt${cuda_version})
        set(CUDA${cuda_version}_STATIC_CUBLAS_LIBS
            ${CUDA${cuda_version}_STATIC_CUBLAS_LIBS}
            ${libcublaslt${cuda_version}}
            CACHE STRING "Cuda${cuda_version} static libs"
            FORCE)
    endif ()

    if (Cuda${cuda_version}_FOUND)
        if (NOT Cuda_FIND_QUIETLY)
            message(STATUS "Found CUDA ${cuda_version}. CUDA${cuda_version}_INCLUDE_DIR=${CUDA${cuda_version}_INCLUDE_DIR}")
            message("Cublas lib: ${libcublas${cuda_version}}")
            message("Cudart lib: ${libcudart${cuda_version}}")
            message("Culibos lib: ${libculibos${cuda_version}}")
            message("CublasLt lib: ${libcublaslt${cuda_version}}")
            message("CUDA lib: ${libcuda${cuda_version}}")
        endif ()
    else ()
        if (Cuda_FIND_REQUIRED)
            message(${CUDA${cuda_version}_INCLUDE_DIR})
            message("Cublas lib: ${libcublas${cuda_version}}")
            message("Cudart lib: ${libcudart${cuda_version}}")
            message("Culibos lib: ${libculibos${cuda_version}}")
            message("CublasLt lib: ${libcublaslt${cuda_version}}")
            message("CUDA lib: ${libcuda${cuda_version}}")
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

if (NOT DEFINED CUDA9_INCLUDE_DIR AND ${CUDA9_ENABLED})
    load_cuda(9)
endif()

if (NOT DEFINED CUDA10_INCLUDE_DIR AND ${CUDA10_ENABLED})
    load_cuda(10)
endif()

if (NOT DEFINED CUDA11_INCLUDE_DIR AND ${CUDA11_ENABLED})
    load_cuda(11)
endif()

