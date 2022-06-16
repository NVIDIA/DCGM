
if (BUILD_TESTING)
    add_library(testing_cuda_stubs STATIC)
    target_sources(testing_cuda_stubs PRIVATE CudaCommonStub.cpp CudaStub.cpp cu_stubs.cpp)
    target_include_directories(testing_cuda_stubs PRIVATE ${CUDA11_INCLUDE_DIR})
    target_link_libraries(testing_cuda_stubs PUBLIC nvvs_interface)
    target_link_libraries(testing_cuda_stubs PRIVATE nvvs_plugins_common_interface)

    add_library(testing_dcgm_cublas_stubs STATIC)
    target_sources(testing_dcgm_cublas_stubs PRIVATE CublasStub.cpp)
    target_include_directories(testing_dcgm_cublas_stubs PRIVATE ${DCGM_ROOT_DIR}/cublas_proxy)
    target_include_directories(testing_dcgm_cublas_stubs PRIVATE ${CUDA11_INCLUDE_DIR})
endif()
