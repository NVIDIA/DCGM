#include <cublas_proxy.hpp>
#include <cublas_v2.h>

namespace Dcgm::CublasProxy
{
cublasStatus_t CublasGetVersion(cublasHandle_t handle, int *version)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const double *alpha, /* host or device pointer */
                           const double *A,
                           int lda,
                           const double *B,
                           int ldb,
                           const double *beta, /* host or device pointer */
                           double *C,
                           int ldc)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const float *alpha, /* host or device pointer */
                           const float *A,
                           int lda,
                           const float *B,
                           int ldb,
                           const float *beta, /* host or device pointer */
                           float *C,
                           int ldc)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasCreate(cublasHandle_t *handle)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasDestroy(cublasHandle_t handle)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const __half *alpha, /* host or device pointer */
                           const __half *A,
                           int lda,
                           const __half *B,
                           int ldb,
                           const __half *beta, /* host or device pointer */
                           __half *C,
                           int ldc)
{
    return CUBLAS_STATUS_SUCCESS;
}


}; // namespace Dcgm::CublasProxy
