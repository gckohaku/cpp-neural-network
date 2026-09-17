#ifndef MKNNLIB_MATRICES_CORE_BLAS_DISPATCHERS_OPENBLAS_HPP
#define MKNNLIB_MATRICES_CORE_BLAS_DISPATCHERS_OPENBLAS_HPP

#include <cblas.h>
#include <openblas_config.h>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_dispatchers/blas_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename Type>
struct BLAS<OpenBLASBackend, Type> {
    static void axpy(

        const blasint n, const Type alpha, const Type* x, const blasint incx, Type* y, const blasint incy) {
        if constexpr (mk_concepts::SingleFloatingPoint<Type>) {
            cblas_saxpy(n, alpha, x, incx, y, incy);
        }
        else if constexpr (mk_concepts::DoubleFloatingPoint<Type>) {
            cblas_daxpy(n, alpha, x, incx, y, incy);
        }
    }

    static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
        const blasint M, const blasint N, const blasint K, const float alpha, const float* A, const blasint lda,
        const float* B, const blasint ldb, const float beta, float* C, const blasint ldc) {
        cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

template <>
struct BLAS<OpenBLASBackend, double> {
    static void axpy(
        const blasint n, const double alpha, const double* x, const blasint incx, double* y, const blasint incy) {
        cblas_daxpy(n, alpha, x, incx, y, incy);
    }

    static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
        const blasint M, const blasint N, const blasint K, const double alpha, const double* A, const blasint lda,
        const double* B, const blasint ldb, const double beta, double* C, const blasint ldc) {
        cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
};

}  // namespace mknnlib::matrix::core
#endif