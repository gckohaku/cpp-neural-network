#ifndef MKNNLIB_MATRICES_CORE_BLAS_DISPATCHERS_OPENBLAS_HPP
#define MKNNLIB_MATRICES_CORE_BLAS_DISPATCHERS_OPENBLAS_HPP

#include <cblas.h>
#include <openblas_config.h>
#include <cstddef>

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
        } else if constexpr (mk_concepts::DoubleFloatingPoint<Type>) {
            cblas_daxpy(n, alpha, x, incx, y, incy);
        }
    }

    static void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
        const blasint M, const blasint N, const blasint K, const Type alpha, const Type* A, const blasint lda,
        const Type* B, const blasint ldb, const Type beta, Type* C, const blasint ldc) {
        if constexpr (mk_concepts::SingleFloatingPoint<Type>) {
            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } else if constexpr (mk_concepts::DoubleFloatingPoint<Type>) {
            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }


    static void add_into(std::size_t elementSize, Type* destination, const Type* source, std::ptrdiff_t destinationStride = 1, std::ptrdiff_t sourceStride = 1) {
        if constexpr (mk_concepts::SingleFloatingPoint<Type>) {
            cblas_saxpy(elementSize, Type{1}, source, sourceStride, destination, destinationStride);
        } else if constexpr (mk_concepts::DoubleFloatingPoint<Type>) {
            cblas_daxpy(elementSize, Type{1}, source, sourceStride, destination, destinationStride);
        }
    }

    static void add_scaled_into(std::size_t elementSize, Type* destination, const Type scaleSize, const Type* source, std::ptrdiff_t destinationStride = 1, std::ptrdiff_t sourceStride = 1) {
        if constexpr (mk_concepts::SingleFloatingPoint<Type>) {
            cblas_saxpy(elementSize, scaleSize, source, sourceStride, destination, destinationStride);
        } else if constexpr (mk_concepts::DoubleFloatingPoint<Type>) {
            cblas_daxpy(elementSize, scaleSize, source, sourceStride, destination, destinationStride);
        }
    }
};
}  // namespace mknnlib::matrix::core
#endif