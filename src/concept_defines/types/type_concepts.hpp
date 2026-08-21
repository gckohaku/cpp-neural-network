#ifndef MKNNLIB_CONCEPT_DEFINES_TYPES_TYPE_CONCEPTS_HPP
#define MKNNLIB_CONCEPT_DEFINES_TYPES_TYPE_CONCEPTS_HPP

#include <cblas.h>
#include <openblas_config.h>

#include <complex>
#include <concepts>

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#endif

#include "src/matrices/core/blas_dispatchers/blas_primary_template.hpp"

namespace mknnlib::mk_concepts {

#if defined(__STDCPP_FLOAT16_T__)
template <typename T>
concept HalfFloatingPoint = std::same_as<T, std::float16_t>;
#endif

template <typename T>
concept SingleFloatingPoint = std::same_as<T, float>;

template <typename T>
concept DoubleFloatingPoint = std::same_as<T, double>;

#if defined(__STDCPP_FLOAT16_T__)
template <typename T>
concept HalfFloatingComplex = std::same_as < T,
        std::complex<std::float16_t>;
#endif

template <typename T>
concept SingleFloatingComplex = std::same_as<T, std::complex<float>>;

template <typename T>
concept DoubleFloatingComplex = std::same_as<T, std::complex<double>>;

template <typename Backend, typename T>
concept BLASSupported = requires { typename matrix::core::BLAS<Backend, T>; };

template <typename Backend, typename T>
concept AxpySupported = requires(const blasint n, const double alpha, const double* x, const blasint incx, double* y,
    const blasint incy) { matrix::core::BLAS<Backend, T>::axpy(n, alpha, x, incx, y, incy); };

template <typename Backend, typename T>
concept GemmSupported =
    requires(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
        const blasint M, const blasint N, const blasint K, const double alpha, const double* A, const blasint lda,
        const double* B, const blasint ldb, const double beta, double* C, const blasint ldc) {
        matrix::core::BLAS<Backend, T>::gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    };

template <typename Backend, typename T>
concept MatrixBLASSupported = BLASSupported<Backend, T> && AxpySupported<Backend, T> && GemmSupported<Backend, T>;
}  // namespace mknnlib::mk_concepts

#endif