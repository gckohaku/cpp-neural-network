#ifndef MKNNLIB_CONCEPT_DEFINES_TYPES_TYPE_CONCEPTS_HPP
#define MKNNLIB_CONCEPT_DEFINES_TYPES_TYPE_CONCEPTS_HPP

#include <complex>
#include <concepts>

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#endif

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

template <typename T>
concept BLASComputable = SingleFloatingPoint<T> || DoubleFloatingPoint<T>;
}  // namespace mknnlib::mk_concepts

#endif