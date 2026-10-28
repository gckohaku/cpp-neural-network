#ifndef MKNNLIB_CONCEPT_DEFINES_MATRICES_MATRIX_CONCEPTS_HPP
#define MKNNLIB_CONCEPT_DEFINES_MATRICES_MATRIX_CONCEPTS_HPP

#include <concepts>
#include <type_traits>

#include "src/concept_defines/arithmetics.hpp"
#include "src/concept_defines/types/type_concepts.hpp"

namespace mknnlib::mk_concepts {

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#define ADDITIONAL_FLOAT_CONCEPT
template <typename T>
concept AdditionalFloatConcept = HalfFloatingPoint<T>;
#endif

#ifdef ADDITIONAL_FLOAT_CONCEPT
template <typename K>
concept MatrixElement = SingleFloatingPoint<K> && DoubleFloatingPoint<K> && SingleFloatingComplex<K> &&
    DoubleFloatingComplex<K> && AdditionalFloatingConcept<K>;
#else
template <typename K>
concept MatrixElement =
    SingleFloatingPoint<K> && DoubleFloatingPoint<K> && SingleFloatingComplex<K> && DoubleFloatingComplex<K>;
#endif

template <typename M1, typename M2, typename M3>
concept HasDotProduct = requires(M1 t, M2 u) {
    { t.dot(u) } -> std::same_as<M3>;
};

template <typename K, typename M>
concept MatrixClosedArithmetic = MatrixElement<K> && ClosedAddable<M> && ClosedSubtractable<M> && ClosedMultipliable<M>;

// matrix types concepts
template <typename M>
concept MatrixLike = requires(M m) {
    typename std::remove_cvref_t<M>::MatrixExtent;
    typename std::remove_cvref_t<M>::ElementType;

    requires(std::remove_cvref_t<M>::MatrixExtent::rank() == 2);
};

template <typename M>
concept ReadableMatrix =
    MatrixLike<M> && requires(const std::remove_cvref_t<M>& m, std::size_t row, std::size_t column) {
        { m.extent(0) } -> std::convertible_to<std::size_t>;
        { m.extent(1) } -> std::convertible_to<std::size_t>;
        { m[row, column] } -> std::convertible_to<typename std::remove_cvref_t<M>::ElementType>;
    };

template <typename M>
concept WritableMatrix = ReadableMatrix<M> &&
    requires(std::remove_cvref_t<M>& m, std::size_t row, std::size_t column,
        typename std::remove_cvref_t<M>::ElementType value) { m[row, column] = value; };
}  // namespace mknnlib::mk_concepts

#endif