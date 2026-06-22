#ifndef MKNNLIB_CONCEPT_DEFINES_MATRICES_MATRIX_CONCEPTS_HPP
#define MKNNLIB_CONCEPT_DEFINES_MATRICES_MATRIX_CONCEPTS_HPP

#include <concepts>

#include "src/concept_defines/arithmetics.hpp"
#include "src/concept_defines/types/type_concepts.hpp"

namespace mknnlib::concepts {

#if defined(__STDCPP_FLOAT16_T__)
#include <stdfloat>
#define ADDITIONAL_FLOAT_CONCEPT
template <typename Tstd>
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
}  // namespace mknnlib::matrix

#endif