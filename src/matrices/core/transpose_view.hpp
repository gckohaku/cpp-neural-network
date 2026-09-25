#ifndef MKNNLIB_MATRICES_CORE_TRANSPOSE_VIEW_HPP
#define MKNNLIB_MATRICES_CORE_TRANSPOSE_VIEW_HPP

#include <type_traits>
#include "src/matrices/core/transposed_state.hpp"

namespace mknnlib::matrix::core {
    template <TransposedState State>
    using ApplyTransposeToState = std::integral_constant<TransposedState, State ^ TransposedState::Transpose>;

    template <TransposedState State>
    using ApplyCTransposeToState = std::integral_constant<TransposedState, State ^ TransposedState::ConjugateTranspose>;
    template <TransposedState State>
    using ApplyHermitianTransposeToState = std::integral_constant<TransposedState, State ^ TransposedState::ConjugateTranspose>;

    template <TransposedState State>
    using ApplyConjugateToState = std::integral_constant<TransposedState, State ^ TransposedState::ConjugateNone>;

    TransposedState a = ApplyTransposeToState<TransposedState::ConjugateNone>::value;
}

#endif