#ifndef MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP
#define MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP

#include <type_traits>
#include "src/concept_defines/matrices/matrix_concepts.hpp"
#include "src/matrices/core/matrix_size_utilities.hpp"

namespace mknnlib::matrix {
template <typename L, typename R, typename Backend>
requires(core::CompatibleMatrixShape<L, R>)
auto MakeBinaryOperatorResultMatrix(L lhs, R rhs) {
    using LeftExtents = core::MatrixExtents<std::remove_cvref_t<L>>;
    using RightExtents = core::MatrixExtents<std::remove_cvref_t<R>>;
    using ElementType = std::remove_cvref_t<L>::ElementType;

    // Matrix<ElementType, core::MergeExtent<, std::size_t R>>
}

template <typename L, typename R, typename Backend>
    requires(mk_concepts::ReadableMatrix<std::remove_cvref_t<L>> && mk_concepts::ReadableMatrix<std::remove_cvref_t<R>>)
auto operator+(L&& lhs, R&& rhs) {
    using Result = core::MatrixSameSizeOperationResult<L, R>;
    using ElementType = std::remove_cvref_t<L>::ElementType;

    if constexpr (mk_concepts::ReusableMatrix<L, Result>) {
        core::BLAS<Backend, ElementType>::add_assign(lhs, rhs);
        return std::forward(lhs);
    }
    else if constexpr (mk_concepts::ReusableMatrix<R, Result>) {
        core::BLAS<Backend, ElementType>::add_assign(rhs, lhs);
        return std::forward(rhs);
    }
    else {
        
    }
}
}  // namespace mknnlib::matrix

#endif