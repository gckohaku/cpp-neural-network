#ifndef MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP
#define MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP

#include <iostream>
#include <type_traits>
#include "src/concept_defines/matrices/matrix_concepts.hpp"
#include "src/matrices/core/matrix_size_utilities.hpp"

namespace mknnlib::matrix {
template <typename L, typename R>
    requires(mk_concepts::ReadableMatrix<std::remove_cvref_t<L>> && mk_concepts::ReadableMatrix<std::remove_cvref_t<R>> && core::CompatibleMatrixShape<L, R>)
auto operator+(L&& lhs, R&& rhs) -> core::MatrixSameSizeOperationResult<L, R> {
    using Result = core::MatrixSameSizeOperationResult<L, R>;
    using ElementType = std::remove_cvref_t<L>::ElementType;
    using Backend = std::remove_cvref_t<L>::MatrixBackend;;

    if constexpr (mk_concepts::ReusableMatrix<L, Result>) {
        std::cout << "lhs is reusable" << std::endl;
        core::BLAS<Backend, ElementType>::add_assign(lhs, rhs);
        return std::forward<L>(lhs);
    }
    else if constexpr (mk_concepts::ReusableMatrix<R, Result>) {
        std::cout << "rhs is reusable" << std::endl;
        core::BLAS<Backend, ElementType>::add_assign(rhs, lhs);
        return std::forward<R>(rhs);
    }
    else {
        std::cout << "both matrix is not reusable" << std::endl;
        Result result(lhs);
        core::BLAS<Backend, ElementType>::add_assign(result, rhs);
        return result;
    }
}
}  // namespace mknnlib::matrix

#endif