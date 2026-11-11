#ifndef MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP
#define MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP

#include <type_traits>
#include "src/concept_defines/matrices/matrix_concepts.hpp"
#include "src/matrices/core/matrix_size_utilities.hpp"

namespace mknnlib::matrix {
template <typename L, typename R>
    requires(mk_concepts::ReadableMatrix<std::remove_cvref_t<L>> && mk_concepts::ReadableMatrix<std::remove_cvref_t<R>>)
auto operator+(L&& lhs, R&& rhs) {
    using Result = core::MatrixSameSizeOperationResult<L, R>;

    
}
}  // namespace mknnlib::matrix

#endif