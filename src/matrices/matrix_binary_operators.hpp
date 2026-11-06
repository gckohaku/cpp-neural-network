#ifndef MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP
#define MKNNLIB_MATRICES_MATRIX_BINARY_OPERATORS_HPP

#include <type_traits>
#include "src/concept_defines/matrices/matrix_concepts.hpp"
namespace mknnlib::matrix {
template <typename L, typename R>
    requires(mk_concepts::ReadableMatrix<std::remove_cvref_t<L>> && mk_concepts::ReadableMatrix<std::remove_cvref_t<R>>)
auto operator+(L&& lhs, R&& rhs) {}
}  // namespace mknnlib::matrix

#endif