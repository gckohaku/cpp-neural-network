#ifndef MKNNLIB_MATRICES_FORWARD_DECLARATIONS_ROW_STATIC_OPERATIONS_HPP
#define MKNNLIB_MATRICES_FORWARD_DECLARATIONS_ROW_STATIC_OPERATIONS_HPP

#include <span>
#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_template_base.hpp"

namespace mknnlib::matrix {
template <typename K_, size_t Row_, typename Backend_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, Row_, std::dynamic_extent, Backend_> operator+(Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, std::dynamic_extent, Backend_> rhs);

template <typename K_, size_t Row_, size_t Col_, typename Backend_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, Row_, Col_, Backend_> operator+(
    Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, Col_, Backend_> rhs);

template <typename K_, size_t Row_, typename Backend_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, Row_, std::dynamic_extent, Backend_> operator-(Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, std::dynamic_extent, Backend_> rhs);

template <typename K_, size_t Row_, size_t Col_, typename Backend_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, Row_, Col_, Backend_> operator-(Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, Col_, Backend_> rhs);
}  // namespace mknnlib::matrices
#endif