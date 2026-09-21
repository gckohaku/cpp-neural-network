#ifndef MKNNLIB_MATRICES_FORWARD_DECLARATIONS_DYNAMIC_OPERATIONS_HPP
#define MKNNLIB_MATRICES_FORWARD_DECLARATIONS_DYNAMIC_OPERATIONS_HPP

#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_template_base.hpp"

namespace mknnlib::matrix {
// Dynamic + Dynamic
template <typename K_, typename Backend_, typename Layout_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_> operator+(
    const Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>& lhs,
    const Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>& rhs);

template <typename K_, typename Backend_, typename Layout_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_> operator+(
    Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>&& lhs,
    const Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>& rhs);

template <typename K_, typename Backend_, typename Layout_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_> operator+(
    const Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>& lhs,
    Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>&& rhs);

template <typename K_, typename Backend_, typename Layout_>
    requires mk_concepts::BLASSupported<Backend_, K_>
Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_> operator+(
    Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>&& lhs,
    Matrix<K_, std::dynamic_extent, std::dynamic_extent, Backend_, Layout_>&& rhs);
}  // namespace mknnlib::matrix

#endif