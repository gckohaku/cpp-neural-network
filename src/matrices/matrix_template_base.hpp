#ifndef MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP
#define MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP

#include <cstddef>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Row = std::dynamic_extent, size_t Col = std::dynamic_extent>
    requires mk_concepts::BLASSupported<K>
class Matrix;

template <typename K, size_t Row, size_t Col>
    requires mk_concepts::BLASSupported<K>
using MatrixStatic = Matrix<K, Row, Col>;

template <typename K, size_t Row>
    requires mk_concepts::BLASSupported<K>
class Matrix<K, Row, std::dynamic_extent>;

template <typename K, size_t Row>
    requires mk_concepts::BLASSupported<K>
using MatrixRowStatic = Matrix<K, Row, std::dynamic_extent>;

template <typename K, size_t Col>
    requires mk_concepts::BLASSupported<K>
class Matrix<K, std::dynamic_extent, Col>;
template <typename K, size_t Col>
    requires mk_concepts::BLASSupported<K>
using MatrixColumnStatic = Matrix<K, std::dynamic_extent, Col>;

template <typename K>
    requires mk_concepts::BLASSupported<K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent>;
template <typename K>
    requires mk_concepts::BLASSupported<K>
using MatrixDynamic = Matrix<K, std::dynamic_extent, std::dynamic_extent>;
}  // namespace mknnlib::matrix

#endif