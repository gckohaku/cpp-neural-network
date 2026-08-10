#ifndef MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP
#define MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP

#include <cstddef>
#include <span>

namespace mknnlib::matrix {
template <typename K, size_t Row = std::dynamic_extent, size_t Col = std::dynamic_extent>
class Matrix;

template <typename K, size_t Row, size_t Col>
using MatrixStatic = Matrix<K, Row, Col>;

template <typename K, size_t Row>
class Matrix<K, Row, std::dynamic_extent>;
template <typename K, size_t Row>
using MatrixRowStatic = Matrix<K, Row, std::dynamic_extent>;

template <typename K, size_t Col>
class Matrix<K, std::dynamic_extent, Col>;
template <typename K, size_t Col>
using MatrixColumnStatic = Matrix<K, std::dynamic_extent, Col>;

template <typename K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent>;
template <typename K>
using MatrixDynamic = Matrix<K, std::dynamic_extent, std::dynamic_extent>;
}

#endif