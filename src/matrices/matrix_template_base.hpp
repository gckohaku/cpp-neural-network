#ifndef MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP
#define MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP

#include <cstddef>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_backends.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Row = std::dynamic_extent, size_t Col = std::dynamic_extent, typename Backend = core::OpenBLASBackend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, size_t Row, size_t Col>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixStaticOpen = Matrix<K, Row, Col, core::OpenBLASBackend>;

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, Row, std::dynamic_extent, Backend>;

template <typename K, size_t Row>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixRowStaticOpen = Matrix<K, Row, std::dynamic_extent, core::OpenBLASBackend>;

template <typename K, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, Col, Backend>;

template <typename K, size_t Col>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixColumnStaticOpen = Matrix<K, std::dynamic_extent, Col, core::OpenBLASBackend>;

template <typename K, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend>;

template <typename K>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixDynamicOpen = Matrix<K, std::dynamic_extent, std::dynamic_extent, core::OpenBLASBackend>;
}  // namespace mknnlib::matrix

#endif