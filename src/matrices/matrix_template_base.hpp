#ifndef MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP
#define MKNNLIB_MATRICES_MATRIX_TEMPLATE_BASE_HPP

#include <cstddef>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/layout.hpp"

namespace mknnlib::matrix {
// primary
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

// both static
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<Backend, K>
using MatrixStatic = Matrix<K, Row, Col, Backend, Layout>;

template <typename K, size_t Row, size_t Col, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixStaticOpen = Matrix<K, Row, Col, core::OpenBLASBackend, Layout>;

template <typename K, size_t Row, size_t Col, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::CuBLASBackend, K>
using MatrixStaticCu = Matrix<K, Row, Col, core::CuBLASBackend, Layout>;

// row static
template <typename K, size_t Row, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, Row, std::dynamic_extent, Backend, Layout>;

template <typename K, size_t Row, typename Backend, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<Backend, K>
using MatrixRowStatic = Matrix<K, Row, std::dynamic_extent, Backend, Layout>;

template <typename K, size_t Row, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixRowStaticOpen = Matrix<K, Row, std::dynamic_extent, core::OpenBLASBackend, Layout>;

template <typename K, size_t Row, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::CuBLASBackend, K>
using MatrixRowStaticCu = Matrix<K, Row, std::dynamic_extent, core::CuBLASBackend, Layout>;

// column static
template <typename K, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, Col, Backend, Layout>;

template <typename K, size_t Col, typename Backend, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<Backend, K>
using MatrixColumnStatic = Matrix<K, std::dynamic_extent, Col, Backend, Layout>;

template <typename K, size_t Col, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixColumnStaticOpen = Matrix<K, std::dynamic_extent, Col, core::OpenBLASBackend, Layout>;

template <typename K, size_t Col, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::CuBLASBackend, K>
using MatrixColumnStaticCu = Matrix<K, std::dynamic_extent, Col, core::CuBLASBackend, Layout>;

// both dynamic
template <typename K, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>;

template <typename K, typename Backend, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<Backend, K>
using MatrixDynamic = Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>;

template <typename K, typename Layout = core::MatrixColumnMajor>
    requires mk_concepts::BLASSupported<core::OpenBLASBackend, K>
using MatrixDynamicOpen = Matrix<K, std::dynamic_extent, std::dynamic_extent, core::OpenBLASBackend, Layout>;

template <typename K, typename Layout>
    requires mk_concepts::BLASSupported<core::CuBLASBackend, K>
using MatrixDynamicCu = Matrix<K, std::dynamic_extent, std::dynamic_extent, core::CuBLASBackend, Layout>;

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
struct MatrixTranspose_base {
    using type = Matrix<K, Col, Row, Backend, core::MatrixTransposeOrderLayout_t<Layout>>;
};

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
using MatrixTranspose = typename MatrixTranspose_base<K, Row, Col, Backend, Layout>::type;
}  // namespace mknnlib::matrix

#endif