#ifndef MKNNLIB_MATRICES_MATRIX_HPP
#define MKNNLIB_MATRICES_MATRIX_HPP

#include <openblas_config.h>

#include <algorithm>
#include <boost/operators.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mdspan>
#include <ostream>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp"  // IWYU pragma: keep
// #include "src/matrices/core/layout.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"
#include "src/matrices/core/transposed_state.hpp"

namespace mknnlib::matrix {
template <typename K, std::size_t Row, std::size_t Col, typename Backend,
    core::TransposedState State = core::TransposedState::None>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix {
    template <typename T, size_t FRow, size_t FCol, typename FBackend, core::TransposedState FState>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;

public:
    // type alias
    using MatrixExtent = std::extents<std::size_t, Row, Col>;
    using ElementType = K;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    MatrixExtent _extents;
    core::Storage<Backend, K, Row * Col> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix()
        requires(Matrix<K, Row, Col, Backend, State>::MatrixExtent::rank_dynamic() == 0);
    /* end constructors declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, State>::Matrix()
    requires(MatrixExtent::rank_dynamic() == 0)
    :
    _extents{},
    _elements{},
    _span(_elements.data(), MatrixExtent{})

{}
/* end constructors definition */
}  // namespace mknnlib::matrix

#endif