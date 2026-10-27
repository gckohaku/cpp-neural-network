#ifndef MKNNLIB_MATRICES_MATRIX_HPP
#define MKNNLIB_MATRICES_MATRIX_HPP

#include <openblas_config.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mdspan>
#include <ostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/concept_defines/matrices/matrix_concepts.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp"  // IWYU pragma: keep
#include "src/matrices/core/transposed_state.hpp"
#include "src/matrices/core/type_utilities.hpp"

namespace mknnlib::matrix {
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend, State>& mat);

template <typename K, std::size_t Row, std::size_t Col, typename Backend,
    core::TransposedState State = core::TransposedState::None>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix {
    template <typename T, std::size_t FRow, std::size_t FCol, typename FBackend, core::TransposedState FState>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;

public:
    // type alias
    using MatrixExtent = std::extents<std::size_t, Row, Col>;
    using ElementType = K;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    MatrixExtent _extents;
    core::Storage<Backend, K, core::MatrixElementSize<MatrixExtent>> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    // both static
    Matrix()
        requires(Matrix::MatrixExtent::rank_dynamic() == 0);
    template <std::size_t ArrSize>
    Matrix(std::array<K, ArrSize> elements)
        requires(Matrix::MatrixExtent::rank_dynamic() == 0 && ArrSize == Row * Col);
    Matrix(std::vector<K> elements)
        requires(Matrix::MatrixExtent::rank_dynamic() == 0);

    // dynamic included
    template <mk_concepts::SizeCompatible... Sizes>
    explicit Matrix(Sizes... sizes)
        requires(Matrix::MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == Matrix::MatrixExtent::rank_dynamic());

    template <std::size_t ArrSize, mk_concepts::SizeCompatible... Sizes>
    explicit Matrix(std::array<K, ArrSize> elements, Sizes... sizes)
        requires(Matrix::MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == Matrix::MatrixExtent::rank_dynamic());

    template <mk_concepts::SizeCompatible... Sizes>
    explicit Matrix(std::vector<K> elements, Sizes... sizes)
        requires(Matrix::MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == Matrix::MatrixExtent::rank_dynamic());

    /* end constructors declaration */

    /* begin operator overloads declaration */
    // arithmetics compound operators
    template <mk_concepts::ReadableMatrix OppM>
    auto operator+=(OppM mat);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    constexpr size_t ColumnSize();
    std::string GetSizeString() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <std::size_t OppCol, core::TransposedState OppState>
    Matrix<K, Row, OppCol, Backend, State> Dot(const Matrix<K, Col, OppCol, Backend, OppState>& mat);
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
// both static
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, State>::Matrix()
    requires(MatrixExtent::rank_dynamic() == 0)
    : _extents{}, _elements{}, _span(_elements.data(), Matrix::MatrixExtent{}) {}

template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
template <std::size_t ArrSize>
Matrix<K, Row, Col, Backend, State>::Matrix(std::array<K, ArrSize> elements)
    requires(MatrixExtent::rank_dynamic() == 0 && ArrSize == Row * Col)
    : _extents{}, _elements{elements}, _span(_elements.data(), Matrix::MatrixExtent{}) {}

template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, State>::Matrix(std::vector<K> elements)
    requires(MatrixExtent::rank_dynamic() == 0)
    : _extents{}, _elements{elements}, _span(_elements.data(), Matrix::MatrixExtent{}) {}

// dynamic included
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
template <mk_concepts::SizeCompatible... Sizes>
Matrix<K, Row, Col, Backend, State>::Matrix(Sizes... sizes)
    requires(MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == MatrixExtent::rank_dynamic())
    : _extents{sizes...}, _elements{}, _span(_elements.data(), Matrix::MatrixExtent{}) {}

template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
template <std::size_t ArrSize, mk_concepts::SizeCompatible... Sizes>
Matrix<K, Row, Col, Backend, State>::Matrix(std::array<K, ArrSize> elements, Sizes... sizes)
    requires(MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == MatrixExtent::rank_dynamic())
    : _extents{sizes...}, _elements{elements}, _span(_elements.data(), Matrix::MatrixExtent{}) {}

template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
template <mk_concepts::SizeCompatible... Sizes>
Matrix<K, Row, Col, Backend, State>::Matrix(std::vector<K> elements, Sizes... sizes)
    requires(MatrixExtent::rank_dynamic() > 0 && sizeof...(Sizes) == MatrixExtent::rank_dynamic())
    : _extents{sizes...}, _elements{elements}, _span(_elements.data(), Matrix::MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// arithmetics compound operators
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
template <mk_concepts::ReadableMatrix OppM>
inline auto Matrix<K, Row, Col, Backend, State>::operator+=(OppM mat) {
#if !defined(NDEBUG)
    if (this->RowSize() != mat.RowSize() || this->ColumnSize() != mat.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    return *this;
}

// ostream
template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend, State>& mat) {
    for (size_t i = 0; i < mat._extents.extent(0); i++) {
        for (size_t j = 0; j < mat._extents.extent(1); j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Row, size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, Row, Col, Backend, State>::GetSizeString() const {
    return "(" + std::to_string(_extents.extent(0)) + ", " + std::to_string(_extents.extent(1)) + ")";
}

template <typename K, size_t Row, size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend, State>::RowSize() {
    return _extents.extent(0);
}

template <typename K, size_t Row, size_t Col, typename Backend, core::TransposedState State>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend, State>::ColumnSize() {
    return _extents.extent(1);
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
// template <typename K, std::size_t Row, std::size_t Col, typename Backend, core::TransposedState State>
// requires mk_concepts::BLASSupported<Backend, K>
// template <std::size_t OppCol, core::TransposedState OppState>
// inline Matrix<K, Row, OppCol, Backend, State> Matrix<K, Row, Col, Backend, State>::Dot(Matrix<K, Col, OppCol,
// Backend, OppState> mat) {

// }
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif