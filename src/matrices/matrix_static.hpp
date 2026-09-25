#ifndef MKNNLIB_MATRICES_MATRIX_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>
#include <span>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/core/blas_dispatchers/blas_dispatchers.hpp"  // IWYU pragma: keep
#include "src/matrices/core/blas_dispatchers/blas_primary_template.hpp"
#include "src/matrices/core/blas_storages/blas_storage.hpp"         // IWYU pragma: keep
#include "src/matrices/forward_declarations/static_operations.hpp"  // IWYU pragma: keep
#include "src/matrices/matrix_template_base.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend, Layout>& mat);

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix :
    private boost::multipliable<Matrix<K, Row, Col, Backend, Layout>>,
    private boost::multipliable<Matrix<K, Row, Col, Backend, Layout>,
        Matrix<K, Row, std::dynamic_extent, Backend, Layout>> {
    static_assert(Row != std::dynamic_extent);
    static_assert(Col != std::dynamic_extent);

    template <typename T, size_t FRow, size_t FCol, typename FBackend, typename FLayout>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;

    // type alias
    using MatrixExtent = std::extents<size_t, Row, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    // std::array<K, Row * Col> _elements;
    core::Storage<Backend, K, Row * Col> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix();
    Matrix(const std::array<K, Row * Col> elements);
    Matrix(const std::vector<K>& elements);
    // copy constructor
    Matrix(const Matrix& mat);
    // move constructor
    Matrix(Matrix&&);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix& operator=(const Matrix& x);
    // move assignment operator
    Matrix& operator=(Matrix&&);
    // arithmetics compound operators
    Matrix& operator+=(const Matrix& x);
    Matrix& operator+=(const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x);
    Matrix& operator-=(const Matrix& x);
    Matrix& operator-=(const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x);
    Matrix& operator*=(const Matrix& x);
    Matrix& operator*=(const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x);

    // arithmetics binary operators

    friend Matrix operator+
        <K, Row, Col, Backend, Layout>(const Matrix& lhs, Matrix<K, Row, std::dynamic_extent, Backend, Layout>&& rhs);

    friend Matrix operator- <K, Row, Col, Backend, Layout>(Matrix lhs, Matrix rhs);
    friend Matrix operator-
        <K, Row, Col, Backend, Layout>(Matrix lhs, Matrix<K, Row, std::dynamic_extent, Backend, Layout> rhs);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    constexpr size_t ColumnSize();
    std::string GetSizeString() const;
    constexpr core::Storage<Backend, K, Row * Col>& Elements() noexcept;
    constexpr const core::Storage<Backend, K, Row * Col>& Elements() const noexcept;
    // constexpr std::array<K, Row * Col>& ElementsRange() noexcept;
    // constexpr const std::array<K, Row * Col>& ElementsRange() const noexcept;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend, Layout> Dot(const Matrix<K, Col, OppCol, Backend, Layout>& mat);

    Matrix<K, Row, std::dynamic_extent, Backend, Layout> Dot(
        const Matrix<K, std::dynamic_extent, Col, Backend, Layout>& mat);

    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend, Layout> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend, Layout>& mat);

    Matrix<K, Row, std::dynamic_extent, Backend, Layout> Dot(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat);
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>::Matrix() :
    _rowSize(Row), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>::Matrix(const std::array<K, Row * Col> elements) :
    _rowSize(Row), _columnSize(Col), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>::Matrix(const std::vector<K>& elements) :
    _rowSize(Row), _columnSize(Col), _span(_elements.data(), MatrixExtent{}) {
    std::copy(elements.begin(), elements.end(), _elements.begin());
}

// copy constructor
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>::Matrix(const Matrix<K, Row, Col, Backend, Layout>& mat) :
    _rowSize(Row), _columnSize(Col), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}

// move constructor
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>::Matrix(Matrix<K, Row, Col, Backend, Layout>&& mat) :
    _rowSize(Row), _columnSize(Col), _elements(std::move(mat._elements)), _span(_elements.data(), MatrixExtent{}) {}

/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>& Matrix<K, Row, Col, Backend, Layout>::operator=(
    const Matrix<K, Row, Col, Backend, Layout>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(_elements.data(), MatrixExtent{});
    }
    return *this;
}

// move assignment operator
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, Col, Backend, Layout>& Matrix<K, Row, Col, Backend, Layout>::operator=(
    Matrix<K, Row, Col, Backend, Layout>&& x) {
    if (this != &x) {
        _elements = std::move(x._elements);
        _span = MdView(_elements.data(), MatrixExtent{});
    }
    return *this;
}

// arithmetics compound operators
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend, Layout>::operator+=(const Matrix& x) -> Matrix& {
    core::BLAS<Backend, K>::axpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend, Layout>::operator+=(
    const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x) -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    core::BLAS<Backend, K>::axpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend, Layout>::operator-=(const Matrix& x) -> Matrix& {
    core::BLAS<Backend, K>::axpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline auto Matrix<K, Row, Col, Backend, Layout>::operator-=(
    const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x) -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    core::BLAS<Backend, K>::axpy(Row * Col, -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
auto Matrix<K, Row, Col, Backend, Layout>::operator*=(const Matrix<K, Row, Col, Backend, Layout>& x) -> Matrix& {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x.Elements(), this->_elements.begin(), std::multiplies<>());
    return *this;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
auto Matrix<K, Row, Col, Backend, Layout>::operator*=(const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& x)
    -> Matrix& {
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x.Elements(), this->_elements.begin(), std::multiplies<>());
    return *this;
}

// arithmetics binary operators
// Static + Static
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>& lhs, Matrix<K, Row, Col, Backend, Layout>& rhs) {
    auto result = lhs;
    result += rhs;
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>&& lhs, Matrix<K, Row, Col, Backend, Layout>& rhs) {
    lhs += rhs;
    return std::move(lhs);
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>& lhs, Matrix<K, Row, Col, Backend, Layout>&& rhs) {
    rhs += lhs;
    return std::move(rhs);
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>&& lhs, Matrix<K, Row, Col, Backend, Layout>&& rhs) {
    lhs += rhs;
    return std::move(lhs);
}

// Static + RowStatic
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    const Matrix<K, Row, Col, Backend, Layout>& lhs, const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = lhs;
    result += rhs;
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>&& lhs, const Matrix<K, Row, std::dynamic_extent, Backend, Layout>& rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    lhs += rhs;
    return std::move(lhs);
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    const Matrix<K, Row, Col, Backend, Layout>& lhs, Matrix<K, Row, std::dynamic_extent, Backend, Layout>&& rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    core::BLAS<Backend, K>::axpy(Row * Col, 1.0, lhs._elements.data(), 1, rhs._elements.data(), 1);
    return std::move(lhs);
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator+(
    Matrix<K, Row, Col, Backend, Layout>&& lhs, Matrix<K, Row, std::dynamic_extent, Backend, Layout>&& rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    lhs += rhs;
    return std::move(lhs);
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
inline Matrix<K, Row, Col, Backend, Layout> operator-(
    Matrix<K, Row, Col, Backend, Layout> lhs, Matrix<K, Row, Col, Backend, Layout> rhs) {
    auto result = lhs;
    core::BLAS<Backend, K>::axpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires(mk_concepts::BLASSupported<Backend, K> && Row != std::dynamic_extent && Col != std::dynamic_extent)
Matrix<K, Row, Col, Backend, Layout> operator-(
    Matrix<K, Row, Col, Backend, Layout> lhs, Matrix<K, Row, std::dynamic_extent, Backend, Layout> rhs) {
#if !defined(NDEBUG)
    if (Col != rhs.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + lhs.GetSizeString() + "\n";
        errorString += "opponent size: " + rhs.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto result = lhs;
    core::BLAS<Backend, K>::axpy(Row * Col, -1.0, rhs._elements.data(), 1, result._elements.data(), 1);
    return result;
}

// ostream
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, Col, Backend, Layout>& mat) {
    for (size_t i = 0; i < Row; i++) {
        for (size_t j = 0; j < Col; j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend, Layout>::RowSize() {
    return Row;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, Col, Backend, Layout>::ColumnSize() {
    return Col;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, Row, Col, Backend, Layout>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(Col) + ")";
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline constexpr core::Storage<Backend, K, Row * Col>& Matrix<K, Row, Col, Backend, Layout>::Elements() noexcept {
    return this->_elements;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline constexpr const core::Storage<Backend, K, Row * Col>& Matrix<K, Row, Col, Backend, Layout>::Elements() const noexcept {
    return this->_elements;
}

// template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
//     requires mk_concepts::BLASSupported<Backend, K>
// inline constexpr std::array<K, Row * Col>& Matrix<K, Row, Col, Backend, Layout>::ElementsRange() noexcept {
//     return this->_elements.elements();
// }

// template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
//     requires mk_concepts::BLASSupported<Backend, K>
// inline constexpr const std::array<K, Row * Col>& Matrix<K, Row, Col, Backend, Layout>::ElementsRange() const noexcept {
//     return this->_elements.elements();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend, Layout> Matrix<K, Row, Col, Backend, Layout>::Dot(
    const Matrix<K, Col, OppCol, Backend, Layout>& mat) {
    Matrix<K, Row, OppCol, Backend, Layout> res = Matrix<K, Row, OppCol, Backend, Layout>();
    core::BLAS<Backend, K>::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(OppCol), Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend, Layout> Matrix<K, Row, Col, Backend, Layout>::Dot(
    const Matrix<K, std::dynamic_extent, Col, Backend, Layout>& mat) {
    // const size_t matColumnSize = mat.ColumnSize();
    // assert(matColumnSize <= INT_MAX);
    auto res = Matrix<K, Row, std::dynamic_extent, Backend, Layout>(mat.ColumnSize());
    core::BLAS<Backend, K>::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
        Col, 1.0, this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
inline Matrix<K, Row, OppCol, Backend, Layout> Matrix<K, Row, Col, Backend, Layout>::Dot(
    const Matrix<K, std::dynamic_extent, OppCol, Backend, Layout>& mat) {
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, OppCol, Backend, Layout>();
    core::BLAS<Backend, K>::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0,
        this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col, typename Backend, typename Layout>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend, Layout> Matrix<K, Row, Col, Backend, Layout>::Dot(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend, Layout>& mat) {
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, std::dynamic_extent, Backend, Layout>(mat.ColumnSize());
    core::BLAS<Backend, K>::gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
        Col, 1.0, this->_elements.data(), Row, mat._elements.data(), Col, 0.0, res._elements.data(), Row);
    return res;
}

/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif