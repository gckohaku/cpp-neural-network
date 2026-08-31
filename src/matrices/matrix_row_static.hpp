#ifndef MKNNLIB_MATRICES_MATRIX_ROW_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_ROW_STATIC_HPP

#include <cblas.h>
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

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_template_base.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, std::dynamic_extent, Backend>& mat);

template <typename K, size_t Row, size_t Col, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix;

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
class Matrix<K, Row, std::dynamic_extent, Backend> :
    private boost::addable<Matrix<K, Row, std::dynamic_extent, Backend>>,
    private boost::subtractable<Matrix<K, Row, std::dynamic_extent, Backend>>,
    private boost::multipliable<Matrix<K, Row, std::dynamic_extent, Backend>> {
    static_assert(Row != std::dynamic_extent);

    template <typename T, size_t FRow, size_t FCol, typename FBackend>
        requires mk_concepts::BLASSupported<FBackend, T>
    friend class Matrix;
    // type alias
    using MatrixExtent = std::extents<size_t, Row, std::dynamic_extent>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = 0;
    std::vector<K> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix();
    Matrix(const size_t columnSize);
    Matrix(const size_t columnSize, const std::vector<K> elements);
    // copy constructor
    Matrix(const Matrix& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix& operator=(const Matrix& x);
    // arithmetics compound operators
    Matrix& operator+=(const Matrix& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix& operator+=(const Matrix& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix& operator-=(const Matrix& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix& operator-=(const Matrix& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix& operator*=(const Matrix& x);

    // arithmetics binary operators
    template <typename K_, size_t Row_, typename Backend_>
    friend Matrix<K_, Row_, std::dynamic_extent, Backend_> operator+(
        Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, std::dynamic_extent, Backend_> rhs);
    template <typename K_, size_t Row_, size_t Col_, typename Backend_>
    friend Matrix<K_, Row_, Col_, Backend_> operator+(
        Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, Col_, Backend_> rhs);
    template <typename K_, size_t Row_, typename Backend_>
    friend Matrix<K_, Row_, std::dynamic_extent, Backend_> operator-(
        Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, std::dynamic_extent, Backend_> rhs);
    template <typename K_, size_t Row_, size_t Col_, typename Backend_>
    friend Matrix<K_, Row_, Col_, Backend_> operator-(
        Matrix<K_, Row_, std::dynamic_extent, Backend_> lhs, Matrix<K_, Row_, Col_, Backend_> rhs);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    std::vector<K>& Elements();
    const std::vector<K>& Elements() const;
    // K* _elements.data();
    // const K* _elements.data() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppRow>
    Matrix Dot(const Matrix<K, OppRow, std::dynamic_extent, Backend> mat);

    template <size_t OppRow, size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, OppRow, OppCol, Backend> mat);

    template <size_t OppCol>
    Matrix<K, Row, OppCol, Backend> Dot(const Matrix<K, std::dynamic_extent, OppCol, Backend> mat);

    Matrix Dot(const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> mat);
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>::Matrix() :
    _rowSize(Row), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>::Matrix(const size_t columnSize) :
    _rowSize(Row), _columnSize(columnSize), _elements(Row * columnSize, K{}), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>::Matrix(const size_t columnSize, const std::vector<K> elements) :
    _rowSize(Row), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>::Matrix(const Matrix<K, Row, std::dynamic_extent, Backend>& mat) :
    _rowSize(Row), _columnSize(mat.ColumnSize()), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(_elements, MatrixExtent{_columnSize});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator+=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(Row * this->_columnSize), 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator+=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(Row * this->_columnSize), 1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator-=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(Row * this->_columnSize), -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator-=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(Row * this->_columnSize), -1.0, x._elements.data(), 1, this->_elements.data(), 1);
    return *this;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend>& Matrix<K, Row, std::dynamic_extent, Backend>::operator*=(
    const Matrix<K, Row, std::dynamic_extent, Backend>& x) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

// ostream
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row, std::dynamic_extent, Backend>& mat) {
    for (size_t i = 0; i < Row; i++) {
        for (size_t j = 0; j < mat._columnSize; j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
constexpr size_t Matrix<K, Row, std::dynamic_extent, Backend>::RowSize() {
    return Row;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, Row, std::dynamic_extent, Backend>::ColumnSize() {
    return this->_columnSize;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
size_t Matrix<K, Row, std::dynamic_extent, Backend>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
std::string Matrix<K, Row, std::dynamic_extent, Backend>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(this->ColumnSize()) + ")";
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline std::vector<K>& Matrix<K, Row, std::dynamic_extent, Backend>::Elements() {
    return this->_elements;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
inline const std::vector<K>& Matrix<K, Row, std::dynamic_extent, Backend>::Elements() const {
    return this->_elements;
}

// template <typename K, size_t Row, typename Backend>
// requires mk_concepts::BLASSupported<Backend, K>
// inline K* Matrix<K, Row, std::dynamic_extent, Backend>::_elements.data() {
//     return this->_elements.data();
// }

// template <typename K, size_t Row, typename Backend>
// requires mk_concepts::BLASSupported<Backend, K>
// inline const K* Matrix<K, Row, std::dynamic_extent, Backend>::_elements.data() const {
//     return this->_elements.data();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow>
Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, std::dynamic_extent, Backend>::Dot(
    const Matrix<K, OppRow, std::dynamic_extent, Backend> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, std::dynamic_extent, Backend>(mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), OppRow, 1.0,
            this->_elements.data(), Row, mat._elements.data(), OppRow, 0.0, res._elements.data(), Row);
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), OppRow, 1.0,
            this->_elements.data(), Row, mat._elements.data(), OppRow, 0.0, res._elements.data(), Row);
    }

    return res;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppRow, size_t OppCol>
Matrix<K, Row, OppCol, Backend> Matrix<K, Row, std::dynamic_extent, Backend>::Dot(
    const Matrix<K, OppRow, OppCol, Backend> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, OppCol>();
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()),
            1.0, this->_elements.data(), Row, mat._elements.data(), static_cast<blasint>(this->ColumnSize()), 0.0,
            res._elements.data(), Row);
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()),
            1.0, this->_elements.data(), Row, mat._elements.data(), static_cast<blasint>(this->ColumnSize()), 0.0,
            res._elements.data(), Row);
    }

    return res;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
template <size_t OppCol>
Matrix<K, Row, OppCol, Backend> Matrix<K, Row, std::dynamic_extent, Backend>::Dot(
    const Matrix<K, std::dynamic_extent, OppCol, Backend> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, OppCol>();
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()),
            1.0, this->_elements.data(), Row, mat._elements.data(), static_cast<blasint>(this->ColumnSize()), 0.0,
            res._elements.data(), Row);
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()),
            1.0, this->_elements.data(), Row, mat._elements.data(), static_cast<blasint>(this->ColumnSize()), 0.0,
            res._elements.data(), Row);
    }

    return res;
}

template <typename K, size_t Row, typename Backend>
    requires mk_concepts::BLASSupported<Backend, K>
Matrix<K, Row, std::dynamic_extent, Backend> Matrix<K, Row, std::dynamic_extent, Backend>::Dot(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent, Backend> mat) {
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    auto res = Matrix<K, Row, std::dynamic_extent, Backend>(mat.ColumnSize());
    if constexpr (mk_concepts::SingleFloatingPoint<K>) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), Row, mat._elements.data(),
            static_cast<blasint>(this->ColumnSize()), 0.0, res._elements.data(), Row);
    } else if (mk_concepts::DoubleFloatingPoint<K>) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
            static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), Row, mat._elements.data(),
            static_cast<blasint>(this->ColumnSize()), 0.0, res._elements.data(), Row);
    }

    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif