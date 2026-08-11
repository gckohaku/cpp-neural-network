#ifndef MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP
#define MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP

#include <cblas.h>
#include <openblas_config.h>

#include <algorithm>
#include <boost/operators.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>
#include <span>
#include <vector>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_template_base.hpp"

namespace mknnlib::matrix {
template <typename K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent>;
template <typename K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent>& mat);

template <typename K, size_t Row, size_t Col>
class Matrix;

template <typename K>
class Matrix<K, std::dynamic_extent, std::dynamic_extent> :
    private boost::addable<Matrix<K, std::dynamic_extent, std::dynamic_extent>>,
    private boost::subtractable<Matrix<K, std::dynamic_extent, std::dynamic_extent>>,
    private boost::multipliable<Matrix<K, std::dynamic_extent, std::dynamic_extent>> {
    template <typename, std::size_t, std::size_t>
    friend class Matrix;
    // type alias
    using MatrixExtent = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = 0;
    size_t _columnSize = 0;
    std::vector<K> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    Matrix<K, std::dynamic_extent, std::dynamic_extent>();
    Matrix<K, std::dynamic_extent, std::dynamic_extent>(const size_t RowSize, const size_t columnSize);
    Matrix<K, std::dynamic_extent, std::dynamic_extent>(
        const size_t RowSize, const size_t columnSize, const std::vector<K> elements);
    // copy constructor
    Matrix<K, std::dynamic_extent, std::dynamic_extent>(const Matrix<K, std::dynamic_extent, std::dynamic_extent>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x);
    // arithmetics
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator+=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator+=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator-=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator-=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, std::dynamic_extent>& operator*=(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(
        std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    size_t RowSize();
    size_t RowSize() const;
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    std::vector<K>& Elements();
    // K* _elements.data();
    // const K* _elements.data() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    Matrix<K, std::dynamic_extent, std::dynamic_extent> Dot(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, std::dynamic_extent, std::dynamic_extent> Dot(
        const Matrix<K, std::dynamic_extent, std::dynamic_extent> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppRow, size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const Matrix<K, OppRow, OppCol>)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppRow, size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const Matrix<K, OppRow, OppCol>)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppRow>
    Matrix<K, std::dynamic_extent, std::dynamic_extent> Dot(const MatrixRowStatic<K, OppRow>)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppRow>
    Matrix<K, std::dynamic_extent, std::dynamic_extent> Dot(const MatrixRowStatic<K, OppRow>)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const MatrixColumnStatic<K, OppCol>)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const MatrixColumnStatic<K, OppCol>)
        requires mk_concepts::DoubleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent>::Matrix() :
    _rowSize(0), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent>::Matrix(const size_t rowSize, const size_t columnSize) :
    _rowSize(rowSize),
    _columnSize(columnSize),
    _elements(rowSize * columnSize, K{}),
    _span(_elements.data(), MatrixExtent{}) {}

template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent>::Matrix(
    const size_t rowSize, const size_t columnSize, const std::vector<K> elements) :
    _rowSize(rowSize), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent>::Matrix(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& mat) :
    _rowSize(mat.RowSize()),
    _columnSize(mat.ColumnSize()),
    _elements(mat._elements),
    _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

// arithmetics
template <typename K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent>&
Matrix<K, std::dynamic_extent, std::dynamic_extent>::operator+=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(this->_rowSize * this->_columnSize), 1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent>&
Matrix<K, std::dynamic_extent, std::dynamic_extent>::operator+=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(this->_rowSize * this->_columnSize), 1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent>&
Matrix<K, std::dynamic_extent, std::dynamic_extent>::operator-=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(static_cast<blasint>(this->_rowSize * this->_columnSize), -1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K>
inline Matrix<K, std::dynamic_extent, std::dynamic_extent>&
Matrix<K, std::dynamic_extent, std::dynamic_extent>::operator-=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(static_cast<blasint>(this->_rowSize * this->_columnSize), -1.0, x._elements.data(), 1,
        this->_elements.data(), 1);
    return *this;
}

template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent>& Matrix<K, std::dynamic_extent, std::dynamic_extent>::operator*=(
    const Matrix<K, std::dynamic_extent, std::dynamic_extent>& x) {
#if !defined(NDEBUG)
    if (this->RowSize() != x.RowSize() || this->ColumnSize() != x.ColumnSize()) {
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
template <typename K>
std::ostream& operator<<(std::ostream& os, const Matrix<K, std::dynamic_extent, std::dynamic_extent>& mat) {
    for (size_t i = 0; i < mat.RowSize(); i++) {
        for (size_t j = 0; j < mat.ColumnSize(); j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}
/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent>::RowSize() {
    return this->_rowSize;
}

template <typename K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent>::RowSize() const {
    return this->_rowSize;
}

template <typename K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent>::ColumnSize() {
    return this->_columnSize;
}

template <typename K>
size_t Matrix<K, std::dynamic_extent, std::dynamic_extent>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K>
std::string Matrix<K, std::dynamic_extent, std::dynamic_extent>::GetSizeString() const {
    return "(" + std::to_string(this->RowSize()) + ", " + std::to_string(this->ColumnSize()) + ")";
}

template <typename K>
inline std::vector<K>& Matrix<K, std::dynamic_extent, std::dynamic_extent>::Elements() {
    return this->_elements;
}

// template <typename K>
// inline K* Matrix<K, std::dynamic_extent, std::dynamic_extent>::_elements.data() {
//     return this->_elements.data();
// }

// template <typename K>
// inline const K* Matrix<K, std::dynamic_extent, std::dynamic_extent>::_elements.data() const {
//     return this->_elements.data();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    Matrix<K, std::dynamic_extent, std::dynamic_extent> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent>(this->RowSize(), mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
        static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
        res._elements.data(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
Matrix<K, std::dynamic_extent, std::dynamic_extent> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    Matrix<K, std::dynamic_extent, std::dynamic_extent> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent>(this->RowSize(), mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
        static_cast<blasint>(this->RowSize()), mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0,
        res._elements.data(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow, size_t OppCol>
MatrixColumnStatic<K, OppCol> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(Matrix<K, OppRow, OppCol> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixColumnStatic<K, OppCol>(this->RowSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
        static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), static_cast<blasint>(this->RowSize()),
        mat._elements.data(), OppRow, 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow, size_t OppCol>
MatrixColumnStatic<K, OppCol> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(Matrix<K, OppRow, OppCol> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixColumnStatic<K, OppCol>(this->RowSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
        static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), static_cast<blasint>(this->RowSize()),
        mat._elements.data(), OppRow, 0.0, res._elements.data(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow>
Matrix<K, std::dynamic_extent, std::dynamic_extent> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    MatrixRowStatic<K, OppRow> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent>(this->RowSize(), mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
        static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow>
Matrix<K, std::dynamic_extent, std::dynamic_extent> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    MatrixRowStatic<K, OppRow> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != OppRow) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = Matrix<K, std::dynamic_extent, std::dynamic_extent>(this->RowSize(), mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(),
        static_cast<blasint>(this->RowSize()), mat._elements.data(), OppRow, 0.0, res._elements.data(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppCol>
MatrixColumnStatic<K, OppCol> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    MatrixColumnStatic<K, OppCol> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixColumnStatic<K, OppCol>(this->RowSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
        static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), static_cast<blasint>(this->RowSize()),
        mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppCol>
MatrixColumnStatic<K, OppCol> Matrix<K, std::dynamic_extent, std::dynamic_extent>::Dot(
    MatrixColumnStatic<K, OppCol> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (this->ColumnSize() != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixColumnStatic<K, OppCol>(this->RowSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol,
        static_cast<blasint>(this->ColumnSize()), 1.0, this->_elements.data(), static_cast<blasint>(this->RowSize()),
        mat._elements.data(), static_cast<blasint>(mat.RowSize()), 0.0, res._elements.data(),
        static_cast<blasint>(this->RowSize()));
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif