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
// template <typename K, size_t Row>
// class Matrix;
template <typename K, size_t Row>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row>& mat);

template <typename K, size_t Row, size_t Col>
class Matrix;

template <typename K, size_t Row>
class Matrix<K, Row, std::dynamic_extent> :
    private boost::addable<Matrix<K, Row>>,
    private boost::subtractable<Matrix<K, Row>>,
    private boost::multipliable<Matrix<K, Row>> {
    static_assert(Row != std::dynamic_extent);

    template <typename, std::size_t, std::size_t>
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
    Matrix(const Matrix<K, Row>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    Matrix<K, Row>& operator=(const Matrix<K, Row>& x);
    // arithmetics
    Matrix<K, Row>& operator+=(const Matrix<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, Row>& operator+=(const Matrix<K, Row>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, Row>& operator-=(const Matrix<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, Row>& operator-=(const Matrix<K, Row>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    Matrix<K, Row>& operator*=(const Matrix<K, Row>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix<K, Row>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    // std::vector<K>& Elements();
    // const std::vector<K>& Elements() const;
    // K* ElementsPointer();
    // const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    // /* begin matrix unique arithmetics declaration */
    template <size_t OppRow>
    Matrix<K, Row> Dot(const Matrix<K, OppRow> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppRow>
    Matrix<K, Row> Dot(const Matrix<K, OppRow> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppRow, size_t OppCol>
    Matrix<K, Row, OppCol> Dot(const Matrix<K, OppRow, OppCol> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppRow, size_t OppCol>
    Matrix<K, Row, OppCol> Dot(const Matrix<K, OppRow, OppCol> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppCol>
    Matrix<K, Row, OppCol> Dot(const MatrixColumnStatic<K, OppCol> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    Matrix<K, Row, OppCol> Dot(const MatrixColumnStatic<K, OppCol> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    Matrix<K, Row> Dot(const MatrixDynamic<K> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    Matrix<K, Row> Dot(const MatrixDynamic<K> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row>
Matrix<K, Row>::Matrix() : _rowSize(Row), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row>
Matrix<K, Row>::Matrix(const size_t columnSize) :
    _rowSize(Row), _columnSize(columnSize), _elements(Row * columnSize, K{}), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row>
Matrix<K, Row>::Matrix(const size_t columnSize, const std::vector<K> elements) :
    _rowSize(Row), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row>
Matrix<K, Row>::Matrix(const Matrix<K, Row>& mat) :
    _rowSize(Row), _columnSize(mat.ColumnSize()), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row>
Matrix<K, Row>& Matrix<K, Row>::operator=(const Matrix<K, Row>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(_elements, MatrixExtent{_columnSize});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row>
inline Matrix<K, Row>& Matrix<K, Row>::operator+=(const Matrix<K, Row>& x)
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
    cblas_saxpy(static_cast<blasint>(Row * this->_columnSize), 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
inline Matrix<K, Row>& Matrix<K, Row>::operator+=(const Matrix<K, Row>& x)
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
    cblas_daxpy(static_cast<blasint>(Row * this->_columnSize), 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
inline Matrix<K, Row>& Matrix<K, Row>::operator-=(const Matrix<K, Row>& x)
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
    cblas_saxpy(
        static_cast<blasint>(Row * this->_columnSize), -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
inline Matrix<K, Row>& Matrix<K, Row>::operator-=(const Matrix<K, Row>& x)
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
    cblas_daxpy(
        static_cast<blasint>(Row * this->_columnSize), -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
Matrix<K, Row>& Matrix<K, Row>::operator*=(const Matrix<K, Row>& x) {
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
template <typename K, size_t Row>
std::ostream& operator<<(std::ostream& os, const Matrix<K, Row>& mat) {
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
template <typename K, size_t Row>
constexpr size_t Matrix<K, Row>::RowSize() {
    return Row;
}

template <typename K, size_t Row>
size_t Matrix<K, Row>::ColumnSize() {
    return this->_columnSize;
}

template <typename K, size_t Row>
size_t Matrix<K, Row>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K, size_t Row>
std::string Matrix<K, Row>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(this->ColumnSize()) + ")";
}

// template <typename K, size_t Row>
// inline std::vector<K>& Matrix<K, Row>::Elements() {
//     return this->_elements;
// }

// template <typename K, size_t Row>
// inline const std::vector<K>& Matrix<K, Row>::Elements() const {
//     return this->_elements;
// }

// template <typename K, size_t Row>
// inline K* Matrix<K, Row>::ElementsPointer() {
//     return this->_elements.data();
// }

// template <typename K, size_t Row>
// inline const K* Matrix<K, Row>::ElementsPointer() const {
//     return this->_elements.data();
// }
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row>
template <size_t OppRow>
Matrix<K, Row> Matrix<K, Row>::Dot(const Matrix<K, OppRow> mat)
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

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), OppRow, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppRow>
Matrix<K, Row> Matrix<K, Row>::Dot(const Matrix<K, OppRow> mat)
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

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), OppRow, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppRow, size_t OppCol>
Matrix<K, Row, OppCol> Matrix<K, Row>::Dot(const Matrix<K, OppRow, OppCol> mat)
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

    auto res = Matrix<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppRow, size_t OppCol>
Matrix<K, Row, OppCol> Matrix<K, Row>::Dot(const Matrix<K, OppRow, OppCol> mat)
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

    auto res = Matrix<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppCol>
Matrix<K, Row, OppCol> Matrix<K, Row>::Dot(const MatrixColumnStatic<K, OppCol> mat)
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

    auto res = Matrix<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppCol>
Matrix<K, Row, OppCol> Matrix<K, Row>::Dot(const MatrixColumnStatic<K, OppCol> mat)
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

    auto res = Matrix<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
Matrix<K, Row> Matrix<K, Row>::Dot(const MatrixDynamic<K> mat)
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

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
        static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(), Row, mat.ElementsPointer(),
        static_cast<blasint>(this->ColumnSize()), 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
Matrix<K, Row> Matrix<K, Row>::Dot(const MatrixDynamic<K> mat)
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

    auto res = Matrix<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
        static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(), Row, mat.ElementsPointer(),
        static_cast<blasint>(this->ColumnSize()), 0.0, res.ElementsPointer(), Row);
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif