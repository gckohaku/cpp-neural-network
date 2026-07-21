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

namespace mknnlib::matrix {
template <typename K, size_t Row>
class MatrixRowStatic;
template <typename K, size_t Row>
std::ostream& operator<<(std::ostream& os, const MatrixRowStatic<K, Row>& mat);

template <typename K, size_t Row, size_t Col>
class MatrixStatic;
template <typename K, size_t Col>
class MatrixColumnStatic;
template <typename K>
class MatrixDynamic;

template <typename K, size_t Row>
class MatrixRowStatic :
    private boost::addable<MatrixRowStatic<K, Row>>,
    private boost::subtractable<MatrixRowStatic<K, Row>>,
    private boost::multipliable<MatrixRowStatic<K, Row>> {
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
    MatrixRowStatic();
    MatrixRowStatic(const size_t columnSize);
    MatrixRowStatic(const size_t columnSize, const std::vector<K> elements);
    // copy constructor
    MatrixRowStatic(const MatrixRowStatic<K, Row>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    MatrixRowStatic<K, Row>& operator=(const MatrixRowStatic<K, Row>& x);
    // arithmetics
    MatrixRowStatic<K, Row>& operator+=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row>& operator-=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row>& operator*=(const MatrixRowStatic<K, Row>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixRowStatic<K, Row>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    std::vector<K>& Elements();
    K* ElementsPointer();
    const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    // /* begin matrix unique arithmetics declaration */
    template <size_t OppRow>
    MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, OppRow> mat)
        requires mk_concepts::SingleFloatingPoint<K>;

    template <size_t OppRow, size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixStatic<K, OppRow, OppCol> mat)
        requires mk_concepts::SingleFloatingPoint<K>;

    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixColumnStatic<K, OppCol> mat)
        requires mk_concepts::SingleFloatingPoint<K>;

    MatrixRowStatic<K, Row> Dot(const MatrixDynamic<K> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic() :
    _rowSize(Row), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic(const size_t columnSize) :
    _rowSize(Row), _columnSize(columnSize), _elements(Row * columnSize, K{}), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic(const size_t columnSize, const std::vector<K> elements) :
    _rowSize(Row), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic(const MatrixRowStatic<K, Row>& mat) :
    _rowSize(Row), _columnSize(mat.ColumnSize()), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row>
MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator=(const MatrixRowStatic<K, Row>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(ElementsPointer(), MatrixExtent{_columnSize});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row>
inline MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator+=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(Row * this->_columnSize, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
inline MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator-=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(Row * this->_columnSize, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator*=(const MatrixRowStatic<K, Row>& x) {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

// ostream
template <typename K, size_t Row>
std::ostream& operator<<(std::ostream& os, const MatrixRowStatic<K, Row>& mat) {
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
constexpr size_t MatrixRowStatic<K, Row>::RowSize() {
    return Row;
}

template <typename K, size_t Row>
size_t MatrixRowStatic<K, Row>::ColumnSize() {
    return this->_columnSize;
}

template <typename K, size_t Row>
size_t MatrixRowStatic<K, Row>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K, size_t Row>
std::string MatrixRowStatic<K, Row>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(this->ColumnSize()) + ")";
}

template <typename K, size_t Row>
inline std::vector<K>& MatrixRowStatic<K, Row>::Elements() {
    return this->_elements;
}

template <typename K, size_t Row>
inline K* MatrixRowStatic<K, Row>::ElementsPointer() {
    return this->_elements.data();
}

template <typename K, size_t Row>
inline const K* MatrixRowStatic<K, Row>::ElementsPointer() const {
    return this->_elements.data();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row>
template <size_t OppRow>
MatrixRowStatic<K, Row> MatrixRowStatic<K, Row>::Dot(const MatrixRowStatic<K, OppRow> mat)
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

    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), OppRow, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppRow, size_t OppCol>
MatrixStatic<K, Row, OppCol> MatrixRowStatic<K, Row>::Dot(const MatrixStatic<K, OppRow, OppCol> mat)
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

    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
template <size_t OppCol>
MatrixStatic<K, Row, OppCol> MatrixRowStatic<K, Row>::Dot(const MatrixColumnStatic<K, OppCol> mat)
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

    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, static_cast<blasint>(this->ColumnSize()), 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), static_cast<blasint>(this->ColumnSize()), 0.0,
        res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row>
MatrixRowStatic<K, Row> MatrixRowStatic<K, Row>::Dot(const MatrixDynamic<K> mat)
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

    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()),
        static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(), Row, mat.ElementsPointer(),
        static_cast<blasint>(this->ColumnSize()), 0.0, res.ElementsPointer(), Row);
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif