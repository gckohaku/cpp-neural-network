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

#include "src/concept_defines/types/type_concepts.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Row, size_t Col>
class MatrixStatic;
template <typename K, size_t Row, size_t Col>
std::ostream& operator<<(std::ostream& os, const MatrixStatic<K, Row, Col>& mat);

template <typename K, size_t Row>
class MatrixRowStatic;
template <typename K, size_t Col>
class MatrixColumnStatic;
template <typename K>
class MatrixDynamic;

template <typename K, size_t Row, size_t Col>
class MatrixStatic :
    private boost::addable<MatrixStatic<K, Row, Col>>,
    private boost::addable<MatrixStatic<K, Row, Col>, MatrixRowStatic<K, Row>>,
    private boost::subtractable<MatrixStatic<K, Row, Col>>,
    private boost::subtractable<MatrixStatic<K, Row, Col>, MatrixRowStatic<K, Row>>,
    private boost::subtractable2_left<MatrixStatic<K, Row, Col>, MatrixRowStatic<K, Row>>,
    private boost::multipliable<MatrixStatic<K, Row, Col>>,
    private boost::multipliable<MatrixStatic<K, Row, Col>, MatrixRowStatic<K, Row>> {
    // type alias
    using MatrixExtent = std::extents<size_t, Row, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    std::array<K, Row * Col> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    MatrixStatic();
    MatrixStatic(const std::array<K, Row * Col> elements);
    // copy constructor
    MatrixStatic(const MatrixStatic<K, Row, Col>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    MatrixStatic<K, Row, Col>& operator=(const MatrixStatic<K, Row, Col>& x);
    // arithmetics
    MatrixStatic<K, Row, Col>& operator+=(const MatrixStatic<K, Row, Col>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator+=(const MatrixStatic<K, Row, Col>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator+=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator+=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator-=(const MatrixStatic<K, Row, Col>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator-=(const MatrixStatic<K, Row, Col>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator-=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator-=(const MatrixRowStatic<K, Row>& x)
        requires mk_concepts::DoubleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator*=(const MatrixStatic<K, Row, Col>& x);
    MatrixStatic<K, Row, Col>& operator*=(const MatrixRowStatic<K, Row>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixStatic<K, Row, Col>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    constexpr size_t RowSize();
    constexpr size_t ColumnSize();
    std::string GetSizeString() const;
    std::array<K, Row * Col>& Elements();
    K* ElementsPointer();
    const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixStatic<K, Col, OppCol> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixStatic<K, Col, OppCol> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, Col>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, Col>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixColumnStatic<K, OppCol>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixColumnStatic<K, OppCol>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    MatrixRowStatic<K, Row> Dot(const MatrixDynamic<K>& mat)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row> Dot(const MatrixDynamic<K>& mat)
        requires mk_concepts::DoubleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic() :
    _rowSize(Row), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(const std::array<K, Row * Col> elements) :
    _rowSize(Row), _columnSize(Col), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(const MatrixStatic<K, Row, Col>& mat) :
    _rowSize(Row), _columnSize(Col), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator=(const MatrixStatic<K, Row, Col>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(ElementsPointer(), MatrixExtent{});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixStatic<K, Row, Col>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(Row * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixStatic<K, Row, Col>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    cblas_daxpy(Row * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(Row * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(Row * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator-=(const MatrixStatic<K, Row, Col>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(Row * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator-=(const MatrixStatic<K, Row, Col>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    cblas_daxpy(Row * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator-=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_saxpy(Row * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
inline MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator-=(const MatrixRowStatic<K, Row>& x)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != x.ColumnSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + x.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif
    cblas_daxpy(Row * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator*=(const MatrixStatic<K, Row, Col>& x) {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator*=(const MatrixRowStatic<K, Row>& x) {
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

// ostream
template <typename K, size_t Row, size_t Col>
std::ostream& operator<<(std::ostream& os, const MatrixStatic<K, Row, Col>& mat) {
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
template <typename K, size_t Row, size_t Col>
constexpr size_t MatrixStatic<K, Row, Col>::RowSize() {
    return Row;
}

template <typename K, size_t Row, size_t Col>
constexpr size_t MatrixStatic<K, Row, Col>::ColumnSize() {
    return Col;
}

template <typename K, size_t Row, size_t Col>
std::string MatrixStatic<K, Row, Col>::GetSizeString() const {
    return "(" + std::to_string(Row) + ", " + std::to_string(Col) + ")";
}

template <typename K, size_t Row, size_t Col>
inline std::array<K, Row * Col>& MatrixStatic<K, Row, Col>::Elements() {
    return this->_elements;
}

template <typename K, size_t Row, size_t Col>
inline K* MatrixStatic<K, Row, Col>::ElementsPointer() {
    return this->_elements.data();
}

template <typename K, size_t Row, size_t Col>
inline const K* MatrixStatic<K, Row, Col>::ElementsPointer() const {
    return this->_elements.data();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Row, size_t Col>
template <size_t OppCol>
inline MatrixStatic<K, Row, OppCol> MatrixStatic<K, Row, Col>::Dot(const MatrixStatic<K, Col, OppCol> mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
template <size_t OppCol>
inline MatrixStatic<K, Row, OppCol> MatrixStatic<K, Row, Col>::Dot(const MatrixStatic<K, Col, OppCol> mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
inline MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixRowStatic<K, Col>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
    // const size_t matColumnSize = mat.ColumnSize();
    // assert(matColumnSize <= INT_MAX);
    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
inline MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixRowStatic<K, Col>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
    // const size_t matColumnSize = mat.ColumnSize();
    // assert(matColumnSize <= INT_MAX);
    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
template <size_t OppCol>
inline MatrixStatic<K, Row, OppCol> MatrixStatic<K, Row, Col>::Dot(const MatrixColumnStatic<K, OppCol>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
template <size_t OppCol>
inline MatrixStatic<K, Row, OppCol> MatrixStatic<K, Row, Col>::Dot(const MatrixColumnStatic<K, OppCol>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
inline MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixDynamic<K>& mat)
    requires mk_concepts::SingleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}

template <typename K, size_t Row, size_t Col>
inline MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixDynamic<K>& mat)
    requires mk_concepts::DoubleFloatingPoint<K>
{
#if !defined(NDEBUG)
    if (Col != mat.RowSize()) {
        std::string errorString = "Mismatch matrix size for matrix product.\n";
        errorString += "this size    : " + this->GetSizeString() + "\n";
        errorString += "opponent size: " + mat.GetSizeString() + ".\n";

        throw std::domain_error(errorString);
    }
#endif

    auto res = MatrixRowStatic<K, Row>(mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Row, static_cast<blasint>(mat.ColumnSize()), Col, 1.0,
        this->ElementsPointer(), Row, mat.ElementsPointer(), Col, 0.0, res.ElementsPointer(), Row);
    return res;
}
/* end matrix unique arithmetics definition */  // namespace mknnlib::matrix
}  // namespace mknnlib::matrix

#endif