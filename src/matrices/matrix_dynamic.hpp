#ifndef MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP
#define MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP

#include <cblas.h>

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

namespace mknnlib::matrix {
template <typename K>
class MatrixDynamic;
template <typename K>
std::ostream& operator<<(std::ostream& os, const MatrixDynamic<K>& mat);

template <typename K, size_t Row, size_t Col>
class MatrixStatic;
template <typename K, size_t Row>
class MatrixRowStatic;
template <typename K, size_t Col>
class MatrixColumnStatic;

template <typename K>
class MatrixDynamic :
    private boost::addable<MatrixDynamic<K>>,
    private boost::subtractable<MatrixDynamic<K>>,
    private boost::multipliable<MatrixDynamic<K>> {
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
    MatrixDynamic();
    MatrixDynamic(const size_t RowSize, const size_t columnSize);
    MatrixDynamic(const size_t RowSize, const size_t columnSize, const std::vector<K> elements);
    // copy constructor
    MatrixDynamic(const MatrixDynamic<K>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    MatrixDynamic<K>& operator=(const MatrixDynamic<K>& x);
    // arithmetics
    MatrixDynamic<K>& operator+=(const MatrixDynamic<K>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixDynamic<K>& operator-=(const MatrixDynamic<K>& x)
        requires mk_concepts::SingleFloatingPoint<K>;
    MatrixDynamic<K>& operator*=(const MatrixDynamic<K>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixDynamic<K>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    size_t RowSize();
    size_t RowSize() const;
    size_t ColumnSize();
    size_t ColumnSize() const;
    std::string GetSizeString() const;
    std::vector<K>& Elements();
    K* ElementsPointer();
    const K* ElementsPointer() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    MatrixDynamic<K> Dot(const MatrixDynamic<K> mat)
        requires mk_concepts::SingleFloatingPoint<K>;
        MatrixDynamic<K> Dot(const MatrixDynamic<K> mat)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppRow, size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const MatrixStatic<K, OppRow, OppCol>)
        requires mk_concepts::SingleFloatingPoint<K>;
        template <size_t OppRow, size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const MatrixStatic<K, OppRow, OppCol>)
        requires mk_concepts::DoubleFloatingPoint<K>;

    template <size_t OppRow>
    MatrixDynamic<K> Dot(const MatrixRowStatic<K, OppRow>)
        requires mk_concepts::SingleFloatingPoint<K>;
        template <size_t OppRow>
    MatrixDynamic<K> Dot(const MatrixRowStatic<K, OppRow>)
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
MatrixDynamic<K>::MatrixDynamic() : _rowSize(0), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K>
MatrixDynamic<K>::MatrixDynamic(const size_t rowSize, const size_t columnSize) :
    _rowSize(rowSize),
    _columnSize(columnSize),
    _elements(rowSize * columnSize, K{}),
    _span(_elements.data(), MatrixExtent{}) {}

template <typename K>
MatrixDynamic<K>::MatrixDynamic(const size_t rowSize, const size_t columnSize, const std::vector<K> elements) :
    _rowSize(rowSize), _columnSize(columnSize), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K>
MatrixDynamic<K>::MatrixDynamic(const MatrixDynamic<K>& mat) :
    _rowSize(mat.RowSize()),
    _columnSize(mat.ColumnSize()),
    _elements(mat._elements),
    _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

// arithmetics
template <typename K>
inline MatrixDynamic<K>& MatrixDynamic<K>::operator+=(const MatrixDynamic<K>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(this->_rowSize * this->_columnSize, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K>
inline MatrixDynamic<K>& MatrixDynamic<K>::operator-=(const MatrixDynamic<K>& x)
    requires mk_concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(this->_rowSize * this->_columnSize, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K>
MatrixDynamic<K>& MatrixDynamic<K>::operator*=(const MatrixDynamic<K>& x) {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

// ostream
template <typename K>
std::ostream& operator<<(std::ostream& os, const MatrixDynamic<K>& mat) {
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
size_t MatrixDynamic<K>::RowSize() {
    return this->_rowSize;
}

template <typename K>
size_t MatrixDynamic<K>::RowSize() const {
    return this->_rowSize;
}

template <typename K>
size_t MatrixDynamic<K>::ColumnSize() {
    return this->_columnSize;
}

template <typename K>
size_t MatrixDynamic<K>::ColumnSize() const {
    return this->_columnSize;
}

template <typename K>
std::string MatrixDynamic<K>::GetSizeString() const {
    return "(" + std::to_string(this->RowSize()) + ", " + std::to_string(this->ColumnSize()) + ")";
}

template <typename K>
inline std::vector<K>& MatrixDynamic<K>::Elements() {
    return this->_elements;
}

template <typename K>
inline K* MatrixDynamic<K>::ElementsPointer() {
    return this->_elements.data();
}

template <typename K>
inline const K* MatrixDynamic<K>::ElementsPointer() const {
    return this->_elements.data();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K>
MatrixDynamic<K> MatrixDynamic<K>::Dot(MatrixDynamic<K> mat)
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

    auto res = MatrixDynamic<K>(this->RowSize(), mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), static_cast<blasint>(mat.RowSize()), 0.0,
        res.ElementsPointer(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
MatrixDynamic<K> MatrixDynamic<K>::Dot(MatrixDynamic<K> mat)
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

    auto res = MatrixDynamic<K>(this->RowSize(), mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), static_cast<blasint>(mat.RowSize()), 0.0,
        res.ElementsPointer(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow, size_t OppCol>
MatrixColumnStatic<K, OppCol> MatrixDynamic<K>::Dot(MatrixStatic<K, OppRow, OppCol> mat)
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
        static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(), static_cast<blasint>(this->RowSize()),
        mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow, size_t OppCol>
MatrixColumnStatic<K, OppCol> MatrixDynamic<K>::Dot(MatrixStatic<K, OppRow, OppCol> mat)
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
        static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(), static_cast<blasint>(this->RowSize()),
        mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(), static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow>
MatrixDynamic<K> MatrixDynamic<K>::Dot(MatrixRowStatic<K, OppRow> mat)
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

    auto res = MatrixDynamic<K>(this->RowSize(), mat.ColumnSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppRow>
MatrixDynamic<K> MatrixDynamic<K>::Dot(MatrixRowStatic<K, OppRow> mat)
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

    auto res = MatrixDynamic<K>(this->RowSize(), mat.ColumnSize());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        static_cast<blasint>(mat.ColumnSize()), static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), OppRow, 0.0, res.ElementsPointer(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppCol>
MatrixColumnStatic<K, OppCol> MatrixDynamic<K>::Dot(MatrixColumnStatic<K, OppCol> mat)
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
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        OppCol, static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), static_cast<blasint>(mat.RowSize()), 0.0, res.ElementsPointer(),
        static_cast<blasint>(this->RowSize()));
    return res;
}

template <typename K>
template <size_t OppCol>
MatrixColumnStatic<K, OppCol> MatrixDynamic<K>::Dot(MatrixColumnStatic<K, OppCol> mat)
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
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()),
        OppCol, static_cast<blasint>(this->ColumnSize()), 1.0, this->ElementsPointer(),
        static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), static_cast<blasint>(mat.RowSize()), 0.0, res.ElementsPointer(),
        static_cast<blasint>(this->RowSize()));
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif