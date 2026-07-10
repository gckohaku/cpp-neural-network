#ifndef MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP

#include <cblas.h>
#include <boost/operators.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>

#include "src/concept_defines/types/type_concepts.hpp"

namespace mknnlib::matrix {
template <typename K, size_t Col>
class MatrixColumnStatic;
template <typename K, size_t Col>
std::ostream& operator<<(std::ostream& os, const MatrixColumnStatic<K, Col>& mat);

template <typename K, size_t Col>
class MatrixColumnStatic :
    private boost::addable<MatrixColumnStatic<K, Col>>,
    private boost::subtractable<MatrixColumnStatic<K, Col>>,
    private boost::multipliable<MatrixColumnStatic<K, Col>> {
    // type alias
    using MatrixExtent = std::extents<size_t, std::dynamic_extent, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = 0;
    size_t _columnSize = Col;
    std::vector<K> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    MatrixColumnStatic();
    MatrixColumnStatic(const size_t rowSize);
    MatrixColumnStatic(const size_t rowSize, const std::vector<K> elements);
    // copy constructor
    MatrixColumnStatic(const MatrixColumnStatic<K, Col>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    MatrixColumnStatic<K, Col>& operator=(const MatrixColumnStatic<K, Col>& x);
    // arithmetics
    MatrixColumnStatic<K, Col>& operator+=(const MatrixColumnStatic<K, Col>& x)
        requires concepts::SingleFloatingPoint<K>;
    MatrixColumnStatic<K, Col>& operator-=(const MatrixColumnStatic<K, Col>& x)
        requires concepts::SingleFloatingPoint<K>;
    MatrixColumnStatic<K, Col>& operator*=(const MatrixColumnStatic<K, Col>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixColumnStatic<K, Col>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    size_t RowSize();
    size_t RowSize() const;
    constexpr size_t ColumnSize();
    std::vector<float>& Elements();
    K* ElementsPointer();
    const K* ElementsPointer() const;
    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    MatrixColumnStatic<K, OppCol> Dot(const MatrixColumnStatic<K, OppCol> mat)
        requires concepts::SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Col>
MatrixColumnStatic<K, Col>::MatrixColumnStatic()
    : _rowSize(0), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Col>
MatrixColumnStatic<K, Col>::MatrixColumnStatic(const size_t rowSize)
    : _rowSize(rowSize), _columnSize(Col), _elements(rowSize * Col, K{}), _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Col>
MatrixColumnStatic<K, Col>::MatrixColumnStatic(const size_t rowSize, const std::vector<K> elements)
    : _rowSize(rowSize), _columnSize(Col), _elements{elements}, _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Col>
MatrixColumnStatic<K, Col>::MatrixColumnStatic(const MatrixColumnStatic<K, Col>& mat)
    : _rowSize(mat.RowSize()), _columnSize(Col), _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Col>
MatrixColumnStatic<K, Col>& MatrixColumnStatic<K, Col>::operator=(const MatrixColumnStatic<K, Col>& x) {
    if (this != &x) {
        _elements = x._elements;
        _span = MdView(ElementsPointer(), MatrixExtent{_rowSize});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Col>
inline MatrixColumnStatic<K, Col>& MatrixColumnStatic<K, Col>::operator+=(const MatrixColumnStatic<K, Col>& x)
    requires concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(this->_rowSize * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Col>
inline MatrixColumnStatic<K, Col>& MatrixColumnStatic<K, Col>::operator-=(const MatrixColumnStatic<K, Col>& x)
    requires concepts::SingleFloatingPoint<K>
{
    cblas_saxpy(this->_rowSize * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Col>
MatrixColumnStatic<K, Col>& MatrixColumnStatic<K, Col>::operator*=(const MatrixColumnStatic<K, Col>& x) {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
    return *this;
}

// ostream
template <typename K, size_t Col>
std::ostream& operator<<(std::ostream& os, const MatrixColumnStatic<K, Col>& mat) {
    for (size_t i = 0; i < mat._rowSize; i++) {
        for (size_t j = 0; j < Col; j++) {
            os << mat._span[i, j] << " ";
        }
        os << std::endl;
    }
    return os;
}

/* end operator overloads definition */

/* begin matrix unique functions definition */
template <typename K, size_t Col>
size_t MatrixColumnStatic<K, Col>::RowSize() {
    return this->_rowSize;
}

template <typename K, size_t Col>
size_t MatrixColumnStatic<K, Col>::RowSize() const {
    return this->_rowSize;
}

template <typename K, size_t Col>
constexpr size_t MatrixColumnStatic<K, Col>::ColumnSize() {
    return Col;
}

template <typename K, size_t Col>
inline std::vector<float>& MatrixColumnStatic<K, Col>::Elements() {
    return this->_elements;
}

template <typename K, size_t Col>
inline K* MatrixColumnStatic<K, Col>::ElementsPointer() {
    return this->_elements.data();
}

template <typename K, size_t Col>
inline const K* MatrixColumnStatic<K, Col>::ElementsPointer() const {
    return this->_elements.data();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
template <typename K, size_t Col>
template <size_t OppCol>
MatrixColumnStatic<K, OppCol> MatrixColumnStatic<K, Col>::Dot(MatrixColumnStatic<K, OppCol> mat) requires concepts::SingleFloatingPoint<K> {
    auto res = MatrixColumnStatic<K, OppCol>(this->RowSize());
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<blasint>(this->RowSize()), OppCol, Col, 1.0,
        this->ElementsPointer(), static_cast<blasint>(this->RowSize()), mat.ElementsPointer(), static_cast<blasint>(mat.RowSize()), 0.0, res.ElementsPointer(), static_cast<blasint>(this->RowSize()));
    return res;
}
/* end matrix unique arithmetics definition */
}  // namespace mknnlib::matrix

#endif