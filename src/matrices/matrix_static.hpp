#include <cblas.h>

#include <array>
#include <boost/operators.hpp>
#include <cstddef>
#include <mdspan>
#include <ostream>

#ifndef MATRIX_STATIC_H
#define MATRIX_STATIC_H

template <typename K, size_t Row, size_t Col>
class MatrixStatic;
template <typename K, size_t Row, size_t Col>
std::ostream& operator<<(std::ostream& os, const MatrixStatic<K, Row, Col>& mat);

template <typename K, size_t Row, size_t Col>
class MatrixStatic :
    private boost::addable<MatrixStatic<K, Row, Col>>,
    private boost::subtractable<MatrixStatic<K, Row, Col>>,
    private boost::multipliable<MatrixStatic<K, Row, Col>> {
    // type alias
    using MatrixExtent = std::extents<size_t, Row, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    std::array<K, Row * Col> _elements;
    MdView _span;

public:
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
    MatrixStatic<K, Row, Col>& operator+=(const MatrixStatic<K, Row, Col>& x);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixStatic<K, Row, Col>& mat);
    /* end operator overloads declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic() : _rowSize(Row), _columnSize(Col) {}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(const std::array<K, Row * Col> elements)
    : _rowSize(Row), _columnSize(Col), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(const MatrixStatic<K, Row, Col>& mat)
    : _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator=(const MatrixStatic<K, Row, Col>& x) {
    if (this != &x) {
        _elements = x.data;
        _span = MdView(_elements.data(), MatrixExtent{});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixStatic<K, Row, Col>& x) {
    cblas_saxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
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

// template <typename K, size_t Row, size_t Col>

#endif