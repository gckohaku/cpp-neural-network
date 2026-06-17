#include <cblas.h>

#include <array>
#include <boost/operators.hpp>
#include <cstddef>
#include <mdspan>
#include <ostream>

#ifndef MATRIX_STATIC_H
#define MATRIX_STATIC_H

template <typename K, size_t Row, size_t Col>
class MatrixStatic :
    private boost::addable<MatrixStatic<K, Row, Col>>,
    private boost::subtractable<MatrixStatic<K, Row, Col>>,
    private boost::multipliable<MatrixStatic<K, Row, Col>> {
private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    std::array<K, Row * Col> _elements;

public:
    MatrixStatic();
    MatrixStatic(std::array<K, Row * Col> elements);
    MatrixStatic<K, Row, Col>& operator+=(const MatrixStatic<K, Row, Col>& x);

    friend std::ostream& operator<<(std::ostream& os, MatrixStatic<K, Row, Col>& mat);
};

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic() : _rowSize(Row), _columnSize(Col) {}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(std::array<K, Row * Col> elements)
    : _rowSize(Row), _columnSize(Col), _elements(elements) {
    }

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixStatic<K, Row, Col>& x) {
    cblas_saxpy(Row * Col, 1.0, x._elements.data(), 1, this->_elements.data(), 1);
}

template <typename K, size_t Row, size_t Col>
std::ostream& operator<<(std::ostream& os, MatrixStatic<K, Row, Col>& mat) {
    for (size_t i = 0; i < Row; i++) {
        for (size_t j = 0; j < Col; j++) {
            os << mat._elements[i * Row + j] << " ";
        }
        os << std::endl;
    }
    return os;
}

// template <typename K, size_t Row, size_t Col>

#endif