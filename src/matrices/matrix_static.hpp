#include <array>
#include <cstddef>
#include <mdspan>

#ifndef MATRIX_STATIC_H
#define MATRIX_STATIC_H

template <typename K, size_t Row, size_t Col>
class MatrixStatic {
private:
    size_t _rowSize = Row;
    size_t _columnSize = Col;
    std::array<K, Row * Col> _elements;

public:
    MatrixStatic(size_t rowSize, size_t columnSize);
    MatrixStatic(size_t rowSize, size_t columnSize, std::array<K, Row * Col> elements);

    
};

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(size_t rowSize, size_t columnSize)
    : _rowSize(rowSize), _columnSize(columnSize) {}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic(size_t rowSize, size_t columnSize, std::array<K, Row * Col> elements)
    : _rowSize(rowSize), _columnSize(columnSize), _elements(elements) {}

#endif