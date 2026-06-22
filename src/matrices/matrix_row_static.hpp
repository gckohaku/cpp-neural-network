#ifndef MATRIX_ROW_STATIC_HPP
#define MATRIX_ROW_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>

#include "src/concept_defines/types/type_concepts.hpp"

template <typename K, size_t Row>
class MatrixRowStatic;
template <typename K, size_t Row>
std::ostream& operator<<(std::ostream& os, const MatrixRowStatic<K, Row>& mat);

template <typename K, size_t Row>
class MatrixRowStatic :
    private boost::addable<MatrixRowStatic<K, Row>>,
    private boost::subtractable<MatrixRowStatic<K, Row>>,
    private boost::multipliable<MatrixRowStatic<K, Row>> {
    // type alias
    using MatrixExtent = std::extents<size_t, Row>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Row;
    size_t _columnSize = 0;
    std::vector<K> _elements;
    MdView _span;

public:
    /* begin constructors declaration */
    MatrixRowStatic();
    MatrixRowStatic(const size_t rowSize, const std::vector<K> elements);
    // copy constructor
    MatrixRowStatic(const MatrixRowStatic<K, Row>& mat);
    /* end constructors declaration */

    /* begin operator overloads declaration */
    // copy assignment operator
    MatrixRowStatic<K, Row>& operator=(const MatrixRowStatic<K, Row>& x);
    // arithmetics
    MatrixRowStatic<K, Row>& operator+=(const MatrixRowStatic<K, Row>& x) requires SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row>& operator-=(const MatrixRowStatic<K, Row>& x) requires SingleFloatingPoint<K>;
    MatrixRowStatic<K, Row>& operator*=(const MatrixRowStatic<K, Row>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixRowStatic<K, Row>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    K* ElementsPointer();
    const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    // template <size_t OppCol>
    // MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, Col> mat) requires SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

#endif