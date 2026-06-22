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

    // /* begin matrix unique arithmetics declaration */
    // template <size_t OppRow>
    // MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, OppRow> mat) requires SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic()
    : _rowSize(Row), _columnSize(0), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic(const size_t column, const std::vector<K> elements)
    : _rowSize(Row), _columnSize(column), _elements(elements), _span(_elements.data(), MatrixExtent{}) {}

// copy constructor
template <typename K, size_t Row>
MatrixRowStatic<K, Row>::MatrixRowStatic(const MatrixRowStatic<K, Row>& mat)
    : _elements(mat._elements), _span(_elements.data(), MatrixExtent{}) {}
/* end constructors definition */

/* begin operator overloads definition */
// copy assignment operator
template <typename K, size_t Row>
MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator=(const MatrixRowStatic<K, Row>& x) {
    if (this != &x) {
        _elements = x.data;
        _span = MdView(ElementsPointer(), MatrixExtent{});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row>
MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator+=(const MatrixRowStatic<K, Row>& x) requires SingleFloatingPoint<K> {
    cblas_saxpy(Row * this->_columnSize, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row>
MatrixRowStatic<K, Row>& MatrixRowStatic<K, Row>::operator-=(const MatrixRowStatic<K, Row>& x) requires SingleFloatingPoint<K> {
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
inline K* MatrixRowStatic<K, Row>::ElementsPointer() {
    return this->_elements.data();
}

template <typename K, size_t Row>
inline const K* MatrixRowStatic<K, Row>::ElementsPointer() const {
    return this->_elements.data();
}
/* end matrix unique functions definition */

/* begin matrix unique arithmetics definition */
// template <typename K, size_t Row>
// template <size_t OppCol>
// MatrixRowStatic<K, Row> MatrixRowStatic<K, Row>::Dot(const MatrixRowStatic<K, Col> mat) requires SingleFloatingPoint<K> {
//     auto res = MatrixStatic<K, Row, OppCol>();
//     cblas_sgemm(
//         CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, this->_columnSize, 1.0, this->ElementsPointer(), Row,
//         mat.ElementsPointer(), Col, 1.0, res.ElementsPointer(), Row
//     );
//     return res;
// }

// template <typename K, size_t Row, size_t Col>
// MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixRowStatic<K, Col> mat) requires SingleFloatingPoint<K> {
//     auto res = MatrixRowStatic<K, Row>();
//     cblas_sgemm(
//         CblasColMajor, CblasNoTrans, CblasNoTrans, Row, mat._columnSize, Col, 1.0, this->ElementsPointer(), Row,
//         mat.ElementsPointer(), Col, 1.0, res.ElementsPointer(), Row
//     );
//     return res;
// }
/* end matrix unique arithmetics definition */

#endif