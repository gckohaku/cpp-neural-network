#ifndef MATRIX_STATIC_HPP
#define MATRIX_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ostream>

#include "src/concept_defines/types/type_concepts.hpp"
#include "src/matrices/matrix_row_static.hpp"

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
    MatrixStatic<K, Row, Col>& operator+=(const MatrixStatic<K, Row, Col>& x) requires SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator-=(const MatrixStatic<K, Row, Col>& x) requires SingleFloatingPoint<K>;
    MatrixStatic<K, Row, Col>& operator*=(const MatrixStatic<K, Row, Col>& x);

    // 2 dimensions index
    K& operator[](const size_t a, const size_t b);

    // ostream
    friend std::ostream& operator<< <>(std::ostream& os, const MatrixStatic<K, Row, Col>& mat);
    /* end operator overloads declaration */

    /* begin matrix unique functions declaration */
    K* ElementsPointer();
    const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    MatrixStatic<K, Row, OppCol> Dot(const MatrixStatic<K, Col, OppCol> mat) requires SingleFloatingPoint<K>;

    MatrixRowStatic<K, Row> Dot(const MatrixRowStatic<K, Col> mat) requires SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};

/* begin constructors definition */
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>::MatrixStatic()
    : _rowSize(Row), _columnSize(Col), _elements{}, _span(_elements.data(), MatrixExtent{}) {}

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
        _span = MdView(ElementsPointer(), MatrixExtent{});
    }
    return *this;
}

// arithmetics
template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator+=(const MatrixStatic<K, Row, Col>& x) requires SingleFloatingPoint<K> {
    cblas_saxpy(Row * Col, 1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator-=(const MatrixStatic<K, Row, Col>& x) requires SingleFloatingPoint<K> {
    cblas_saxpy(Row * Col, -1.0, x.ElementsPointer(), 1, this->ElementsPointer(), 1);
    return *this;
}

template <typename K, size_t Row, size_t Col>
MatrixStatic<K, Row, Col>& MatrixStatic<K, Row, Col>::operator*=(const MatrixStatic<K, Row, Col>& x) {
    // hadamard product is not into BLAS
    std::ranges::transform(this->_elements, x._elements, this->_elements.begin(), std::multiplies<>());
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
MatrixStatic<K, Row, OppCol> MatrixStatic<K, Row, Col>::Dot(const MatrixStatic<K, Col, OppCol> mat) requires SingleFloatingPoint<K> {
    auto res = MatrixStatic<K, Row, OppCol>();
    cblas_sgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans, Row, OppCol, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 1.0, res.ElementsPointer(), Row
    );
    return res;
}

template <typename K, size_t Row, size_t Col>
MatrixRowStatic<K, Row> MatrixStatic<K, Row, Col>::Dot(const MatrixRowStatic<K, Col> mat) requires SingleFloatingPoint<K> {
    auto res = MatrixRowStatic<K, Row>();
    cblas_sgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans, Row, mat._columnSize, Col, 1.0, this->ElementsPointer(), Row,
        mat.ElementsPointer(), Col, 1.0, res.ElementsPointer(), Row
    );
    return res;
}
/* end matrix unique arithmetics definition */

#endif