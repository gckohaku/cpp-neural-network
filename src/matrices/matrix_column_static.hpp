#ifndef MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP
#define MKNNLIB_MATRICES_MATRIX_COLUMN_STATIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
#include <boost/operators.hpp>
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
    using MatrixExtent = std::extents<size_t, Col>;
    using MdView = std::mdspan<K, MatrixExtent, std::layout_left>;

private:
    size_t _rowSize = Col;
    size_t _columnSize = 0;
    std::vector<K> _elements;
    MdView _span;

public:
    // TODO: ドキュメントをちゃんと書く
    /* begin constructors declaration */
    MatrixColumnStatic();
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
    K* ElementsPointer();
    const K* ElementsPointer() const;

    /* end matrix unique functions declaration */

    /* begin matrix unique arithmetics declaration */
    template <size_t OppCol>
    MatrixColumnStatic<K, Col> Dot(const MatrixColumnStatic<K, Col> mat)
        requires concepts::SingleFloatingPoint<K>;
    /* end matrix unique arithmetics declaration */
};
}  // namespace mknnlib::matrix

#endif