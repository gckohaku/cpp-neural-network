#ifndef MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP
#define MKNNLIB_MATRICES_MATRIX_DYNAMIC_HPP

#include <cblas.h>

#include <algorithm>
#include <array>
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
};
}  // namespace mknnlib::matrix

#endif