#ifndef MKNNLIB_MATRICES_CORE_MATRIX_SIZE_UTILITIES_HPP
#define MKNNLIB_MATRICES_CORE_MATRIX_SIZE_UTILITIES_HPP

#include <cstddef>
#include <span>
#include <type_traits>

#include "src/concept_defines/matrices/matrix_concepts.hpp"

namespace mknnlib::matrix::core {
template <std::size_t L, std::size_t R>
concept CompatibleExtent = L == R || L == std::dynamic_extent || R == std::dynamic_extent;

template <std::size_t L, std::size_t R>
    requires CompatibleExtent<L, R>
inline constexpr std::size_t MergeExtent = L == std::dynamic_extent ? R : L;

template <mk_concepts::MatrixLike M>
using MatrixExtents = typename std::remove_cvref_t<M>::MatrixExtent;

template <std::size_t LRow, std::size_t LCol, std::size_t RRow, std::size_t RCol>
concept CompatibleShape = CompatibleExtent<LRow, RRow> && CompatibleExtent<RRow, RCol>;

template <typename ML, typename MR>
concept CompatibleMatrixShape = mk_concepts::MatrixLike<ML> && mk_concepts::MatrixLike<MR> &&
    CompatibleShape<MatrixExtents<ML>::static_extent(0), MatrixExtents<ML>::static_extent(1),
        MatrixExtents<MR>::static_extent(0), MatrixExtents<MR>::static_extent(1)>;

}  // namespace mknnlib::matrix::core

#endif