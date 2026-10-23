#ifndef MKNNLIB_MATRICES_CORE_TYPE_UTILITIES_HPP
#define MKNNLIB_MATRICES_CORE_TYPE_UTILITIES_HPP

#include <span>
namespace mknnlib::matrix::core {
    template <typename Extents>
    constexpr std::size_t MatrixElementSize = (Extents::rank_dynamic() == 0) ? Extents::static_extent(0) * Extents::static_extent(1) : std::dynamic_extent;
}

#endif