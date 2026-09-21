#ifndef MKNNLIB_MATRICES_CORE_LAYOUT_HPP
#define MKNNLIB_MATRICES_CORE_LAYOUT_HPP

#include <concepts>
namespace mknnlib::matrix::core {
struct MatrixColumnMajor;
struct MatrixRowMajor;

template <typename T>
concept MatrixStorageOrderLayout = std::same_as<T, MatrixColumnMajor> || std::same_as<T, MatrixRowMajor>;

template <typename T>
    requires MatrixStorageOrderLayout<T>
struct MatrixTransposeOrderLayout;

template <>
struct MatrixTransposeOrderLayout<MatrixColumnMajor> {
    using type = MatrixRowMajor;
};

template <>
struct MatrixTransposeOrderLayout<MatrixRowMajor> {
    using type = MatrixColumnMajor;
};

template <typename T>
using MatrixTransposeOrderLayout_t = typename MatrixTransposeOrderLayout<T>::type;
}  // namespace mknnlib::matrix::core

#endif