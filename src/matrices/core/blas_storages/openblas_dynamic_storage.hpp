#ifndef MKNNLIB_MATRICES_CORE_BLAS_STORAGES_OPENBLAS_DYNAMIC_STORAGE
#define MKNNLIB_MATRICES_CORE_BLAS_STORAGES_OPENBLAS_DYNAMIC_STORAGE

#include <cstddef>
#include <span>
#include <vector>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename T>
struct Storage<OpenBLASBackend, T, std::dynamic_extent> {
    std::vector<T> _data;

    constexpr size_t size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;
};

template <typename T>
inline constexpr size_t Storage<OpenBLASBackend, T, std::dynamic_extent>::size() const noexcept {
    return _data.size();
}

template <typename T>
inline constexpr T* Storage<OpenBLASBackend, T, std::dynamic_extent>::data() noexcept {
    return _data.data();
}

template <typename T>
inline constexpr const T* Storage<OpenBLASBackend, T, std::dynamic_extent>::data() const noexcept {
    return _data.data();
}
}  // namespace mknnlib::matrix::core

#endif