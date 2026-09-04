#ifndef MKNNLIB_MATRICES_CORE_BLAS_STORAGES_OPENBLAS_STATIC_STORAGE
#define MKNNLIB_MATRICES_CORE_BLAS_STORAGES_OPENBLAS_STATIC_STORAGE

#include <array>
#include <cstddef>
#include <vector>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename T, size_t Size>
struct Storage<OpenBLASBackend, T, Size> {
    std::array<T, Size> _data;

    Storage();

    Storage(std::array<T, Size> elements);
    Storage(std::vector<T> elements);

    constexpr size_t size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;

    std::vector<T> GetVector();
};

template <typename  T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage() : _data(T{}) {};

template <typename  T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage(std::array<T, Size> elements) : _data(elements) {};

template <typename  T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage(std::vector<T> elements) : _data(Size, elements) {}

template <typename T, size_t Size>
inline constexpr size_t Storage<OpenBLASBackend, T, Size>::size() const noexcept {
    return _data.size();
}

template <typename T, size_t Size>
inline constexpr T* Storage<OpenBLASBackend, T, Size>::data() noexcept {
    return _data.data();
}

template <typename T, size_t Size>
inline constexpr const T* Storage<OpenBLASBackend, T, Size>::data() const noexcept {
    return _data.data();
}

template <typename T, size_t Size>
std::vector<T> Storage<OpenBLASBackend, T, Size>::GetVector() {
    std::vector<T> vec(this->_data.begin(), this->_data.end());
    return vec;
}
}  // namespace mknnlib::matrix::core

#endif