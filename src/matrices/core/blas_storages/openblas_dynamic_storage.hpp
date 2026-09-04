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

    Storage();

    template <size_t Size>
    Storage(std::array<T, Size> elements);

    Storage(std::vector<T> elements);
    Storage(size_t size);
    Storage(size_t size, std::vector<T> elements);

    constexpr size_t size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;

    std::vector<T> GetVector();
};

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage() : _data(0, T{}) {};

template <typename T>
template <size_t Size>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(std::array<T, Size> elements) :
    _data(elements.begin(), elements.end()) {};

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(std::vector<T> elements) :
    _data(elements.begin(), elements.end()) {}

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(size_t size) : _data(size, T{}) {}

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(size_t size, std::vector<T> elements) :
    _data(size, elements) {}

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

template <typename T>
std::vector<T> Storage<OpenBLASBackend, T, std::dynamic_extent>::GetVector() {
    std::vector<T> vec(this->_data.begin(), this->_data.end());
    return vec;
}
}  // namespace mknnlib::matrix::core

#endif