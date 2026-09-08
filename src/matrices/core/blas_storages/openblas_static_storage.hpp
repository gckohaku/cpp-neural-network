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
public:
    using iterator = std::array<T, Size>::iterator;
    using const_iterator = std::array<T, Size>::const_iterator;
    using size_type = std::array<T, Size>::size_type;

    std::array<T, Size> _elements;

    Storage();

    Storage(std::array<T, Size> elements);
    Storage(std::vector<T> elements);

    constexpr std::array<T, Size>& elements() noexcept;
    constexpr std::array<T, Size>& elements() const noexcept;

    constexpr size_type size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;

    constexpr iterator begin() noexcept;
    constexpr const_iterator begin() const noexcept;

    std::vector<T> GetVector();
    std::vector<T> GetVector() const;
};

template <typename T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage() : _elements({}) {};

template <typename T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage(std::array<T, Size> elements) : _elements(elements) {};

template <typename T, size_t Size>
Storage<OpenBLASBackend, T, Size>::Storage(std::vector<T> elements) : _elements(Size, elements) {}

template <typename T, size_t Size>
inline constexpr std::array<T, Size>& Storage<OpenBLASBackend, T, Size>::elements() noexcept {
    return this->_elements;
}

template <typename T, size_t Size>
inline constexpr std::array<T, Size>& Storage<OpenBLASBackend, T, Size>::elements() const noexcept {
    return this->_elements;
}

template <typename T, size_t Size>
inline constexpr auto Storage<OpenBLASBackend, T, Size>::size() const noexcept -> size_type {
    return _elements.size();
}

template <typename T, size_t Size>
inline constexpr T* Storage<OpenBLASBackend, T, Size>::data() noexcept {
    return _elements.data();
}

template <typename T, size_t Size>
inline constexpr const T* Storage<OpenBLASBackend, T, Size>::data() const noexcept {
    return _elements.data();
}

template <typename T, size_t Size>
inline constexpr auto Storage<OpenBLASBackend, T, Size>::begin() noexcept -> iterator {
    return _elements.begin();
}

template <typename T, size_t Size>
inline constexpr auto Storage<OpenBLASBackend, T, Size>::begin() const noexcept -> const_iterator {
    return _elements.begin();
}

template <typename T, size_t Size>
std::vector<T> Storage<OpenBLASBackend, T, Size>::GetVector() {
    std::vector<T> vec(this->_elements.begin(), this->_elements.end());
    return vec;
}

template <typename T, size_t Size>
std::vector<T> Storage<OpenBLASBackend, T, Size>::GetVector() const {
    std::vector<T> vec(this->_elements.begin(), this->_elements.end());
    return vec;
}
}  // namespace mknnlib::matrix::core

#endif