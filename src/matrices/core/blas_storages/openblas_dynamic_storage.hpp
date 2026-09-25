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
public:
    using iterator = std::vector<T>::iterator;
    using const_iterator = std::vector<T>::const_iterator;
    using size_type = std::vector<T>::size_type;
    std::vector<T> _elements;

    Storage();

    template <size_t Size>
    Storage(std::array<T, Size> elements);

    Storage(std::vector<T> elements);
    Storage(size_t size);
    Storage(size_t size, std::vector<T> elements);
    Storage(const Storage&) = default;
    Storage(Storage&&) = default;

    Storage& operator=(const Storage&) = default;
    Storage& operator=(Storage&&) = default;

    constexpr std::vector<T>& elements() noexcept;
    constexpr const std::vector<T>& elements() const noexcept;

    constexpr size_type size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;

    constexpr iterator begin() noexcept;
    constexpr const_iterator begin() const noexcept;

    constexpr iterator end() noexcept;
    constexpr const_iterator end() const noexcept;

    std::vector<T> GetVector();
    std::vector<T> GetVector() const;
};

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage() : _elements(0, T{}) {};

template <typename T>
template <size_t Size>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(std::array<T, Size> elements) :
    _elements(elements.begin(), elements.end()) {};

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(std::vector<T> elements) :
    _elements(elements.begin(), elements.end()) {}

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(size_t size) : _elements(size, T{}) {}

template <typename T>
Storage<OpenBLASBackend, T, std::dynamic_extent>::Storage(size_t size, std::vector<T> elements) :
    _elements(size, elements) {}

template <typename T>
inline constexpr std::vector<T>& Storage<OpenBLASBackend, T, std::dynamic_extent>::elements() noexcept {
    return this->_elements;
}

template <typename T>
inline constexpr const std::vector<T>& Storage<OpenBLASBackend, T, std::dynamic_extent>::elements() const noexcept {
    return this->_elements;
}

template <typename T>
inline constexpr auto Storage<OpenBLASBackend, T, std::dynamic_extent>::size() const noexcept -> size_type {
    return _elements.size();
}

template <typename T>
inline constexpr T* Storage<OpenBLASBackend, T, std::dynamic_extent>::data() noexcept {
    return _elements.data();
}

template <typename T>
inline constexpr const T* Storage<OpenBLASBackend, T, std::dynamic_extent>::data() const noexcept {
    return _elements.data();
}

template <typename T>
inline constexpr auto Storage<OpenBLASBackend, T, std::dynamic_extent>::begin() noexcept -> iterator {
    return _elements.begin();
}

template <typename T>
inline constexpr auto Storage<OpenBLASBackend, T, std::dynamic_extent>::begin() const noexcept -> const_iterator {
    return _elements.begin();
}

template <typename T>
inline constexpr auto Storage<OpenBLASBackend, T, std::dynamic_extent>::end() noexcept -> iterator {
    return _elements.end();
}

template <typename T>
inline constexpr auto Storage<OpenBLASBackend, T, std::dynamic_extent>::end() const noexcept -> const_iterator {
    return _elements.end();
}

template <typename T>
std::vector<T> Storage<OpenBLASBackend, T, std::dynamic_extent>::GetVector() {
    std::vector<T> vec(this->_elements.begin(), this->_elements.end());
    return vec;
}

template <typename T>
std::vector<T> Storage<OpenBLASBackend, T, std::dynamic_extent>::GetVector() const {
    std::vector<T> vec(this->_elements.begin(), this->_elements.end());
    return vec;
}
}  // namespace mknnlib::matrix::core

#endif