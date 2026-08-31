#include <array>
#include <cstddef>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename T, size_t Size>
struct Storage<OpenBLASBackend, T, Size> {
    std::array<T, Size> _data;

    constexpr size_t size() const noexcept;
    constexpr T* data() noexcept;
    constexpr const T* data() const noexcept;
};

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
}  // namespace mknnlib::matrix::core