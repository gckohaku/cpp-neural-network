#include <array>
#include <cstddef>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename T, size_t Size>
struct Storage<OpenBLASBackend, T, Size> {
    std::array<T, Size> data;
};
}  // namespace mknnlib::matrix::core