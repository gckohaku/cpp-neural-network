#include <cstddef>
#include <span>
#include <vector>

#include "src/matrices/core/blas_backends.hpp"
#include "src/matrices/core/blas_storages/storage_primary_template.hpp"

namespace mknnlib::matrix::core {
template <typename T>
struct Storage<OpenBLASBackend, T, std::dynamic_extent> {
    std::vector<T> data;
};
}  // namespace mknnlib::matrix::core