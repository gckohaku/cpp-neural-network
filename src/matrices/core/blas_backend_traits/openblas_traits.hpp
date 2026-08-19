#include "src/matrices/core/blas_backend_traits/backend_trait_primary_template.hpp"
#include "src/matrices/core/blas_backends.hpp"

namespace mknnlib::matrix::core {
template <>
struct BackendTraits<OpenBLASBackend> {
    static constexpr MemorySpace memorySpace = MemorySpace::Host;
};
}  // namespace mknnlib::matrix::core