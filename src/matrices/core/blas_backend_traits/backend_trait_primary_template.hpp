namespace mknnlib::matrix::core {
enum class MemorySpace {
    Host,
    Device,
};

template <typename Backend>
struct BackendTraits;
}