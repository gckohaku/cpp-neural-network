#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <cstdlib>
template <typename T, size_t N>
void CheckCloseEachArrayElement(
    const std::array<T, N> left, const std::array<T, N> right, const std::array<T, N> tolerance
) {
    for (size_t i = 0; i < N; i++) {
        BOOST_CHECK_MESSAGE(
            std::abs(left[i] - right[i]) <= tolerance[i],
            "\nExceeds tolerance in index of " << i << ".\n"
                                             << "Elements: (" << left[i] << ", " << right[i] << ")\n"
                                             << "Expected error: " << tolerance[i] << "\n"
                                             << "Actual Error" << std::abs(left[i] - right[i])
                                             << "is exceeds expected error."
        );
    }
}