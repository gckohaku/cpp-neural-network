#include <array>
#include <boost/test/included/unit_test.hpp>
#include <cstddef>
#include <cstdlib>
#include <source_location>
#include <sstream>
template <typename T, size_t N>
void CheckCloseEachArrayElement(
    const std::array<T, N> left, const std::array<T, N> right, const std::array<T, N> tolerance,
    const std::source_location& location = std::source_location::current()
) {
    for (size_t i = 0; i < N; i++) {
        std::ostringstream message;
        message << "\nFile: " << location.file_name() << " at " << location.line() << ":" << location.column() << "\n";
        message << "Exceeds tolerance in index of " << i << ".\n";
        message << "Elements: (" << left[i] << ", " << right[i] << "),\n";
        message << "Expected error: " << tolerance[i] << ",\n";
        message << "Actual Error: " << std::abs(left[i] - right[i]) << ".\n";
        message << "Actual error exceeds expected error.";
        BOOST_CHECK_MESSAGE(std::abs(left[i] - right[i]) <= tolerance[i], message.str());
    }
}