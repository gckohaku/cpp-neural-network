#ifndef TESTS_TEST_UTILITIES_HPP
#define TESTS_TEST_UTILITIES_HPP

#include <array>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <source_location>
#include <sstream>

template <typename T, size_t N>
void CheckCloseEachArrayElement(const std::array<T, N> left, const std::array<T, N> right,
    const std::array<T, N> tolerance, const std::source_location& location = std::source_location::current()) {
    for (size_t i = 0; i < N; i++) {
        std::ostringstream message;
        message << "\nFile: " << location.file_name() << ":" << location.line() << ":" << location.column() << ":\n";
        message << "Exceeds tolerance in index of " << i << ".\n";
        message << "Elements: (" << std::fixed << std::setprecision(8) << left[i] << ", " << right[i] << "),\n";
        message << "Expected error: " << std::defaultfloat << tolerance[i] << ",\n";
        message << "Actual Error: " << std::abs(left[i] - right[i]) << ".\n";
        message << "Actual error exceeds expected error.";
        BOOST_CHECK_MESSAGE(std::abs(left[i] - right[i]) <= tolerance[i], message.str());
    }
}

template <typename T>
void CheckCloseEachVectorElement(const std::vector<T> left, const std::vector<T> right, const std::vector<T> tolerance,
    const std::source_location& location = std::source_location::current()) {
    if (left.size() != right.size()) {
        std::ostringstream message;
        message << "\nFile: " << location.file_name() << ":" << location.line() << ":" << location.column() << ":\n";
        message << "Vector size is not match.\n";
        message << "Left: " << left.size() << ", Right: " << right.size() << ".";
        BOOST_ERROR(message.str());
        return;
    }
    const size_t N = left.size();
    for (size_t i = 0; i < N; i++) {
        std::ostringstream message;
        message << "\nFile: " << location.file_name() << ":" << location.line() << ":" << location.column() << ":\n";
        message << "Exceeds tolerance in index of " << i << ".\n";
        message << "Elements: (" << std::fixed << std::setprecision(8) << left[i] << ", " << right[i] << "),\n";
        message << "Expected error: " << std::defaultfloat << tolerance[i] << ",\n";
        message << "Actual Error: " << std::abs(left[i] - right[i]) << ".\n";
        message << "Actual error exceeds expected error.";
        BOOST_CHECK_MESSAGE(std::abs(left[i] - right[i]) <= tolerance[i], message.str());
    }
}

template<typename T, size_t Size>
std::vector<T> ArrayToVector(std::array<T, Size> arr) {
    return std::vector<T>(arr.begin(), arr.end());
}

template<typename T, size_t Size>
std::array<T, Size> VectorToArray(std::vector<T> vec) {
    auto arr = std::array<T, Size>();
    std::copy_n(vec.begin(), Size, arr.begin());

    return arr;
}

#endif