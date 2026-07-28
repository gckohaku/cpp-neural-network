#ifndef TESTS_TEST_MATRIX_SUB_ARRAY_DEFINES_HPP
#define TESTS_TEST_MATRIX_SUB_ARRAY_DEFINES_HPP

#include <array>
#include <cstddef>

namespace matrix_test_sub {
/* define A and B */
template <typename T>
std::array<T, 3 * 3> MakeArrayA() {
    return std::array<T, 3 * 3>{static_cast<T>(5.1), static_cast<T>(3), static_cast<T>(5.4), static_cast<T>(5.7),
        static_cast<T>(2.4), static_cast<T>(3.4), static_cast<T>(1.5), static_cast<T>(2.7), static_cast<T>(1.6)};
}

template <typename T>
std::array<T, 3 * 3> MakeArrayB() {
    return std::array<T, 3 * 3>{static_cast<T>(5.3), static_cast<T>(2.8), static_cast<T>(3.4), static_cast<T>(1.4),
        static_cast<T>(3), static_cast<T>(5.7), static_cast<T>(2.7), static_cast<T>(4.2), static_cast<T>(3.3)};
}

template <typename T>
std::array<T, 3 * 3> MakeExpectedArrayAAndB() {
    return std::array<T, 3 * 3>{static_cast<T>(-0.2), static_cast<T>(0.2), static_cast<T>(2), static_cast<T>(4.3),
        static_cast<T>(-0.6), static_cast<T>(-2.3), static_cast<T>(-1.2), static_cast<T>(-1.5), static_cast<T>(-1.7)};
}

template <typename T>
std::array<T, 3 * 3> MakeAcceptableErrorArrayAAndB() {
    return std::array<T, 3 * 3>{static_cast<T>(8), static_cast<T>(4), static_cast<T>(8), static_cast<T>(8),
        static_cast<T>(4), static_cast<T>(8), static_cast<T>(4), static_cast<T>(8), static_cast<T>(4)};
}

/* define C and D */
template <typename T>
std::array<T, 2 * 5> MakeArrayC() {
    return std::array<T, 2 * 5>{static_cast<T>(5.2), static_cast<T>(4.3), static_cast<T>(1.8), static_cast<T>(2.5),
        static_cast<T>(3.6), static_cast<T>(3.7), static_cast<T>(4.1), static_cast<T>(6.3), static_cast<T>(3),
        static_cast<T>(7.7)};
}

template <typename T>
std::array<T, 2 * 5> MakeArrayD() {
    return std::array<T, 2 * 5>{static_cast<T>(3.6), static_cast<T>(4.8), static_cast<T>(5.7), static_cast<T>(3.4),
        static_cast<T>(1.5), static_cast<T>(2.7), static_cast<T>(6.7), static_cast<T>(3.5), static_cast<T>(2.5),
        static_cast<T>(1.1)};
}

template <typename T>
std::array<T, 2 * 5> MakeExpectedArrayCAndD() {
    return std::array<T, 2 * 5>{static_cast<T>(1.6), static_cast<T>(-0.5), static_cast<T>(-3.9), static_cast<T>(-0.9),
        static_cast<T>(2.1), static_cast<T>(1), static_cast<T>(-2.6), static_cast<T>(2.8), static_cast<T>(0.5),
        static_cast<T>(6.6)};
}

template <typename T>
std::array<T, 2 * 5> MakeAcceptableErrorArrayCAndD() {
    return std::array<T, 2 * 5>{static_cast<T>(8), static_cast<T>(8), static_cast<T>(8), static_cast<T>(4),
        static_cast<T>(4), static_cast<T>(4), static_cast<T>(8), static_cast<T>(8), static_cast<T>(4),
        static_cast<T>(8)};
}

template <typename T>
std::array<T, 3 * 2> MakeArrayE() {
    return std::array<T, 3 * 2>{static_cast<T>(2.4), static_cast<T>(5.1), static_cast<T>(5.7), static_cast<T>(4.3),
        static_cast<T>(1.7), static_cast<T>(4.9)};
}

template <typename T>
std::array<T, 3 * 3> MakeArrayF() {
    return std::array<T, 3 * 3>{
        static_cast<T>(5.1),
        static_cast<T>(3.9),
        static_cast<T>(6.7),
        static_cast<T>(3.7),
        static_cast<T>(6.2),
        static_cast<T>(4.2),
        static_cast<T>(5.1),
        static_cast<T>(2.8),
        static_cast<T>(2.0),
    };
}

template <typename T>
std::array<T, 3 * 3> MakeArrayG() {
    return std::array<T, 3 * 3>{
        static_cast<T>(3.7),
        static_cast<T>(5.7),
        static_cast<T>(2.3),
        static_cast<T>(4.0),
        static_cast<T>(1.7),
        static_cast<T>(2.7),
        static_cast<T>(2.6),
        static_cast<T>(1.2),
        static_cast<T>(3.4),
    };
}
}  // namespace matrix_test_sub
#endif