#ifndef TESTS_TEST_MATRIX_MUL_ARRAY_DEFINES_HPP
#define TESTS_TEST_MATRIX_MUL_ARRAY_DEFINES_HPP

#include <array>
#include <cstddef>

namespace matrix_test_mul {
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
    return std::array<T, 3 * 3>{static_cast<T>(27.03), static_cast<T>(8.4), static_cast<T>(18.36), static_cast<T>(7.98),
        static_cast<T>(7.2), static_cast<T>(19.38), static_cast<T>(4.05), static_cast<T>(11.34), static_cast<T>(5.28)};
}

template <typename T>
std::array<T, 3 * 3> MakeAcceptableErrorArrayAAndB() {
    return std::array<T, 3 * 3>{static_cast<T>(41.6), static_cast<T>(11.6), static_cast<T>(24.4), static_cast<T>(11.3),
        static_cast<T>(10.8), static_cast<T>(25), static_cast<T>(5.7), static_cast<T>(19.2), static_cast<T>(6.5)};
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
    return std::array<T, 2 * 5>{static_cast<T>(18.72), static_cast<T>(20.64), static_cast<T>(10.26), static_cast<T>(8.5), static_cast<T>(5.4), static_cast<T>(9.99), static_cast<T>(27.47), static_cast<T>(22.05), static_cast<T>(7.5), static_cast<T>(8.47)};
}

template <typename T>
std::array<T, 2 * 5> MakeAcceptableErrorArrayCAndD() {
    return std::array<T, 2 * 5>{static_cast<T>(24.8), static_cast<T>(36.4), static_cast<T>(12.9), static_cast<T>(11.8), static_cast<T>(6.6), static_cast<T>(12.8), static_cast<T>(43.2), static_cast<T>(26.6), static_cast<T>(11), static_cast<T>(12.1)};
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
}  // namespace matrix_test_mul
#endif