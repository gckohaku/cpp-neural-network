#ifndef TESTS_TEST_MATRIX_ADD_VECTOR_DEFINES_HPP
#define TESTS_TEST_MATRIX_ADD_VECTOR_DEFINES_HPP

#include <cstddef>
#include <vector>

namespace matrix_test_add {
/* define A and B */
template <typename T>
std::vector<T> MakeVectorA() {
    return std::vector<T>{static_cast<T>(5.1), static_cast<T>(3), static_cast<T>(5.4), static_cast<T>(5.7),
        static_cast<T>(2.4), static_cast<T>(3.4), static_cast<T>(1.5), static_cast<T>(2.7), static_cast<T>(1.6)};
}

template <typename T>
std::vector<T> MakeVectorB() {
    return std::vector<T>{static_cast<T>(5.3), static_cast<T>(2.8), static_cast<T>(3.4), static_cast<T>(1.4),
        static_cast<T>(3), static_cast<T>(5.7), static_cast<T>(2.7), static_cast<T>(4.2), static_cast<T>(3.3)};
}

template <typename T>
std::vector<T> MakeExpectedVectorAAndB() {
    return std::vector<T>{static_cast<T>(10.4), static_cast<T>(5.8), static_cast<T>(8.8), static_cast<T>(7.1),
        static_cast<T>(5.4), static_cast<T>(9.1), static_cast<T>(4.2), static_cast<T>(6.9), static_cast<T>(4.9)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorAAndB() {
    return std::vector<T>{static_cast<T>(8), static_cast<T>(4), static_cast<T>(8), static_cast<T>(8),
        static_cast<T>(4), static_cast<T>(8), static_cast<T>(4), static_cast<T>(8), static_cast<T>(4)};
}

/* define C and D */
template <typename T>
std::vector<T> MakeVectorC() {
    return std::vector<T>{static_cast<T>(5.2), static_cast<T>(4.3), static_cast<T>(1.8), static_cast<T>(2.5),
        static_cast<T>(3.6), static_cast<T>(3.7), static_cast<T>(4.1), static_cast<T>(6.3), static_cast<T>(3),
        static_cast<T>(7.7)};
}

template <typename T>
std::vector<T> MakeVectorD() {
    return std::vector<T>{static_cast<T>(3.6), static_cast<T>(4.8), static_cast<T>(5.7), static_cast<T>(3.4),
        static_cast<T>(1.5), static_cast<T>(2.7), static_cast<T>(6.7), static_cast<T>(3.5), static_cast<T>(2.5),
        static_cast<T>(1.1)};
}

template <typename T>
std::vector<T> MakeExpectedVectorCAndD() {
    return std::vector<T>{static_cast<T>(8.8), static_cast<T>(9.1), static_cast<T>(7.5), static_cast<T>(5.9),
        static_cast<T>(5.1), static_cast<T>(6.4), static_cast<T>(10.8), static_cast<T>(9.8), static_cast<T>(5.5),
        static_cast<T>(8.8)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorCAndD() {
    return std::vector<T>{static_cast<T>(8), static_cast<T>(8), static_cast<T>(8), static_cast<T>(4),
        static_cast<T>(4), static_cast<T>(4), static_cast<T>(8), static_cast<T>(8), static_cast<T>(4),
        static_cast<T>(8)};
}

template <typename T>
std::vector<T> MakeVectorE() {
    return std::vector<T>{static_cast<T>(2.4), static_cast<T>(5.1), static_cast<T>(5.7), static_cast<T>(4.3),
        static_cast<T>(1.7), static_cast<T>(4.9)};
}

template <typename T>
std::vector<T> MakeVectorF() {
    return std::vector<T>{
        static_cast<T>(5.1),
        static_cast<T>(3.9),
        static_cast<T>(6.7),
        static_cast<T>(3.7),
        static_cast<T>(6.2),
        static_cast<T>(4.2),
    };
}

template <typename T>
std::vector<T> MakeVectorG() {
    return std::vector<T>{
        static_cast<T>(3.7),
        static_cast<T>(5.7),
        static_cast<T>(2.3),
        static_cast<T>(4.0),
        static_cast<T>(1.7),
        static_cast<T>(2.7),
    };
}

template <typename T>
std::vector<T> MakeVectorH() {
    return std::vector<T>{static_cast<T>(6.7), static_cast<T>(1.7), static_cast<T>(2.4), static_cast<T>(3)};
}
}  // namespace matrix_test_add
#endif