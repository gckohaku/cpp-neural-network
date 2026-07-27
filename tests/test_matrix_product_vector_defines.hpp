#ifndef TESTS_TEST_MATRIX_PRODUCT_VECTOR_DEFINES_HPP
#define TESTS_TEST_MATRIX_PRODUCT_VECTOR_DEFINES_HPP

#include <cstddef>
#include <vector>

namespace matrix_test_matrix_product {
/* define A and B */
template <typename T>
std::vector<T> MakeVectorA() {
    return std::vector<T>{static_cast<T>(5.3), static_cast<T>(6.7), static_cast<T>(3.7), static_cast<T>(4.1),
        static_cast<T>(7.4), static_cast<T>(8), static_cast<T>(3.4), static_cast<T>(8.2), static_cast<T>(4.5)};
}

template <typename T>
std::vector<T> MakeVectorB() {
    return std::vector<T>{static_cast<T>(5.7), static_cast<T>(5.5), static_cast<T>(0.3), static_cast<T>(8.3),
        static_cast<T>(2), static_cast<T>(1.4), static_cast<T>(7.2), static_cast<T>(3.1), static_cast<T>(7.9)};
}

template <typename T>
std::vector<T> MakeExpectedVectorAB() {
    return std::vector<T>{static_cast<T>(53.78), static_cast<T>(81.35), static_cast<T>(66.44), static_cast<T>(56.95),
        static_cast<T>(81.89), static_cast<T>(53.01), static_cast<T>(77.73), static_cast<T>(135.96),
        static_cast<T>(86.99)};
}

template <typename T>
std::vector<T> MakeExpectedVectorBA() {
    return std::vector<T>{static_cast<T>(112.46), static_cast<T>(54.02), static_cast<T>(40.2), static_cast<T>(142.39),
        static_cast<T>(62.15), static_cast<T>(74.79), static_cast<T>(119.84), static_cast<T>(49.05),
        static_cast<T>(48.05)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorAB() {
    return std::vector<T>{static_cast<T>(83.85), static_cast<T>(105.65), static_cast<T>(104.525), static_cast<T>(98),
        static_cast<T>(129), static_cast<T>(88.3), static_cast<T>(100), static_cast<T>(178.8), static_cast<T>(119.6)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorBA() {
    return std::vector<T>{static_cast<T>(160), static_cast<T>(78.2), static_cast<T>(45.425), static_cast<T>(221.2),
        static_cast<T>(102), static_cast<T>(110.425), static_cast<T>(203.8), static_cast<T>(78.4),
        static_cast<T>(70.45)};
}

/* define C and D */
template <typename T>
std::vector<T> MakeVectorC() {
    return std::vector<T>{static_cast<T>(2.1), static_cast<T>(1.5), static_cast<T>(0.7), static_cast<T>(6),
        static_cast<T>(2.6), static_cast<T>(3.4), static_cast<T>(3.4), static_cast<T>(5.1), static_cast<T>(4.2),
        static_cast<T>(4.3), static_cast<T>(6.3), static_cast<T>(1.7), static_cast<T>(6.7), static_cast<T>(7.3),
        static_cast<T>(3.7), static_cast<T>(5.4), static_cast<T>(3.4), static_cast<T>(2.4), static_cast<T>(3.5),
        static_cast<T>(1.4), static_cast<T>(3.7), static_cast<T>(5.6), static_cast<T>(5.2), static_cast<T>(1.4),
        static_cast<T>(0.8)};
}

template <typename T>
std::vector<T> MakeVectorD() {
    return std::vector<T>{static_cast<T>(2.4), static_cast<T>(1.5), static_cast<T>(3.6), static_cast<T>(6.3),
        static_cast<T>(1.4), static_cast<T>(5.4), static_cast<T>(2.4), static_cast<T>(0.4), static_cast<T>(2.2),
        static_cast<T>(3.2), static_cast<T>(7), static_cast<T>(3), static_cast<T>(2.3), static_cast<T>(4.7),
        static_cast<T>(4.4), static_cast<T>(3.6), static_cast<T>(7.5), static_cast<T>(0.8), static_cast<T>(3),
        static_cast<T>(0.3), static_cast<T>(3), static_cast<T>(3.1), static_cast<T>(2.4), static_cast<T>(0.1),
        static_cast<T>(0.8)};
}

template <typename T>
std::vector<T> MakeExpectedVectorCD() {
    return std::vector<T>{static_cast<T>(72.02), static_cast<T>(44.08), static_cast<T>(55.85), static_cast<T>(70.99),
        static_cast<T>(35.95), static_cast<T>(45.74), static_cast<T>(42.34), static_cast<T>(40.62),
        static_cast<T>(57.58), static_cast<T>(31.48), static_cast<T>(81.05), static_cast<T>(65.23),
        static_cast<T>(69.77), static_cast<T>(94), static_cast<T>(49.71), static_cast<T>(55.41), static_cast<T>(44.14),
        static_cast<T>(54.89), static_cast<T>(69.86), static_cast<T>(49.01), static_cast<T>(35.46),
        static_cast<T>(23.94), static_cast<T>(38.39), static_cast<T>(50.01), static_cast<T>(30.79)};
}

template <typename T>
std::vector<T> MakeExpectedVectorDC() {
    return std::vector<T>{static_cast<T>(47.44), static_cast<T>(61.91), static_cast<T>(20.81), static_cast<T>(38.08),
        static_cast<T>(14.7), static_cast<T>(90.24), static_cast<T>(73.39), static_cast<T>(39.01), static_cast<T>(65.9),
        static_cast<T>(42.78), static_cast<T>(108.58), static_cast<T>(99.85), static_cast<T>(53.49),
        static_cast<T>(97.19), static_cast<T>(48.89), static_cast<T>(64.92), static_cast<T>(54.05),
        static_cast<T>(32.48), static_cast<T>(63.42), static_cast<T>(31.17), static_cast<T>(82.96),
        static_cast<T>(47.57), static_cast<T>(30.56), static_cast<T>(64.35), static_cast<T>(47.04)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorCD() {
    return std::vector<T>{static_cast<T>(95.7), static_cast<T>(56.2), static_cast<T>(74.5), static_cast<T>(90.2),
        static_cast<T>(48.3), static_cast<T>(67.375), static_cast<T>(59.025), static_cast<T>(60.975),
        static_cast<T>(84.425), static_cast<T>(49.325), static_cast<T>(121), static_cast<T>(94.5),
        static_cast<T>(108.5), static_cast<T>(129.6), static_cast<T>(72.7), static_cast<T>(70.675),
        static_cast<T>(52.25), static_cast<T>(73.45), static_cast<T>(93.7), static_cast<T>(69.2),
        static_cast<T>(49.5875), static_cast<T>(31.2125), static_cast<T>(54.65), static_cast<T>(70.91875),
        static_cast<T>(45.3875)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorDC() {
    return std::vector<T>{static_cast<T>(64.3), static_cast<T>(78.8), static_cast<T>(30.925), static_cast<T>(55.7125),
        static_cast<T>(21.7), static_cast<T>(127.8), static_cast<T>(108), static_cast<T>(58.55),
        static_cast<T>(97.66875), static_cast<T>(65), static_cast<T>(131.6), static_cast<T>(116.3),
        static_cast<T>(69.475), static_cast<T>(128.63125), static_cast<T>(69.375), static_cast<T>(88.4),
        static_cast<T>(68.7), static_cast<T>(44.8), static_cast<T>(90.1875), static_cast<T>(45.575),
        static_cast<T>(114.5), static_cast<T>(66.15), static_cast<T>(41.5), static_cast<T>(92.9),
        static_cast<T>(70.35)};
}

/* define E and F */
template <typename T>
std::vector<T> MakeVectorE() {
    return std::vector<T>{static_cast<T>(3.1), static_cast<T>(2.1), static_cast<T>(6.3), static_cast<T>(4.6),
        static_cast<T>(3.7), static_cast<T>(4.7), static_cast<T>(5.1), static_cast<T>(2.2), static_cast<T>(5.7),
        static_cast<T>(3.4), static_cast<T>(2.7), static_cast<T>(3.1)};
}

template <typename T>
std::vector<T> MakeVectorF() {
    return std::vector<T>{static_cast<T>(1.7), static_cast<T>(4.5), static_cast<T>(5.3), static_cast<T>(1.7),
        static_cast<T>(3.4), static_cast<T>(3.7), static_cast<T>(0.5), static_cast<T>(3.5), static_cast<T>(1.5),
        static_cast<T>(1.4), static_cast<T>(3.4), static_cast<T>(4.3)};
}

template <typename T>
std::vector<T> MakeExpectedVectorEF() {
    return std::vector<T>{static_cast<T>(58.78), static_cast<T>(36.47), static_cast<T>(67.34), static_cast<T>(42.01),
        static_cast<T>(31.38), static_cast<T>(52.51), static_cast<T>(43.05), static_cast<T>(27.42),
        static_cast<T>(48.74)};
}

template <typename T>
std::vector<T> MakeExpectedVectorFE() {
    return std::vector<T>{static_cast<T>(21.86), static_cast<T>(30.54), static_cast<T>(38.9), static_cast<T>(39.71),
        static_cast<T>(27.45), static_cast<T>(40.97), static_cast<T>(42.21), static_cast<T>(40.98),
        static_cast<T>(24.7), static_cast<T>(39.07), static_cast<T>(47.51), static_cast<T>(40.88),
        static_cast<T>(19.61), static_cast<T>(29.63), static_cast<T>(29.91), static_cast<T>(28.56)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorEF() {
    return std::vector<T>{static_cast<T>(91.3), static_cast<T>(54.8), static_cast<T>(100.4), static_cast<T>(55.35),
        static_cast<T>(40.3), static_cast<T>(68.45), static_cast<T>(62.3), static_cast<T>(42.2), static_cast<T>(68.6)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorFE() {
    return std::vector<T>{static_cast<T>(29.8), static_cast<T>(44.9), static_cast<T>(51.25), static_cast<T>(60.1),
        static_cast<T>(36.3), static_cast<T>(61.5), static_cast<T>(65.45), static_cast<T>(61.8), static_cast<T>(34.8),
        static_cast<T>(61.5), static_cast<T>(68.7), static_cast<T>(63.3), static_cast<T>(25.1), static_cast<T>(41.3),
        static_cast<T>(39.55), static_cast<T>(40.2)};
}

/* define G and H */
template <typename T>
std::vector<T> MakeVectorG() {
    return std::vector<T>{static_cast<T>(3.5), static_cast<T>(6.2), static_cast<T>(4), static_cast<T>(5.4)};
}

template <typename T>
std::vector<T> MakeVectorH() {
    return std::vector<T>{static_cast<T>(1.2), static_cast<T>(3.4), static_cast<T>(5.1), static_cast<T>(3.4),
        static_cast<T>(3.4), static_cast<T>(1.4), static_cast<T>(3.6), static_cast<T>(3.6), static_cast<T>(5.1),
        static_cast<T>(0.7), static_cast<T>(7.2), static_cast<T>(2.5), static_cast<T>(2), static_cast<T>(3.4),
        static_cast<T>(3.4), static_cast<T>(1.4)};
}

template <typename T>
std::vector<T> MakeExpectedVectorHG() {
    return std::vector<T>{static_cast<T>(56.48), static_cast<T>(41.74), static_cast<T>(87.33), static_cast<T>(51.78)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorHG() {
    return std::vector<T>{static_cast<T>(87.1), static_cast<T>(54.8), static_cast<T>(120.2), static_cast<T>(69.6)};
}

/* define I and J */
template <typename T>
std::vector<T> MakeVectorI() {
    return std::vector<T>{static_cast<T>(4.1), static_cast<T>(3.6), static_cast<T>(1.7), static_cast<T>(2.4),
        static_cast<T>(2.7), static_cast<T>(3.7), static_cast<T>(0.8), static_cast<T>(3), static_cast<T>(5.4),
        static_cast<T>(3.5), static_cast<T>(3.2), static_cast<T>(1.7), static_cast<T>(1.7), static_cast<T>(4.6),
        static_cast<T>(3.3)};
}

template <typename T>
std::vector<T> MakeVectorJ() {
    return std::vector<T>{static_cast<T>(2.7), static_cast<T>(1.6), static_cast<T>(3.5), static_cast<T>(2.2),
        static_cast<T>(1.7), static_cast<T>(0.9)};
}

template <typename T>
std::vector<T> MakeExpectedVectorJI() {
    return std::vector<T>{static_cast<T>(26.56), static_cast<T>(16.01), static_cast<T>(22.22), static_cast<T>(13.11),
        static_cast<T>(21.84), static_cast<T>(12.74), static_cast<T>(23.54), static_cast<T>(14.17),
        static_cast<T>(26.3), static_cast<T>(15.81)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorJI() {
    return std::vector<T>{static_cast<T>(36.6), static_cast<T>(23.85), static_cast<T>(29.7), static_cast<T>(19.05),
        static_cast<T>(28.15), static_cast<T>(18.3), static_cast<T>(29.2), static_cast<T>(19.25), static_cast<T>(36),
        static_cast<T>(24.75)};
}

/* define K */
template <typename T>
std::vector<T> MakeVectorK() {
    return std::vector<T>{static_cast<T>(0.0), static_cast<T>(-1.0), static_cast<T>(1.0), static_cast<T>(0.0)};
}

template <typename T>
std::vector<T> MakeExpectedVectorKK() {
    return std::vector<T>{static_cast<T>(-1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(-1.0)};
}

template <typename T>
std::vector<T> MakeAcceptableErrorVectorKK() {
    return std::vector<T>{static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0)};
}

/* define M and N */
template <typename T>
std::vector<T> MakeVectorM() {
    return std::vector<T>{static_cast<T>(2.4), static_cast<T>(5.4), static_cast<T>(7.1), static_cast<T>(3.1),
        static_cast<T>(2.4), static_cast<T>(4.8)};
}

template <typename T>
std::vector<T> MakeVectorN() {
    return std::vector<T>{static_cast<T>(2.3), static_cast<T>(2.9), static_cast<T>(3.5), static_cast<T>(5.7),
        static_cast<T>(3.8), static_cast<T>(7.3), static_cast<T>(4.2), static_cast<T>(1.5), static_cast<T>(2.7)};
}
}  // namespace matrix_test_matrix_product
#endif