#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixStatic;

BOOST_AUTO_TEST_CASE(matrix_static_test) {
    // 正方行列同士の積
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = std::array<float, elementSize1>{5.3f, 6.7f, 3.7f, 4.1f, 7.4f, 8.0f, 3.4f, 8.2f, 4.5f};
    auto arrayB = std::array<float, elementSize1>{5.7f, 5.5f, 0.3f, 8.3f, 2.0f, 1.4f, 7.2f, 3.1f, 7.9f};

    auto expectedAB =
        std::array<float, elementSize1>{53.78f, 81.35f, 66.44f, 56.95f, 81.89f, 53.01f, 77.73f, 135.96f, 86.99f};
    auto expectedBA =
        std::array<float, elementSize1>{112.46f, 54.02f, 40.2f, 142.39f, 62.15f, 74.79f, 119.84f, 49.05f, 48.05f};

    auto A = MatrixStatic<float, row1, column1>(arrayA);
    auto B = MatrixStatic<float, row1, column1>(arrayB);

    auto acceptableErrorAB =
        std::array<float, elementSize1>{83.85f, 105.65f, 104.525f, 98.0f, 129.0f, 88.3f, 100.0f, 178.8f, 119.6f};
    auto acceptableErrorBA =
        std::array<float, elementSize1>{160.0f, 78.2f, 45.425f, 221.2f, 102.0f, 110.425f, 203.8f, 78.4f, 70.45f};

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<float>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<float>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixStatic<float, row1, column1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixStatic<float, row1, column1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachArrayElement<float, elementSize1>(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachArrayElement<float, elementSize1>(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = std::array<float, elementSize2>{2.1f, 1.5f, 0.7f, 6.0f, 2.6f, 3.4f, 3.4f, 5.1f, 4.2f, 4.3f, 6.3f,
        1.7f, 6.7f, 7.3f, 3.7f, 5.4f, 3.4f, 2.4f, 3.5f, 1.4f, 3.7f, 5.6f, 5.2f, 1.4f, 0.8f};
    auto arrayD = std::array<float, elementSize2>{2.4f, 1.5f, 3.6f, 6.3f, 1.4f, 5.4f, 2.4f, 0.4f, 2.2f, 3.2f, 7.0f,
        3.0f, 2.3f, 4.7f, 4.4f, 3.6f, 7.5f, 0.8f, 3.0f, 0.3f, 3.0f, 3.1f, 2.4f, 0.1f, 0.8f};

    auto expectedCD = std::array<float, elementSize2>{72.02f, 44.08f, 55.85f, 70.99f, 35.95f, 45.74f, 42.34f, 40.62f,
        57.58f, 31.48f, 81.05f, 65.23f, 69.77f, 94.0f, 49.71f, 55.41f, 44.14f, 54.89f, 69.86f, 49.01f, 35.46f, 23.94f,
        38.39f, 50.01f, 30.79f};
    auto expectedDC = std::array<float, elementSize2>{47.44f, 61.91f, 20.81f, 38.08f, 14.7f, 90.24f, 73.39f, 39.01f,
        65.9f, 42.78f, 108.58f, 99.85f, 53.49f, 97.19f, 48.89f, 64.92f, 54.05f, 32.48f, 63.42f, 31.17f, 82.96f, 47.57f,
        30.56f, 64.35f, 47.04f};

    auto C = MatrixStatic<float, row2, column2>(arrayC);
    auto D = MatrixStatic<float, row2, column2>(arrayD);

    auto acceptableErrorCD = std::array<float, elementSize2>{95.7f, 56.2f, 74.5f, 90.2f, 48.3f, 67.375f, 59.025f,
        60.975f, 84.425f, 49.325f, 121.0f, 94.5f, 108.5f, 129.6f, 72.7f, 70.675f, 52.25f, 73.45f, 93.7f, 69.2f,
        49.5875f, 31.2125f, 54.65f, 70.91875f, 45.3875f};
    auto acceptableErrorDC = std::array<float, elementSize2>{64.3f, 78.8f, 30.925f, 55.7125f, 21.7f, 127.8f, 108.0f,
        58.55f, 97.66875f, 65.0f, 131.6f, 116.3f, 69.475f, 128.63125f, 69.375f, 88.4f, 68.7f, 44.8f, 90.1875f, 45.575f,
        114.5f, 66.15f, 41.5f, 92.9f, 70.35f};

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<float>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<float>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixStatic<float, row2, column2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixStatic<float, row2, column2>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachArrayElement(CD.Elements(), expectedCD, acceptableErrorCD);
    CheckCloseEachArrayElement(DC.Elements(), expectedDC, acceptableErrorDC);
}