#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_row_static.hpp"
#include "tests/test_utilities.hpp"
#include "tests/test_defines.hpp"

using mknnlib::matrix::MatrixRowStatic;

BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_row_static_test, T, MatrixProductCheckType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = std::vector<T>{5.3f, 6.7f, 3.7f, 4.1f, 7.4f, 8.0f, 3.4f, 8.2f, 4.5f};
    auto arrayB = std::vector<T>{5.7f, 5.5f, 0.3f, 8.3f, 2.0f, 1.4f, 7.2f, 3.1f, 7.9f};

    auto expectedAB = std::vector<T>{53.78f, 81.35f, 66.44f, 56.95f, 81.89f, 53.01f, 77.73f, 135.96f, 86.99f};
    auto expectedBA = std::vector<T>{112.46f, 54.02f, 40.2f, 142.39f, 62.15f, 74.79f, 119.84f, 49.05f, 48.05f};

    auto A = MatrixRowStatic<T, row1>(column1, arrayA);
    auto B = MatrixRowStatic<T, row1>(column1, arrayB);

    auto acceptableErrorAB =
        std::vector<T>{83.85f, 105.65f, 104.525f, 98.0f, 129.0f, 88.3f, 100.0f, 178.8f, 119.6f};
    auto acceptableErrorBA =
        std::vector<T>{160.0f, 78.2f, 45.425f, 221.2f, 102.0f, 110.425f, 203.8f, 78.4f, 70.45f};

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixRowStatic<T, row1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixRowStatic<T, row1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachVectorElement(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = std::vector<T>{2.1f, 1.5f, 0.7f, 6.0f, 2.6f, 3.4f, 3.4f, 5.1f, 4.2f, 4.3f, 6.3f, 1.7f, 6.7f, 7.3f,
        3.7f, 5.4f, 3.4f, 2.4f, 3.5f, 1.4f, 3.7f, 5.6f, 5.2f, 1.4f, 0.8f};
    auto arrayD = std::vector<T>{2.4f, 1.5f, 3.6f, 6.3f, 1.4f, 5.4f, 2.4f, 0.4f, 2.2f, 3.2f, 7.0f, 3.0f, 2.3f, 4.7f,
        4.4f, 3.6f, 7.5f, 0.8f, 3.0f, 0.3f, 3.0f, 3.1f, 2.4f, 0.1f, 0.8f};

    auto expectedCD = std::vector<T>{72.02f, 44.08f, 55.85f, 70.99f, 35.95f, 45.74f, 42.34f, 40.62f, 57.58f, 31.48f,
        81.05f, 65.23f, 69.77f, 94.0f, 49.71f, 55.41f, 44.14f, 54.89f, 69.86f, 49.01f, 35.46f, 23.94f, 38.39f, 50.01f,
        30.79f};
    auto expectedDC = std::vector<T>{47.44f, 61.91f, 20.81f, 38.08f, 14.7f, 90.24f, 73.39f, 39.01f, 65.9f, 42.78f,
        108.58f, 99.85f, 53.49f, 97.19f, 48.89f, 64.92f, 54.05f, 32.48f, 63.42f, 31.17f, 82.96f, 47.57f, 30.56f, 64.35f,
        47.04f};

    auto C = MatrixRowStatic<T, row2>(column2, arrayC);
    auto D = MatrixRowStatic<T, row2>(column2, arrayD);

    auto acceptableErrorCD = std::vector<T>{95.7f, 56.2f, 74.5f, 90.2f, 48.3f, 67.375f, 59.025f, 60.975f, 84.425f,
        49.325f, 121.0f, 94.5f, 108.5f, 129.6f, 72.7f, 70.675f, 52.25f, 73.45f, 93.7f, 69.2f, 49.5875f, 31.2125f,
        54.65f, 70.91875f, 45.3875f};
    auto acceptableErrorDC = std::vector<T>{64.3f, 78.8f, 30.925f, 55.7125f, 21.7f, 127.8f, 108.0f, 58.55f,
        97.66875f, 65.0f, 131.6f, 116.3f, 69.475f, 128.63125f, 69.375f, 88.4f, 68.7f, 44.8f, 90.1875f, 45.575f, 114.5f,
        66.15f, 41.5f, 92.9f, 70.35f};

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixRowStatic<T, row2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixRowStatic<T, row2>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachVectorElement(CD.Elements(), expectedCD, acceptableErrorCD);
    CheckCloseEachVectorElement(DC.Elements(), expectedDC, acceptableErrorDC);

    // non-square x non-square -> square
    constexpr size_t rowE = 3;
    constexpr size_t columnE = 4;
    constexpr size_t rowF = 4;
    constexpr size_t columnF = 3;
    constexpr size_t resultSizeEF = rowE * columnF;
    constexpr size_t resultSizeFE = rowF * columnE;

    auto arrayE = std::vector<T>{3.1f, 2.1f, 6.3f, 4.6f, 3.7f, 4.7f, 5.1f, 2.2f, 5.7f, 3.4f, 2.7f, 3.1f};
    auto arrayF = std::vector<T>{1.7f, 4.5f, 5.3f, 1.7f, 3.4f, 3.7f, 0.5f, 3.5f, 1.5f, 1.4f, 3.4f, 4.3f};

    auto expectedEF = std::vector<T>{58.78f, 36.47f, 67.34f, 42.01f, 31.38f, 52.51f, 43.05f, 27.42f, 48.74f};
    auto expectedFE = std::vector<T>{21.86f, 30.54f, 38.9f, 39.71f, 27.45f, 40.97f, 42.21f, 40.98f, 24.7f, 39.07f,
        47.51f, 40.88f, 19.61f, 29.63f, 29.91f, 28.56f};

    auto E = MatrixRowStatic<T, rowE>(columnE, arrayE);
    auto F = MatrixRowStatic<T, rowF>(columnF, arrayF);

    auto acceptableErrorEF = std::vector<T>{91.3f, 54.8f, 100.4f, 55.35f, 40.3f, 68.45f, 62.3f, 42.2f, 68.6f};
    auto acceptableErrorFE = std::vector<T>{29.8f, 44.9f, 51.25f, 60.1f, 36.3f, 61.5f, 65.45f, 61.8f, 34.8f, 61.5f,
        68.7f, 63.3f, 25.1f, 41.3f, 39.55f, 40.2f};

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<T>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<T>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixRowStatic<T, rowE>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixRowStatic<T, rowF>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachVectorElement(EF.Elements(), expectedEF, acceptableErrorEF);
    CheckCloseEachVectorElement(FE.Elements(), expectedFE, acceptableErrorFE);

    // other matrix x matrix
    constexpr size_t rowG = 4;
    constexpr size_t columnG = 1;
    constexpr size_t rowH = 4;
    constexpr size_t columnH = 4;
    constexpr size_t resultSizeHG = rowH * columnG;

    auto arrayG = std::vector<T>{3.5f, 6.2f, 4.0f, 5.4f};
    auto arrayH = std::vector<T>{
        1.2f, 3.4f, 5.1f, 3.4f, 3.4f, 1.4f, 3.6f, 3.6f, 5.1f, 0.7f, 7.2f, 2.5f, 2.0f, 3.4f, 3.4f, 1.4f};

    auto expectedHG = std::vector<T>{56.48f, 41.74f, 87.33f, 51.78f};

    auto G = MatrixRowStatic<T, rowG>(columnG, arrayG);
    auto H = MatrixRowStatic<T, rowH>(columnH, arrayH);

    auto acceptableErrorHG = std::vector<T>{87.1f, 54.8f, 120.2f, 69.6f};
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixRowStatic<T, rowH>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachVectorElement(HG.Elements(), expectedHG, acceptableErrorHG);

    // other matrix x matrix
    constexpr size_t rowI = 3;
    constexpr size_t columnI = 5;
    constexpr size_t rowJ = 2;
    constexpr size_t columnJ = 3;
    constexpr size_t resultSizeJI = rowJ * columnI;

    auto arrayI =
        std::vector<T>{4.1f, 3.6f, 1.7f, 2.4f, 2.7f, 3.7f, 0.8f, 3.0f, 5.4f, 3.5f, 3.2f, 1.7f, 1.7f, 4.6f, 3.3f};
    auto arrayJ = std::vector<T>{2.7f, 1.6f, 3.5f, 2.2f, 1.7f, 0.9f};

    auto expectedJI = std::vector<T>{26.56f, 16.01f, 22.22f, 13.11f, 21.84f, 12.74f, 23.54f, 14.17f, 26.3f, 15.81f};

    auto I = MatrixRowStatic<T, rowI>(columnI, arrayI);
    auto J = MatrixRowStatic<T, rowJ>(columnJ, arrayJ);

    auto acceptableErrorJI =
        std::vector<T>{36.6f, 23.85f, 29.7f, 19.05f, 28.15f, 18.3f, 29.2f, 19.25f, 36.0f, 24.75f};
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixRowStatic<T, rowJ>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachVectorElement(JI.Elements(), expectedJI, acceptableErrorJI);

    constexpr size_t rowK = 2;
    constexpr size_t columnK = 2;
    constexpr size_t elementSizeK = rowK * columnK;

    auto arrayK = std::vector<T>{0.0f, -1.0f, 1.0f, 0.0f};

    auto expectedKK = std::vector<T>{-1.0f, 0.0f, 0.0f, -1.0f};

    auto K = MatrixRowStatic<T, rowK>(columnK, arrayK);

    auto acceptableErrorKK = std::vector<T>{0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < elementSizeK; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto KK = K.Dot(K);
    auto typeCheckKK = std::is_same<decltype(KK), MatrixRowStatic<T, rowK>>::value;
    BOOST_CHECK(typeCheckKK);
    CheckCloseEachVectorElement(KK.Elements(), expectedKK, acceptableErrorKK);
}