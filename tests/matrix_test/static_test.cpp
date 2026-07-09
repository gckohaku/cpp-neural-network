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
    // square x square
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

    // non-square x non-square -> square
    constexpr size_t rowE = 3;
    constexpr size_t columnE = 4;
    constexpr size_t elementSizeE = rowE * columnE;
    constexpr size_t rowF = 4;
    constexpr size_t columnF = 3;
    constexpr size_t elementSizeF = rowF * columnF;
    constexpr size_t resultSizeEF = rowE * columnF;
    constexpr size_t resultSizeFE = rowF * columnE;

    auto arrayE =
        std::array<float, elementSizeE>{3.1f, 2.1f, 6.3f, 4.6f, 3.7f, 4.7f, 5.1f, 2.2f, 5.7f, 3.4f, 2.7f, 3.1f};
    auto arrayF =
        std::array<float, elementSizeF>{1.7f, 4.5f, 5.3f, 1.7f, 3.4f, 3.7f, 0.5f, 3.5f, 1.5f, 1.4f, 3.4f, 4.3f};

    auto expectedEF =
        std::array<float, resultSizeEF>{58.78f, 36.47f, 67.34f, 42.01f, 31.38f, 52.51f, 43.05f, 27.42f, 48.74f};
    auto expectedFE = std::array<float, resultSizeFE>{21.86f, 30.54f, 38.9f, 39.71f, 27.45f, 40.97f, 42.21f, 40.98f,
        24.7f, 39.07f, 47.51f, 40.88f, 19.61f, 29.63f, 29.91f, 28.56f};

    auto E = MatrixStatic<float, rowE, columnE>(arrayE);
    auto F = MatrixStatic<float, rowF, columnF>(arrayF);

    auto acceptableErrorEF =
        std::array<float, resultSizeEF>{91.3f, 54.8f, 100.4f, 55.35f, 40.3f, 68.45f, 62.3f, 42.2f, 68.6f};
    auto acceptableErrorFE = std::array<float, resultSizeFE>{29.8f, 44.9f, 51.25f, 60.1f, 36.3f, 61.5f, 65.45f, 61.8f,
        34.8f, 61.5f, 68.7f, 63.3f, 25.1f, 41.3f, 39.55f, 40.2f};

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<float>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<float>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixStatic<float, rowE, columnF>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixStatic<float, rowF, columnE>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachArrayElement(EF.Elements(), expectedEF, acceptableErrorEF);
    CheckCloseEachArrayElement(FE.Elements(), expectedFE, acceptableErrorFE);

    // other matrix x matrix
    constexpr size_t rowG = 4;
    constexpr size_t columnG = 1;
    constexpr size_t elementSizeG = rowG * columnG;
    constexpr size_t rowH = 4;
    constexpr size_t columnH = 4;
    constexpr size_t elementSizeH = rowH * columnH;
    constexpr size_t resultSizeHG = rowH * columnG;

    auto arrayG = std::array<float, elementSizeG>{3.5f, 6.2f, 4.0f, 5.4f};
    auto arrayH = std::array<float, elementSizeH>{
        1.2f, 3.4f, 5.1f, 3.4f, 3.4f, 1.4f, 3.6f, 3.6f, 5.1f, 0.7f, 7.2f, 2.5f, 2.0f, 3.4f, 3.4f, 1.4f};

    auto expectedHG = std::array<float, resultSizeHG>{56.48f, 41.74f, 87.33f, 51.78f};

    auto G = MatrixStatic<float, rowG, columnG>(arrayG);
    auto H = MatrixStatic<float, rowH, columnH>(arrayH);

    auto acceptableErrorHG = std::array<float, resultSizeHG>{87.1f, 54.8f, 120.2f, 69.6f};
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<float>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixStatic<float, rowH, columnG>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachArrayElement(HG.Elements(), expectedHG, acceptableErrorHG);

    // other matrix x matrix
    constexpr size_t rowI = 3;
    constexpr size_t columnI = 5;
    constexpr size_t elementSizeI = rowI * columnI;
    constexpr size_t rowJ = 2;
    constexpr size_t columnJ = 3;
    constexpr size_t elementSizeJ = rowJ * columnJ;
    constexpr size_t resultSizeJI = rowJ * columnI;

    auto arrayI = std::array<float, elementSizeI>{
        4.1f, 3.6f, 1.7f, 2.4f, 2.7f, 3.7f, 0.8f, 3.0f, 5.4f, 3.5f, 3.2f, 1.7f, 1.7f, 4.6f, 3.3f};
    auto arrayJ = std::array<float, elementSizeJ>{2.7f, 1.6f, 3.5f, 2.2f, 1.7f, 0.9f};

    auto expectedJI =
        std::array<float, resultSizeJI>{26.56f, 16.01f, 22.22f, 13.11f, 21.84f, 12.74f, 23.54f, 14.17f, 26.3f, 15.81f};

    auto I = MatrixStatic<float, rowI, columnI>(arrayI);
    auto J = MatrixStatic<float, rowJ, columnJ>(arrayJ);

    auto acceptableErrorJI =
        std::array<float, resultSizeJI>{36.6f, 23.85f, 29.7f, 19.05f, 28.15f, 18.3f, 29.2f, 19.25f, 36.0f, 24.75f};
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<float>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixStatic<float, rowJ, columnI>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachArrayElement(JI.Elements(), expectedJI, acceptableErrorJI);

    constexpr size_t rowK = 2;
    constexpr size_t columnK = 2;
    constexpr size_t elementSizeK = rowK * columnK;

    auto arrayK = std::array<float, elementSizeK>{0.0f, -1.0f, 1.0f, 0.0f};

    auto expectedKK = std::array<float, elementSizeK>{-1.0f, 0.0f, 0.0f, -1.0f};

    auto K = MatrixStatic<float, rowK, columnK>(arrayK);

    auto acceptableErrorKK = std::array<float, elementSizeK>{0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < elementSizeK; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<float>::epsilon();
    }

    auto KK = K.Dot(K);
    auto typeCheckKK = std::is_same<decltype(KK), MatrixStatic<float, rowK, columnK>>::value;
    BOOST_CHECK(typeCheckKK);
    CheckCloseEachArrayElement(KK.Elements(), expectedKK, acceptableErrorKK);
}