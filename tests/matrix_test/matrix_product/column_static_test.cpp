#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>

#include "src/matrices/matrix_column_static.hpp"
#include "tests/test_matrix_product_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixColumnStatic;

BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_column_static_test, T, MatrixProductCheckType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeArrayA<T>();
    auto arrayB = MakeArrayB<T>();

    auto expectedAB = MakeExpectedAB<T>();
    auto expectedBA = MakeExpectedBA<T>();

    auto A = MatrixColumnStatic<T, column1>(row1, arrayA);
    auto B = MatrixColumnStatic<T, column1>(row1, arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorAB<T>();
    auto acceptableErrorBA = MakeAcceptableErrorBA<T>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixColumnStatic<T, column1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixColumnStatic<T, column1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachVectorElement(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = std::vector<T>{static_cast<T>(2.1), static_cast<T>(1.5), static_cast<T>(0.7), static_cast<T>(6),
        static_cast<T>(2.6), static_cast<T>(3.4), static_cast<T>(3.4), static_cast<T>(5.1), static_cast<T>(4.2),
        static_cast<T>(4.3), static_cast<T>(6.3), static_cast<T>(1.7), static_cast<T>(6.7), static_cast<T>(7.3),
        static_cast<T>(3.7), static_cast<T>(5.4), static_cast<T>(3.4), static_cast<T>(2.4), static_cast<T>(3.5),
        static_cast<T>(1.4), static_cast<T>(3.7), static_cast<T>(5.6), static_cast<T>(5.2), static_cast<T>(1.4),
        static_cast<T>(0.8)};
    auto arrayD = std::vector<T>{static_cast<T>(2.4), static_cast<T>(1.5), static_cast<T>(3.6), static_cast<T>(6.3),
        static_cast<T>(1.4), static_cast<T>(5.4), static_cast<T>(2.4), static_cast<T>(0.4), static_cast<T>(2.2),
        static_cast<T>(3.2), static_cast<T>(7), static_cast<T>(3), static_cast<T>(2.3), static_cast<T>(4.7),
        static_cast<T>(4.4), static_cast<T>(3.6), static_cast<T>(7.5), static_cast<T>(0.8), static_cast<T>(3),
        static_cast<T>(0.3), static_cast<T>(3), static_cast<T>(3.1), static_cast<T>(2.4), static_cast<T>(0.1),
        static_cast<T>(0.8)};

    auto expectedCD = std::vector<T>{static_cast<T>(72.02), static_cast<T>(44.08), static_cast<T>(55.85),
        static_cast<T>(70.99), static_cast<T>(35.95), static_cast<T>(45.74), static_cast<T>(42.34),
        static_cast<T>(40.62), static_cast<T>(57.58), static_cast<T>(31.48), static_cast<T>(81.05),
        static_cast<T>(65.23), static_cast<T>(69.77), static_cast<T>(94), static_cast<T>(49.71), static_cast<T>(55.41),
        static_cast<T>(44.14), static_cast<T>(54.89), static_cast<T>(69.86), static_cast<T>(49.01),
        static_cast<T>(35.46), static_cast<T>(23.94), static_cast<T>(38.39), static_cast<T>(50.01),
        static_cast<T>(30.79)};
    auto expectedDC = std::vector<T>{static_cast<T>(47.44), static_cast<T>(61.91), static_cast<T>(20.81),
        static_cast<T>(38.08), static_cast<T>(14.7), static_cast<T>(90.24), static_cast<T>(73.39),
        static_cast<T>(39.01), static_cast<T>(65.9), static_cast<T>(42.78), static_cast<T>(108.58),
        static_cast<T>(99.85), static_cast<T>(53.49), static_cast<T>(97.19), static_cast<T>(48.89),
        static_cast<T>(64.92), static_cast<T>(54.05), static_cast<T>(32.48), static_cast<T>(63.42),
        static_cast<T>(31.17), static_cast<T>(82.96), static_cast<T>(47.57), static_cast<T>(30.56),
        static_cast<T>(64.35), static_cast<T>(47.04)};

    auto C = MatrixColumnStatic<T, column2>(row2, arrayC);
    auto D = MatrixColumnStatic<T, column2>(row2, arrayD);

    auto acceptableErrorCD = std::vector<T>{static_cast<T>(95.7), static_cast<T>(56.2), static_cast<T>(74.5),
        static_cast<T>(90.2), static_cast<T>(48.3), static_cast<T>(67.375), static_cast<T>(59.025),
        static_cast<T>(60.975), static_cast<T>(84.425), static_cast<T>(49.325), static_cast<T>(121),
        static_cast<T>(94.5), static_cast<T>(108.5), static_cast<T>(129.6), static_cast<T>(72.7),
        static_cast<T>(70.675), static_cast<T>(52.25), static_cast<T>(73.45), static_cast<T>(93.7),
        static_cast<T>(69.2), static_cast<T>(49.5875), static_cast<T>(31.2125), static_cast<T>(54.65),
        static_cast<T>(70.91875), static_cast<T>(45.3875)};
    auto acceptableErrorDC = std::vector<T>{static_cast<T>(64.3), static_cast<T>(78.8), static_cast<T>(30.925),
        static_cast<T>(55.7125), static_cast<T>(21.7), static_cast<T>(127.8), static_cast<T>(108),
        static_cast<T>(58.55), static_cast<T>(97.66875), static_cast<T>(65), static_cast<T>(131.6),
        static_cast<T>(116.3), static_cast<T>(69.475), static_cast<T>(128.63125), static_cast<T>(69.375),
        static_cast<T>(88.4), static_cast<T>(68.7), static_cast<T>(44.8), static_cast<T>(90.1875),
        static_cast<T>(45.575), static_cast<T>(114.5), static_cast<T>(66.15), static_cast<T>(41.5),
        static_cast<T>(92.9), static_cast<T>(70.35)};

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixColumnStatic<T, column2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixColumnStatic<T, column2>>::value;
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

    auto arrayE = std::vector<T>{static_cast<T>(3.1), static_cast<T>(2.1), static_cast<T>(6.3), static_cast<T>(4.6),
        static_cast<T>(3.7), static_cast<T>(4.7), static_cast<T>(5.1), static_cast<T>(2.2), static_cast<T>(5.7),
        static_cast<T>(3.4), static_cast<T>(2.7), static_cast<T>(3.1)};
    auto arrayF = std::vector<T>{static_cast<T>(1.7), static_cast<T>(4.5), static_cast<T>(5.3), static_cast<T>(1.7),
        static_cast<T>(3.4), static_cast<T>(3.7), static_cast<T>(0.5), static_cast<T>(3.5), static_cast<T>(1.5),
        static_cast<T>(1.4), static_cast<T>(3.4), static_cast<T>(4.3)};

    auto expectedEF = std::vector<T>{static_cast<T>(58.78), static_cast<T>(36.47), static_cast<T>(67.34),
        static_cast<T>(42.01), static_cast<T>(31.38), static_cast<T>(52.51), static_cast<T>(43.05),
        static_cast<T>(27.42), static_cast<T>(48.74)};
    auto expectedFE =
        std::vector<T>{static_cast<T>(21.86), static_cast<T>(30.54), static_cast<T>(38.9), static_cast<T>(39.71),
            static_cast<T>(27.45), static_cast<T>(40.97), static_cast<T>(42.21), static_cast<T>(40.98),
            static_cast<T>(24.7), static_cast<T>(39.07), static_cast<T>(47.51), static_cast<T>(40.88),
            static_cast<T>(19.61), static_cast<T>(29.63), static_cast<T>(29.91), static_cast<T>(28.56)};

    auto E = MatrixColumnStatic<T, columnE>(rowE, arrayE);
    auto F = MatrixColumnStatic<T, columnF>(rowF, arrayF);

    auto acceptableErrorEF = std::vector<T>{static_cast<T>(91.3), static_cast<T>(54.8), static_cast<T>(100.4),
        static_cast<T>(55.35), static_cast<T>(40.3), static_cast<T>(68.45), static_cast<T>(62.3), static_cast<T>(42.2),
        static_cast<T>(68.6)};
    auto acceptableErrorFE = std::vector<T>{static_cast<T>(29.8), static_cast<T>(44.9), static_cast<T>(51.25),
        static_cast<T>(60.1), static_cast<T>(36.3), static_cast<T>(61.5), static_cast<T>(65.45), static_cast<T>(61.8),
        static_cast<T>(34.8), static_cast<T>(61.5), static_cast<T>(68.7), static_cast<T>(63.3), static_cast<T>(25.1),
        static_cast<T>(41.3), static_cast<T>(39.55), static_cast<T>(40.2)};

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<T>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<T>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixColumnStatic<T, columnF>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixColumnStatic<T, columnE>>::value;
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

    auto arrayG = std::vector<T>{static_cast<T>(3.5), static_cast<T>(6.2), static_cast<T>(4), static_cast<T>(5.4)};
    auto arrayH = std::vector<T>{static_cast<T>(1.2), static_cast<T>(3.4), static_cast<T>(5.1), static_cast<T>(3.4),
        static_cast<T>(3.4), static_cast<T>(1.4), static_cast<T>(3.6), static_cast<T>(3.6), static_cast<T>(5.1),
        static_cast<T>(0.7), static_cast<T>(7.2), static_cast<T>(2.5), static_cast<T>(2), static_cast<T>(3.4),
        static_cast<T>(3.4), static_cast<T>(1.4)};

    auto expectedHG =
        std::vector<T>{static_cast<T>(56.48), static_cast<T>(41.74), static_cast<T>(87.33), static_cast<T>(51.78)};

    auto G = MatrixColumnStatic<T, columnG>(rowG, arrayG);
    auto H = MatrixColumnStatic<T, columnH>(rowH, arrayH);

    auto acceptableErrorHG =
        std::vector<T>{static_cast<T>(87.1), static_cast<T>(54.8), static_cast<T>(120.2), static_cast<T>(69.6)};
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixColumnStatic<T, columnG>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachVectorElement(HG.Elements(), expectedHG, acceptableErrorHG);

    // other matrix x matrix
    constexpr size_t rowI = 3;
    constexpr size_t columnI = 5;
    constexpr size_t rowJ = 2;
    constexpr size_t columnJ = 3;
    constexpr size_t resultSizeJI = rowJ * columnI;

    auto arrayI = std::vector<T>{static_cast<T>(4.1), static_cast<T>(3.6), static_cast<T>(1.7), static_cast<T>(2.4),
        static_cast<T>(2.7), static_cast<T>(3.7), static_cast<T>(0.8), static_cast<T>(3), static_cast<T>(5.4),
        static_cast<T>(3.5), static_cast<T>(3.2), static_cast<T>(1.7), static_cast<T>(1.7), static_cast<T>(4.6),
        static_cast<T>(3.3)};
    auto arrayJ = std::vector<T>{static_cast<T>(2.7), static_cast<T>(1.6), static_cast<T>(3.5), static_cast<T>(2.2),
        static_cast<T>(1.7), static_cast<T>(0.9)};

    auto expectedJI = std::vector<T>{static_cast<T>(26.56), static_cast<T>(16.01), static_cast<T>(22.22),
        static_cast<T>(13.11), static_cast<T>(21.84), static_cast<T>(12.74), static_cast<T>(23.54),
        static_cast<T>(14.17), static_cast<T>(26.3), static_cast<T>(15.81)};

    auto I = MatrixColumnStatic<T, columnI>(rowI, arrayI);
    auto J = MatrixColumnStatic<T, columnJ>(rowJ, arrayJ);

    auto acceptableErrorJI = std::vector<T>{static_cast<T>(36.6), static_cast<T>(23.85), static_cast<T>(29.7),
        static_cast<T>(19.05), static_cast<T>(28.15), static_cast<T>(18.3), static_cast<T>(29.2), static_cast<T>(19.25),
        static_cast<T>(36), static_cast<T>(24.75)};
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixColumnStatic<T, columnI>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachVectorElement(JI.Elements(), expectedJI, acceptableErrorJI);

    constexpr size_t rowK = 2;
    constexpr size_t columnK = 2;
    constexpr size_t elementSizeK = rowK * columnK;

    auto arrayK = std::vector<T>{static_cast<T>(0.0), static_cast<T>(-1.0), static_cast<T>(1.0), static_cast<T>(0.0)};

    auto expectedKK = std::vector<T>{static_cast<T>(-1.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(-1.0)};

    auto K = MatrixColumnStatic<T, columnK>(rowK, arrayK);

    auto acceptableErrorKK = std::vector<T>{static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0)};
    for (size_t i = 0; i < elementSizeK; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto KK = K.Dot(K);
    auto typeCheckKK = std::is_same<decltype(KK), MatrixColumnStatic<T, columnK>>::value;
    BOOST_CHECK(typeCheckKK);
    CheckCloseEachVectorElement(KK.Elements(), expectedKK, acceptableErrorKK);

    // throw exception test
    auto mArray = std::vector<T>{static_cast<T>(2.4), static_cast<T>(5.4), static_cast<T>(7.1), static_cast<T>(3.1), static_cast<T>(2.4), static_cast<T>(4.8)};
    auto nArray = std::vector<T>{static_cast<T>(2.3), static_cast<T>(2.9), static_cast<T>(3.5), static_cast<T>(5.7), static_cast<T>(3.8), static_cast<T>(7.3), static_cast<T>(4.2), static_cast<T>(1.5), static_cast<T>(2.7)};
    auto M = MatrixColumnStatic<T, 2>(3, mArray);
    auto N = MatrixColumnStatic<T, 3>(3, nArray);

    BOOST_CHECK_THROW(M.Dot(N), std::domain_error);
    BOOST_CHECK_NO_THROW(N.Dot(M));
}