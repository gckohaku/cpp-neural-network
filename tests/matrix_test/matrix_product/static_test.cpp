#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixStatic;

namespace matrix_test_matrix_product {
BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_product_matrix_add_static_test, T, CheckMatrixElementType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeArrayA<T, elementSize1>();
    auto arrayB = MakeArrayB<T, elementSize1>();

    auto expectedAB = MakeExpectedArrayAB<T, elementSize1>();
    auto expectedBA = MakeExpectedArrayBA<T, elementSize1>();

    auto A = MatrixStatic<T, row1, column1>(arrayA);
    auto B = MatrixStatic<T, row1, column1>(arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorArrayAB<T, elementSize1>();
    auto acceptableErrorBA = MakeAcceptableErrorArrayBA<T, elementSize1>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixStatic<T, row1, column1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixStatic<T, row1, column1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachArrayElement<T, elementSize1>(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachArrayElement<T, elementSize1>(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = MakeArrayC<T, elementSize2>();
    auto arrayD = MakeArrayD<T, elementSize2>();

    auto expectedCD = MakeExpectedArrayCD<T, elementSize2>();
    auto expectedDC = MakeExpectedArrayDC<T, elementSize2>();

    auto C = MatrixStatic<T, row2, column2>(arrayC);
    auto D = MatrixStatic<T, row2, column2>(arrayD);

    auto acceptableErrorCD = MakeAcceptableErrorArrayCD<T, elementSize2>();
    auto acceptableErrorDC = MakeAcceptableErrorArrayDC<T, elementSize2>();

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixStatic<T, row2, column2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixStatic<T, row2, column2>>::value;
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

    auto arrayE = MakeArrayE<T, elementSizeE>();
    auto arrayF = MakeArrayF<T, elementSizeF>();

    auto expectedEF = MakeExpectedArrayEF<T, resultSizeEF>();
    auto expectedFE = MakeExpectedArrayFE<T, resultSizeFE>();

    auto E = MatrixStatic<T, rowE, columnE>(arrayE);
    auto F = MatrixStatic<T, rowF, columnF>(arrayF);

    auto acceptableErrorEF = MakeAcceptableErrorArrayEF<T, resultSizeEF>();
    auto acceptableErrorFE = MakeAcceptableErrorArrayFE<T, resultSizeFE>();

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<T>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<T>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixStatic<T, rowE, columnF>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixStatic<T, rowF, columnE>>::value;
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

    auto arrayG = MakeArrayG<T, elementSizeG>();
    auto arrayH = MakeArrayH<T, elementSizeH>();

    auto expectedHG = MakeExpectedArrayHG<T, resultSizeHG>();

    auto G = MatrixStatic<T, rowG, columnG>(arrayG);
    auto H = MatrixStatic<T, rowH, columnH>(arrayH);

    auto acceptableErrorHG = MakeAcceptableErrorArrayHG<T, resultSizeHG>();
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixStatic<T, rowH, columnG>>::value;
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

    auto arrayI = MakeArrayI<T, elementSizeI>();
    auto arrayJ = MakeArrayJ<T, elementSizeJ>();

    auto expectedJI = MakeExpectedArrayJI<T, resultSizeJI>();

    auto I = MatrixStatic<T, rowI, columnI>(arrayI);
    auto J = MatrixStatic<T, rowJ, columnJ>(arrayJ);

    auto acceptableErrorJI = MakeAcceptableErrorArrayJI<T, resultSizeJI>();
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixStatic<T, rowJ, columnI>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachArrayElement(JI.Elements(), expectedJI, acceptableErrorJI);

    constexpr size_t rowK = 2;
    constexpr size_t columnK = 2;
    constexpr size_t elementSizeK = rowK * columnK;

    auto arrayK = MakeArrayK<T, elementSizeK>();

    auto expectedKK = MakeExpectedArrayKK<T, elementSizeK>();

    auto K = MatrixStatic<T, rowK, columnK>(arrayK);

    auto acceptableErrorKK = MakeAcceptableErrorArrayKK<T, elementSizeK>();
    for (size_t i = 0; i < elementSizeK; i++) {
        acceptableErrorKK[i] *= std::numeric_limits<T>::epsilon();
    }

    auto KK = K.Dot(K);
    auto typeCheckKK = std::is_same<decltype(KK), MatrixStatic<T, rowK, columnK>>::value;
    BOOST_CHECK(typeCheckKK);
    CheckCloseEachArrayElement(KK.Elements(), expectedKK, acceptableErrorKK);
}
}  // namespace matrix_test_matrix_product