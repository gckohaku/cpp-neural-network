#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>

#include "src/matrices/matrix_template_base.hpp"
#include "tests/test_matrix_product_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixColumnStatic;

namespace matrix_test_matrix_product {
BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_product_matrix_column_static_test, Cases, CheckTypeCases) {
    using T = typename Cases::first;
    using Backend = typename Cases::second;

    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeVectorA<T>();
    auto arrayB = MakeVectorB<T>();

    auto expectedAB = MakeExpectedVectorAB<T>();
    auto expectedBA = MakeExpectedVectorBA<T>();

    auto A = MatrixColumnStatic<T, column1, Backend>(row1, arrayA);
    auto B = MatrixColumnStatic<T, column1, Backend>(row1, arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorVectorAB<T>();
    auto acceptableErrorBA = MakeAcceptableErrorVectorBA<T>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixColumnStatic<T, column1, Backend>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixColumnStatic<T, column1, Backend>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement(AB.Elements().GetVector(), expectedAB, acceptableErrorAB);
    CheckCloseEachVectorElement(BA.Elements().GetVector(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = MakeVectorC<T>();
    auto arrayD = MakeVectorD<T>();

    auto expectedCD = MakeExpectedVectorCD<T>();
    auto expectedDC = MakeExpectedVectorDC<T>();

    auto C = MatrixColumnStatic<T, column2, Backend>(row2, arrayC);
    auto D = MatrixColumnStatic<T, column2, Backend>(row2, arrayD);

    auto acceptableErrorCD = MakeAcceptableErrorVectorCD<T>();
    auto acceptableErrorDC = MakeAcceptableErrorVectorDC<T>();

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixColumnStatic<T, column2, Backend>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixColumnStatic<T, column2, Backend>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachVectorElement(CD.Elements().GetVector(), expectedCD, acceptableErrorCD);
    CheckCloseEachVectorElement(DC.Elements().GetVector(), expectedDC, acceptableErrorDC);

    // non-square x non-square -> square
    constexpr size_t rowE = 3;
    constexpr size_t columnE = 4;
    constexpr size_t rowF = 4;
    constexpr size_t columnF = 3;
    constexpr size_t resultSizeEF = rowE * columnF;
    constexpr size_t resultSizeFE = rowF * columnE;

    auto arrayE = MakeVectorE<T>();
    auto arrayF = MakeVectorF<T>();

    auto expectedEF = MakeExpectedVectorEF<T>();
    auto expectedFE = MakeExpectedVectorFE<T>();

    auto E = MatrixColumnStatic<T, columnE, Backend>(rowE, arrayE);
    auto F = MatrixColumnStatic<T, columnF, Backend>(rowF, arrayF);

    auto acceptableErrorEF = MakeAcceptableErrorVectorEF<T>();
    auto acceptableErrorFE = MakeAcceptableErrorVectorFE<T>();

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<T>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<T>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixColumnStatic<T, columnF, Backend>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixColumnStatic<T, columnE, Backend>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachVectorElement(EF.Elements().GetVector(), expectedEF, acceptableErrorEF);
    CheckCloseEachVectorElement(FE.Elements().GetVector(), expectedFE, acceptableErrorFE);

    // other matrix x matrix
    constexpr size_t rowG = 4;
    constexpr size_t columnG = 1;
    constexpr size_t rowH = 4;
    constexpr size_t columnH = 4;
    constexpr size_t resultSizeHG = rowH * columnG;

    auto arrayG = MakeVectorG<T>();
    auto arrayH = MakeVectorH<T>();

    auto expectedHG = MakeExpectedVectorHG<T>();

    auto G = MatrixColumnStatic<T, columnG, Backend>(rowG, arrayG);
    auto H = MatrixColumnStatic<T, columnH, Backend>(rowH, arrayH);

    auto acceptableErrorHG = MakeAcceptableErrorVectorHG<T>();
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixColumnStatic<T, columnG, Backend>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachVectorElement(HG.Elements().GetVector(), expectedHG, acceptableErrorHG);

    // other matrix x matrix
    constexpr size_t rowI = 3;
    constexpr size_t columnI = 5;
    constexpr size_t rowJ = 2;
    constexpr size_t columnJ = 3;
    constexpr size_t resultSizeJI = rowJ * columnI;

    auto arrayI = MakeVectorI<T>();
    auto arrayJ = MakeVectorJ<T>();

    auto expectedJI = MakeExpectedVectorJI<T>();

    auto I = MatrixColumnStatic<T, columnI, Backend>(rowI, arrayI);
    auto J = MatrixColumnStatic<T, columnJ, Backend>(rowJ, arrayJ);

    auto acceptableErrorJI = MakeAcceptableErrorVectorJI<T>();

    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixColumnStatic<T, columnI, Backend>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachVectorElement(JI.Elements().GetVector(), expectedJI, acceptableErrorJI);

    constexpr size_t rowK = 2;
    constexpr size_t columnK = 2;
    constexpr size_t elementSizeK = rowK * columnK;

    auto arrayK = MakeVectorK<T>();

    auto expectedKK = MakeExpectedVectorKK<T>();

    auto K = MatrixColumnStatic<T, columnK, Backend>(rowK, arrayK);

    auto acceptableErrorKK = MakeAcceptableErrorVectorKK<T>();
    for (size_t i = 0; i < elementSizeK; i++) {
        acceptableErrorKK[i] *= std::numeric_limits<T>::epsilon();
    }

    auto KK = K.Dot(K);
    auto typeCheckKK = std::is_same<decltype(KK), MatrixColumnStatic<T, columnK, Backend>>::value;
    BOOST_CHECK(typeCheckKK);
    CheckCloseEachVectorElement(KK.Elements().GetVector(), expectedKK, acceptableErrorKK);

    // throw exception test
    auto mArray = MakeVectorM<T>();
    auto nArray = MakeVectorN<T>();
    auto M = MatrixColumnStatic<T, 2, Backend>(3, mArray);
    auto N = MatrixColumnStatic<T, 3, Backend>(3, nArray);

    BOOST_CHECK_THROW(M.Dot(N), std::domain_error);
    BOOST_CHECK_NO_THROW(N.Dot(M));
}
}  // namespace matrix_test_matrix_product