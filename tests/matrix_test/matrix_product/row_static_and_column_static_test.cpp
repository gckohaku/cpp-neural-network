#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>

#include "src/matrices/matrix_column_static.hpp"
#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_product_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixColumnStatic;
using mknnlib::matrix::MatrixDynamic;
using mknnlib::matrix::MatrixRowStatic;
using mknnlib::matrix::MatrixStatic;

BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_row_static_and_matrix_column_static_test, T, MatrixProductCheckType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeVectorA<T>();
    auto arrayB = MakeVectorB<T>();

    auto expectedAB = MakeExpectedVectorAB<T>();
    auto expectedBA = MakeExpectedVectorBA<T>();

    auto A = MatrixRowStatic<T, row1>(column1, arrayA);
    auto B = MatrixColumnStatic<T, column1>(row1, arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorVectorAB<T>();
    auto acceptableErrorBA = MakeAcceptableErrorVectorBA<T>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixStatic<T, row1, column1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixDynamic<T>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement<T>(ArrayToVector<T>(AB.Elements()), expectedAB, acceptableErrorAB);
    CheckCloseEachVectorElement<T>(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = MakeVectorC<T>();
    auto arrayD = MakeVectorD<T>();

    auto expectedCD = MakeExpectedVectorCD<T>();
    auto expectedDC = MakeExpectedVectorDC<T>();

    auto C = MatrixRowStatic<T, row2>(column2, arrayC);
    auto D = MatrixColumnStatic<T, column2>(row2, arrayD);

    auto acceptableErrorCD = MakeAcceptableErrorVectorCD<T>();
    auto acceptableErrorDC = MakeAcceptableErrorVectorDC<T>();

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixStatic<T, row2, column2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixDynamic<T>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachVectorElement(ArrayToVector<T>(CD.Elements()), expectedCD, acceptableErrorCD);
    CheckCloseEachVectorElement(DC.Elements(), expectedDC, acceptableErrorDC);

    // non-square x non-square -> square
    constexpr size_t rowE = 3;
    constexpr size_t columnE = 4;
    constexpr size_t elementSizeE = rowE * columnE;
    constexpr size_t rowF = 4;
    constexpr size_t columnF = 3;
    // constexpr size_t elementSizeF = rowF * columnF;
    constexpr size_t resultSizeEF = rowE * columnF;
    constexpr size_t resultSizeFE = rowF * columnE;

    auto arrayE = MakeVectorE<T>();
    auto arrayF = MakeVectorF<T>();

    auto expectedEF = MakeExpectedVectorEF<T>();
    auto expectedFE = MakeExpectedVectorFE<T>();

    auto E = MatrixRowStatic<T, rowE>(columnE, arrayE);
    auto F = MatrixColumnStatic<T, columnF>(rowF, arrayF);

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

    auto typeCheckEF = std::is_same<decltype(EF), MatrixStatic<T, rowE, columnF>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixDynamic<T>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachVectorElement(ArrayToVector<T>(EF.Elements()), expectedEF, acceptableErrorEF);
    CheckCloseEachVectorElement(FE.Elements(), expectedFE, acceptableErrorFE);

    // other matrix x matrix
    constexpr size_t rowG = 4;
    constexpr size_t columnG = 1;
    constexpr size_t elementSizeG = rowG * columnG;
    constexpr size_t rowH = 4;
    constexpr size_t columnH = 4;
    // constexpr size_t elementSizeH = rowH * columnH;
    constexpr size_t resultSizeHG = rowH * columnG;

    auto arrayG = MakeVectorG<T>();
    auto arrayH = MakeVectorH<T>();

    auto expectedHG = MakeExpectedVectorHG<T>();

    auto G = MatrixRowStatic<T, rowG>(columnG, arrayG);
    auto H = MatrixColumnStatic<T, columnH>(rowH, arrayH);

    auto acceptableErrorHG = MakeAcceptableErrorVectorHG<T>();
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixDynamic<T>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachVectorElement(HG.Elements(), expectedHG, acceptableErrorHG);

    constexpr size_t rowI = 3;
    constexpr size_t columnI = 5;
    constexpr size_t elementSizeI = rowI * columnI;
    constexpr size_t rowJ = 2;
    constexpr size_t columnJ = 3;
    constexpr size_t elementSizeJ = rowJ * columnJ;
    // constexpr size_t elementSizeJ = rowJ * columnJ;
    constexpr size_t resultSizeJI = rowJ * columnI;

    auto arrayI = MakeVectorI<T>();
    auto arrayJ = MakeVectorJ<T>();

    auto expectedJI = MakeExpectedVectorJI<T>();

    auto I = MatrixRowStatic<T, rowI>(columnI, arrayI);
    auto J = MatrixColumnStatic<T, columnJ>(rowJ, arrayJ);

    auto acceptableErrorJI = MakeAcceptableErrorVectorJI<T>();
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixDynamic<T>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachVectorElement(JI.Elements(), expectedJI, acceptableErrorJI);

    auto K = MatrixColumnStatic<T, columnI>(rowI, std::vector<T>(arrayI.begin(), arrayI.end()));
    auto arrayL = std::array<T, elementSizeJ>();
    std::copy_n(arrayJ.begin(), elementSizeJ, arrayL.begin());
    auto L = MatrixRowStatic<T, rowJ>(columnJ, ArrayToVector(arrayL));

    auto LK = L.Dot(K);

    auto typeCheckKL = std::is_same<decltype(LK), MatrixStatic<T, rowJ, columnI>>::value;
    BOOST_CHECK(typeCheckKL);
    CheckCloseEachVectorElement(ArrayToVector<T>(LK.Elements()), expectedJI, acceptableErrorJI);

    // throw exception test
    auto mArray = MakeVectorM<T>();
    auto nArray = MakeVectorN<T>();
    auto M = MatrixRowStatic<T, 3>(2, mArray);
    auto N = MatrixColumnStatic<T, 3>(3, nArray);

    BOOST_CHECK_THROW(M.Dot(N), std::domain_error);
    BOOST_CHECK_NO_THROW(N.Dot(M));
}