#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "src/matrices/matrix_column_static.hpp"
#include "src/matrices/matrix_dynamic.hpp"
#include "src/matrices/matrix_row_static.hpp"
#include "src/matrices/matrix_static.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"
#include "tests/test_matrix_product_array_defines.hpp"
#include "tests/test_matrix_product_vector_defines.hpp"

using mknnlib::matrix::MatrixColumnStatic;
using mknnlib::matrix::MatrixDynamic;
using mknnlib::matrix::MatrixRowStatic;
using mknnlib::matrix::MatrixStatic;

BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_static_and_matrix_dynamic_test, T, MatrixProductCheckType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeArrayA<T, elementSize1>();
    auto arrayB = MakeVectorB<T>();

    auto expectedAB = MakeExpectedVectorAB<T>();
    auto expectedBA = MakeExpectedVectorBA<T>();

    auto A = MatrixStatic<T, row1, column1>(arrayA);
    auto B = MatrixDynamic<T>(row1, column1, arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorVectorAB<T>();
    auto acceptableErrorBA = MakeAcceptableErrorVectorBA<T>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixRowStatic<T, row1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixColumnStatic<T, column1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement<T>(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachVectorElement<T>(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = MakeArrayC<T, elementSize2>();
    auto arrayD = MakeVectorD<T>();

    auto expectedCD = MakeExpectedVectorCD<T>();
    auto expectedDC = MakeExpectedVectorDC<T>();

    auto C = MatrixStatic<T, row2, column2>(arrayC);
    auto D = MatrixDynamic<T>(row2, column2, arrayD);

    auto acceptableErrorCD = MakeAcceptableErrorVectorCD<T>();
    auto acceptableErrorDC = MakeAcceptableErrorVectorDC<T>();

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixRowStatic<T, row2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixColumnStatic<T, column2>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachVectorElement(CD.Elements(), expectedCD, acceptableErrorCD);
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

    auto arrayE = MakeArrayE<T, elementSizeE>();
    auto arrayF = MakeVectorF<T>();

    auto expectedEF = MakeExpectedVectorEF<T>();
    auto expectedFE = MakeExpectedVectorFE<T>();

    auto E = MatrixStatic<T, rowE, columnE>(arrayE);
    auto F = MatrixDynamic<T>(rowF, columnF, arrayF);

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

    auto typeCheckEF = std::is_same<decltype(EF), MatrixRowStatic<T, rowE>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixColumnStatic<T, columnE>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachVectorElement(EF.Elements(), expectedEF, acceptableErrorEF);
    CheckCloseEachVectorElement(FE.Elements(), expectedFE, acceptableErrorFE);

    // other matrix x matrix
    constexpr size_t rowG = 4;
    constexpr size_t columnG = 1;
    constexpr size_t elementSizeG = rowG * columnG;
    constexpr size_t rowH = 4;
    constexpr size_t columnH = 4;
    // constexpr size_t elementSizeH = rowH * columnH;
    constexpr size_t resultSizeHG = rowH * columnG;

    auto arrayG = MakeArrayG<T, elementSizeG>();
    auto arrayH = MakeVectorH<T>();

    auto expectedHG = MakeExpectedVectorHG<T>();

    auto G = MatrixStatic<T, rowG, columnG>(arrayG);
    auto H = MatrixDynamic<T>(rowH, columnH, arrayH);

    auto acceptableErrorHG = MakeAcceptableErrorVectorHG<T>();
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixColumnStatic<T, columnG>>::value;
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

    auto arrayI = MakeArrayI<T, elementSizeI>();
    auto arrayJ = MakeVectorJ<T>();

    auto expectedJI = MakeExpectedVectorJI<T>();

    auto I = MatrixStatic<T, rowI, columnI>(arrayI);
    auto J = MatrixDynamic<T>(rowJ, columnJ, arrayJ);

    auto acceptableErrorJI = MakeAcceptableErrorVectorJI<T>();
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixColumnStatic<T, columnI>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachVectorElement(JI.Elements(), expectedJI, acceptableErrorJI);

    auto K = MatrixDynamic<T>(rowI, columnI, std::vector<T>(arrayI.begin(), arrayI.end()));
    auto arrayL = std::array<T, elementSizeJ>();
    std::copy_n(arrayJ.begin(), elementSizeJ, arrayL.begin());
    auto L = MatrixStatic<T, rowJ, columnJ>(arrayL);

    auto LK = L.Dot(K);

    auto typeCheckKL = std::is_same<decltype(LK), MatrixRowStatic<T, rowJ>>::value;
    BOOST_CHECK(typeCheckKL);
    CheckCloseEachVectorElement(LK.Elements(), expectedJI, acceptableErrorJI);

    // throw exception test
    auto mArray = MakeArrayM<T, 3 * 2>();
    auto nArray = MakeVectorN<T>();
    auto M = MatrixStatic<T, 3, 2>(mArray);
    auto N = MatrixDynamic<T>(3, 3, nArray);

    BOOST_CHECK_THROW(M.Dot(N), std::domain_error);
    BOOST_CHECK_NO_THROW(N.Dot(M));

    auto O = MatrixDynamic<T>(3, 2, ArrayToVector<T>(mArray));
    auto P = MatrixStatic<T, 3, 3>(VectorToArray<T, 9>(nArray));

    BOOST_CHECK_THROW(O.Dot(P), std::domain_error);
    BOOST_CHECK_NO_THROW(P.Dot(O));
}