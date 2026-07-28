#include <algorithm>
#include <array>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "src/matrices/matrix_row_static.hpp"
#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_product_array_defines.hpp"
#include "tests/test_matrix_product_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixRowStatic;
using mknnlib::matrix::MatrixStatic;

namespace matrix_test_matrix_product {
BOOST_AUTO_TEST_CASE_TEMPLATE(matrix_product_matrix_static_and_matrix_row_static_test, T, MatrixProductCheckType) {
    // square x square
    // 3x3
    constexpr size_t row1 = 3;
    constexpr size_t column1 = 3;
    constexpr size_t elementSize1 = row1 * column1;

    auto arrayA = MakeArrayA<T, elementSize1>();
    auto arrayB = MakeVectorB<T>();

    auto expectedAB = MakeExpectedVectorAB<T>();
    auto expectedBA = MakeExpectedArrayBA<T, elementSize1>();

    auto A = MatrixStatic<T, row1, column1>(arrayA);
    auto B = MatrixRowStatic<T, row1>(column1, arrayB);

    auto acceptableErrorAB = MakeAcceptableErrorVectorAB<T>();
    auto acceptableErrorBA = MakeAcceptableErrorArrayBA<T, elementSize1>();

    for (size_t i = 0; i < elementSize1; i++) {
        acceptableErrorAB[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorBA[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AB = A.Dot(B);
    auto BA = B.Dot(A);

    auto typeCheckAB = std::is_same<decltype(AB), MatrixRowStatic<T, row1>>::value;
    auto typeCheckBA = std::is_same<decltype(BA), MatrixStatic<T, row1, column1>>::value;
    BOOST_CHECK(typeCheckAB);
    BOOST_CHECK(typeCheckBA);
    CheckCloseEachVectorElement<T>(AB.Elements(), expectedAB, acceptableErrorAB);
    CheckCloseEachArrayElement<T>(BA.Elements(), expectedBA, acceptableErrorBA);

    // 5x5
    constexpr size_t row2 = 5;
    constexpr size_t column2 = 5;
    constexpr size_t elementSize2 = row2 * column2;

    auto arrayC = MakeArrayC<T, elementSize2>();
    auto arrayD = MakeVectorD<T>();

    auto expectedCD = MakeExpectedVectorCD<T>();
    auto expectedDC = MakeExpectedArrayDC<T, elementSize2>();

    auto C = MatrixStatic<T, row2, column2>(arrayC);
    auto D = MatrixRowStatic<T, row2>(column2, arrayD);

    auto acceptableErrorCD = MakeAcceptableErrorVectorCD<T>();
    auto acceptableErrorDC = MakeAcceptableErrorArrayDC<T, elementSize2>();

    for (size_t i = 0; i < elementSize2; i++) {
        acceptableErrorCD[i] *= std::numeric_limits<T>::epsilon();
        acceptableErrorDC[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CD = C.Dot(D);
    auto DC = D.Dot(C);

    auto typeCheckCD = std::is_same<decltype(CD), MatrixRowStatic<T, row2>>::value;
    auto typeCheckDC = std::is_same<decltype(DC), MatrixStatic<T, row2, column2>>::value;
    BOOST_CHECK(typeCheckCD);
    BOOST_CHECK(typeCheckDC);
    CheckCloseEachVectorElement(CD.Elements(), expectedCD, acceptableErrorCD);
    CheckCloseEachArrayElement(DC.Elements(), expectedDC, acceptableErrorDC);

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
    auto expectedFE = MakeExpectedArrayFE<T, resultSizeFE>();

    auto E = MatrixStatic<T, rowE, columnE>(arrayE);
    auto F = MatrixRowStatic<T, rowF>(columnF, arrayF);

    auto acceptableErrorEF = MakeAcceptableErrorVectorEF<T>();
    auto acceptableErrorFE = MakeAcceptableErrorArrayFE<T, resultSizeFE>();

    for (size_t i = 0; i < resultSizeEF; i++) {
        acceptableErrorEF[i] *= std::numeric_limits<T>::epsilon();
    }
    for (size_t i = 0; i < resultSizeFE; i++) {
        acceptableErrorFE[i] *= std::numeric_limits<T>::epsilon();
    }

    auto EF = E.Dot(F);
    auto FE = F.Dot(E);

    auto typeCheckEF = std::is_same<decltype(EF), MatrixRowStatic<T, rowE>>::value;
    auto typeCheckFE = std::is_same<decltype(FE), MatrixStatic<T, rowF, columnE>>::value;
    BOOST_CHECK(typeCheckEF);
    BOOST_CHECK(typeCheckFE);
    CheckCloseEachVectorElement(EF.Elements(), expectedEF, acceptableErrorEF);
    CheckCloseEachArrayElement(FE.Elements(), expectedFE, acceptableErrorFE);

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

    auto expectedHG = MakeExpectedArrayHG<T, resultSizeHG>();

    auto G = MatrixStatic<T, rowG, columnG>(arrayG);
    auto H = MatrixRowStatic<T, rowH>(columnH, arrayH);

    auto acceptableErrorHG = MakeAcceptableErrorArrayHG<T, resultSizeHG>();
    for (size_t i = 0; i < resultSizeHG; i++) {
        acceptableErrorHG[i] *= std::numeric_limits<T>::epsilon();
    }

    auto HG = H.Dot(G);

    auto typeCheckHG = std::is_same<decltype(HG), MatrixStatic<T, rowH, columnG>>::value;
    BOOST_CHECK(typeCheckHG);
    CheckCloseEachArrayElement(HG.Elements(), expectedHG, acceptableErrorHG);

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

    auto expectedJI = MakeExpectedArrayJI<T, resultSizeJI>();

    auto I = MatrixStatic<T, rowI, columnI>(arrayI);
    auto J = MatrixRowStatic<T, rowJ>(columnJ, arrayJ);

    auto acceptableErrorJI = MakeAcceptableErrorArrayJI<T, resultSizeJI>();
    for (size_t i = 0; i < resultSizeJI; i++) {
        acceptableErrorJI[i] *= std::numeric_limits<T>::epsilon();
    }

    auto JI = J.Dot(I);

    auto typeCheckJI = std::is_same<decltype(JI), MatrixStatic<T, rowJ, columnI>>::value;
    BOOST_CHECK(typeCheckJI);
    CheckCloseEachArrayElement(JI.Elements(), expectedJI, acceptableErrorJI);

    auto K = MatrixRowStatic<T, rowI>(columnI, std::vector<T>(arrayI.begin(), arrayI.end()));
    auto arrayL = std::array<T, elementSizeJ>();
    std::copy_n(arrayJ.begin(), elementSizeJ, arrayL.begin());
    auto L = MatrixStatic<T, rowJ, columnJ>(arrayL);

    auto LK = L.Dot(K);

    auto typeCheckKL = std::is_same<decltype(LK), MatrixRowStatic<T, rowJ>>::value;
    BOOST_CHECK(typeCheckKL);
    CheckCloseEachArrayElement(VectorToArray<T, resultSizeJI>(LK.Elements()), expectedJI, acceptableErrorJI);

    // throw exception test
    auto mArray = MakeVectorM<T>();
    auto nArray = MakeArrayN<T, 3 * 3>();
    auto M = MatrixRowStatic<T, 3>(2, mArray);
    auto N = MatrixStatic<T, 3, 3>(nArray);

    BOOST_CHECK_THROW(M.Dot(N), std::domain_error);
    BOOST_CHECK_NO_THROW(N.Dot(M));
}
}  // namespace matrix_test_matrix_product