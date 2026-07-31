#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_row_static.hpp"
#include "tests/test_matrix_sub_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixRowStatic;

namespace matrix_test_sub {
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_matrix_row_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeVectorA<T>();
    auto arrayB = MakeVectorB<T>();

    auto A = MatrixRowStatic<T, rowAAndB>(columnAAndB, arrayA);
    auto B = MatrixRowStatic<T, rowAAndB>(columnAAndB, arrayB);

    auto expectedAMinusB = MakeExpectedVectorAAndB<T>();

    auto acceptableErrorAMinusB = MakeAcceptableErrorVectorAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAMinusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AMinusB = A - B;

    auto typeCheckAMinusB = std::is_same<decltype(AMinusB), MatrixRowStatic<T, rowAAndB>>::value;
    BOOST_CHECK(typeCheckAMinusB);
    CheckCloseEachVectorElement<T>(AMinusB.Elements(), expectedAMinusB, acceptableErrorAMinusB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeVectorC<T>();
    auto arrayD = MakeVectorD<T>();

    auto C = MatrixRowStatic<T, rowCAndD>(columnCAndD, arrayC);
    auto D = MatrixRowStatic<T, rowCAndD>(columnCAndD, arrayD);

    auto expectedCMinusD = MakeExpectedVectorCAndD<T>();

    auto acceptableErrorCMinusD = MakeAcceptableErrorVectorCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCMinusD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CMinusD = C - D;

    auto typeCheckCMinusD = std::is_same<decltype(CMinusD), MatrixRowStatic<T, rowCAndD>>::value;
    BOOST_CHECK(typeCheckCMinusD);
    CheckCloseEachVectorElement<T>(CMinusD.Elements(), expectedCMinusD, acceptableErrorCMinusD);

    // throw exception test
    auto arrayE = MakeVectorE<T>();
    auto arrayF = MakeVectorF<T>();
    auto arrayG = MakeVectorG<T>();
    auto arrayH = MakeVectorH<T>();

    auto E = MatrixRowStatic<T, 3>(2, arrayE);
    auto F = MatrixRowStatic<T, 2>(3, arrayF);
    auto G = MatrixRowStatic<T, 2>(3, arrayG);
    auto H = MatrixRowStatic<T, 2>(2, arrayH);

    // BOOST_CHECK_THROW(E - F, std::domain_error);
    // BOOST_CHECK_THROW(F - E, std::domain_error);
    BOOST_CHECK_NO_THROW(F - G);
    BOOST_CHECK_NO_THROW(G - F);
    // BOOST_CHECK_THROW(E - H, std::domain_error);
    // BOOST_CHECK_THROW(H - E, std::domain_error);
    BOOST_CHECK_THROW(F - H, std::domain_error);
    BOOST_CHECK_THROW(H - F, std::domain_error);
}
}  // namespace matrix_test_add