#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_sub_array_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::Matrix;

namespace matrix_test_sub {
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_matrix_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeArrayA<T>();
    auto arrayB = MakeArrayB<T>();

    auto A = Matrix<T, rowAAndB, columnAAndB>(arrayA);
    auto B = Matrix<T, rowAAndB, columnAAndB>(arrayB);

    auto expectedAMinusB = MakeExpectedArrayAAndB<T>();

    auto acceptableErrorAMinusB = MakeAcceptableErrorArrayAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAMinusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto AMinusB = A - B;

    auto typeCheckAMinusB = std::is_same<decltype(AMinusB), Matrix<T, rowAAndB, columnAAndB>>::value;
    BOOST_CHECK(typeCheckAMinusB);
    CheckCloseEachArrayElement<T, elementSizeAAndB>(AMinusB.Elements(), expectedAMinusB, acceptableErrorAMinusB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeArrayC<T>();
    auto arrayD = MakeArrayD<T>();

    auto C = Matrix<T, rowCAndD, columnCAndD>(arrayC);
    auto D = Matrix<T, rowCAndD, columnCAndD>(arrayD);

    auto expectedCMinusD = MakeExpectedArrayCAndD<T>();

    auto acceptableErrorCMinusD = MakeAcceptableErrorArrayCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCMinusD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CMinusD = C - D;

    auto typeCheckCMinusD = std::is_same<decltype(CMinusD), Matrix<T, rowCAndD, columnCAndD>>::value;
    BOOST_CHECK(typeCheckCMinusD);
    CheckCloseEachArrayElement<T, elementSizeCAndD>(CMinusD.Elements(), expectedCMinusD, acceptableErrorCMinusD);

    // throw exception test
    // auto arrayE = MakeArrayE<T>();
    auto arrayF = MakeArrayF<T>();
    auto arrayG = MakeArrayG<T>();

    // auto E = MatrixStaticOpen<T, 3, 2>(arrayE);
    auto F = Matrix<T, 2, 3>(arrayF);
    auto G = Matrix<T, 2, 3>(arrayG);

    // BOOST_CHECK_THROW(E - F, std::domain_error);
    // BOOST_CHECK_THROW(F - E, std::domain_error);
    BOOST_CHECK_NO_THROW(F - G);
    BOOST_CHECK_NO_THROW(G - F);
}
}  // namespace matrix_test_add