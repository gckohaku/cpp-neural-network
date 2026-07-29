#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_mul_array_defines.hpp"
#include "tests/test_type_defines.hpp"

using mknnlib::matrix::MatrixStatic;

namespace matrix_test_mul {
BOOST_AUTO_TEST_CASE_TEMPLATE(mul_matrix_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeArrayA<T>();
    auto arrayB = MakeArrayB<T>();

    auto A = MatrixStatic<T, rowAAndB, columnAAndB>(arrayA);
    auto B = MatrixStatic<T, rowAAndB, columnAAndB>(arrayB);

    auto expectedAPlusB = MakeExpectedArrayAAndB<T>();

    auto acceptableErrorAPlusB = MakeAcceptableErrorArrayAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAPlusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto APlusB = A * B;

    auto typeCheckAPlusB = std::is_same<decltype(APlusB), MatrixStatic<T, rowAAndB, columnAAndB>>::value;
    BOOST_CHECK(typeCheckAPlusB);
    CheckCloseEachArrayElement<T, elementSizeAAndB>(APlusB.Elements(), expectedAPlusB, acceptableErrorAPlusB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeArrayC<T>();
    auto arrayD = MakeArrayD<T>();

    auto C = MatrixStatic<T, rowCAndD, columnCAndD>(arrayC);
    auto D = MatrixStatic<T, rowCAndD, columnCAndD>(arrayD);

    auto expectedCPlusD = MakeExpectedArrayCAndD<T>();

    auto acceptableErrorCPlusD = MakeAcceptableErrorArrayCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCPlusD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CPlusD = C * D;

    auto typeCheckCPlusD = std::is_same<decltype(CPlusD), MatrixStatic<T, rowCAndD, columnCAndD>>::value;
    BOOST_CHECK(typeCheckCPlusD);
    CheckCloseEachArrayElement<T, elementSizeCAndD>(CPlusD.Elements(), expectedCPlusD, acceptableErrorCPlusD);

    // throw exception test
    // auto arrayE = MakeArrayE<T>();
    auto arrayF = MakeArrayF<T>();
    auto arrayG = MakeArrayG<T>();

    // auto E = MatrixStatic<T, 3, 2>(arrayE);
    auto F = MatrixStatic<T, 3, 3>(arrayF);
    auto G = MatrixStatic<T, 3, 3>(arrayG);

    // BOOST_CHECK_THROW(E * F, std::domain_error);
    // BOOST_CHECK_THROW(F * E, std::domain_error);
    BOOST_CHECK_NO_THROW(F * G);
    BOOST_CHECK_NO_THROW(G * F);
}
}  // namespace matrix_test_mul