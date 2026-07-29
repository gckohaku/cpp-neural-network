#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_static.hpp"
#include "tests/test_matrix_add_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixRowStatic;

namespace matrix_test_add {
BOOST_AUTO_TEST_CASE_TEMPLATE(add_matrix_row_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeVectorA<T>();
    auto arrayB = MakeVectorB<T>();

    auto A = MatrixRowStatic<T, rowAAndB>(columnAAndB, arrayA);
    auto B = MatrixRowStatic<T, rowAAndB>(columnAAndB, arrayB);

    auto expectedAPlusB = MakeExpectedVectorAAndB<T>();

    auto acceptableErrorAPlusB = MakeAcceptableErrorVectorAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAPlusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto APlusB = A + B;

    auto typeCheckAPlusB = std::is_same<decltype(APlusB), MatrixRowStatic<T, rowAAndB>>::value;
    BOOST_CHECK(typeCheckAPlusB);
    CheckCloseEachVectorElement<T>(APlusB.Elements(), expectedAPlusB, acceptableErrorAPlusB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeVectorC<T>();
    auto arrayD = MakeVectorD<T>();

    auto C = MatrixRowStatic<T, rowCAndD>(columnCAndD, arrayC);
    auto D = MatrixRowStatic<T, rowCAndD>(columnCAndD, arrayD);

    auto expectedCPlusD = MakeExpectedVectorCAndD<T>();

    auto acceptableErrorCPlusD = MakeAcceptableErrorVectorCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCPlusD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CPlusD = C + D;

    auto typeCheckCPlusD = std::is_same<decltype(CPlusD), MatrixRowStatic<T, rowCAndD>>::value;
    BOOST_CHECK(typeCheckCPlusD);
    CheckCloseEachVectorElement<T>(CPlusD.Elements(), expectedCPlusD, acceptableErrorCPlusD);

    // throw exception test
    auto arrayE = MakeVectorE<T>();
    auto arrayF = MakeVectorF<T>();
    auto arrayG = MakeVectorG<T>();

    auto E = MatrixRowStatic<T, 3>(2, arrayE);
    auto F = MatrixRowStatic<T, 3>(3, arrayF);
    auto G = MatrixRowStatic<T, 3>(3, arrayG);

    BOOST_CHECK_THROW(E + F, std::domain_error);
    BOOST_CHECK_THROW(F + E, std::domain_error);
    BOOST_CHECK_NO_THROW(F + G);
    BOOST_CHECK_NO_THROW(G + F);
}
}  // namespace matrix_test_add