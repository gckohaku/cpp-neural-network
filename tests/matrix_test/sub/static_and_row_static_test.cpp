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
#include "tests/test_matrix_sub_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixStatic;
using mknnlib::matrix::MatrixRowStatic;

namespace matrix_test_sub {
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_matrix_static_and_row_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeArrayA<T>();
    auto arrayB = MakeVectorB<T>();

    auto A = MatrixStatic<T, rowAAndB, columnAAndB>(arrayA);
    auto B = MatrixRowStatic<T, rowAAndB>(columnAAndB, arrayB);

    auto expectedAPlusB = MakeExpectedVectorAAndB<T>();

    auto acceptableErrorAPlusB = MakeAcceptableErrorVectorAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorAPlusB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto APlusB = A - B;
    auto BPlusA = B - A;

    auto typeCheckAPlusB = std::is_same<decltype(APlusB), MatrixStatic<T, rowAAndB, columnAAndB>>::value;
    auto typeCheckBPlusA = std::is_same<decltype(BPlusA), MatrixStatic<T, rowAAndB, columnAAndB>>::value;
    BOOST_CHECK(typeCheckAPlusB);
    BOOST_CHECK(typeCheckBPlusA);
    CheckCloseEachVectorElement<T>(ArrayToVector(APlusB.Elements()), expectedAPlusB, acceptableErrorAPlusB);
    CheckCloseEachVectorElement<T>(ArrayToVector(BPlusA.Elements()), expectedAPlusB, acceptableErrorAPlusB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeArrayC<T>();
    auto arrayD = MakeVectorD<T>();

    auto C = MatrixStatic<T, rowCAndD, columnCAndD>(arrayC);
    auto D = MatrixRowStatic<T, rowCAndD>(columnCAndD, arrayD);

    auto expectedCPlusD = MakeExpectedVectorCAndD<T>();

    auto acceptableErrorCPlusD = MakeAcceptableErrorVectorCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCPlusD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CPlusD = C - D;
    auto DPlusC = D - C;

    auto typeCheckCPlusD = std::is_same<decltype(CPlusD), MatrixStatic<T, rowCAndD, columnCAndD>>::value;
    auto typeCheckDPlusC = std::is_same<decltype(DPlusC), MatrixStatic<T, rowCAndD, columnCAndD>>::value;
    BOOST_CHECK(typeCheckCPlusD);
    BOOST_CHECK(typeCheckDPlusC);
    CheckCloseEachVectorElement<T>(ArrayToVector(CPlusD.Elements()), expectedCPlusD, acceptableErrorCPlusD);
    CheckCloseEachVectorElement<T>(ArrayToVector(DPlusC.Elements()), expectedCPlusD, acceptableErrorCPlusD);

    // throw exception test
    // auto arrayE = MakeArrayE<T>();
    auto vectorF = MakeVectorF<T>();
    auto arrayG = MakeArrayG<T>();
    auto arrayH = MakeArrayH<T>();
    auto vectorH = MakeVectorH<T>();

    // auto staticE = MatrixStatic<T, 3, 2>(arrayE);
    auto rowStaticF = MatrixRowStatic<T, 2>(3, vectorF);
    auto staticG = MatrixStatic<T, 2, 3>(arrayG);
    auto rowStaticH = MatrixRowStatic<T, 2>(2, vectorH);
    auto staticH = MatrixStatic<T, 2, 2>(arrayH);

    // BOOST_CHECK_THROW(staticE - rowStaticF, std::domain_error);
    // BOOST_CHECK_THROW(rowStaticF - staticE, std::domain_error);
    BOOST_CHECK_NO_THROW(rowStaticF - staticG);
    BOOST_CHECK_NO_THROW(staticG - rowStaticF);
    // BOOST_CHECK_THROW(staticE - rowStaticH, std::domain_error);
    // BOOST_CHECK_THROW(rowStaticH - staticE, std::domain_error);
    BOOST_CHECK_THROW(rowStaticF - staticH, std::domain_error);
    BOOST_CHECK_THROW(staticH - rowStaticF, std::domain_error);
}
}  // namespace matrix_test_add