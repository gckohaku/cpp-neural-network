#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/interface.hpp>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/matrices/matrix_row_static.hpp"
#include "tests/test_matrix_mul_vector_defines.hpp"
#include "tests/test_type_defines.hpp"
#include "tests/test_utilities.hpp"

using mknnlib::matrix::MatrixRowStaticOpen;

namespace matrix_test_mul {
BOOST_AUTO_TEST_CASE_TEMPLATE(mul_matrix_row_static_test, T, CheckMatrixElementType) {
    // 3x3
    constexpr size_t rowAAndB = 3;
    constexpr size_t columnAAndB = 3;
    constexpr size_t elementSizeAAndB = rowAAndB * columnAAndB;

    auto arrayA = MakeVectorA<T>();
    auto arrayB = MakeVectorB<T>();

    auto A = MatrixRowStaticOpen<T, rowAAndB>(columnAAndB, arrayA);
    auto B = MatrixRowStaticOpen<T, rowAAndB>(columnAAndB, arrayB);

    auto expectedATimesB = MakeExpectedVectorAAndB<T>();

    auto acceptableErrorATimesB = MakeAcceptableErrorVectorAAndB<T>();

    for (size_t i = 0; i < elementSizeAAndB; i++) {
        acceptableErrorATimesB[i] *= std::numeric_limits<T>::epsilon();
    }

    auto ATimesB = A * B;

    auto typeCheckATimesB = std::is_same<decltype(ATimesB), MatrixRowStaticOpen<T, rowAAndB>>::value;
    BOOST_CHECK(typeCheckATimesB);
    CheckCloseEachStorageElement<T>(ATimesB.Elements(), expectedATimesB, acceptableErrorATimesB);

    // 2x5
    constexpr size_t rowCAndD = 2;
    constexpr size_t columnCAndD = 5;
    constexpr size_t elementSizeCAndD = rowCAndD * columnCAndD;

    auto arrayC = MakeVectorC<T>();
    auto arrayD = MakeVectorD<T>();

    auto C = MatrixRowStaticOpen<T, rowCAndD>(columnCAndD, arrayC);
    auto D = MatrixRowStaticOpen<T, rowCAndD>(columnCAndD, arrayD);

    auto expectedCTimesD = MakeExpectedVectorCAndD<T>();

    auto acceptableErrorCTimesD = MakeAcceptableErrorVectorCAndD<T>();
    for (size_t i = 0; i < elementSizeCAndD; i++) {
        acceptableErrorCTimesD[i] *= std::numeric_limits<T>::epsilon();
    }

    auto CTimesD = C * D;

    auto typeCheckCTimesD = std::is_same<decltype(CTimesD), MatrixRowStaticOpen<T, rowCAndD>>::value;
    BOOST_CHECK(typeCheckCTimesD);
    CheckCloseEachStorageElement<T>(CTimesD.Elements(), expectedCTimesD, acceptableErrorCTimesD);

    // throw exception test
    auto arrayE = MakeVectorE<T>();
    auto arrayF = MakeVectorF<T>();
    auto arrayG = MakeVectorG<T>();
    auto arrayH = MakeVectorH<T>();

    auto E = MatrixRowStaticOpen<T, 3>(2, arrayE);
    auto F = MatrixRowStaticOpen<T, 2>(3, arrayF);
    auto G = MatrixRowStaticOpen<T, 2>(3, arrayG);
    auto H = MatrixRowStaticOpen<T, 2>(2, arrayH);

    // BOOST_CHECK_THROW(E * F, std::domain_error);
    // BOOST_CHECK_THROW(F * E, std::domain_error);
    BOOST_CHECK_NO_THROW(F * G);
    BOOST_CHECK_NO_THROW(G * F);
    // BOOST_CHECK_THROW(E * H, std::domain_error);
    // BOOST_CHECK_THROW(H * E, std::domain_error);
    BOOST_CHECK_THROW(F * H, std::domain_error);
    BOOST_CHECK_THROW(H * F, std::domain_error);
}
}  // namespace matrix_test_add